"""Training-only fixed-coefficient controls for supplied LLM-SR ODE transcriptions.

The native Python programs are identity evidence only and are never executed.
Filter/lag/Euler transcriptions approximate discrete programs; their native scores
do not transfer. Coefficient branches were frozen at the supplied values, so this
milestone deliberately has no optimizer or parameter-override interface.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler
from autoformalism.expressions import (
    CompiledModel,
    RuntimeExpressionError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import (
    simulate_trajectory,
    trajectory_forcing,
    trajectory_initial_state,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit, Sha256

PROTOCOL = "llm-sr-fixed-transcription-replay-1"
REFIT_LIMIT = (
    "Refit deferred: the supplied transcription resolves native coefficient "
    "branches only at its sealed values; no reviewed parameter domain preserves "
    "every branch, clamp, and dropped guard. A new domain contract is required."
)


class ProgramBinding(StrictSchema):
    """Byte-exact native source digest and native fitted coefficient vector."""

    function_sha256: Sha256
    parameters: tuple[FiniteFloat, ...] = Field(min_length=1, max_length=64)


class SourceProgram(StrictSchema):
    """Source text is hashed as data, without parsing or executing Python."""

    function: str = Field(min_length=1, max_length=200_000)
    parameters: tuple[FiniteFloat, ...] = Field(min_length=1, max_length=64)
    variables: tuple[Identifier, ...] = Field(min_length=1)

    # StrictSchema strips strings, which would invalidate a byte-exact binding.
    model_config = ConfigDict(str_strip_whitespace=False)


class HiddenState(StrictSchema):
    """One supplied recursion approximation and its causal initial expression."""

    rhs: str = Field(min_length=1, max_length=4096)
    initial: str = Field(min_length=1, max_length=4096)
    program_name: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class Transcription(StrictSchema):
    """Restricted expression data for one frozen native selection."""

    task: int = Field(ge=0)
    benchmark_id: str = Field(min_length=1)
    repetition: int = Field(ge=0)
    programs: dict[Identifier, ProgramBinding] = Field(min_length=1)
    parameters: dict[Identifier, FiniteFloat] = Field(default_factory=dict)
    processes: dict[Identifier, str] = Field(default_factory=dict)
    hidden_states: dict[Identifier, HiddenState] = Field(default_factory=dict)
    targets: dict[Identifier, str] = Field(min_length=1)
    rules: tuple[str, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def distinct_names(self) -> Transcription:
        """Reject ambiguous symbol rosters and unbound target equations."""
        if set(self.targets) != set(self.programs):
            raise ValueError("each target needs exactly one native program binding")
        names = [*self.parameters, *self.processes, *self.hidden_states, *self.targets]
        if len(names) != len(set(names)):
            raise ValueError("transcription symbol names must be distinct")
        if not {h.rule for h in self.hidden_states.values()} <= set(self.rules):
            raise ValueError("hidden state uses an undeclared transcription rule")
        return self


class TranscriptionSet(StrictSchema):
    """The supplied config, retaining its actual review status and source plan."""

    protocol: Literal["phase-c-llm-sr-ode-transcriptions-1"]
    status: Literal["proposed_pending_review", "reviewed"]
    source_run: str = Field(min_length=1)
    source_plan_sha256: Sha256
    rules: dict[str, str] = Field(min_length=1)
    transcriptions: tuple[Transcription, ...]

    @model_validator(mode="after")
    def unique_tasks(self) -> TranscriptionSet:
        """Every task is unique and uses only declared rules."""
        tasks = [item.task for item in self.transcriptions]
        if len(tasks) != len(set(tasks)):
            raise ValueError("duplicate transcription task")
        if any(set(item.rules) - set(self.rules) for item in self.transcriptions):
            raise ValueError("transcription uses undeclared rules")
        return self

    def for_task(self, task: int) -> Transcription | None:
        """Return the supplied item without substituting a different repetition."""
        return next((item for item in self.transcriptions if item.task == task), None)


class ReplaySettings(StrictSchema):
    """Bound one full-training replay; these are solver settings, not fit knobs."""

    method: Literal["Radau", "DOP853"] = "Radau"
    relative_tolerance: FiniteFloat = Field(default=1e-7, gt=0, le=1e-5)
    absolute_tolerance: FiniteFloat = Field(default=1e-9, gt=0, le=1e-7)
    seconds: FiniteFloat = Field(default=180, gt=0, le=300)


def load_transcriptions(path: Path) -> TranscriptionSet:
    """Load expression data with strict schemas, never as importable Python."""
    return TranscriptionSet.model_validate(json.loads(path.read_text("utf-8")))


def bind_programs(item: Transcription, programs: Mapping[str, SourceProgram]) -> None:
    """Require exactly the supplied native source and native coefficients."""
    if set(item.programs) != set(programs):
        raise ValueError("native program target roster differs")
    for target, binding in item.programs.items():
        source = programs[target]
        digest = hashlib.sha256(source.function.encode("utf-8")).hexdigest()
        if digest != binding.function_sha256 or source.parameters != binding.parameters:
            raise ValueError(f"{target}: native source or coefficients differ")


def compile_transcription(
    item: Transcription, context: ValidationContext
) -> CompiledModel:
    """Compile only restricted expressions, preserving the supplied initializer."""
    if tuple(item.targets) != context.targets:
        raise ValueError("transcription target order differs from the public context")
    if context.lagged_targets:
        raise ValueError("fixed replay forbids measured target forcing")
    context = ValidationContext.model_validate(
        context.model_dump() | {"fitted_initialization": True}
    )
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": f"llm_sr_task{item.task}_supplied_transcription",
            "parent_candidate_id": None,
            "change_summary": "Supplied ODE transcription; native equivalence unproven",
            "states": [{"name": name, "kind": "observed"} for name in item.targets]
            + [{"name": name, "kind": "latent"} for name in item.hidden_states],
            "processes": [
                {"name": name, "expression": expression}
                for name, expression in item.processes.items()
            ],
            "state_equations": [
                {"state": name, "rhs": expression}
                for name, expression in item.targets.items()
            ]
            + [
                {"state": name, "rhs": state.rhs}
                for name, state in item.hidden_states.items()
            ],
            "observation_mappings": [
                {"channel": name, "expression": name} for name in item.targets
            ],
            "parameters": [
                {"name": name, "scope": "global"} for name in item.parameters
            ],
            "initial_conditions": [
                {"state": name, "scope": "global", "expression": name}
                for name in item.targets
            ]
            + [
                {"state": name, "scope": "global", "expression": state.initial}
                for name, state in item.hidden_states.items()
            ],
        }
    )
    return compile_candidate(candidate, context)


def checked_training(
    training: DatasetSplit | PublicSplit, context: ValidationContext
) -> tuple[PublicSplit, DatasetSplit]:
    """Reject nontraining data before packing; validate every channel and row."""
    if isinstance(training, DatasetSplit):
        if training.name is not SplitName.TRAIN:
            raise ValueError("numerical controls accept training only")
        training = public.pack_split(training)
    training = PublicSplit.model_validate(training.model_dump(mode="json"))
    if training.name != "train":
        raise ValueError("numerical controls accept training only")
    for row in training.rows:
        for role in ("targets", "auxiliaries", "external_inputs", "fixed_covariates"):
            if set(getattr(row, role)) != set(getattr(context, role)):
                raise ValueError(f"{row.trajectory_id}: {role} differ from context")
    return training, public.unpack_split(training)


def _denominator_audit(model, row, parameters, states, deadline) -> dict:
    """Inspect accepted grid values; this is not a continuous-domain certificate."""
    denominators = {}
    for prefix, expressions in (
        ("process", model.validated.process_expressions),
        ("rhs", model.validated.equation_expressions),
    ):
        for name, expression in expressions.items():
            for i, node in enumerate(ast.walk(expression.tree)):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    denominators[f"{prefix}:{name}:{i}"] = node.right
    summaries = {
        name: {
            "expression": ast.unparse(node),
            "minimum_absolute_value": None,
            "guarded_samples": 0,
            "sign_changes": 0,
        }
        for name, node in denominators.items()
    }
    previous = {}
    forcing = trajectory_forcing(model, row)
    count = 0
    for index, time in enumerate(row.time[: states.shape[1]]):
        if monotonic() >= deadline:
            break
        for name, node in denominators.items():
            value = float(
                model.evaluate_expression(
                    ParsedExpression(
                        ast.unparse(node), ast.Expression(node), frozenset()
                    ),
                    float(time),
                    states[:, index],
                    parameters,
                    forcing,
                )
            )
            summary = summaries[name]
            minimum = summary["minimum_absolute_value"]
            summary["minimum_absolute_value"] = (
                abs(value) if minimum is None else min(minimum, abs(value))
            )
            summary["guarded_samples"] += int(abs(value) < SAFE_DIVISION_EPSILON)
            summary["sign_changes"] += int(
                name in previous and value * previous[name] < 0
            )
            previous[name] = value
        count += 1
    return {
        "scope": "accepted output grid; failed rollout checks initial point only",
        "continuous_domain_certified": False,
        "sampled_points": count,
        "samples_complete": count == states.shape[1],
        "denominators": summaries,
        "possible_singularity": any(
            row["guarded_samples"] or row["sign_changes"] for row in summaries.values()
        ),
    }


def replay_training(
    item: Transcription,
    context: ValidationContext,
    training: DatasetSplit | PublicSplit,
    settings: ReplaySettings,
    *,
    checkpoint: Callable[[dict], None] | None = None,
    model_factory: Callable[
        [Transcription, ValidationContext], CompiledModel
    ] = compile_transcription,
) -> dict:
    """Replay the supplied values once with no optimizer or fitted initial states."""
    settings = ReplaySettings.model_validate(settings.model_dump(mode="json"))
    wall, cpu = monotonic(), process_time()
    deadline = wall + settings.seconds
    packed, split = checked_training(training, context)
    model = model_factory(item, context)
    scaler = TrainingScaler().fit(split)
    scales = {
        t: scaler.scales[f"target:{t}"].standard_deviation for t in context.targets
    }
    config = FitConfig(
        integration_backend="solve_ivp",
        integration_method=settings.method,
        relative_tolerance=settings.relative_tolerance,
        absolute_tolerance=settings.absolute_tolerance,
        maximum_wall_time_seconds=settings.seconds,
        allow_derivative_regression=False,
    )
    squared = {target: [] for target in context.targets}
    rows = []
    failures = {}
    for row in split.trajectories:
        record = {"trajectory_id": row.trajectory_id, "predictions": {}}
        try:
            simulation = simulate_trajectory(
                model,
                row,
                item.parameters,
                {},
                config,
                deadline=deadline,
                reset_observed_states=False,
            )
            if not simulation.success:
                raise RuntimeError(simulation.message or "integration failed")
            with np.errstate(over="ignore", invalid="ignore"):
                pieces = {
                    t: ((simulation.predictions[t] - row.targets[t]) / scales[t]) ** 2
                    for t in context.targets
                }
            if any(not np.isfinite(piece).all() for piece in pieces.values()):
                raise ArithmeticError("nonfinite normalized prediction error")
            record.update(
                status="complete",
                predictions={
                    t: simulation.predictions[t].tolist() for t in context.targets
                },
                per_target_normalized_mse={
                    t: float(np.mean(v)) for t, v in pieces.items()
                },
            )
            for target, piece in pieces.items():
                squared[target].append(piece)
            states = simulation.states
        except (TimeoutError, RuntimeError, ArithmeticError, ValueError) as error:
            reason = f"{type(error).__name__}: {error}"
            failures[row.trajectory_id] = reason
            record.update(
                status="failed",
                error=reason,
                per_target_failures=dict.fromkeys(context.targets, reason),
            )
            states = None
        try:
            if states is None:
                states = trajectory_initial_state(
                    model, row, {}, parameters=item.parameters
                )[:, None]
            record["domain_audit"] = _denominator_audit(
                model, row, item.parameters, states, deadline
            )
        except (
            RuntimeError,
            RuntimeExpressionError,
            ArithmeticError,
            ValueError,
        ) as error:
            record["domain_audit"] = {
                "available": False,
                "error": str(error),
                "continuous_domain_certified": False,
            }
        rows.append(record)
        if checkpoint is not None:
            checkpoint(
                {
                    "completed_trajectories": len(rows),
                    "failures": failures,
                    "timing": {
                        "elapsed_wall_seconds": monotonic() - wall,
                        "elapsed_cpu_seconds": process_time() - cpu,
                    },
                }
            )
    scores = (
        {t: float(np.mean(np.concatenate(v))) for t, v in squared.items()}
        if not failures
        else {}
    )
    return {
        "protocol": PROTOCOL,
        "status": "complete" if not failures else "replay_failed",
        "training_content_sha256": public.content_sha256(packed),
        "supplied_parameters": dict(item.parameters),
        "settings": settings.model_dump(),
        "normalized_mse": float(np.mean(list(scores.values()))) if scores else None,
        "per_target_normalized_mse": scores,
        "failed_trajectories": failures,
        "scales": scales,
        "trajectories": rows,
        "budget_exhausted": monotonic() >= deadline,
        "wall_seconds": monotonic() - wall,
        "cpu_seconds": process_time() - cpu,
        "optimizer_calls": 0,
        "refit_ready": False,
        "refit_limitation": REFIT_LIMIT,
        "native_program_executed": False,
        "native_equivalence_certified": False,
        "interpretation": (
            "Supplied ODE approximation only; native scores do not transfer."
        ),
        "validation_or_test_access": False,
    }
