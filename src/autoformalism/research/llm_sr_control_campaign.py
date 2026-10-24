"""Frozen two-repetition T2-easy numerical controls, with no native execution.

The same training observations and supplied coefficients are replayed twice.
Every operation has a durable budget reservation; interrupted work cannot obtain
a fresh allowance. Neither accuracy nor solver agreement selects a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit, Sha256

PROTOCOL = "llm-sr-numerical-controls-1"
BENCHMARK = "phase_c_dalla_man_t2_canonical_named_easy_rates_v1"


class Task(StrictSchema):
    """One preselected repetition, including an explicit missing-transcription row."""

    task: int
    benchmark_id: Literal["phase_c_dalla_man_t2_canonical_named_easy_rates_v1"]
    repetition: int
    programs: dict[Identifier, control.SourceProgram]
    unavailable_reason: str | None = Field(default=None, min_length=1)


class Inputs(StrictSchema):
    """Public training, exact native source bindings, and supplied expression data."""

    protocol: Literal["llm-sr-numerical-controls-input-1"]
    source_plan_sha256: Sha256
    transcription_set: control.TranscriptionSet
    context: ValidationContext
    training: PublicSplit
    tasks: tuple[Task, Task]

    @model_validator(mode="after")
    def fixed_roster(self) -> Inputs:
        """Freeze both same-cell repetitions before computing any error."""
        if [(t.task, t.repetition) for t in self.tasks] != [(4, 0), (5, 1)]:
            raise ValueError("controls require tasks 4 and 5, repetitions 0 and 1")
        if self.context.targets != ("Gp", "I", "U"):
            raise ValueError("controls require the complete T2-easy target roster")
        if self.source_plan_sha256 != self.transcription_set.source_plan_sha256:
            raise ValueError("transcription source plan differs")
        control.checked_training(self.training, self.context)
        channels = (
            *self.context.targets,
            *self.context.auxiliaries,
            *self.context.external_inputs,
            *self.context.fixed_covariates,
        )
        for task in self.tasks:
            item = self.transcription_set.for_task(task.task)
            if item is None:
                if task.programs or task.unavailable_reason is None:
                    raise ValueError(
                        "missing transcription needs an explicit unavailable row"
                    )
                continue
            if task.unavailable_reason is not None:
                raise ValueError("available transcription cannot be marked unavailable")
            if (item.benchmark_id, item.repetition) != (
                task.benchmark_id,
                task.repetition,
            ):
                raise ValueError("transcription cell or repetition differs")
            control.bind_programs(item, task.programs)
            if any(program.variables != channels for program in task.programs.values()):
                raise ValueError(
                    "native variable order differs from the public context"
                )
            control.compile_transcription(item, self.context)
        return self


class Policy(StrictSchema):
    """An independent solver/tolerance ladder with bounded per-stage budgets."""

    protocol: Literal["llm-sr-numerical-control-policy-1"] = (
        "llm-sr-numerical-control-policy-1"
    )
    primary: control.ReplaySettings = Field(default_factory=control.ReplaySettings)
    verification: control.ReplaySettings = Field(
        default_factory=lambda: control.ReplaySettings(
            method="DOP853",
            relative_tolerance=1e-9,
            absolute_tolerance=1e-11,
            seconds=300,
        )
    )
    solver_agreement_tolerance: FiniteFloat = Field(default=1e-5, gt=0, le=1e-3)

    @model_validator(mode="after")
    def independent_ladder(self) -> Policy:
        """Do not silently replace the second solver with the same numerical call."""
        if self.primary.method != "Radau" or self.verification.method != "DOP853":
            raise ValueError("verification requires Radau then independent DOP853")
        if (
            self.verification.relative_tolerance >= self.primary.relative_tolerance
            or self.verification.absolute_tolerance >= self.primary.absolute_tolerance
        ):
            raise ValueError("verification must tighten both tolerances")
        return self


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None) -> dict:
    """Seal all inputs/settings and the fixed roster without a numerical replay."""
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = Policy.model_validate((policy or Policy()).model_dump(mode="json"))
    body = {
        "protocol": PROTOCOL,
        "inputs": inputs.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "input_sha256": public.content_sha256(inputs),
        "training_content_sha256": public.content_sha256(inputs.training),
        "transcription_config_sha256": public.content_sha256(inputs.transcription_set),
        "refit_ready": False,
        "refit_limitation": control.REFIT_LIMIT,
        "interpretation": (
            "ODE transcription approximation; native equivalence unproven"
        ),
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError("cannot prepare a nonempty unsealed control directory")
        seal(root / "plan.json", body)
    return report(root)


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, Inputs, Policy]:
    """Authenticate the entire input and the exact executing source/runtime."""
    plan = read_seal(root / "plan.json")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("unknown numerical-control protocol")
    inputs, policy = (
        Inputs.model_validate(plan["inputs"]),
        Policy.model_validate(plan["policy"]),
    )
    if (
        plan["input_sha256"] != public.content_sha256(inputs)
        or plan["training_content_sha256"] != public.content_sha256(inputs.training)
        or plan["transcription_config_sha256"]
        != public.content_sha256(inputs.transcription_set)
    ):
        raise ValueError("control input identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("control execution source/runtime differs")
    return plan, inputs, policy


def _identity(plan: dict, task: Task) -> dict:
    return {
        "plan_sha256": public.content_sha256(plan),
        "task": task.task,
        "benchmark_id": task.benchmark_id,
        "repetition": task.repetition,
    }


def _agreement(operations: dict, policy: Policy) -> dict:
    """Compare every predicted training target; never compare native scores."""
    values = [operations[name].get("value") for name in ("primary", "verification")]
    if any(
        operation["status"] != "complete" for operation in operations.values()
    ) or any(value is None or value["status"] != "complete" for value in values):
        return {"available": False, "verified": False, "reason": "incomplete replay"}
    primary, verification = values
    maxima = dict.fromkeys(primary["scales"], 0.0)
    for first, second in zip(
        primary["trajectories"], verification["trajectories"], strict=True
    ):
        if first["trajectory_id"] != second["trajectory_id"]:
            raise ValueError("replay trajectory roster differs")
        for target, scale in primary["scales"].items():
            a, b = (
                np.asarray(first["predictions"][target]),
                np.asarray(second["predictions"][target]),
            )
            maxima[target] = max(maxima[target], float(np.max(np.abs(a - b) / scale)))
    return {
        "available": True,
        "verified": max(maxima.values()) <= policy.solver_agreement_tolerance,
        "per_target_maximum_scaled_difference": maxima,
        "tolerance": policy.solver_agreement_tolerance,
        "scope": "two solvers at supplied coefficients; no native equivalence claim",
    }


def run_task(root: Path, index: int) -> dict:
    """Run one frozen roster position once; a resumed interruption stays charged."""
    plan, inputs, policy = verify(root)
    if index not in (0, 1):
        raise ValueError("task index must be 0 or 1")
    task = inputs.tasks[index]
    identity = _identity(plan, task)
    directory = root / "tasks" / str(index)
    with public._lock(directory):
        result_path = directory / "result.json"
        if result_path.exists():
            return _task_result(directory, identity)
        base = {
            "identity": identity,
            "native_program_executed": False,
            "native_equivalence_certified": False,
            "validation_or_test_access": False,
            "optimizer_calls": 0,
            "refit_ready": False,
            "refit_limitation": control.REFIT_LIMIT,
            "model_selected": False,
        }
        item = inputs.transcription_set.for_task(task.task)
        if item is None:
            result = base | {
                "status": "unavailable_transcription",
                "operations": {},
                "reason": task.unavailable_reason,
                "stage_cost": journal._cost({}),
            }
        else:
            operations = {}
            for name, settings in (
                ("primary", policy.primary),
                ("verification", policy.verification),
            ):
                operations[name] = journal._operation(
                    directory,
                    identity,
                    name,
                    settings.seconds,
                    lambda save, settings=settings: control.replay_training(
                        item,
                        inputs.context,
                        inputs.training,
                        settings,
                        checkpoint=save,
                    ),
                )
            agreement = _agreement(operations, policy)
            result = base | {
                "status": "replay_verified"
                if agreement["verified"]
                else "replay_unverified",
                "transcription_review_status": inputs.transcription_set.status,
                "operations": {
                    name: public.content_sha256(value)
                    for name, value in operations.items()
                },
                "solver_agreement": agreement,
                "stage_cost": journal._cost(operations),
                "interpretation": (
                    "Supplied ODE approximation only; native scores do not transfer."
                ),
            }
        seal(result_path, result)
        return _task_result(directory, identity)


def _task_result(directory: Path, identity: dict) -> dict:
    """Authenticate operation receipts before reporting a retained task."""
    result = read_seal(directory / "result.json")
    if result["identity"] != identity:
        raise ValueError("control task identity differs")
    stages = {}
    for name, digest in result["operations"].items():
        folder = directory / "operations" / name
        bound = read_seal(folder / "started.json")
        if bound["campaign"] != identity or bound["stage"] != name:
            raise ValueError("control stage identity differs")
        operation = journal._read_operation(folder, bound)
        if public.content_sha256(operation) != digest:
            raise ValueError("control stage digest differs")
        value = operation.get("value") or {}
        stages[name] = {
            "operation_status": operation["status"],
            "replay_status": value.get("status"),
            "normalized_mse": value.get("normalized_mse"),
            "per_target_normalized_mse": value.get("per_target_normalized_mse", {}),
            "failed_trajectories": value.get("failed_trajectories", {}),
            "budget_exhausted": value.get("budget_exhausted"),
            "wall_seconds": operation["wall_seconds"],
            "cpu_seconds": operation["cpu_seconds"],
            "budget_charge_seconds": operation["budget_charge_seconds"],
            "budget_restarted": operation["budget_restarted"],
            "per_trajectory_diagnostics": [
                {k: v for k, v in row.items() if k != "predictions"}
                for row in value.get("trajectories", [])
            ],
        }
    return result | {"stages": stages}


def report(root: Path) -> dict:
    """Return both frozen repetitions without dropping failure/unavailable rows."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for index, task in enumerate(inputs.tasks):
        identity = _identity(plan, task)
        directory = root / "tasks" / str(index)
        rows.append(
            _task_result(directory, identity)
            if (directory / "result.json").exists()
            else {"identity": identity, "status": "not_started"}
        )
    return {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "training_content_sha256": plan["training_content_sha256"],
        "policy": policy.model_dump(mode="json"),
        "tasks": rows,
        "status": "complete"
        if all(row["status"] != "not_started" for row in rows)
        else "pending",
        "refit_ready": False,
        "refit_limitation": control.REFIT_LIMIT,
        "model_selection_performed": False,
        "validation_or_test_access": False,
    }
