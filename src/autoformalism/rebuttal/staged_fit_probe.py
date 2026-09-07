"""Source-bound numerical probe of one scientifically reviewed staged candidate."""

from __future__ import annotations

import re
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.baselines.raw_data_agent import (
    fit_result_payload,
    raw_agent_validation_context,
)
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, DevelopmentDataset
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.fitter_diagnostic import (
    PUBLIC_FILES,
    _finite_payload,
    _write_bytes,
    read_json,
    replay_parameters,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.staged_topology import content_hash


class StagedFitPlan(StrictSchema):
    """One prespecified candidate and a bounded numerical feasibility budget."""

    protocol: Literal["scientific-staged-fit-1"]
    function_plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    function_task_id: Identifier
    function_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    benchmark_id: str = Field(pattern=r"^phase_b_[a-z0-9_]+$")
    tier: Literal["easy", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fit_config: FitConfig
    screen_seconds: FiniteFloat = Field(gt=0)
    replay_seconds: FiniteFloat = Field(gt=0)
    supervisor_grace_seconds: FiniteFloat = Field(gt=0)
    selection_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def bounded_rollout_only(self) -> StagedFitPlan:
        """Keep this gate separate from optimizer/initializer ablations."""
        settings = self.fit_config
        if (
            settings.maximum_wall_time_seconds is None
            or settings.integration_backend != "solve_ivp"
            or settings.allow_derivative_regression
            or settings.nonlinear_initializer != "none"
            or settings.parameter_fit_strategy != "bounded_nonlinear"
        ):
            raise ValueError("probe requires bounded solve_ivp rollout fitting only")
        return self

    @property
    def worker_seconds(self) -> float:
        """Include replay and numerical-library finalization in the hard cap."""
        return (
            float(self.fit_config.maximum_wall_time_seconds)
            + self.screen_seconds
            + self.replay_seconds
            + self.supervisor_grace_seconds
        )


def launcher_hash() -> str:
    """Bind the supervisor and both scheduler entry points for safe resume."""
    repository = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            name: sha256(repository / name)
            for name in (
                "scripts/run_staged_fit_probe.py",
                "scripts/hpc/staged_fit_probe_delta.slurm",
                "scripts/hpc/submit_staged_fit_probe_delta.sh",
            )
        }
    )


def load_data(
    root: Path, plan: StagedFitPlan
) -> tuple[DevelopmentDataset, ValidationContext]:
    """Read only the frozen train/validation release, never the test table."""
    registry = BenchmarkRegistry()
    spec = registry.get(plan.benchmark_id)
    if spec.one_step_target_history or spec.data_layout != "tidy_split_file":
        raise ValueError("probe requires Phase-B free-rollout development data")
    dataset = BenchmarkLoader(registry).load_development(
        DataConfig(benchmark_id=plan.benchmark_id, tier=plan.tier, root=root / "public")
    )
    return dataset, raw_agent_validation_context(dataset, spec)


def prepare_probe(
    plan: StagedFitPlan,
    function_plan: Path,
    function_results: Path,
    public_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze exact model provenance and public development assets before fitting."""
    parent = read_json(function_plan)
    if (
        parent.get("plan_sha256") != plan.function_plan_sha256
        or content_hash({k: v for k, v in parent.items() if k != "plan_sha256"})
        != plan.function_plan_sha256
    ):
        raise ValueError("function plan digest differs")
    matches = [t for t in parent["tasks"] if t["task_id"] == plan.function_task_id]
    if len(matches) != 1 or matches[0]["kind"] != "benchmark":
        raise ValueError("selected function task is not one public benchmark")
    task = matches[0]
    source_root = function_results / plan.function_task_id
    source = read_json(source_root / "result.json")
    terminal = read_json(source_root / "terminal.json")
    if (
        content_hash(source) != plan.function_result_sha256
        or terminal.get("result") != source
        or terminal.get("identity") != content_hash([plan.function_plan_sha256, task])
        or not source.get("complete_model")
        or source.get("test_data_opened") is not False
        or source.get("private_reference_opened") is not False
        or source.get("parameter_fitting_performed") is not False
    ):
        raise ValueError("selected function result or terminal identity differs")

    relative = Path("phase_b_v1") / plan.benchmark_id
    hashes = {name: sha256(public_root / relative / name) for name in PUBLIC_FILES}
    prompt = (public_root / relative / "proposer_prompt.txt").read_text()
    scientific_text = re.split(r"(?m)^F\.\s+Required response\s*$", prompt, maxsplit=1)
    if (
        hashes["proposer_prompt.txt"] != plan.public_prompt_sha256
        or len(scientific_text) != 2
        or scientific_text[0].rstrip() != task["brief"]["scientific_context"]
    ):
        raise ValueError("public data prompt differs from the generation brief")
    assets: dict[str, str] = {}
    for name in PUBLIC_FILES:
        destination = output / "public" / relative / name
        _write_bytes(
            destination, (public_root / relative / name).read_bytes(), immutable=True
        )
        if sha256(destination) != hashes[name]:
            raise ValueError("public input changed during snapshot")
        assets[str(destination.relative_to(output))] = hashes[name]
    dataset, context = load_data(output, plan)
    candidate = CandidateModel.model_validate(source["candidate"])
    if any(p.scope.value != "global" for p in candidate.parameters):
        raise ValueError("probe requires globally shared parameters")
    if any(initial.fixed_value is None for initial in candidate.initial_conditions):
        raise ValueError("this initial probe preserves explicitly fixed initial values")
    model = compile_candidate(candidate, context)
    if model.validated.context.targets != tuple(task["context"]["targets"]):
        raise ValueError("source and data target channels differ")
    del dataset
    for name, value in (
        ("candidate.json", candidate.model_dump(mode="json")),
        (
            "source_function.json",
            {"task": task, "result": source, "terminal": terminal},
        ),
    ):
        write_json(output / name, value, immutable=True)
        assets[name] = sha256(output / name)
    frozen = {
        "protocol": plan.protocol,
        "plan": plan.model_dump(mode="json"),
        "runtime": runtime_identity(),
        "launcher_sha256": launcher_hash(),
        "assets": assets,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_probe(output: Path) -> dict[str, Any]:
    """Reject changed code, dependencies or inputs before any checkpoint reuse."""
    frozen = read_json(output / "freeze.json")
    StagedFitPlan.model_validate(frozen["plan"])
    if (
        frozen["runtime"] != runtime_identity()
        or frozen["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("runtime or launcher differs from freeze")
    for relative, digest in frozen["assets"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
            raise ValueError(f"frozen asset differs: {relative}")
    return frozen


def read_checkpoint(path: Path, identity: str) -> dict[str, Any] | None:
    """Check the experiment identity before accepting a completed phase."""
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("freeze_sha256") != identity:
        raise ValueError("checkpoint belongs to another frozen probe")
    return value


def execute_probe(output: Path) -> dict[str, Any]:
    """Replay a default vector, fit on train, and independently replay the result."""
    frozen = verify_probe(output)
    plan = StagedFitPlan.model_validate(frozen["plan"])
    identity = sha256(output / "freeze.json")
    existing = read_checkpoint(output / "result.json", identity)
    if existing is not None:
        return existing
    dataset, context = load_data(output, plan)
    candidate = CandidateModel.model_validate(read_json(output / "candidate.json"))
    model = compile_candidate(candidate, context)
    screen_path = output / "default_replay.json"
    screen = read_checkpoint(screen_path, identity)
    if screen is None:
        parameters = {p.name: 1.0 for p in candidate.parameters}
        screen = {
            "freeze_sha256": identity,
            "parameters": parameters,
            "replay": replay_parameters(
                model, dataset, parameters, plan.fit_config, plan.screen_seconds
            ),
        }
        write_json(screen_path, _finite_payload(screen))
    if any(
        screen["replay"][split]["normalized_mse"] is None
        for split in ("train", "validation")
    ):
        result = {
            "freeze_sha256": identity,
            "status": "screen_failed",
            "default_replay": screen,
            "error": "all-one vector did not produce finite complete split rollouts",
            "test_data_opened": False,
            "private_reference_opened": False,
            "llm_calls": 0,
        }
        write_json(output / "result.json", result)
        return result
    fit_path = output / "fit.json"
    fitted = read_checkpoint(fit_path, identity)
    if fitted is None:
        started = monotonic()
        fit = fit_candidate(
            model,
            dataset.train,
            dataset.validation,
            plan.fit_config,
            initial_global_parameters=screen["parameters"],
        )
        fitted = {
            "freeze_sha256": identity,
            "fit_seconds": monotonic() - started,
            "fit": _finite_payload(fit_result_payload(fit)),
        }
        write_json(fit_path, fitted)
    parameters = fitted["fit"]["global_parameters"]
    replay_path = output / "final_replay.json"
    replay_record = read_checkpoint(replay_path, identity)
    if replay_record is None:
        replay_record = {
            "freeze_sha256": identity,
            "replay": replay_parameters(
                model, dataset, parameters, plan.fit_config, plan.replay_seconds
            ),
        }
        write_json(replay_path, _finite_payload(replay_record))
    replay = replay_record["replay"]
    complete = all(
        replay[s]["normalized_mse"] is not None for s in ("train", "validation")
    )
    result = _finite_payload(
        {
            "freeze_sha256": identity,
            "status": (
                "fit_failed"
                if not fitted["fit"]["success"]
                else "complete"
                if complete
                else "rollout_failed"
            ),
            "default_replay": screen,
            **fitted,
            "replay": replay,
            "test_data_opened": False,
            "private_reference_opened": False,
            "llm_calls": 0,
        }
    )
    write_json(output / "result.json", result)
    return result
