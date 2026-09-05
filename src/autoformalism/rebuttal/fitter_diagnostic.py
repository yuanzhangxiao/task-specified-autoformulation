"""Development-only, frozen-structure numerical fitter diagnostic.

No proposal generation, parameter selection on validation, or test loader is used.
The production fitter and restricted compiler remain the only execution engines.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.baselines.raw_data_agent import (
    fit_result_payload,
    raw_agent_validation_context,
)
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    BenchmarkSpec,
    DevelopmentDataset,
    TrainingScaler,
)
from autoformalism.expressions import (
    CompiledModel,
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, fit_candidate, simulate_trajectory
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, StrictSchema

ARMS = ("agent_replay", "warm_1", "cold_1", "cold_3")
PUBLIC_FILES = ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv")
SOURCE_FILES = ("run_config.json", "candidate.json", "evaluation.json")


class DiagnosticCell(StrictSchema):
    """A development cell selected before looking at diagnostic outcomes."""

    benchmark_id: str = Field(pattern=r"^phase_b_[a-z0-9_]+$")
    tier: Literal["easy", "hard"]
    source: Literal["historical", "refresh"]
    public_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class DiagnosticPlan(StrictSchema):
    """Budget-matched initialization ablation with an immutable task matrix."""

    schema_version: Literal["fitter-diagnostic-plan-1"] = "fitter-diagnostic-plan-1"
    cells: tuple[DiagnosticCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = (0, 1, 2)
    total_max_nfev: int = Field(default=300, ge=3)
    fit_seconds: FiniteFloat = Field(default=900, gt=0)
    replay_seconds: FiniteFloat = Field(default=180, gt=0)
    supervisor_grace_seconds: FiniteFloat = Field(default=60, gt=0)
    seed: int = Field(default=20260905, ge=0)

    @model_validator(mode="after")
    def validate_matrix(self) -> DiagnosticPlan:
        """Reject duplicate cells and unequal evaluation allocations."""
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate diagnostic cell")
        if (
            not self.repetitions
            or min(self.repetitions) < 0
            or len(set(self.repetitions)) != len(self.repetitions)
        ):
            raise ValueError("repetitions must be distinct nonnegative integers")
        if self.total_max_nfev % 3:
            raise ValueError("total_max_nfev must be divisible by three")
        return self

    def fit_config(self, arm: str, repetition: int) -> FitConfig:
        """Use the same integrator, seed and total budget in every fitting arm."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        starts = 3 if arm == "cold_3" else 1
        return FitConfig(
            number_of_starts=starts,
            random_seed=self.seed + repetition,
            integration_backend="solve_ivp",
            allow_derivative_regression=False,
            maximum_function_evaluations=self.total_max_nfev // starts,
            maximum_wall_time_seconds=self.fit_seconds,
        )


class FrozenCandidate(StrictSchema):
    """Minimal executable source; historical metrics are deliberately omitted."""

    candidate: CandidateModel
    parameters: dict[str, FiniteFloat]


def read_json(path: Path) -> dict[str, Any]:
    """Read an object, allowing historical nonfinite metrics outside our payload."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def json_bytes(value: object) -> bytes:
    """Produce stable, standards-compliant checkpoint bytes."""
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def sha256(path: Path) -> str:
    """Hash a required artifact without following any directory recursively."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object, *, immutable: bool = False) -> None:
    """Atomically publish a JSON checkpoint; reject changed frozen artifacts."""
    _write_bytes(path, json_bytes(value), immutable=immutable)


def _write_bytes(path: Path, data: bytes, *, immutable: bool = False) -> None:
    """Publish a complete file so a killed snapshot can be resumed safely."""
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen artifact differs: {path}; use a new output root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def runtime_identity() -> dict[str, object]:
    """Pin loaded source and numerical dependency versions for safe resume."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode() + b"\0")
        digest.update(path.read_bytes())
    runner = package.parents[1] / "scripts" / "run_phase_b_fitter_diagnostic.py"
    digest.update(runner.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "pydantic")
        },
    }


def _source_payload(
    run: Path, cell: DiagnosticCell, repetition: int, public_hashes: dict[str, str]
) -> FrozenCandidate:
    config = read_json(run / "run_config.json")
    expected = {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "repetition": repetition,
        "parameter_refit_applied": False,
        "test_data_opened": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("source run identity or split policy differs")
    if config.get("agent_config", {}).get("output_contract") != "fitted_model":
        raise ValueError("source is not an agent-fitted model")
    for name in PUBLIC_FILES[1:]:
        if config.get("public_input_hashes", {}).get(name) != public_hashes[name]:
            raise ValueError(f"source/public input hash differs: {name}")
    evaluation = read_json(run / "evaluation.json")
    if (
        evaluation.get("schema_version") != "raw-data-agent-fitted-evaluation-1"
        or evaluation.get("parameter_refit_applied") is not False
        or evaluation.get("test_data_opened") is not False
    ):
        raise ValueError("evaluation is not an untouched agent parameter vector")
    candidate = CandidateModel.model_validate_json(
        (run / "candidate.json").read_bytes()
    )
    payload = FrozenCandidate(
        candidate=candidate, parameters=evaluation["fitted_parameter_values"]
    )
    if set(payload.parameters) != {item.name for item in candidate.parameters}:
        raise ValueError("source parameter names differ from frozen candidate")
    return payload


def prepare_diagnostic(
    plan: DiagnosticPlan,
    *,
    public_root: Path,
    historical_root: Path,
    refresh_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Snapshot only public development files and all prespecified candidates.

    Missing/invalid source models remain in the denominator. Public package errors
    are fatal. This function must run under the CLI's preparation lock.
    """
    assets: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    for cell in plan.cells:
        relative = Path("phase_b_v1") / cell.benchmark_id
        hashes = {name: sha256(public_root / relative / name) for name in PUBLIC_FILES}
        if hashes["proposer_prompt.txt"] != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")
        for name in PUBLIC_FILES:
            destination = output_root / "public" / relative / name
            data = (public_root / relative / name).read_bytes()
            _write_bytes(destination, data, immutable=True)
            assets[str(destination.relative_to(output_root))] = hashes[name]
        spec, dataset, context = load_development(output_root, cell)
        del spec, dataset
        for repetition in plan.repetitions:
            run_name = (
                f"openai_gpt-5-6-sol_{cell.benchmark_id}_{cell.tier}_rep{repetition}"
            )
            run = (
                historical_root / run_name
                if cell.source == "historical"
                else refresh_root / "runs" / run_name
            )
            record: dict[str, Any] = {
                "index": len(sources),
                "cell": cell.model_dump(mode="json"),
                "repetition": repetition,
                "source_path": str(run.resolve()),
                "source_hashes": {
                    name: sha256(run / name) if (run / name).is_file() else None
                    for name in SOURCE_FILES
                },
            }
            try:
                payload = _source_payload(run, cell, repetition, hashes)
                if any(
                    sha256(run / name) != record["source_hashes"][name]
                    for name in SOURCE_FILES
                ):
                    raise ValueError("source changed during snapshot")
                compiled = compile_candidate(payload.candidate, context)
                # Exact agent replay has no separately supplied fitted initials.
                # Do not introduce fitted latent/validation initials in refit arms.
                if any(
                    initial.initialization_range is not None
                    and initial.state not in compiled.observed_state_channels
                    for initial in payload.candidate.initial_conditions
                ):
                    raise ValueError(
                        "unresolved fitted initial conditions are unsupported"
                    )
                frozen = Path("candidates") / f"{len(sources):03d}.json"
                write_json(
                    output_root / frozen,
                    payload.model_dump(mode="json"),
                    immutable=True,
                )
                assets[str(frozen)] = sha256(output_root / frozen)
                record.update(status="ready", candidate_path=str(frozen))
            except FileNotFoundError as exc:
                record.update(status="source_missing", error=str(exc))
            except (ValueError, KeyError, TypeError, ModelValidationError) as exc:
                record.update(status="source_invalid", error=str(exc))
            sources.append(record)
    tasks = [
        {"index": index, "source_index": source["index"], "arm": arm}
        for index, (source, arm) in enumerate(
            (source, arm) for source in sources for arm in ARMS
        )
    ]
    manifest = {
        "schema_version": "fitter-diagnostic-freeze-1",
        "plan": plan.model_dump(mode="json"),
        "runtime": runtime_identity(),
        "assets": assets,
        "sources": sources,
        "tasks": tasks,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    write_json(output_root / "freeze.json", manifest, immutable=True)
    return manifest


def verify_freeze(output_root: Path) -> dict[str, Any]:
    """Validate inputs and runtime before using any existing result checkpoint."""
    manifest = read_json(output_root / "freeze.json")
    DiagnosticPlan.model_validate(manifest["plan"])
    if runtime_identity() != manifest["runtime"]:
        raise ValueError(
            "runtime differs from freeze; restore it or use a new output root"
        )
    for relative, expected in manifest["assets"].items():
        path = (output_root / relative).resolve()
        if not path.is_relative_to(output_root.resolve()) or sha256(path) != expected:
            raise ValueError(f"frozen input hash differs: {relative}")
    return manifest


def load_development(
    output_root: Path,
    cell: DiagnosticCell,
) -> tuple[BenchmarkSpec, DevelopmentDataset, ValidationContext]:
    """Load only the snapshotted train/validation tables through the public loader."""
    registry = BenchmarkRegistry()
    spec = registry.get(cell.benchmark_id)
    if spec.one_step_target_history or spec.data_layout != "tidy_split_file":
        raise ValueError("diagnostic requires Phase-B free-rollout development data")
    dataset = BenchmarkLoader(registry).load_development(
        DataConfig(
            benchmark_id=cell.benchmark_id,
            tier=cell.tier,
            root=(output_root / "public").resolve(),
        )
    )
    return spec, dataset, raw_agent_validation_context(dataset, spec)


def _finite_payload(value: Any) -> Any:
    """Represent nonfinite optimizer diagnostics as JSON null, never as a score."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_payload(item) for item in value]
    return value


def replay_parameters(
    model: CompiledModel,
    dataset: DevelopmentDataset,
    parameters: Mapping[str, float],
    config: FitConfig,
    seconds: float,
) -> dict[str, Any]:
    """Score exact parameters with train scales and no fitted/reset hidden states."""
    scaler = TrainingScaler().fit(dataset.train)
    scales = {
        channel: scaler.scales[f"target:{channel}"].standard_deviation
        for channel in dataset.roles.targets
    }
    deadline = monotonic() + seconds
    result: dict[str, Any] = {"target_scales": scales}
    for label, split in (("train", dataset.train), ("validation", dataset.validation)):
        squared: dict[str, list[np.ndarray]] = {channel: [] for channel in scales}
        failures: list[dict[str, str]] = []
        for trajectory in split.trajectories:
            try:
                simulation = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    config,
                    deadline=deadline,
                    reset_observed_states=False,
                )
            except TimeoutError:
                failures.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "message": "replay wall-clock limit reached",
                    }
                )
                continue
            if not simulation.success:
                failures.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "message": str(simulation.message),
                    }
                )
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                errors = {
                    channel: (
                        (simulation.predictions[channel] - trajectory.targets[channel])
                        / scale
                    )
                    ** 2
                    for channel, scale in scales.items()
                }
            if any(not np.isfinite(values).all() for values in errors.values()):
                failures.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "message": "nonfinite normalized squared error",
                    }
                )
                continue
            for channel, values in errors.items():
                squared[channel].append(values)
        per_target = {
            channel: float(np.mean(np.concatenate(values), dtype=np.longdouble))
            if values
            else None
            for channel, values in squared.items()
        }
        result[label] = {
            "normalized_mse": (
                float(np.mean(list(per_target.values()), dtype=np.longdouble))
                if not failures
                else None
            ),
            "per_target_normalized_mse": per_target if not failures else None,
            "failures": failures,
            "trajectory_count": len(split.trajectories),
        }
    return result


def execute_task(output_root: Path, index: int) -> dict[str, Any]:
    """Execute one arm in the supervised worker; checkpoint a completed fit first."""
    manifest = verify_freeze(output_root)
    if not 0 <= index < len(manifest["tasks"]):
        raise ValueError("task index outside frozen matrix")
    task = manifest["tasks"][index]
    source = manifest["sources"][task["source_index"]]
    result: dict[str, Any] = {
        **task,
        "freeze_sha256": sha256(output_root / "freeze.json"),
        "test_data_opened": False,
        "llm_calls": 0,
    }
    if source["status"] != "ready":
        return {**result, "status": source["status"], "error": source["error"]}
    plan = DiagnosticPlan.model_validate(manifest["plan"])
    _, dataset, context = load_development(
        output_root, DiagnosticCell.model_validate(source["cell"])
    )
    payload = FrozenCandidate.model_validate(
        read_json(output_root / source["candidate_path"])
    )
    model = compile_candidate(payload.candidate, context)
    config = plan.fit_config(task["arm"], source["repetition"])
    result["fit_config"] = config.model_dump(mode="json")
    parameters = payload.parameters
    if task["arm"] != "agent_replay":
        checkpoint = output_root / "results" / f"{index:03d}.fit.json"
        if checkpoint.exists():
            fitted = read_json(checkpoint)
            if fitted["freeze_sha256"] != result["freeze_sha256"] or any(
                fitted.get(key) != task[key] for key in task
            ):
                raise ValueError("fit checkpoint belongs to another freeze")
        else:
            started = monotonic()
            fit = fit_candidate(
                model,
                dataset.train,
                dataset.validation,
                config,
                initial_global_parameters=(
                    parameters if task["arm"] == "warm_1" else None
                ),
            )
            fitted = {
                **task,
                "freeze_sha256": result["freeze_sha256"],
                "fit_seconds": monotonic() - started,
                "fit": _finite_payload(fit_result_payload(fit)),
            }
            write_json(checkpoint, fitted)
        result.update(fitted)
        parameters = fitted["fit"]["global_parameters"]
    result["parameters"] = parameters
    result["replay"] = replay_parameters(
        model, dataset, parameters, config, plan.replay_seconds
    )
    complete = all(
        result["replay"][split]["normalized_mse"] is not None
        for split in ("train", "validation")
    )
    result["status"] = "complete" if complete else "rollout_failed"
    if task["arm"] != "agent_replay" and not result["fit"]["success"]:
        result["status"] = "fit_failed"
    return _finite_payload(result)


def summarize_diagnostic(output_root: Path) -> dict[str, Any]:
    """Report every prespecified model/arm, including missing and failed tasks."""
    manifest = verify_freeze(output_root)
    frozen_hash = sha256(output_root / "freeze.json")
    rows = []
    for source in manifest["sources"]:
        arms: dict[str, Any] = {}
        for task in manifest["tasks"]:
            if task["source_index"] != source["index"]:
                continue
            path = output_root / "results" / f"{task['index']:03d}.json"
            value = read_json(path) if path.exists() else {"status": "missing"}
            if path.exists() and (
                value.get("freeze_sha256") != frozen_hash
                or any(value.get(key) != task[key] for key in task)
            ):
                raise ValueError(f"result checkpoint identity differs: {path}")
            arms[task["arm"]] = {
                "status": value["status"],
                "train_nmse": value.get("replay", {})
                .get("train", {})
                .get("normalized_mse"),
                "validation_nmse": value.get("replay", {})
                .get("validation", {})
                .get("normalized_mse"),
                "fit_seconds": value.get("fit_seconds"),
                "error": value.get("error"),
            }
        reference = arms["agent_replay"]["validation_nmse"]
        for arm in ARMS[1:]:
            observed = arms[arm]["validation_nmse"]
            arms[arm]["validation_ratio_to_agent"] = (
                observed / reference
                if reference is not None
                and reference > 0
                and observed is not None
                and arms[arm]["status"] == "complete"
                and arms["agent_replay"]["status"] == "complete"
                else None
            )
        rows.append(
            {
                "benchmark_id": source["cell"]["benchmark_id"],
                "repetition": source["repetition"],
                "arms": arms,
            }
        )
    counts = {
        arm: sum(row["arms"][arm]["status"] == "complete" for row in rows)
        for arm in ARMS
    }
    return _finite_payload(
        {
            "schema_version": "fitter-diagnostic-summary-1",
            "rows": rows,
            "expected_models": len(rows),
            "expected_tasks": len(manifest["tasks"]),
            "complete_by_arm": counts,
            "freeze_sha256": frozen_hash,
            "test_data_opened": False,
            "llm_calls": 0,
        }
    )
