"""Matched numerical-rescue campaign for the six frozen staged candidates."""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from time import monotonic, process_time
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.baselines.raw_data_agent import fit_result_payload
from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import (
    FitConfig,
    estimate_profiled_warm_start_from_public_derivatives,
    fit_candidate,
    simulate_trajectory,
)
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel, ParameterDomain, ParameterRole
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class FitterRescueConfig(StrictSchema):
    """Frozen source and bounded train-only numerical-rescue policy."""

    protocol: Literal["scientific-staged-fitter-rescue-1"]
    purpose: str = Field(min_length=1)
    source_fitting_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fitting_summary_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_candidate_count: Literal[6]
    screen_trajectory_count: int = Field(ge=1, le=4)
    screen_sample_count: int = Field(ge=4, le=64)
    selected_start_count: int = Field(ge=1, le=3)
    screen_seconds_per_start: float = Field(gt=0.0, le=60.0)
    score_seconds_per_split: float = Field(gt=0.0, le=180.0)
    initializer_fit_config: FitConfig
    screen_fit_config: FitConfig
    rescue_fit_config: FitConfig
    score_fit_config: FitConfig

    @model_validator(mode="after")
    def frozen_policy(self) -> FitterRescueConfig:
        """Keep estimated derivatives confined to the warm-start stage."""
        initializer = self.initializer_fit_config
        if (
            initializer.parameter_fit_strategy != "profiled_latent_basis_linear_ridge"
            or not initializer.allow_derivative_regression
            or initializer.maximum_wall_time_seconds is None
        ):
            raise ValueError("initializer must be bounded profiled derivative fitting")
        for name, settings in (
            ("screen", self.screen_fit_config),
            ("rescue", self.rescue_fit_config),
            ("score", self.score_fit_config),
        ):
            if (
                settings.parameter_fit_strategy != "bounded_nonlinear"
                or settings.allow_derivative_regression
            ):
                raise ValueError(f"{name} must use derivative-free bounded rollout")
        if self.screen_fit_config.integration_backend != "fixed_rk4":
            raise ValueError("screen must use deterministic fixed-step rollout")
        if self.rescue_fit_config.number_of_starts != 1:
            raise ValueError("each selected rescue start must be fit separately")
        return self


def launcher_hash() -> str:
    """Bind every command that creates or consumes rescue artifacts."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_fitter_rescue_campaign.py",
        "scripts/hpc/staged_fitter_rescue_prepare_cpu.slurm",
        "scripts/hpc/staged_fitter_rescue_worker_cpu.slurm",
        "scripts/hpc/staged_fitter_rescue_summary_cpu.slurm",
        "scripts/hpc/submit_staged_fitter_rescue_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_campaign(
    config_path: Path,
    source_fitting_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze the exact prior candidates, fit results, and copied public data."""
    config = FitterRescueConfig.model_validate_json(config_path.read_text())
    source_plan_path = source_fitting_root / "plan.json"
    source_summary_path = source_fitting_root / "summary" / "summary.json"
    source_plan = _read_object(source_plan_path)
    source_summary = _read_object(source_summary_path)
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    )
    if (
        source_digest != source_plan.get("plan_sha256")
        or source_digest != config.source_fitting_plan_sha256
    ):
        raise ValueError("source fitting plan differs")
    if _sha256(source_summary_path) != config.source_fitting_summary_file_sha256:
        raise ValueError("source fitting summary differs")
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("terminal_results") != config.expected_candidate_count
        or source_summary.get("candidate_regeneration_performed") is not False
        or source_summary.get("scientific_judge_called") is not False
        or source_summary.get("test_data_opened") is not False
        or source_summary.get("private_reference_opened") is not False
    ):
        raise ValueError("source fitting campaign is not an eligible frozen source")

    frozen = output_root / "frozen"
    public_assets: dict[str, str] = {}
    for relative, expected_hash in source_plan["public_asset_ledger"].items():
        source = source_fitting_root / relative
        if _sha256(source) != expected_hash:
            raise ValueError(f"source public asset differs: {relative}")
        destination = frozen / "public" / Path(relative).relative_to("frozen/public")
        _copy_once(source, destination)
        public_assets[str(destination.relative_to(output_root))] = _sha256(destination)

    tasks: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, str]] = []
    for index, source_task in enumerate(source_plan.get("tasks", [])):
        if int(source_task["task_index"]) != index:
            raise ValueError("source task ordering differs")
        candidate_source = source_fitting_root / source_task["candidate_path"]
        if _sha256(candidate_source) != source_task["candidate_file_sha256"]:
            raise ValueError(f"source candidate differs: {index}")
        prior_result_source = source_fitting_root / "tasks" / f"task_{index:03d}.json"
        prior_result = _read_object(prior_result_source)
        if (
            prior_result.get("plan_sha256") != source_digest
            or prior_result.get("candidate_file_sha256")
            != source_task["candidate_file_sha256"]
            or prior_result.get("status") != "complete"
        ):
            raise ValueError(f"source fitting result differs: {index}")
        candidate = CandidateModel.model_validate_json(candidate_source.read_text())
        if any(
            item.initialization_range is not None
            for item in candidate.initial_conditions
        ):
            raise ValueError("rescue scorer does not fit latent initial conditions")
        candidate_path = frozen / "candidates" / f"candidate_{index:03d}.json"
        result_path = frozen / "source_results" / f"task_{index:03d}.json"
        _copy_once(candidate_source, candidate_path)
        _copy_once(prior_result_source, result_path)
        source_artifacts.extend(
            (
                {
                    "path": str(candidate_path.relative_to(output_root)),
                    "sha256": _sha256(candidate_path),
                },
                {
                    "path": str(result_path.relative_to(output_root)),
                    "sha256": _sha256(result_path),
                },
            )
        )
        tasks.append(
            {
                "task_index": index,
                "task_id": f"{source_task['task_id']}_rescue",
                "benchmark_id": source_task["benchmark_id"],
                "tier": source_task["tier"],
                "seed": source_task["seed"],
                "candidate_path": str(candidate_path.relative_to(output_root)),
                "candidate_file_sha256": _sha256(candidate_path),
                "source_result_path": str(result_path.relative_to(output_root)),
                "source_result_file_sha256": _sha256(result_path),
            }
        )
    if len(tasks) != config.expected_candidate_count:
        raise ValueError("source candidate count differs")

    _write_once_json(frozen / "source_plan.json", source_plan)
    _write_once_json(frozen / "source_summary.json", source_summary)
    plan = {
        "schema_version": "scientific-staged-fitter-rescue-plan-1",
        "config": config.model_dump(mode="json"),
        "source_fitting_plan_sha256": source_digest,
        "source_fitting_summary_file_sha256": _sha256(source_summary_path),
        "source_artifact_ledger": source_artifacts,
        "source_artifact_ledger_sha256": content_hash(source_artifacts),
        "public_asset_ledger": public_assets,
        "public_asset_ledger_sha256": content_hash(public_assets),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": launcher_hash(),
        "tasks": tasks,
        "candidate_regeneration_performed": False,
        "topology_or_function_revision_performed": False,
        "estimated_derivatives_used_only_for_initialization": True,
        "validation_used_for_start_selection": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
        "multiple_round_search_performed": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    _write_once_json(output_root / "plan.json", plan)
    freeze = {
        "schema_version": "scientific-staged-fitter-rescue-freeze-1",
        "status": "frozen_before_rescue",
        "plan_sha256": plan["plan_sha256"],
        "source_artifact_ledger_sha256": plan["source_artifact_ledger_sha256"],
        "public_asset_ledger_sha256": plan["public_asset_ledger_sha256"],
        "task_count": len(tasks),
        "candidate_regeneration_performed": False,
        "topology_or_function_revision_performed": False,
        "estimated_derivatives_used_only_for_initialization": True,
        "validation_used_for_start_selection": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "multiple_round_search_performed": False,
    }
    _write_once_json(frozen / "freeze_manifest.json", freeze)
    return freeze


def run_task(output_root: Path, task_index: int) -> dict[str, Any]:
    """Run source rescoring, derivative initialization, and bounded rescue fits."""
    plan = _verified_plan(output_root)
    if not 0 <= task_index < len(plan["tasks"]):
        raise ValueError(f"task index out of range: {task_index}")
    task = plan["tasks"][task_index]
    destination = output_root / "tasks" / f"task_{task_index:03d}.json"
    if destination.exists():
        existing = _read_object(destination)
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("rescue checkpoint belongs to another plan")
        return existing
    candidate_path = output_root / task["candidate_path"]
    source_result_path = output_root / task["source_result_path"]
    if _sha256(candidate_path) != task["candidate_file_sha256"]:
        raise ValueError("frozen rescue candidate differs")
    if _sha256(source_result_path) != task["source_result_file_sha256"]:
        raise ValueError("frozen source fit result differs")
    candidate = CandidateModel.model_validate_json(candidate_path.read_text())
    source_result = _read_object(source_result_path)
    dataset, context = load_public_data(
        output_root / "frozen" / "public", task["benchmark_id"], task["tier"]
    )
    model = compile_candidate(candidate, context)
    config = FitterRescueConfig.model_validate(plan["config"])
    scales = _target_scales(dataset.train, tuple(context.targets))
    started = monotonic()
    cpu_started = process_time()

    source_parameters = _fit_parameters(source_result)
    source_rescore = _score_train_then_validation(
        model,
        dataset.train,
        dataset.validation,
        source_parameters,
        scales,
        config.score_fit_config,
        config.score_seconds_per_split,
    )
    initialization = _derivative_initialization(
        model,
        dataset.train,
        config.initializer_fit_config.model_copy(update={"random_seed": task["seed"]}),
        _role_start(candidate, dataset.train, "small"),
    )
    proposed_starts: list[tuple[str, Mapping[str, float]]] = []
    if source_parameters is not None:
        proposed_starts.append(("source_fit", source_parameters))
    if initialization["usable"]:
        proposed_starts.append(
            ("estimated_derivative_profiled", initialization["parameters"])
        )
    proposed_starts.extend(
        (
            ("runtime_role_small", _role_start(candidate, dataset.train, "small")),
            ("runtime_role_unit", _role_start(candidate, dataset.train, "unit")),
        )
    )
    unique_starts = _unique_valid_starts(candidate, proposed_starts)
    screen_split = _screen_split(
        dataset.train,
        config.screen_trajectory_count,
        config.screen_sample_count,
    )
    screens = [
        {
            "label": label,
            "parameters": dict(parameters),
            **_score_split(
                model,
                screen_split,
                parameters,
                scales,
                config.screen_fit_config,
                config.screen_seconds_per_start,
            ),
        }
        for label, parameters in unique_starts
    ]
    ranked = sorted(
        screens,
        key=lambda row: (
            not row["complete"],
            row["failed_trajectory_count"],
            row["normalized_mse"],
            row["label"],
        ),
    )
    selected = ranked[: config.selected_start_count]
    train_only_validation = _training_only_validation_surrogate(dataset.train)
    attempts: list[dict[str, Any]] = []
    for attempt_index, screen in enumerate(selected):
        attempt_started = monotonic()
        settings = config.rescue_fit_config.model_copy(
            update={"random_seed": task["seed"] * 100 + attempt_index}
        )
        try:
            fitted = fit_candidate(
                model,
                dataset.train,
                train_only_validation,
                settings,
                initial_global_parameters=screen["parameters"],
            )
        except Exception as exc:  # candidate and numerical paths remain untrusted
            attempts.append(
                {
                    "label": screen["label"],
                    "fit_call_completed": False,
                    "fit": None,
                    "fresh_training_score": None,
                    "fit_wall_seconds": monotonic() - attempt_started,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:8000],
                }
            )
            continue
        fit_payload = _train_only_fit_payload(fitted)
        fresh_training = _score_split(
            model,
            dataset.train,
            fitted.global_parameters,
            scales,
            config.score_fit_config,
            config.score_seconds_per_split,
        )
        attempts.append(
            {
                "label": screen["label"],
                "fit_call_completed": True,
                "fit": fit_payload,
                "fresh_training_score": fresh_training,
                "fit_wall_seconds": monotonic() - attempt_started,
                "error_type": None,
                "error": fitted.message,
            }
        )
    eligible = [
        (index, row)
        for index, row in enumerate(attempts)
        if (row.get("fresh_training_score") or {}).get("complete") is True
    ]
    if eligible:
        selected_index, selected_attempt = min(
            eligible,
            key=lambda item: (
                item[1]["fresh_training_score"]["normalized_mse"],
                item[0],
            ),
        )
        selected_parameters = selected_attempt["fit"]["global_parameters"]
        validation_score = _score_split(
            model,
            dataset.validation,
            selected_parameters,
            scales,
            config.score_fit_config,
            config.score_seconds_per_split,
        )
        rescue_score = {
            "complete": bool(validation_score["complete"]),
            "selected_attempt_index": selected_index,
            "selected_start_label": selected_attempt["label"],
            "training": selected_attempt["fresh_training_score"],
            "validation": validation_score,
            "parameters": selected_parameters,
        }
    else:
        rescue_score = {
            "complete": False,
            "selected_attempt_index": None,
            "selected_start_label": None,
            "training": None,
            "validation": None,
            "parameters": None,
        }
    attribution = _attribution(source_result, source_rescore, rescue_score)
    result = {
        "schema_version": "scientific-staged-fitter-rescue-task-1",
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        **{
            key: task[key]
            for key in ("task_index", "task_id", "benchmark_id", "tier", "seed")
        },
        "candidate_file_sha256": task["candidate_file_sha256"],
        "source_fit_success": bool(source_result.get("fit_success")),
        "source_fit_error": source_result.get("error"),
        "source_rescore": source_rescore,
        "estimated_derivative_initializer": initialization,
        "start_screens": screens,
        "selected_start_labels": [row["label"] for row in selected],
        "rescue_attempts": attempts,
        "rescue_score": rescue_score,
        "attribution": attribution,
        "fit_wall_seconds": monotonic() - started,
        "fit_process_cpu_seconds": process_time() - cpu_started,
        "candidate_regeneration_performed": False,
        "topology_or_function_revision_performed": False,
        "estimated_derivatives_used_only_for_initialization": True,
        "validation_used_for_start_selection": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "multiple_round_search_performed": False,
    }
    _write_once_json(destination, result)
    return result


def summarize_campaign(output_root: Path) -> dict[str, Any]:
    """Report rescue and attribution endpoints without declaring a model winner."""
    plan = _verified_plan(output_root)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        path = output_root / "tasks" / f"task_{task['task_index']:03d}.json"
        if not path.exists():
            rows.append({"task_index": task["task_index"], "result_present": False})
            continue
        result = _read_object(path)
        if result.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(f"rescue task belongs to another plan: {path}")
        initializer = result["estimated_derivative_initializer"]
        source_score = result["source_rescore"]
        rescue = result["rescue_score"]
        selected_attempt = (
            None
            if rescue["selected_attempt_index"] is None
            else result["rescue_attempts"][rescue["selected_attempt_index"]]
        )
        rows.append(
            {
                "task_index": task["task_index"],
                "benchmark_id": task["benchmark_id"],
                "seed": task["seed"],
                "result_present": True,
                "source_fit_success": result["source_fit_success"],
                "source_rescore_complete": source_score["complete"],
                "source_rescore_training_normalized_mse": _nested(
                    source_score, "training", "normalized_mse"
                ),
                "source_rescore_validation_normalized_mse": _nested(
                    source_score, "validation", "normalized_mse"
                ),
                "estimated_derivative_initializer_usable": initializer["usable"],
                "stable_screen_count": sum(
                    row["complete"] for row in result["start_screens"]
                ),
                "rescue_complete": rescue["complete"],
                "rescue_optimizer_fit_success": bool(
                    selected_attempt and selected_attempt["fit"]["success"]
                ),
                "rescue_training_normalized_mse": _nested(
                    rescue, "training", "normalized_mse"
                ),
                "rescue_validation_normalized_mse": _nested(
                    rescue, "validation", "normalized_mse"
                ),
                "selected_start_label": rescue["selected_start_label"],
                "attribution": result["attribution"],
                "fit_wall_seconds": result["fit_wall_seconds"],
                "fit_process_cpu_seconds": result["fit_process_cpu_seconds"],
            }
        )
        artifacts.append(
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    terminal = sum(row["result_present"] for row in rows)
    complete = [row for row in rows if row.get("rescue_complete")]
    attribution_counts = Counter(
        row["attribution"] for row in rows if row["result_present"]
    )
    summary = {
        "schema_version": "scientific-staged-fitter-rescue-summary-1",
        "status": "complete" if terminal == len(rows) else "incomplete",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": terminal,
        "source_fit_success_rate": _rate(
            sum(bool(row.get("source_fit_success")) for row in rows), len(rows)
        ),
        "source_retained_parameter_rescore_rate": _rate(
            sum(bool(row.get("source_rescore_complete")) for row in rows), len(rows)
        ),
        "estimated_derivative_initializer_usable_rate": _rate(
            sum(
                bool(row.get("estimated_derivative_initializer_usable")) for row in rows
            ),
            len(rows),
        ),
        "tasks_with_at_least_one_stable_screen_rate": _rate(
            sum(int(row.get("stable_screen_count", 0)) > 0 for row in rows), len(rows)
        ),
        "rescue_complete_rate": _rate(len(complete), len(rows)),
        "rescue_optimizer_fit_success_rate": _rate(
            sum(bool(row.get("rescue_optimizer_fit_success")) for row in rows),
            len(rows),
        ),
        "median_rescue_validation_normalized_mse_conditional_on_complete": _median(
            [row["rescue_validation_normalized_mse"] for row in complete]
        ),
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "rows": rows,
        "candidate_regeneration_performed": False,
        "topology_or_function_revision_performed": False,
        "estimated_derivatives_used_only_for_initialization": True,
        "validation_used_for_start_selection": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
        "multiple_round_search_performed": False,
    }
    summary_root = output_root / "summary"
    _write_once(
        summary_root / "task_artifact_ledger.jsonl",
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in artifacts).encode(),
    )
    summary["task_artifact_ledger_sha256"] = _sha256(
        summary_root / "task_artifact_ledger.jsonl"
    )
    _write_once_json(summary_root / "summary.json", summary)
    _write_once(summary_root / "summary.md", _summary_markdown(summary).encode())
    return summary


def _derivative_initialization(
    model: Any,
    training: DatasetSplit,
    settings: FitConfig,
    preferred: Mapping[str, float],
) -> dict[str, Any]:
    started = monotonic()
    try:
        fitted = estimate_profiled_warm_start_from_public_derivatives(
            model,
            training,
            settings,
            initial_global_parameters=preferred,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "usable": False,
            "parameters": None,
            "fit": None,
            "wall_seconds": monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc)[:8000],
        }
    payload = _train_only_fit_payload(fitted)
    best = fitted.diagnostics[fitted.best_start_index]
    parameters = dict(fitted.global_parameters)
    usable = bool(
        best.status != -2
        and best.affine_design_rank is not None
        and set(parameters) == set(model.parameter_names)
        and all(np.isfinite(value) for value in parameters.values())
    )
    return {
        "attempted": True,
        "usable": usable,
        "parameters": parameters,
        "fit": payload,
        "wall_seconds": monotonic() - started,
        "error_type": None,
        "error": fitted.message,
    }


def _train_only_fit_payload(fitted: Any) -> dict[str, Any]:
    """Serialize a fit without mislabeling duplicated training as validation."""
    raw = fit_result_payload(fitted)
    return {
        "success": raw.get("success"),
        "message": raw.get("message"),
        "global_parameters": raw.get("global_parameters"),
        "training_rollout_normalized_mse": raw.get("training_normalized_mse"),
        "training_per_target_normalized_mse": raw.get(
            "training_per_target_normalized_mse"
        ),
        "training_failed_trajectories": raw.get("training_failed_trajectories"),
        "diagnostics": raw.get("diagnostics", []),
        "best_start_index": raw.get("best_start_index"),
        "validation_data_opened": False,
    }


def _fit_parameters(source_result: Mapping[str, Any]) -> dict[str, float] | None:
    fit = source_result.get("fit") or {}
    parameters = fit.get("global_parameters")
    if not isinstance(parameters, dict) or not parameters:
        return None
    if not all(
        isinstance(value, (int, float)) and np.isfinite(value)
        for value in parameters.values()
    ):
        return None
    return {str(name): float(value) for name, value in parameters.items()}


def _role_start(
    candidate: CandidateModel,
    training: DatasetSplit,
    policy: Literal["small", "unit"],
) -> dict[str, float]:
    spans = [float(item.time[-1] - item.time[0]) for item in training.trajectories]
    steps = [float(np.median(np.diff(item.time))) for item in training.trajectories]
    span = max(float(np.median(spans)), 1e-6)
    step = max(float(np.median(steps)), 1e-6)
    time_scale = max(step, span / (10.0 if policy == "small" else 2.0))
    result: dict[str, float] = {}
    for parameter in candidate.parameters:
        if parameter.role is ParameterRole.TIME_CONSTANT:
            value = time_scale
        elif parameter.role is ParameterRole.RATE:
            value = 1.0 / time_scale
        elif parameter.role in {ParameterRole.SCALE, ParameterRole.POSITIVE_SHAPE}:
            value = 0.1 if policy == "small" else 1.0
        elif parameter.role is ParameterRole.OFFSET:
            value = 0.0
        elif parameter.domain is ParameterDomain.REAL:
            value = 0.1 if parameter.role is ParameterRole.COEFFICIENT else 1.0
        else:
            value = 0.1 if policy == "small" else 1.0
        if parameter.bounds is not None:
            value = min(max(value, parameter.bounds.lower), parameter.bounds.upper)
        if parameter.domain is ParameterDomain.POSITIVE:
            value = max(value, np.finfo(float).eps)
        elif parameter.domain is ParameterDomain.NONNEGATIVE:
            value = max(value, 0.0)
        result[parameter.name] = float(value)
    return result


def _unique_valid_starts(
    candidate: CandidateModel,
    starts: list[tuple[str, Mapping[str, float]]],
) -> list[tuple[str, dict[str, float]]]:
    expected = {item.name for item in candidate.parameters}
    seen: set[tuple[tuple[str, float], ...]] = set()
    result: list[tuple[str, dict[str, float]]] = []
    specs = {item.name: item for item in candidate.parameters}
    for label, proposed in starts:
        if set(proposed) != expected:
            continue
        normalized = {name: float(proposed[name]) for name in sorted(expected)}
        valid = all(np.isfinite(value) for value in normalized.values())
        for name, value in normalized.items():
            spec = specs[name]
            valid &= spec.domain is not ParameterDomain.POSITIVE or value > 0.0
            valid &= spec.domain is not ParameterDomain.NONNEGATIVE or value >= 0.0
            if spec.bounds is not None:
                valid &= spec.bounds.lower <= value <= spec.bounds.upper
        key = tuple(normalized.items())
        if valid and key not in seen:
            seen.add(key)
            result.append((label, normalized))
    if not result:
        raise ValueError("no valid runtime rescue starts")
    return result


def _screen_split(
    split: DatasetSplit, trajectory_count: int, sample_count: int
) -> DatasetSplit:
    trajectories = tuple(
        _truncate_trajectory(item, sample_count)
        for item in split.trajectories[:trajectory_count]
    )
    return DatasetSplit(SplitName.TRAIN, trajectories, f"{split.fingerprint}:screen")


def _training_only_validation_surrogate(training: DatasetSplit) -> DatasetSplit:
    """Satisfy the fit API without exposing the real validation trajectories."""
    if training.name is not SplitName.TRAIN:
        raise ValueError("training-only validation surrogate requires training data")
    return DatasetSplit(
        SplitName.VALIDATION,
        training.trajectories,
        f"{training.fingerprint}:training-only-validation-surrogate",
    )


def _truncate_trajectory(trajectory: Trajectory, sample_count: int) -> Trajectory:
    count = min(sample_count, trajectory.number_of_rows)
    return Trajectory(
        trajectory_id=trajectory.trajectory_id,
        time=np.asarray(trajectory.time[:count], dtype=float).copy(),
        targets={
            name: np.asarray(values[:count], dtype=float).copy()
            for name, values in trajectory.targets.items()
        },
        auxiliaries={
            name: np.asarray(values[:count], dtype=float).copy()
            for name, values in trajectory.auxiliaries.items()
        },
        external_inputs={
            name: np.asarray(values[:count], dtype=float).copy()
            for name, values in trajectory.external_inputs.items()
        },
        fixed_covariates=dict(trajectory.fixed_covariates),
        derivatives={
            name: np.asarray(values[:count], dtype=float).copy()
            for name, values in trajectory.derivatives.items()
        },
        derivative_provenance=trajectory.derivative_provenance,
    )


def _target_scales(
    training: DatasetSplit, targets: tuple[str, ...]
) -> dict[str, float]:
    scaler = TrainingScaler().fit(training)
    return {
        channel: scaler.scales[f"target:{channel}"].standard_deviation
        for channel in targets
    }


def _score_train_then_validation(
    model: Any,
    training: DatasetSplit,
    validation: DatasetSplit,
    parameters: Mapping[str, float] | None,
    scales: Mapping[str, float],
    config: FitConfig,
    seconds: float,
) -> dict[str, Any]:
    if parameters is None:
        return {
            "complete": False,
            "training": None,
            "validation": None,
            "reason": "source has no finite parameter vector",
        }
    training_score = _score_split(model, training, parameters, scales, config, seconds)
    validation_score = (
        _score_split(model, validation, parameters, scales, config, seconds)
        if training_score["complete"]
        else None
    )
    return {
        "complete": bool(validation_score and validation_score["complete"]),
        "training": training_score,
        "validation": validation_score,
        "reason": None,
    }


def _score_split(
    model: Any,
    split: DatasetSplit,
    parameters: Mapping[str, float],
    scales: Mapping[str, float],
    config: FitConfig,
    seconds: float,
) -> dict[str, Any]:
    deadline = monotonic() + seconds
    squared: dict[str, list[np.ndarray]] = {
        channel: [] for channel in model.validated.context.targets
    }
    failures: list[dict[str, str]] = []
    started = monotonic()
    for trajectory in split.trajectories:
        try:
            simulation = simulate_trajectory(
                model,
                trajectory,
                parameters,
                {},
                config,
                deadline=deadline,
            )
        except TimeoutError as exc:
            failures.append(
                {"trajectory_id": trajectory.trajectory_id, "message": str(exc)}
            )
            break
        if not simulation.success:
            failures.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "message": simulation.message or "simulation failed",
                }
            )
            continue
        trajectory_squared: dict[str, np.ndarray] = {}
        for channel in squared:
            start = 1 if model.validated.context.lagged_targets else 0
            residual = (
                simulation.predictions[channel][start:]
                - trajectory.targets[channel][start:]
            ) / scales[channel]
            with np.errstate(over="ignore", invalid="ignore"):
                squared_residual = np.asarray(np.square(residual), dtype=float)
            if not np.all(np.isfinite(squared_residual)):
                failures.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "message": (
                            "causal rollout produced non-finite normalized "
                            f"squared residuals for target {channel}"
                        ),
                    }
                )
                break
            trajectory_squared[channel] = squared_residual
        else:
            for channel, values in trajectory_squared.items():
                squared[channel].append(values)
    complete = not failures and all(squared.values())
    per_target = {
        channel: float(np.mean(np.concatenate(values))) if values else None
        for channel, values in squared.items()
    }
    observed = [value for value in per_target.values() if value is not None]
    return {
        "complete": bool(complete),
        "normalized_mse": float(np.mean(observed)) if observed else float("inf"),
        "per_target_normalized_mse": per_target,
        "failed_trajectory_count": len(failures),
        "failures": failures,
        "wall_seconds": monotonic() - started,
    }


def _attribution(
    source_result: Mapping[str, Any],
    source_rescore: Mapping[str, Any],
    rescue: Mapping[str, Any],
) -> str:
    if bool(source_result.get("fit_success")):
        return "source_fit_already_finite"
    if bool(source_rescore.get("complete")):
        return "source_timeout_policy_limited"
    if bool(rescue.get("complete")):
        return "fitter_initialization_or_search_limited"
    return "unresolved_after_bounded_rescue"


def _verified_plan(output_root: Path) -> dict[str, Any]:
    plan = _read_object(output_root / "plan.json")
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen rescue plan digest differs")
    if plan.get("runtime_source_sha256") != runtime_source_hash():
        raise ValueError("runtime source differs from frozen rescue plan")
    if plan.get("launcher_sha256") != launcher_hash():
        raise ValueError("launcher differs from frozen rescue plan")
    for item in plan["source_artifact_ledger"]:
        if _sha256(output_root / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen source artifact differs: {item['path']}")
    for relative, expected in plan["public_asset_ledger"].items():
        if _sha256(output_root / relative) != expected:
            raise ValueError(f"frozen public artifact differs: {relative}")
    return plan


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen staged fitter-rescue test",
        "",
        "The exact six candidates and source fits were retained. Estimated "
        "public training derivatives were used only to propose warm starts; "
        "every reported error came from a fresh causal rollout. Validation "
        "never selected a start or fit.",
        "",
        "| Benchmark | Seed | Source rescore | Rescue | Rescue train NMSE | "
        "Rescue validation NMSE | Attribution |",
        "|:---|---:|:---:|:---:|---:|---:|:---|",
    ]
    for row in summary["rows"]:
        if not row["result_present"]:
            lines.append(
                f"| missing | {row['task_index']} | no | no | N/A | N/A | missing |"
            )
            continue
        lines.append(
            f"| {row['benchmark_id']} | {row['seed']} | "
            f"{'yes' if row['source_rescore_complete'] else 'no'} | "
            f"{'yes' if row['rescue_complete'] else 'no'} | "
            f"{_number(row['rescue_training_normalized_mse'])} | "
            f"{_number(row['rescue_validation_normalized_mse'])} | "
            f"{row['attribution']} |"
        )
    lines.extend(
        (
            "",
            f"Fresh rescue completion rate: {summary['rescue_complete_rate']:.3f}",
            "",
            "No candidate winner is defined. Numerical rescue is an attribution "
            "test; topology/function revision and multiple-round search remain "
            "separate subsequent milestones.",
        )
    )
    return "\n".join(lines) + "\n"


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _median(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return statistics.median(finite) if finite else None


def _number(value: Any) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{float(value):.6g}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"existing frozen file differs: {destination}")
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Any) -> None:
    _write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
