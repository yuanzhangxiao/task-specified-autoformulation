#!/usr/bin/env python3
"""Run one frozen public-only CasADi initializer comparison task."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import monotonic, process_time
from typing import Any

from autoformalism.execution import ExecutionArguments, _context, _load_inputs
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.casadi_initializer_pilot import (
    canonical_casadi_initializer_plan_sha256,
    casadi_initializer_task_identity,
    load_casadi_initializer_pilot_plan,
)
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity


def run_task(
    experiment_root: Path,
    public_data_root: Path,
    task_index: int,
) -> dict[str, object]:
    """Fit one frozen candidate and checkpoint successful or failed outcomes."""
    experiment_root = experiment_root.expanduser().resolve()
    public_data_root = public_data_root.expanduser().resolve()
    output = experiment_root / "tasks" / f"task_{task_index:03d}.json"
    plan = load_casadi_initializer_pilot_plan(
        experiment_root / "frozen" / "plan.json"
    )
    plan_sha256 = canonical_casadi_initializer_plan_sha256(plan)
    if output.is_file():
        payload = _read_object(output)
        if (
            payload.get("schema_version")
            != "phase-b-casadi-initializer-pilot-task-1"
            or payload.get("status") != "complete"
            or payload.get("task_index") != task_index
            or payload.get("plan_sha256") != plan_sha256
        ):
            raise ValueError(f"CasADi pilot checkpoint differs: {output}")
        return payload

    freeze = _read_object(experiment_root / "frozen" / "freeze_manifest.json")
    if freeze.get("plan_sha256") != plan_sha256:
        raise ValueError("CasADi pilot freeze differs from plan")
    condition, cell, repetition, candidate_index = (
        casadi_initializer_task_identity(plan, task_index)
    )
    candidate_manifest = {
        int(item["candidate_index"]): item
        for item in _read_jsonl(experiment_root / "frozen" / "candidate_manifest.jsonl")
    }
    public_manifest = {
        str(item["benchmark_id"]): item
        for item in _read_jsonl(
            experiment_root / "frozen" / "public_input_manifest.jsonl"
        )
    }
    candidate_path = (
        experiment_root
        / "frozen"
        / "candidates"
        / f"candidate_{candidate_index:03d}.json"
    )
    candidate_row = candidate_manifest[candidate_index]
    if _sha256(candidate_path) != candidate_row["frozen_candidate_sha256"]:
        raise ValueError("frozen CasADi pilot candidate differs")
    candidate = CandidateModel.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
    source_identity = candidate_identity(candidate)
    candidate, removed_range_field_count = _without_parameter_ranges(candidate)
    fit_identity = candidate_identity(candidate)
    if (
        source_identity.topology_sha256 != fit_identity.topology_sha256
        or source_identity.functional_sha256 != fit_identity.functional_sha256
    ):
        raise ValueError("removing parameter ranges changed scientific identity")

    arguments = ExecutionArguments(
        data_root=public_data_root,
        benchmark_id=cell.benchmark_id,
        tier=cell.tier,
        seed=repetition,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=experiment_root / ".context",
        resume=False,
        dry_run=True,
        mock_llm=True,
        use_clean_observations=False,
        use_judge=False,
        development_only=True,
    )
    dataset, _test_loader, prompt, _judge_prompt = _load_inputs(arguments)
    public_root = public_data_root / "phase_b_v1" / cell.benchmark_id
    public_row = public_manifest[cell.benchmark_id]
    if hashlib.sha256(prompt.encode()).hexdigest() != cell.public_prompt_sha256:
        raise ValueError("CasADi pilot public prompt differs")
    for split in ("train", "validation"):
        if _sha256(public_root / f"{split}.csv") != public_row[f"{split}_sha256"]:
            raise ValueError(f"CasADi pilot public {split} split differs")

    settings = FitConfig(
        number_of_starts=condition.number_of_starts,
        random_seed=repetition,
        integration_backend="fixed_rk4",
        allow_derivative_regression=False,
        parameter_fit_strategy="bounded_nonlinear",
        nonlinear_initializer=condition.nonlinear_initializer,
        nonlinear_initializer_failure_policy=(
            condition.nonlinear_initializer_failure_policy
        ),
        casadi_shooting_interval_count=condition.casadi_shooting_interval_count,
        casadi_maximum_intervals_per_trajectory=(
            condition.casadi_maximum_intervals_per_trajectory
        ),
        casadi_maximum_iterations=condition.casadi_maximum_iterations,
        casadi_maximum_wall_time_seconds=max(
            condition.initializer_wall_time_seconds, 1.0
        ),
        affine_parameter_bound_policy="suggested",
        runtime_parameter_start_center=1.0,
        runtime_parameter_start_half_width=2.0,
        maximum_function_evaluations=condition.maximum_function_evaluations,
        maximum_wall_time_seconds=condition.core_fit_wall_time_seconds,
    )
    started = monotonic()
    cpu_started = process_time()
    try:
        model = compile_candidate(candidate, _context(arguments, dataset))
        fitted = fit_candidate(model, dataset.train, dataset.validation, settings)
    except Exception as exc:  # frozen proposer candidates remain untrusted input
        fit = {
            "fit_contract_compatible": False,
            "fit_success": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:8000],
            "initialization_diagnostics": [],
            "function_evaluations": 0,
            "integration_failures": 0,
        }
    else:
        fit = {
            "fit_contract_compatible": True,
            "fit_success": fitted.success,
            "error_type": None,
            "error": fitted.message,
            "global_parameters": dict(fitted.global_parameters),
            "training_normalized_mse": fitted.training_metrics.normalized_mse,
            "validation_normalized_mse": fitted.validation_metrics.normalized_mse,
            "diagnostics": [asdict(item) for item in fitted.diagnostics],
            "initialization_diagnostics": [
                asdict(item) for item in fitted.initialization_diagnostics
            ],
            "function_evaluations": sum(
                item.function_evaluations for item in fitted.diagnostics
            ),
            "integration_failures": sum(
                item.integration_failures for item in fitted.diagnostics
            ),
        }
    payload = {
        "schema_version": "phase-b-casadi-initializer-pilot-task-1",
        "status": "complete",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_available_to_fitter": False,
        "observed_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "plan_sha256": plan_sha256,
        "task_index": task_index,
        "candidate_index": candidate_index,
        "candidate_sha256": _sha256(candidate_path),
        "source_candidate_identity": source_identity.model_dump(mode="json"),
        "fit_candidate_identity": fit_identity.model_dump(mode="json"),
        "legacy_parameter_range_field_count_removed": removed_range_field_count,
        "condition": condition.model_dump(mode="json"),
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "repetition": repetition,
        "train_public_fingerprint": dataset.train.fingerprint,
        "validation_public_fingerprint": dataset.validation.fingerprint,
        "fit_wall_seconds": monotonic() - started,
        "fit_process_cpu_seconds": process_time() - cpu_started,
        **fit,
    }
    _write_once(
        output,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def _without_parameter_ranges(
    candidate: CandidateModel,
) -> tuple[CandidateModel, int]:
    """Remove obsolete proposer numeric hints without changing the model."""
    removed = sum(
        int(item.bounds is not None) + int(item.initialization_range is not None)
        for item in candidate.parameters
    )
    parameters = tuple(
        item.model_copy(update={"bounds": None, "initialization_range": None})
        for item in candidate.parameters
    )
    return candidate.model_copy(update={"parameters": parameters}), removed


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"CasADi pilot checkpoint differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    result = run_task(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
