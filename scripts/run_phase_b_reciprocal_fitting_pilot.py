#!/usr/bin/env python3
"""Run one frozen shared-candidate reciprocal-coordinate fitting task."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import monotonic, process_time
from typing import Any

from autoformalism.data import attach_exact_derivative_overlay
from autoformalism.execution import ExecutionArguments, _context, _load_inputs
from autoformalism.expressions import (
    certify_reciprocal_transformations,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.reciprocal_fitting_pilot import (
    canonical_reciprocal_fitting_plan_sha256,
    load_reciprocal_fitting_pilot_plan,
    reciprocal_fitting_task_identity,
)
from autoformalism.schemas import CandidateModel


def run_task(
    experiment_root: Path,
    public_data_root: Path,
    task_index: int,
) -> dict[str, object]:
    """Fit one candidate under one condition and checkpoint every outcome."""
    experiment_root = experiment_root.expanduser().resolve()
    public_data_root = public_data_root.expanduser().resolve()
    output = experiment_root / "tasks" / f"task_{task_index:03d}.json"
    plan_path = experiment_root / "frozen" / "plan.json"
    plan = load_reciprocal_fitting_pilot_plan(plan_path)
    plan_sha256 = canonical_reciprocal_fitting_plan_sha256(plan)
    if output.is_file():
        payload = _read_object(output)
        if (
            payload.get("schema_version")
            != "phase-b-reciprocal-fitting-pilot-task-1"
            or payload.get("status") != "complete"
            or payload.get("task_index") != task_index
            or payload.get("plan_sha256") != plan_sha256
        ):
            raise ValueError(f"reciprocal pilot checkpoint differs: {output}")
        return payload
    freeze = _read_object(experiment_root / "frozen" / "freeze_manifest.json")
    if freeze.get("plan_sha256") != plan_sha256:
        raise ValueError("reciprocal fitting freeze differs from plan")
    condition, cell, repetition, candidate_index = reciprocal_fitting_task_identity(
        plan, task_index
    )
    candidate_manifest = {
        int(item["candidate_index"]): item
        for item in _read_jsonl(experiment_root / "frozen" / "candidate_manifest.jsonl")
    }
    derivative_manifest = {
        (str(item["benchmark_id"]), str(item["split"])): item
        for item in _read_jsonl(
            experiment_root / "frozen" / "derivative_manifest.jsonl"
        )
    }
    candidate_path = (
        experiment_root
        / "frozen"
        / "candidates"
        / f"candidate_{candidate_index:03d}.json"
    )
    if _sha256(candidate_path) != candidate_manifest[candidate_index][
        "frozen_candidate_sha256"
    ]:
        raise ValueError("frozen reciprocal pilot candidate differs")
    candidate = CandidateModel.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
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
    if hashlib.sha256(prompt.encode()).hexdigest() != cell.public_prompt_sha256:
        raise ValueError("reciprocal pilot public prompt differs")
    train_overlay = (
        experiment_root
        / "frozen"
        / "derivatives"
        / cell.benchmark_id
        / "train.csv"
    )
    training = attach_exact_derivative_overlay(
        dataset.train,
        train_overlay,
        expected_sha256=str(
            derivative_manifest[(cell.benchmark_id, "train")][
                "derivative_overlay_sha256"
            ]
        ),
    )
    validation = dataset.validation
    context = _context(arguments, dataset)
    settings = FitConfig(
        number_of_starts=condition.number_of_starts,
        random_seed=repetition,
        integration_backend="fixed_rk4",
        allow_derivative_regression=condition.allow_derivative_regression,
        parameter_fit_strategy=condition.parameter_fit_strategy,
        derivative_ridge_regularization=1e-8,
        # Preserve the frozen v1 pilot's bounded inner solve. Later profiled
        # experiments use the new suggested-range policy explicitly.
        affine_parameter_bound_policy="hard",
        maximum_function_evaluations=condition.maximum_function_evaluations,
        maximum_wall_time_seconds=condition.maximum_wall_time_seconds,
        use_certified_reciprocal_coordinates=(
            condition.use_certified_reciprocal_coordinates
        ),
    )
    started = monotonic()
    cpu_started = process_time()
    try:
        model = compile_candidate(candidate, context)
        reciprocal = certify_reciprocal_transformations(model.validated)
        fitted = fit_candidate(model, training, validation, settings)
    except Exception as exc:  # proposer candidates are untrusted experiment inputs
        fit = {
            "fit_contract_compatible": False,
            "fit_success": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:8000],
            "certified_reciprocal_transformations": [],
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
            "function_evaluations": sum(
                item.function_evaluations for item in fitted.diagnostics
            ),
            "integration_failures": sum(
                item.integration_failures for item in fitted.diagnostics
            ),
            "certified_reciprocal_transformations": [
                asdict(item) for item in reciprocal
            ],
        }
    payload = {
        "schema_version": "phase-b-reciprocal-fitting-pilot-task-1",
        "status": "complete",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_available_to_fitter": False,
        "exact_training_observed_derivatives_supplied": True,
        "validation_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "plan_sha256": plan_sha256,
        "task_index": task_index,
        "candidate_index": candidate_index,
        "candidate_sha256": _sha256(candidate_path),
        "condition": condition.model_dump(mode="json"),
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "repetition": repetition,
        "train_public_fingerprint": dataset.train.fingerprint,
        "validation_public_fingerprint": dataset.validation.fingerprint,
        "train_derivative_fingerprint": training.fingerprint,
        "fit_wall_seconds": monotonic() - started,
        "fit_process_cpu_seconds": process_time() - cpu_started,
        **fit,
    }
    _write_once(
        output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    return payload


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
            raise ValueError(f"reciprocal pilot checkpoint differs: {path}")
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
