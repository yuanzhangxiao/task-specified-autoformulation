"""Evaluate one development-frozen failure-recovery selection exactly once."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
)
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, evaluate_fitted_candidate, fit_candidate
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "result.json"
    if result_path.is_file():
        print(result_path.read_text(encoding="utf-8"))
        return
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    validate_frozen_manifest(manifest)
    result = evaluate(args, manifest)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def validate_frozen_manifest(manifest: dict) -> None:
    if manifest.get("stage") != "development_selection_frozen":
        raise ValueError("manifest is not a frozen development selection")
    if manifest.get("uses_test_data") is not False:
        raise ValueError("manifest must explicitly record uses_test_data=false")
    if manifest.get("selection_metric") != "validation_normalized_mse":
        raise ValueError("unexpected recovery selection metric")


def evaluate(args: argparse.Namespace, manifest: dict) -> dict:
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    data_config = DataConfig(
        root=args.data_root.resolve(),
        benchmark_id=manifest["benchmark_id"],
        tier=manifest["tier"],
    )
    development = loader.load_development(data_config)
    execution_arguments = ExecutionArguments(
        data_root=data_config.root,
        benchmark_id=manifest["benchmark_id"],
        tier=manifest["tier"],
        seed=int(manifest["seed"]),
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=args.output_root,
        resume=False,
        dry_run=False,
        mock_llm=True,
        use_clean_observations=False,
    )
    candidate = CandidateModel.model_validate(manifest["candidate"])
    model = compile_candidate(candidate, _context(execution_arguments, development))
    combined = DatasetSplit(
        SplitName.TRAIN,
        tuple(
            replace(item, trajectory_id=f"train:{item.trajectory_id}")
            for item in development.train.trajectories
        )
        + tuple(
            replace(item, trajectory_id=f"validation:{item.trajectory_id}")
            for item in development.validation.trajectories
        ),
        f"recovery:{manifest['structural_hash']}",
    )
    settings = FitConfig(
        number_of_starts=1,
        random_seed=int(manifest["seed"]),
        integration_backend="solve_ivp",
        maximum_function_evaluations=args.max_nfev,
        maximum_wall_time_seconds=args.timeout_seconds,
    )
    fitted = fit_candidate(
        model,
        combined,
        development.validation,
        settings,
        initial_global_parameters=manifest["global_parameters"],
    )
    if not fitted.success:
        raise RuntimeError(f"recovery final refit failed: {fitted.message}")

    claim = args.output_root / "test_access.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(manifest["structural_hash"])
    test = loader.load_test(data_config)
    test_initials, test_metrics = evaluate_fitted_candidate(
        model,
        test,
        global_parameters=fitted.global_parameters,
        global_initial_conditions=fitted.global_initial_conditions,
        target_scales=fitted.target_scales,
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    if test_metrics.failed_trajectories:
        failed = ", ".join(test_metrics.failed_trajectories)
        raise RuntimeError(f"recovery test rollout failed for {failed}")
    return {
        "status": "complete",
        "stage": "complete",
        "benchmark_id": manifest["benchmark_id"],
        "tier": manifest["tier"],
        "seed": manifest["seed"],
        "candidate_id": manifest["candidate_id"],
        "structural_hash": manifest["structural_hash"],
        "selection_validation_normalized_mse": manifest[
            "validation_normalized_mse"
        ],
        "final_training_normalized_mse": fitted.training_metrics.normalized_mse,
        "test_normalized_mse": test_metrics.normalized_mse,
        "test_per_target_normalized_mse": dict(
            test_metrics.per_target_normalized_mse
        ),
        "test_failed_trajectories": list(test_metrics.failed_trajectories),
        "global_parameters": dict(fitted.global_parameters),
        "global_initial_conditions": dict(fitted.global_initial_conditions),
        "test_initial_conditions": {
            key: dict(value) for key, value in test_initials.items()
        },
        "final_fit": {
            "global_parameters": dict(fitted.global_parameters),
            "global_initial_conditions": dict(fitted.global_initial_conditions),
            "training_trajectory_initial_conditions": {
                key: dict(value)
                for key, value in fitted.training_trajectory_initial_conditions.items()
            },
        },
        "test_initials": {
            key: dict(value) for key, value in test_initials.items()
        },
        "frozen": {"candidate": manifest["candidate"]},
        "recovery_protocol": {
            "selection": "validation_normalized_mse",
            "initialization_profile": manifest["initialization_profile"],
            "final_backend": "solve_ivp",
            "final_max_nfev": args.max_nfev,
            "final_timeout_seconds": args.timeout_seconds,
            "new_llm_calls": 0,
        },
    }


if __name__ == "__main__":
    main()
