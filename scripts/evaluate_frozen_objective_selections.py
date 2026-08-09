"""Refit and test objective-selected structures after a manifest is frozen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
)
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import (
    FitConfig,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.rebuttal.artifacts import CandidateArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--frozen-selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-nfev", type=int, default=150)
    args = parser.parse_args()
    selections = json.loads(args.frozen_selection_manifest.read_text(encoding="utf-8"))
    if not isinstance(selections, list):
        raise SystemExit("frozen selection manifest must contain a JSON list")
    pool = {
        item.artifact_id: item
        for line in args.candidate_pool.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in (CandidateArtifact.model_validate_json(line),)
    }
    selected_ids = {str(item["artifact_id"]) for item in selections}
    missing = selected_ids - set(pool)
    if missing:
        raise SystemExit(
            f"selection manifest references unknown artifacts: {sorted(missing)}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for artifact_id in sorted(selected_ids):
        artifact = pool[artifact_id]
        result_path = args.output_root / artifact_id / "result.json"
        if result_path.is_file():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            row = _evaluate(args, artifact, result_path.parent / "test_access.claim")
        except Exception as exc:
            row = {
                "artifact_id": artifact.artifact_id,
                "benchmark_id": artifact.benchmark_id,
                "tier": artifact.tier,
                "seed": artifact.seed,
                "validation_mse": artifact.validation_mse,
                "test_mse": None,
                "term_count": artifact.term_count,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
            }
        result_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(row)
    with (args.output_root / "frozen_test_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        scalar_rows = [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
            for row in rows
        ]
        fields = (
            tuple(sorted({key for row in scalar_rows for key in row}))
            if rows
            else ("artifact_id", "test_mse")
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _evaluate(
    args: argparse.Namespace, artifact: CandidateArtifact, claim: Path
) -> dict:
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    from autoformalism.config import DataConfig

    data_config = DataConfig(
        root=args.data_root.expanduser().resolve(),
        benchmark_id=artifact.benchmark_id,
        tier=artifact.tier,
    )
    development = loader.load_development(data_config)
    execution_arguments = ExecutionArguments(
        data_root=data_config.root,
        benchmark_id=artifact.benchmark_id,
        tier=artifact.tier,
        seed=artifact.seed,
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
    compiled = compile_candidate(
        artifact.candidate, _context(execution_arguments, development)
    )
    checkpoint = json.loads(
        Path(artifact.source_checkpoint).read_text(encoding="utf-8")
    )
    search_fit = checkpoint.get("pruned_fit") or checkpoint["record"]["pruned_fit"]
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
        f"frozen:{artifact.artifact_id}",
    )
    settings = FitConfig(
        number_of_starts=1,
        random_seed=artifact.seed,
        integration_backend="solve_ivp",
        maximum_function_evaluations=args.max_nfev,
        maximum_wall_time_seconds=args.timeout_seconds,
    )
    fitted = fit_candidate(
        compiled,
        combined,
        development.validation,
        settings,
        initial_global_parameters=search_fit["global_parameters"],
    )
    if not fitted.success:
        raise RuntimeError(f"frozen refit failed for {artifact.artifact_id}")
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(artifact.artifact_id)
    test = loader.load_test(data_config)
    test_initials, test_metrics = evaluate_fitted_candidate(
        compiled,
        test,
        global_parameters=fitted.global_parameters,
        global_initial_conditions=fitted.global_initial_conditions,
        target_scales=fitted.target_scales,
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    return {
        "artifact_id": artifact.artifact_id,
        "benchmark_id": artifact.benchmark_id,
        "tier": artifact.tier,
        "seed": artifact.seed,
        "validation_mse": artifact.validation_mse,
        "test_mse": test_metrics.normalized_mse,
        "term_count": artifact.term_count,
        "status": "complete",
        "error": None,
        "stage": "complete",
        "frozen": {"candidate": artifact.candidate.model_dump(mode="json")},
        "final_fit": {
            "global_parameters": dict(fitted.global_parameters),
            "global_initial_conditions": dict(fitted.global_initial_conditions),
            "training_trajectory_initial_conditions": {
                key: dict(value)
                for key, value in fitted.training_trajectory_initial_conditions.items()
            },
            "target_scales": dict(fitted.target_scales),
        },
        "test_initials": {key: dict(value) for key, value in test_initials.items()},
    }


if __name__ == "__main__":
    main()
