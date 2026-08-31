"""Run one frozen, development-only search fit-recovery case."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Any

from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
)
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.search_fit_recovery import (
    canonical_plan_sha256,
    load_search_fit_recovery_plan,
    verify_source_plan,
)
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()

    plan = load_search_fit_recovery_plan(args.config)
    verify_source_plan(plan, args.search_root)
    if not 0 <= args.task_index < len(plan.cases):
        raise ValueError(f"task index is out of range: {args.task_index}")
    case = plan.cases[args.task_index]
    run = args.search_root / case.run_directory
    if not run.is_dir():
        raise ValueError(f"missing frozen search run: {run}")

    loader = BenchmarkLoader(BenchmarkRegistry())
    data_config = DataConfig(
        root=args.data_root.resolve(),
        benchmark_id=case.benchmark_id,
        tier=case.tier,
    )
    development = loader.load_development(data_config)
    context = _context(
        ExecutionArguments(
            data_root=data_config.root,
            benchmark_id=case.benchmark_id,
            tier=case.tier,
            seed=case.seed,
            proposer_model=None,
            judge_model=None,
            iteration_budget=1,
            beam_size=1,
            output_root=args.output_root,
            resume=False,
            dry_run=False,
            mock_llm=True,
            use_clean_observations=False,
            development_only=True,
        ),
        development,
    )
    results: list[dict[str, Any]] = []
    profiles = plan.profiles_for(case.mode)
    for round_index in case.round_indices:
        checkpoint = run / "checkpoints" / f"round_{round_index:04d}.json"
        payload = _read_object(checkpoint)
        candidate, preferred = _candidate_and_parameters(
            payload,
            mode=case.mode,
            final_checkpoint=run / "checkpoints" / "final.json",
        )
        compiled = compile_candidate(candidate, context)
        for profile in profiles:
            training = (
                _combined_training(development.train, development.validation)
                if case.mode == "final_refit"
                else development.train
            )
            settings = FitConfig(
                number_of_starts=profile.number_of_starts,
                random_seed=case.seed,
                integration_backend=profile.integration_backend,
                maximum_function_evaluations=(
                    profile.maximum_function_evaluations
                ),
                maximum_wall_time_seconds=profile.maximum_wall_time_seconds,
                allow_derivative_regression=False,
            )
            started = monotonic()
            fitted = fit_candidate(
                compiled,
                training,
                development.validation,
                settings,
                initial_global_parameters=preferred,
            )
            result = {
                "case_id": case.case_id,
                "benchmark_id": case.benchmark_id,
                "tier": case.tier,
                "seed": case.seed,
                "mode": case.mode,
                "round_index": round_index,
                "candidate_id": candidate.candidate_id,
                "profile_id": profile.profile_id,
                "success": fitted.success,
                "elapsed_seconds": monotonic() - started,
                "training_normalized_mse": (
                    fitted.training_metrics.normalized_mse
                ),
                "validation_normalized_mse": (
                    fitted.validation_metrics.normalized_mse
                ),
                "global_parameters": dict(fitted.global_parameters),
                "function_evaluations": sum(
                    item.function_evaluations for item in fitted.diagnostics
                ),
                "integration_failures": sum(
                    item.integration_failures for item in fitted.diagnostics
                ),
                "message": fitted.message,
            }
            results.append(result)
            print(
                f"{case.case_id} round={round_index} "
                f"profile={profile.profile_id} success={fitted.success} "
                f"validation_nmse={fitted.validation_metrics.normalized_mse:.6g}",
                flush=True,
            )
            if fitted.success and plan.stop_after_first_success:
                break

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"task_{args.task_index}.json"
    _write_object(
        output,
        {
            "schema_version": "phase-b-search-fit-recovery-task-1",
            "status": "complete",
            "development_only": True,
            "new_llm_calls": False,
            "test_data_opened": False,
            "task_index": args.task_index,
            "plan_sha256": canonical_plan_sha256(plan),
            "case": case.model_dump(mode="json"),
            "results": results,
        },
    )


def _candidate_and_parameters(
    checkpoint: dict[str, Any],
    *,
    mode: str,
    final_checkpoint: Path,
) -> tuple[CandidateModel, dict[str, float] | None]:
    record = checkpoint.get("record")
    if mode == "final_refit":
        final = _read_object(final_checkpoint)
        frozen = final.get("frozen")
        candidate_payload = (
            frozen.get("candidate") if isinstance(frozen, dict) else None
        )
        if candidate_payload is None and isinstance(record, dict):
            candidate_payload = record.get("pruned_candidate")
        fit_payload = record.get("pruned_fit") if isinstance(record, dict) else None
    else:
        candidate_payload = checkpoint.get("candidate")
        fit_payload = checkpoint.get("fit")
    if not isinstance(candidate_payload, dict):
        raise ValueError("frozen checkpoint has no recoverable candidate")
    preferred = None
    if isinstance(fit_payload, dict) and isinstance(
        fit_payload.get("global_parameters"), dict
    ):
        preferred = {
            str(key): float(value)
            for key, value in fit_payload["global_parameters"].items()
        }
    return CandidateModel.model_validate(candidate_payload), preferred


def _combined_training(
    training: DatasetSplit, validation: DatasetSplit
) -> DatasetSplit:
    fingerprint = hashlib.sha256(
        f"{training.fingerprint}:{validation.fingerprint}".encode()
    ).hexdigest()
    return DatasetSplit(
        SplitName.TRAIN,
        tuple(
            replace(item, trajectory_id=f"train:{item.trajectory_id}")
            for item in training.trajectories
        )
        + tuple(
            replace(item, trajectory_id=f"validation:{item.trajectory_id}")
            for item in validation.trajectories
        ),
        fingerprint,
    )


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing frozen checkpoint: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a JSON object: {path}")
    return payload


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
