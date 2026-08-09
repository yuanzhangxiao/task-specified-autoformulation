#!/usr/bin/env python3
"""Refit exact-contract frozen structures on Phase-B development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic
from typing import Any

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.artifacts import (
    CandidateArtifact,
    candidate_warm_start_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-benchmark", required=True)
    parser.add_argument("--destination-benchmark", required=True)
    parser.add_argument("--tier", required=True, choices=("easy", "hard"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=20)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--refine-top-k", type=int, default=0)
    parser.add_argument("--refine-max-nfev", type=int, default=25)
    parser.add_argument("--refine-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        args.maximum_candidates < 1
        or args.max_nfev < 1
        or args.refine_top_k < 0
        or args.refine_max_nfev < 1
    ):
        raise SystemExit("candidate and fitting budgets must be positive")
    if args.timeout_seconds <= 0 or args.refine_timeout_seconds <= 0:
        raise SystemExit("timeout must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    artifacts = _load_pool(args.candidate_pool)
    source = [
        item
        for item in artifacts
        if item.benchmark_id == args.source_benchmark and item.tier == args.tier
    ]
    if not source:
        raise SystemExit("candidate pool contains no matching source artifacts")
    development, context = _development_context(args)
    eligibility = []
    eligible: dict[str, CandidateArtifact] = {}
    for artifact in sorted(source, key=lambda item: item.artifact_id):
        try:
            compile_candidate(artifact.candidate, context)
        except Exception as exc:
            eligibility.append(
                _eligibility_row(artifact, False, f"{type(exc).__name__}: {exc}")
            )
            continue
        eligibility.append(_eligibility_row(artifact, True, None))
        previous = eligible.get(artifact.structural_hash)
        if previous is None or (
            artifact.validation_mse,
            artifact.artifact_id,
        ) < (previous.validation_mse, previous.artifact_id):
            eligible[artifact.structural_hash] = artifact
    _write_jsonl(args.output_root / "eligibility.jsonl", eligibility)

    selected = sorted(
        eligible.values(),
        key=lambda item: (item.validation_mse, item.artifact_id),
    )[: args.maximum_candidates]
    if not selected:
        raise SystemExit("no exact-contract candidate is eligible for replay")
    results_root = args.output_root / "results"
    results_root.mkdir(exist_ok=True)
    for index, artifact in enumerate(selected, start=1):
        path = results_root / f"{artifact.artifact_id}.json"
        if path.is_file():
            continue
        row = _refit(
            args,
            artifact,
            development,
            context,
            max_nfev=args.max_nfev,
            timeout_seconds=args.timeout_seconds,
            stage="screen",
        )
        _write_json(path, row)
        print(
            f"replay {index}/{len(selected)} {artifact.candidate.candidate_id} "
            f"success={row['success']} val={row['validation_nmse']}",
            flush=True,
        )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_root.glob("*.json"))
    ]
    refined_rows = _refine_screening_winners(
        args,
        rows,
        {item.artifact_id: item for item in selected},
        development,
        context,
    )
    refined_by_id = {item["artifact_id"]: item for item in refined_rows}
    rows = [
        _better_result(item, refined_by_id.get(item["artifact_id"]))
        for item in rows
    ]
    successful = [item for item in rows if item["success"]]
    winner = min(
        successful,
        key=lambda item: (item["validation_nmse"], item["artifact_id"]),
        default=None,
    )
    manifest = {
        "schema_version": "phase_b_frozen_replay_v1",
        "stage": "development_selection_frozen" if winner else "no_valid_fit",
        "source_benchmark": args.source_benchmark,
        "destination_benchmark": args.destination_benchmark,
        "tier": args.tier,
        "uses_llm_calls": False,
        "uses_test_data": False,
        "fit_protocol": "open_loop_rollout",
        "derivative_regression": False,
        "screening_budget": {
            "maximum_candidates": args.maximum_candidates,
            "maximum_function_evaluations": args.max_nfev,
            "timeout_seconds": args.timeout_seconds,
        },
        "refinement_budget": {
            "top_k": args.refine_top_k,
            "maximum_function_evaluations": args.refine_max_nfev,
            "timeout_seconds": args.refine_timeout_seconds,
        },
        "source_artifacts": len(source),
        "exact_contract_artifacts": sum(item["eligible"] for item in eligibility),
        "unique_eligible_structures": len(eligible),
        "replayed_structures": len(rows),
        "refined_structures": len(refined_rows),
        "successful_replays": len(successful),
        "selected": winner,
    }
    _write_json(args.output_root / "replay_manifest.json", manifest)
    _write_report(args.output_root / "replay_report.md", manifest, eligibility)
    if winner is None:
        raise SystemExit("no frozen structure passed Phase-B rollout refitting")


def _development_context(args: argparse.Namespace):
    config = DataConfig(
        root=args.public_data_root.expanduser().resolve(),
        benchmark_id=args.destination_benchmark,
        tier=args.tier,
        use_clean_observations=False,
    )
    development = BenchmarkLoader().load_development(config)
    execution = ExecutionArguments(
        data_root=config.root,
        benchmark_id=config.benchmark_id,
        tier=config.tier,
        seed=args.seed,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=args.output_root,
        resume=False,
        dry_run=False,
        mock_llm=True,
        use_clean_observations=False,
        use_derivative_fit_fast_path=False,
    )
    return development, _context(execution, development)


def _refit(
    args,
    artifact,
    development,
    context,
    *,
    max_nfev: int,
    timeout_seconds: float,
    stage: str,
    preferred_parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = monotonic()
    try:
        compiled = compile_candidate(artifact.candidate, context)
        settings = FitConfig(
            number_of_starts=1,
            random_seed=args.seed,
            integration_backend="fixed_rk4",
            fixed_step_substeps=1,
            maximum_function_evaluations=max_nfev,
            maximum_wall_time_seconds=timeout_seconds,
            allow_derivative_regression=False,
        )
        preferred = preferred_parameters or candidate_warm_start_parameters(artifact)
        fitted = fit_candidate(
            compiled,
            development.train,
            development.validation,
            settings,
            initial_global_parameters=preferred or None,
        )
        return {
            "artifact_id": artifact.artifact_id,
            "candidate_id": artifact.candidate.candidate_id,
            "structural_hash": artifact.structural_hash,
            "source_checkpoint": artifact.source_checkpoint,
            "source_validation_nmse": artifact.validation_mse,
            "stage": stage,
            "success": fitted.success,
            "training_nmse": fitted.training_metrics.normalized_mse,
            "validation_nmse": fitted.validation_metrics.normalized_mse,
            "global_parameters": dict(fitted.global_parameters),
            "elapsed_seconds": monotonic() - started,
            "message": fitted.message,
            "error": None,
        }
    except Exception as exc:
        return {
            "artifact_id": artifact.artifact_id,
            "candidate_id": artifact.candidate.candidate_id,
            "structural_hash": artifact.structural_hash,
            "source_checkpoint": artifact.source_checkpoint,
            "source_validation_nmse": artifact.validation_mse,
            "stage": stage,
            "success": False,
            "training_nmse": None,
            "validation_nmse": None,
            "global_parameters": {},
            "elapsed_seconds": monotonic() - started,
            "message": "replay failed",
            "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
        }


def _refine_screening_winners(
    args,
    rows: list[dict[str, Any]],
    artifacts: dict[str, CandidateArtifact],
    development,
    context,
) -> list[dict[str, Any]]:
    if args.refine_top_k == 0:
        return []
    finalists = sorted(
        (item for item in rows if item["success"]),
        key=lambda item: (item["validation_nmse"], item["artifact_id"]),
    )[: args.refine_top_k]
    root = args.output_root / "refinement_results"
    root.mkdir(exist_ok=True)
    result = []
    for index, screening in enumerate(finalists, start=1):
        path = root / f"{screening['artifact_id']}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = _refit(
                args,
                artifacts[screening["artifact_id"]],
                development,
                context,
                max_nfev=args.refine_max_nfev,
                timeout_seconds=args.refine_timeout_seconds,
                stage="refine",
                preferred_parameters=screening["global_parameters"],
            )
            _write_json(path, row)
        result.append(row)
        print(
            f"refine {index}/{len(finalists)} {row['candidate_id']} "
            f"success={row['success']} val={row['validation_nmse']}",
            flush=True,
        )
    return result


def _better_result(
    screening: dict[str, Any], refinement: dict[str, Any] | None
) -> dict[str, Any]:
    """Retain a valid screening result unless refinement improves it."""
    if refinement is None or not refinement["success"]:
        return screening
    if not screening["success"]:
        return refinement
    if refinement["validation_nmse"] < screening["validation_nmse"]:
        return refinement
    return screening


def _eligibility_row(
    artifact: CandidateArtifact, eligible: bool, reason: str | None
) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "structural_hash": artifact.structural_hash,
        "candidate_id": artifact.candidate.candidate_id,
        "source_validation_nmse": artifact.validation_mse,
        "eligible": eligible,
        "reason": None if reason is None else reason[:4000],
    }


def _load_pool(path: Path) -> tuple[CandidateArtifact, ...]:
    return tuple(
        CandidateArtifact.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    eligibility: list[dict[str, Any]],
) -> None:
    failures: dict[str, int] = {}
    for item in eligibility:
        if item["eligible"]:
            continue
        reason = str(item["reason"]).split(" at ", maxsplit=1)[0]
        failures[reason] = failures.get(reason, 0) + 1
    selected = manifest["selected"]
    lines = [
        "# Phase-B frozen-structure rollout replay",
        "",
        "This development-only pilot made no LLM calls and did not open test data.",
        "",
        f"- source artifacts: {manifest['source_artifacts']}",
        f"- exact-contract artifacts: {manifest['exact_contract_artifacts']}",
        f"- unique eligible structures: {manifest['unique_eligible_structures']}",
        f"- replayed structures: {manifest['replayed_structures']}",
        f"- refined structures: {manifest['refined_structures']}",
        f"- successful rollout refits: {manifest['successful_replays']}",
        "",
        "## Eligibility failures",
        "",
        *(
            [f"- {count}: {reason}" for reason, count in sorted(failures.items())]
            or ["- none"]
        ),
        "",
        "## Validation-selected replay",
        "",
        (
            "- none"
            if selected is None
            else (
                f"- `{selected['candidate_id']}`: validation NMSE "
                f"{selected['validation_nmse']:.6g}"
            )
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
