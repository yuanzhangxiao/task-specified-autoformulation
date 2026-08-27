#!/usr/bin/env python3
"""Apply the checkpointed common evaluator to one frozen candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoformalism.baselines.common_refit import (
    CommonRefitConfig,
    evaluate_common_refit,
)
from autoformalism.baselines.raw_data_agent import (
    fit_result_payload,
    raw_agent_validation_context,
)
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.schemas import CandidateModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-run",
        type=Path,
        help="Raw-agent run directory containing run_config.json and candidate.json.",
    )
    source.add_argument(
        "--source-summary",
        type=Path,
        help="Autoformalism summary.json containing selected_candidate.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenCandidateSource:
    """One immutable candidate plus the metadata needed by the evaluator."""

    source_kind: str
    source_path: Path
    run_name: str
    benchmark_id: str
    tier: str
    provider: str
    model: str
    repetition: int
    candidate: CandidateModel
    source_sha256: str
    candidate_sha256: str


def _candidate_digest(candidate: CandidateModel) -> str:
    payload = candidate.model_dump_json().encode()
    return hashlib.sha256(payload).hexdigest()


def _load_frozen_source(
    *, source_run: Path | None, source_summary: Path | None
) -> FrozenCandidateSource:
    """Load either baseline output form without consulting generated metrics."""
    if source_run is not None:
        run = source_run.expanduser().resolve()
        config_path = run / "run_config.json"
        candidate_path = run / "candidate.json"
        for path in (config_path, candidate_path):
            if not path.is_file():
                raise ValueError(f"required input is missing: {path}")
        config = _read_json(config_path)
        candidate = CandidateModel.model_validate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        return FrozenCandidateSource(
            source_kind="raw_data_agent_run",
            source_path=run,
            run_name=run.name,
            benchmark_id=str(config["benchmark_id"]),
            tier=str(config["tier"]),
            provider=str(config["provider"]),
            model=str(config["model"]),
            repetition=int(config["repetition"]),
            candidate=candidate,
            source_sha256=_sha256(config_path),
            candidate_sha256=_sha256(candidate_path),
        )
    if source_summary is None:  # pragma: no cover - argparse enforces this.
        raise ValueError("one frozen candidate source is required")
    summary_path = source_summary.expanduser().resolve()
    if not summary_path.is_file():
        raise ValueError(f"required input is missing: {summary_path}")
    summary = _read_json(summary_path)
    if summary.get("selected_candidate") is None:
        raise ValueError(f"summary has no selected_candidate: {summary_path}")
    candidate = CandidateModel.model_validate(summary["selected_candidate"])
    repetition = int(summary.get("seed", 0))
    return FrozenCandidateSource(
        source_kind="autoformalism_summary",
        source_path=summary_path,
        run_name=f"autoformalism_{summary_path.parent.name}",
        benchmark_id=str(summary["benchmark_id"]),
        tier=str(summary["tier"]),
        provider="autoformalism",
        model=str(summary.get("proposer_model", "search_pipeline")),
        repetition=repetition,
        candidate=candidate,
        source_sha256=_sha256(summary_path),
        candidate_sha256=_candidate_digest(candidate),
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seeded_config(payload: dict[str, Any], seed: int) -> CommonRefitConfig:
    normalized = {
        "schema_version": payload["schema_version"],
        "screening_fit": dict(payload["screening_fit"]),
        "final_fit": dict(payload["final_fit"]),
    }
    normalized["screening_fit"]["random_seed"] = seed
    normalized["final_fit"]["random_seed"] = seed
    return CommonRefitConfig.model_validate(normalized)


def main() -> None:
    args = _parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    protocol_path = args.protocol_config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not protocol_path.is_file():
        raise SystemExit(f"required input is missing: {protocol_path}")
    try:
        source = _load_frozen_source(
            source_run=args.source_run,
            source_summary=args.source_summary,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    protocol = _read_json(protocol_path)
    config = _seeded_config(protocol, source.repetition)
    run_directory = output_root / source.run_name
    refit_config = {
        "schema_version": "common-candidate-refit-run-1",
        "source_kind": source.source_kind,
        "source_path": str(source.source_path),
        "source_sha256": source.source_sha256,
        "source_candidate_sha256": source.candidate_sha256,
        "protocol_config_sha256": _sha256(protocol_path),
        "benchmark_id": source.benchmark_id,
        "tier": source.tier,
        "provider": source.provider,
        "model": source.model,
        "repetition": source.repetition,
        "common_refit": config.model_dump(mode="json"),
        "test_data_opened": False,
        "pruning_applied": False,
    }
    saved_config = run_directory / "refit_config.json"
    if saved_config.is_file():
        if _read_json(saved_config) != refit_config:
            raise SystemExit("refit resume configuration differs from checkpoint")
        status_path = run_directory / "status.json"
        if status_path.is_file():
            print(status_path.read_text(encoding="utf-8"), end="")
            return
    _atomic_json(saved_config, refit_config)
    if args.dry_run:
        status = {"status": "dry_run", "run_directory": str(run_directory)}
        _atomic_json(run_directory / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    registry = BenchmarkRegistry()
    spec = registry.get(source.benchmark_id)
    dataset = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=data_root,
            benchmark_id=source.benchmark_id,
            tier=source.tier,
        )
    )
    context = raw_agent_validation_context(dataset, spec)
    try:
        result = evaluate_common_refit(source.candidate, dataset, context, config)
        evaluation = {
            "schema_version": "common-candidate-refit-evaluation-1",
            "candidate_id": result.candidate.candidate_id,
            "repairs": list(result.repairs),
            "validation_warnings": list(result.warnings),
            "screening_fit": fit_result_payload(result.screening_fit),
            "final_fit": (
                None
                if result.final_fit is None
                else fit_result_payload(result.final_fit)
            ),
            "test_data_opened": False,
            "pruning_applied": False,
        }
        final = result.final_fit
        status = {
            "status": (
                "complete"
                if final is not None and final.success
                else (
                    "screening_fit_failed"
                    if not result.screening_fit.success
                    else "final_fit_failed"
                )
            ),
            "candidate_id": result.candidate.candidate_id,
            "screening_success": result.screening_fit.success,
            "screening_validation_normalized_mse": (
                result.screening_fit.validation_metrics.normalized_mse
            ),
            "final_success": None if final is None else final.success,
            "final_validation_normalized_mse": (
                None if final is None else final.validation_metrics.normalized_mse
            ),
            "test_data_opened": False,
        }
        _atomic_json(
            run_directory / "candidate.json",
            result.candidate.model_dump(mode="json"),
        )
        _atomic_json(run_directory / "evaluation.json", evaluation)
        _atomic_json(run_directory / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "test_data_opened": False,
        }
        _atomic_json(run_directory / "status.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
