#!/usr/bin/env python3
"""Adapt prespecified method artifacts into common frozen evaluation subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterOutcome,
    SourceAdapterRequest,
    adapt_source,
    source_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    requests = _read_requests(args.requests)
    subjects, outcomes = export_requests(requests, args.data_root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_path = output_root / "frozen_evaluation_subjects.jsonl"
    outcomes_path = output_root / "source_adapter_outcomes.jsonl"
    _write_jsonl(subjects_path, subjects)
    _write_jsonl(outcomes_path, outcomes)
    manifest = {
        "schema_version": "phase-b-source-adapter-manifest-1",
        "status": "complete",
        "request_count": len(requests),
        "adapted_count": len(subjects),
        "failed_count": sum(item.status == "failed" for item in outcomes),
        "selection_frozen": True,
        "test_data_opened": False,
        "requests_sha256": _sha256(args.requests),
        "subjects_sha256": _sha256(subjects_path),
        "outcomes_sha256": _sha256(outcomes_path),
    }
    _write_text_atomic(
        output_root / "source_adapter_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"adapted {len(subjects)} of {len(requests)} requested artifacts; "
        f"failures={manifest['failed_count']}"
    )


def export_requests(
    requests: tuple[SourceAdapterRequest, ...],
    data_root: Path,
) -> tuple[tuple[object, ...], tuple[SourceAdapterOutcome, ...]]:
    """Adapt every request while retaining source-level completion failures."""
    root = data_root.expanduser().resolve()
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    contexts: dict[tuple[str, str], ValidationContext] = {}
    subjects: list[object] = []
    outcomes: list[SourceAdapterOutcome] = []
    subject_ids: set[str] = set()
    for request in requests:
        try:
            benchmark_id, tier, _ = source_identity(request)
            key = (benchmark_id, tier)
            if key not in contexts:
                config = DataConfig(
                    root=root,
                    benchmark_id=benchmark_id,
                    tier=tier,
                )
                development = loader.load_development(config)
                contexts[key] = raw_agent_validation_context(
                    development,
                    registry.get(benchmark_id),
                )
            subject = adapt_source(request, contexts[key])
            if subject.subject_id in subject_ids:
                raise ValueError(f"duplicate adapted subject: {subject.subject_id}")
            subject_ids.add(subject.subject_id)
            subjects.append(subject)
            outcomes.append(
                SourceAdapterOutcome(
                    request_id=request.request_id,
                    source_kind=request.source_kind,
                    source_path=str(request.source_path),
                    status="adapted",
                    subject_id=subject.subject_id,
                )
            )
        except Exception as exc:  # source artifacts are untrusted inputs
            outcomes.append(
                SourceAdapterOutcome(
                    request_id=request.request_id,
                    source_kind=request.source_kind,
                    source_path=str(request.source_path),
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
    return tuple(subjects), tuple(outcomes)


def _read_requests(path: Path) -> tuple[SourceAdapterRequest, ...]:
    requests: list[SourceAdapterRequest] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            requests.append(SourceAdapterRequest.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid request at line {line_number}: {exc}") from exc
    identifiers = [item.request_id for item in requests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("source adapter request identifiers must be unique")
    return tuple(requests)


def _write_jsonl(path: Path, values: tuple[object, ...]) -> None:
    lines = [value.model_dump_json() for value in values]  # type: ignore[attr-defined]
    _write_text_atomic(path, "".join(f"{line}\n" for line in lines))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
