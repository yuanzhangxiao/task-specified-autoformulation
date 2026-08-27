#!/usr/bin/env python3
"""Freeze unlabeled raw-agent versus Autoformalism mechanism pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.expressions import (
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_method_pairs(
    raw_candidates: Sequence[tuple[Path, str, str, int, CandidateModel]],
    references: Mapping[tuple[str, str], tuple[Path, CandidateModel]],
    contexts: Mapping[tuple[str, str], ValidationContext],
) -> tuple[AdversarialPair, ...]:
    """Build identity-blinded, unlabeled scientific-comparison pairs."""
    output = []
    for _source, benchmark_id, tier, repetition, raw_candidate in raw_candidates:
        task = (benchmark_id, tier)
        if task not in references:
            continue
        _reference_source, reference_candidate = references[task]
        context = contexts[task]
        reference, _ = repair_protected_declarations(reference_candidate, context)
        raw, _ = repair_protected_declarations(raw_candidate, context)
        compile_candidate(reference, context)
        compile_candidate(raw, context)
        identity = {
            "schema_version": "raw-data-agent-method-pair-1",
            "benchmark_id": benchmark_id,
            "tier": tier,
            "repetition": repetition,
            "raw_candidate": raw.model_dump(mode="json"),
            "reference_candidate": reference.model_dump(mode="json"),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        output.append(
            AdversarialPair(
                pair_id=f"rawmethod_{digest}",
                benchmark_id=benchmark_id,
                tier=tier,
                mutation_type="raw_agent_vs_autoformalism_unlabeled",
                valid_candidate=reference,
                adversarial_candidate=raw,
            )
        )
    pair_ids = [pair.pair_id for pair in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("raw-agent comparison pair identifiers must be unique")
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-runs-root", type=Path, required=True)
    parser.add_argument(
        "--reference-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    raw_root = args.raw_runs_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    reference_paths = tuple(
        path.expanduser().resolve() for path in args.reference_summary
    )
    if len(reference_paths) != len(set(reference_paths)):
        raise ValueError("reference summary paths must be unique")

    references: dict[tuple[str, str], tuple[Path, CandidateModel]] = {}
    for path in reference_paths:
        payload = _read_json(path)
        task = (str(payload["benchmark_id"]), str(payload["tier"]))
        if task in references:
            raise ValueError(f"duplicate reference task: {task}")
        references[task] = (
            path,
            CandidateModel.model_validate(payload["selected_candidate"]),
        )

    raw_candidates = []
    raw_sources = []
    for run in sorted(raw_root.iterdir()):
        config_path = run / "run_config.json"
        candidate_path = run / "candidate.json"
        if (
            not run.is_dir()
            or not config_path.is_file()
            or not candidate_path.is_file()
        ):
            continue
        config = _read_json(config_path)
        if args.provider is not None and config.get("provider") != args.provider:
            continue
        if args.model is not None and config.get("model") != args.model:
            continue
        raw_candidates.append(
            (
                candidate_path,
                str(config["benchmark_id"]),
                str(config["tier"]),
                int(config["repetition"]),
                CandidateModel.model_validate_json(
                    candidate_path.read_text(encoding="utf-8")
                ),
            )
        )
        raw_sources.append(
            {
                "run": str(run),
                "run_config_sha256": _sha256(config_path),
                "candidate_sha256": _sha256(candidate_path),
            }
        )

    registry = BenchmarkRegistry()
    contexts = {}
    for benchmark_id, tier in references:
        spec = registry.get(benchmark_id)
        development = BenchmarkLoader(registry).load_development(
            DataConfig(root=data_root, benchmark_id=benchmark_id, tier=tier)
        )
        contexts[(benchmark_id, tier)] = raw_agent_validation_context(
            development, spec
        )
    pairs = build_method_pairs(raw_candidates, references, contexts)
    if not pairs:
        raise ValueError("no raw candidates matched a reference benchmark and tier")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "raw_agent_method_pairs_manifest.json"
    )
    manifest = {
        "schema_version": "raw-data-agent-method-pairs-manifest-1",
        "status": "frozen_before_judge_calls",
        "pair_count": len(pairs),
        "pair_ids": [pair.pair_id for pair in pairs],
        "pair_truth": "unlabeled",
        "candidate_identity_disclosure": False,
        "fit_metrics_disclosed_to_judge": False,
        "source_raw_runs": raw_sources,
        "reference_summaries": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in reference_paths
        ],
        "pairs_sha256": _sha256(args.output),
        "test_data_opened": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(pairs)} unlabeled mechanism pairs to {args.output}; "
        f"references={len(references)}"
    )


if __name__ == "__main__":
    main()
