#!/usr/bin/env python3
"""Freeze identity self-pairs for fit-free scientific audits of raw-agent models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_self_audit_pairs(
    candidates: Sequence[tuple[str, str, int, CandidateModel]],
    contexts: Mapping[tuple[str, str], ValidationContext],
) -> tuple[AdversarialPair, ...]:
    """Duplicate each structure so absolute questions get two blinded readings."""
    pairs = []
    for benchmark_id, tier, repetition, candidate in candidates:
        compile_candidate(candidate, contexts[(benchmark_id, tier)])
        identity = {
            "benchmark_id": benchmark_id,
            "tier": tier,
            "repetition": repetition,
            "candidate": candidate.model_dump(mode="json"),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        pairs.append(
            AdversarialPair(
                pair_id=f"rawaudit_{digest}",
                benchmark_id=benchmark_id,
                tier=tier,
                mutation_type="raw_agent_identity_scientific_audit",
                valid_candidate=candidate,
                adversarial_candidate=candidate,
            )
        )
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    raw_root = args.raw_runs_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    candidates = []
    sources = []
    tasks = set()
    for run in sorted(raw_root.iterdir()):
        config_path = run / "run_config.json"
        candidate_path = run / "candidate.json"
        if not config_path.is_file() or not candidate_path.is_file():
            continue
        config = _read_json(config_path)
        if config.get("provider") != args.provider or config.get("model") != args.model:
            continue
        benchmark_id = str(config["benchmark_id"])
        tier = str(config["tier"])
        candidates.append(
            (
                benchmark_id,
                tier,
                int(config["repetition"]),
                CandidateModel.model_validate_json(
                    candidate_path.read_text(encoding="utf-8")
                ),
            )
        )
        tasks.add((benchmark_id, tier))
        sources.append(
            {
                "run": str(run),
                "run_config_sha256": _sha256(config_path),
                "candidate_sha256": _sha256(candidate_path),
            }
        )
    registry = BenchmarkRegistry()
    contexts = {}
    for benchmark_id, tier in sorted(tasks):
        spec = registry.get(benchmark_id)
        development = BenchmarkLoader(registry).load_development(
            DataConfig(root=data_root, benchmark_id=benchmark_id, tier=tier)
        )
        contexts[(benchmark_id, tier)] = raw_agent_validation_context(
            development, spec
        )
    pairs = build_self_audit_pairs(candidates, contexts)
    if not pairs:
        raise ValueError("no matching raw-agent candidates were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{pair.model_dump_json()}\n" for pair in pairs),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_name(
        "raw_agent_scientific_audit_manifest.json"
    )
    manifest = {
        "schema_version": "raw-agent-scientific-audit-manifest-1",
        "status": "frozen_before_judge_calls",
        "pair_count": len(pairs),
        "pair_ids": [pair.pair_id for pair in pairs],
        "evaluation_scope": "structure_only_no_parameter_fitting",
        "pair_design": "identity_self_pair_for_repeated_absolute_assessment",
        "comparative_outcomes_interpretable": False,
        "accuracy_claimed": False,
        "source_runs": sources,
        "pairs_sha256": _sha256(args.output),
        "test_data_opened": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} fit-free scientific self-audit pairs")


if __name__ == "__main__":
    main()
