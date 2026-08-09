#!/usr/bin/env python3
"""Deterministically merge Phase-B screening shards and refine global winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.replay_phase_b_frozen_candidates import (
        _better_result,
        _build_manifest,
        _development_context,
        _load_pool,
        _prepare_candidates,
        _refine_screening_winners,
        _write_json,
        _write_jsonl,
        _write_report,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from replay_phase_b_frozen_candidates import (  # type: ignore[no-redef]
        _better_result,
        _build_manifest,
        _development_context,
        _load_pool,
        _prepare_candidates,
        _refine_screening_winners,
        _write_json,
        _write_jsonl,
        _write_report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-benchmark", required=True)
    parser.add_argument("--destination-benchmark", required=True)
    parser.add_argument("--tier", required=True, choices=("easy", "hard"))
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=100)
    parser.add_argument("--refine-top-k", type=int, default=5)
    parser.add_argument("--refine-max-nfev", type=int, default=10)
    parser.add_argument("--refine-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.maximum_candidates < 1 or args.refine_top_k < 0:
        raise SystemExit("candidate and refinement budgets must be nonnegative")
    if args.refine_max_nfev < 1 or args.refine_timeout_seconds <= 0:
        raise SystemExit("refinement fitting budgets must be positive")
    # Compatibility with helpers shared by the unsharded driver.
    args.max_nfev = 3
    args.timeout_seconds = 60.0
    args.output_root.mkdir(parents=True, exist_ok=True)

    artifacts = _load_pool(args.candidate_pool)
    development, context = _development_context(args)
    source, eligibility, eligible, selected = _prepare_candidates(
        args, artifacts, context
    )
    rows = _read_complete_shards(args.shards_root, selected)
    artifacts_by_id = {item.artifact_id: item for item in selected}
    refined_rows = _refine_screening_winners(
        args, rows, artifacts_by_id, development, context
    )
    refined_by_id = {item["artifact_id"]: item for item in refined_rows}
    combined = [
        _better_result(item, refined_by_id.get(item["artifact_id"])) for item in rows
    ]
    successful = [item for item in combined if item["success"]]
    winner = min(
        successful,
        key=lambda item: (item["validation_nmse"], item["artifact_id"]),
        default=None,
    )
    manifest = _build_manifest(
        args=args,
        source=source,
        eligibility=eligibility,
        eligible=eligible,
        rows=combined,
        refined_rows=refined_rows,
        winner=winner,
    )
    manifest["schema_version"] = "phase_b_frozen_sharded_replay_v1"
    manifest["screening_shards_root"] = str(args.shards_root)
    _write_jsonl(args.output_root / "eligibility.jsonl", eligibility)
    _write_json(args.output_root / "replay_manifest.json", manifest)
    _write_report(args.output_root / "replay_report.md", manifest, eligibility)
    if winner is None:
        raise SystemExit("no frozen structure passed Phase-B rollout refitting")


def _read_complete_shards(shards_root: Path, selected) -> list[dict]:
    expected = {item.artifact_id for item in selected}
    manifests = sorted(shards_root.glob("shard-*/shard_manifest.json"))
    if not manifests:
        raise SystemExit("no completed shard manifests found")
    rows: dict[str, dict] = {}
    declared: set[str] = set()
    shard_counts: set[int] = set()
    shard_indices: set[int] = set()
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("stage") != "screen_complete":
            raise SystemExit(f"incomplete shard manifest: {path}")
        shard_counts.add(int(manifest["shard_count"]))
        shard_indices.add(int(manifest["shard_index"]))
        assigned = set(manifest["assigned_artifact_ids"])
        completed = set(manifest["completed_artifact_ids"])
        if assigned != completed:
            raise SystemExit(f"shard is missing results: {path}")
        if declared & assigned:
            raise SystemExit(f"duplicate artifact assignment in {path}")
        declared.update(assigned)
        for result_path in sorted(path.parent.joinpath("results").glob("*.json")):
            row = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_id = row["artifact_id"]
            if artifact_id in rows:
                raise SystemExit(f"duplicate result for {artifact_id}")
            rows[artifact_id] = row
    if len(shard_counts) != 1:
        raise SystemExit("shards disagree on shard count")
    shard_count = next(iter(shard_counts))
    if shard_indices != set(range(shard_count)):
        raise SystemExit("not all declared shards have completed")
    if declared != expected or set(rows) != expected:
        missing = sorted(expected - set(rows))
        unexpected = sorted(set(rows) - expected)
        raise SystemExit(
            "shard coverage mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return [rows[artifact_id] for artifact_id in sorted(rows)]


if __name__ == "__main__":
    main()
