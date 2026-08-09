"""Build a development-only candidate pool from completed checkpoints."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal import index_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = index_artifacts(tuple(args.roots))
    args.output_root.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_root / "candidate_pool.jsonl"
    serialized = "".join(item.model_dump_json() + "\n" for item in records)
    jsonl.write_text(serialized, encoding="utf-8")
    fields = (
        "artifact_id",
        "benchmark_id",
        "tier",
        "seed",
        "round_index",
        "structural_hash",
        "training_mse",
        "validation_mse",
        "judge_score",
        "state_count",
        "latent_state_count",
        "process_count",
        "parameter_count",
        "term_count",
        "use_judge",
        "source_checkpoint",
    )
    with (args.output_root / "run_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            payload = item.model_dump(mode="json")
            writer.writerow({name: payload[name] for name in fields})
    cell_counts: collections.Counter[tuple[str, str, int]] = collections.Counter(
        (item.benchmark_id, item.tier, item.seed) for item in records
    )
    structure_counts: dict[tuple[str, str, int], set[str]] = collections.defaultdict(
        set
    )
    judged_counts: collections.Counter[tuple[str, str, int]] = collections.Counter()
    for item in records:
        key = (item.benchmark_id, item.tier, item.seed)
        structure_counts[key].add(item.structural_hash)
        judged_counts[key] += int(item.use_judge)
    with (args.output_root / "candidate_pool_completeness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "benchmark_id",
                "tier",
                "seed",
                "candidate_count",
                "unique_structure_count",
                "judged_candidate_count",
            ),
        )
        writer.writeheader()
        for key in sorted(cell_counts):
            writer.writerow(
                {
                    "benchmark_id": key[0],
                    "tier": key[1],
                    "seed": key[2],
                    "candidate_count": cell_counts[key],
                    "unique_structure_count": len(structure_counts[key]),
                    "judged_candidate_count": judged_counts[key],
                }
            )
    report = {
        "candidate_count": len(records),
        "judged_candidate_count": sum(item.use_judge for item in records),
        "unique_structure_count": len({item.structural_hash for item in records}),
        "covered_cell_count": len(cell_counts),
        "candidate_pool_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "benchmarks": sorted({item.benchmark_id for item in records}),
        "roots": [str(path.expanduser().resolve()) for path in args.roots],
    }
    (args.output_root / "artifact_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
