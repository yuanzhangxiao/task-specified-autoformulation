"""Build a development-only candidate pool from completed checkpoints."""

from __future__ import annotations

import argparse
import csv
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
    jsonl.write_text(
        "".join(item.model_dump_json() + "\n" for item in records),
        encoding="utf-8",
    )
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
    report = {
        "candidate_count": len(records),
        "judged_candidate_count": sum(item.use_judge for item in records),
        "benchmarks": sorted({item.benchmark_id for item in records}),
        "roots": [str(path.expanduser().resolve()) for path in args.roots],
    }
    (args.output_root / "artifact_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
