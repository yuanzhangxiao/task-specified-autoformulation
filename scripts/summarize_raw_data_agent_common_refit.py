#!/usr/bin/env python3
"""Summarize common-evaluator refits without opening test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "provider",
    "model",
    "benchmark_id",
    "tier",
    "repetition",
    "status",
    "candidate_id",
    "screening_success",
    "screening_validation_normalized_mse",
    "final_initialization",
    "final_success",
    "final_validation_normalized_mse",
    "error_type",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = []
    for status_path in sorted(root.glob("*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = json.loads(
            (status_path.parent / "refit_config.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "provider": config["provider"],
                "model": config["model"],
                "benchmark_id": config["benchmark_id"],
                "tier": config["tier"],
                "repetition": config["repetition"],
                "status": status["status"],
                "candidate_id": status.get("candidate_id", ""),
                "screening_success": status.get("screening_success", ""),
                "screening_validation_normalized_mse": status.get(
                    "screening_validation_normalized_mse", ""
                ),
                "final_initialization": status.get("final_initialization", ""),
                "final_success": status.get("final_success", ""),
                "final_validation_normalized_mse": status.get(
                    "final_validation_normalized_mse", ""
                ),
                "error_type": status.get("error_type", ""),
            }
        )
    output = root / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    for row in rows:
        print("\t".join(str(row[field]) for field in FIELDS))


if __name__ == "__main__":
    main()
