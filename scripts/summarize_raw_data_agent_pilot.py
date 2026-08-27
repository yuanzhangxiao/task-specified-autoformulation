#!/usr/bin/env python3
"""Summarize completed raw-data agent baseline runs without opening test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for status_path in sorted(args.root.glob("*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = json.loads(
            (status_path.parent / "run_config.json").read_text(encoding="utf-8")
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
                "validation_normalized_mse": status.get(
                    "validation_normalized_mse", ""
                ),
                "agent_latency_seconds": status.get("agent_latency_seconds", ""),
                "tool_call_count": status.get("tool_call_count", ""),
                "requested_max_tool_calls": status.get(
                    "requested_max_tool_calls",
                    config.get("agent_config", {}).get("max_tool_calls", ""),
                ),
                "tool_call_limit_exceeded": status.get(
                    "tool_call_limit_exceeded",
                    (
                        status.get("tool_call_count", 0)
                        > config.get("agent_config", {}).get(
                            "max_tool_calls", float("inf")
                        )
                    ),
                ),
                "error_type": status.get("error_type", ""),
            }
        )
    output = args.root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    for row in rows:
        print("\t".join(str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
