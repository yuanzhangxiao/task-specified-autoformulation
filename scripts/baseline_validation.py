#!/usr/bin/env python3
"""Inventory, freeze and replay saved external models on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.baseline_validation import (
    ReplaySettings,
    inventory,
    prepare,
    report,
    run,
)


def main() -> None:
    """Keep inventory and preparation CPU-only; never dispatch an LLM or test access."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inventory")
    scan.add_argument("--root", type=Path, action="append", required=True)
    scan.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--roster", type=Path, required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument("--legacy-root", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--trajectory-seconds", type=float, default=120)
    worker = sub.add_parser("run")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--shard", type=int, default=0)
    worker.add_argument("--shards", type=int, default=1)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        value = inventory(args.root, args.output)
        value = {**value, "rows_found": len(value["rows"])}
        value.pop("rows")
    elif args.command == "prepare":
        value = prepare(
            args.inventory,
            args.roster,
            args.public_root,
            args.legacy_root,
            args.output_root,
            ReplaySettings(trajectory_seconds=args.trajectory_seconds),
        )
        value = {
            "plan_sha256": value["artifact_sha256"],
            "expected": len(value["rows"]),
            "ready": sum(r["status"] == "ready" for r in value["rows"]),
            "unavailable": [
                {
                    k: r.get(k)
                    for k in (
                        "index",
                        "source_kind",
                        "benchmark_id",
                        "repetition",
                        "error",
                        "rejected_sources",
                    )
                }
                for r in value["rows"]
                if r["status"] != "ready"
            ],
        }
    else:
        value = (
            run(args.root, args.shard, args.shards)
            if args.command == "run"
            else report(args.root)
        )
        # Only explicit report writes summary; workers never race on this file.
        if args.command == "report":
            atomic_json(args.root / "summary.json", value)
        value = {key: item for key, item in value.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
