#!/usr/bin/env python3
"""Freeze, run, and summarize a fresh Phase-B D3 validation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.phase_b_d3 import prepare, report, run


def main() -> None:
    """Discovery is explicit; prepare/report never call a provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument("--model", required=True, help="Explicit OpenAI API model ID")
    freeze.add_argument("--root", type=Path, required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config, args.public_root, args.root, args.model)
        value = {
            "plan_sha256": result["artifact_sha256"],
            "expected": len(result["rows"]),
            "maximum_logical_calls": result["maximum_logical_calls"],
            "model": result["model"],
        }
    else:
        result = (
            run(args.root, args.index) if args.command == "run" else report(args.root)
        )
        value = {key: item for key, item in result.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
