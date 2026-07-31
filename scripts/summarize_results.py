#!/usr/bin/env python3
"""Summarize one experiment directory or all completed runs below a root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the result-summary command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="run directory or output root")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def load_summaries(path: Path) -> list[dict[str, Any]]:
    """Load deterministic summaries without opening raw data or LLM logs."""
    resolved = path.expanduser().resolve()
    direct = resolved / "summary.json"
    paths = [direct] if direct.is_file() else sorted(resolved.glob("*/summary.json"))
    if not paths:
        raise SystemExit(f"no summary.json files found below {resolved}")
    return [json.loads(item.read_text(encoding="utf-8")) for item in paths]


def main() -> None:
    """Print selected structures and validation/test metrics."""
    args = build_parser().parse_args()
    summaries = load_summaries(args.path)
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return
    header = (
        "benchmark",
        "tier",
        "seed",
        "stop",
        "iterations",
        "validation_mse",
        "test_mse",
    )
    rows = [
        (
            item["benchmark_id"],
            item["tier"],
            str(item["seed"]),
            item["stopping_reason"],
            str(item["completed_iterations"]),
            f"{item['selection_validation_normalized_mse']:.6g}",
            f"{item['test_normalized_mse']:.6g}",
        )
        for item in summaries
    ]
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(len(header))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )


if __name__ == "__main__":
    main()
