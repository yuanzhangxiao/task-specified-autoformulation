#!/usr/bin/env python3
"""Resolve one frozen raw-agent matrix array index to a benchmark repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_task(config: dict[str, object], task_id: int) -> tuple[str, str, int]:
    """Return benchmark identifier, tier, and repetition for one array task."""
    benchmarks = config.get("benchmarks")
    repetitions = config.get("repetitions")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("config must contain a nonempty benchmarks list")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("config repetitions must be an integer")
    if repetitions < 1:
        raise ValueError("config repetitions must be positive")
    task_count = len(benchmarks) * repetitions
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"task_id must be in [0, {task_count - 1}]")
    item = benchmarks[task_id // repetitions]
    if not isinstance(item, dict):
        raise ValueError("each benchmark entry must be an object")
    benchmark_id = item.get("benchmark_id")
    tier = item.get("tier")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("benchmark entry has no benchmark_id")
    if tier not in {"easy", "hard"}:
        raise ValueError("benchmark tier must be easy or hard")
    return benchmark_id, str(tier), task_id % repetitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    benchmark_id, tier, repetition = resolve_task(config, args.task_id)
    print(benchmark_id)
    print(tier)
    print(repetition)


if __name__ == "__main__":
    main()
