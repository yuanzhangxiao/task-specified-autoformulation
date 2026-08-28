#!/usr/bin/env python3
"""List frozen matrix task IDs whose raw-agent run has no terminal model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.resolve_raw_data_agent_matrix_task import resolve_task
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from resolve_raw_data_agent_matrix_task import resolve_task


def retry_task_ids(config: dict[str, object], runs_root: Path) -> tuple[int, ...]:
    """Return task IDs with a missing or failed status, excluding rollouts."""
    benchmarks = config.get("benchmarks")
    repetitions = config.get("repetitions")
    if not isinstance(benchmarks, list) or not isinstance(repetitions, int):
        raise ValueError("invalid matrix configuration")
    status_index: dict[tuple[str, str, int], str] = {}
    for config_path in sorted(runs_root.glob("*/run_config.json")):
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        status_path = config_path.with_name("status.json")
        status = (
            json.loads(status_path.read_text(encoding="utf-8")).get("status")
            if status_path.is_file()
            else "missing"
        )
        key = (
            str(run_config["benchmark_id"]),
            str(run_config["tier"]),
            int(run_config["repetition"]),
        )
        status_index[key] = str(status)
    task_count = len(benchmarks) * repetitions
    result = []
    for task_id in range(task_count):
        key = resolve_task(config, task_id)
        if status_index.get(key, "missing") in {"failed", "missing"}:
            result.append(task_id)
    return tuple(result)


def compact_array_spec(task_ids: tuple[int, ...]) -> str:
    """Render deterministic comma-separated task identifiers for Slurm."""
    return ",".join(str(item) for item in task_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    task_ids = retry_task_ids(config, args.runs_root)
    print(compact_array_spec(task_ids))


if __name__ == "__main__":
    main()
