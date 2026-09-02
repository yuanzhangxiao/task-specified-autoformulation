#!/usr/bin/env python3
"""List incomplete identities in a frozen public baseline task matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.baselines.models import (
    BaselineDevelopmentResult,
    BaselineRunStatus,
)
from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask


def main() -> None:
    """Print deterministic method-grouped retry indices as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    report = find_incomplete_tasks(args.task_plan, args.runs_root)
    print(json.dumps(report, indent=2, sort_keys=True))


def find_incomplete_tasks(task_plan: Path, runs_root: Path) -> dict[str, object]:
    """Return only tasks without a valid complete result/status pair."""
    tasks = tuple(
        BaselinePilotTask.model_validate_json(line)
        for line in task_plan.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    grouped: dict[str, list[int]] = {
        method: [] for method in sorted({task.method for task in tasks})
    }
    failures: list[dict[str, object]] = []
    for task in tasks:
        run = (
            runs_root.expanduser().resolve()
            / task.method
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        try:
            result = BaselineDevelopmentResult.model_validate_json(
                (run / "result.json").read_text(encoding="utf-8")
            )
            status = BaselineRunStatus.model_validate_json(
                (run / "run_status.json").read_text(encoding="utf-8")
            )
            identity = (
                result.method,
                result.benchmark_id,
                result.tier,
                result.seed,
            )
            expected = (
                task.method,
                task.benchmark_id,
                task.tier,
                task.repetition,
            )
            if (
                identity != expected
                or result.status != "development_complete"
                or result.test_data_opened is not False
                or status.status != "complete"
                or not result.equations
            ):
                raise ValueError("result/status does not match the frozen task")
        except (OSError, ValueError) as exc:
            grouped[task.method].append(task.task_index)
            failures.append(
                {
                    "task_index": task.task_index,
                    "method": task.method,
                    "benchmark_id": task.benchmark_id,
                    "tier": task.tier,
                    "repetition": task.repetition,
                    "reason": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            )
    return {
        "schema_version": "phase-b-public-baseline-resume-audit-1",
        "planned_task_count": len(tasks),
        "complete_task_count": len(tasks) - len(failures),
        "incomplete_task_count": len(failures),
        "incomplete_indices_by_method": grouped,
        "incomplete_tasks": failures,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


if __name__ == "__main__":
    main()
