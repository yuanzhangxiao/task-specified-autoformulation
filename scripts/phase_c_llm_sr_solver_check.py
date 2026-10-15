#!/usr/bin/env python3
"""Roll LLM-SR's sealed picks out with the shared evaluator's solver as well.

A diagnostic for picks that read only the current row. The sealed rollout
stepped every program along the data grid with the trapezoid rule, while an
equation of the current values, such as PySR's, goes to the shared
evaluator's adaptive solver. For each sealed task whose picks are programs,
this checks that every program gives each training row the same output alone
as with the whole table; if they all do, it prints the sealed grid error,
the same grid rollout recomputed here, and the adaptive solver's error.

It reads the frozen plan, the sealed results and the development data only,
and changes nothing. ``--release`` points at this machine's copy of the
frozen release, whose receipt must match the plan's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from autoformalism.rebuttal.llm_sr_campaign import training_rows
from autoformalism.rebuttal.llm_sr_programs import (
    check_program,
    program_rollout_error,
    program_solver_rollout_error,
)
from autoformalism.rebuttal.phase_c_vendored_campaign import load_development
from autoformalism.rebuttal.prefit_replay import sealed_read


def check_task(sealed: dict, root: Path, index: int, *, seconds: float) -> dict | None:
    """One task's errors, or ``None`` when it has no sealed programs."""
    path = root / "results" / str(index) / "result.json"
    if not path.is_file():
        return None
    result = sealed_read(path)
    records = result.get("programs")
    if not records:
        return None
    row = sealed["rows"][index]
    development, context = load_development(sealed, row)
    channels = tuple(row["channels"])
    chosen = {
        target: (
            check_program(record["function"], record["variables"]),
            tuple(record["parameters"]),
        )
        for target, record in records.items()
    }
    values, _ = training_rows(sealed["plan"], row, development.train, context)
    started = time.monotonic()
    solver = program_solver_rollout_error(
        chosen, channels, context, development.train, development.validation,
        values, seconds=seconds,
    )
    solver_seconds = time.monotonic() - started
    report = {
        "index": index,
        "benchmark_id": row["benchmark_id"],
        "tier": row["tier"],
        "repetition": row["repetition"],
        "status": result["status"],
        "sealed_grid_error": result.get("development_rollout_error"),
        "alone": solver.alone,
        "solver_error": solver.error,
        "solver_failure": solver.failure,
        "solver_rhs_calls": solver.rhs_calls,
        "solver_seconds": round(solver_seconds, 1),
    }
    if solver.failure is None or not solver.failure.startswith("reads rows"):
        # Only picks of the current values: a history-reading pick's grid
        # rollout is long, and the sealed value already states it.
        grid = program_rollout_error(
            chosen, channels, context, development.train, development.validation,
            step=result["grid_step"], seconds=seconds,
        )
        report.update(grid_error_here=grid.error, grid_failure_here=grid.failure)
    return report


def main() -> None:
    """Check the chosen tasks, or every sealed task with programs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="a run's directory, or a copy of its plan and results")
    parser.add_argument("--release", type=Path,
                        help="this machine's copy of the frozen release")
    parser.add_argument("--index", type=int, action="append",
                        help="a task to check; repeat for more (default: all)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="time limit per trajectory, as the sealed rollout's")
    args = parser.parse_args()
    sealed = sealed_read(args.root / "plan.json")
    if args.release is not None:
        # The receipt and every cell's identity are still checked against
        # the plan's, so only the frozen release passes.
        sealed = {**sealed, "release": str(args.release)}
    indices = args.index or range(len(sealed["rows"]))
    reports = [
        report for index in indices
        if (report := check_task(sealed, args.root, index, seconds=args.seconds))
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
