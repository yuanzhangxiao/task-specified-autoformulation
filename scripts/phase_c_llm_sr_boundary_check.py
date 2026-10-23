#!/usr/bin/env python3
"""Measure what LLM-SR's picks carry across training-trajectory boundaries.

LLM-SR's evaluator gives a program the whole training table, the trajectories'
rows one after another, so a program that reads its history carries it from
one trajectory into the next. Its rollout gives the program one trajectory at
a time. For every sealed program pick, with its sealed coefficients, this runs
the program both ways on the training rows and reports, per target, the error
their evaluator minimizes (mean squared error against the derivative labels)
under each, and how many rows change.

Development data only: it changes nothing and opens no test file. The programs
run in the restricted worker, as the sealed run ran them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.llm_sr_campaign import training_rows
from autoformalism.rebuttal.llm_sr_programs import check_program, run_worker
from autoformalism.rebuttal.phase_c_vendored_campaign import load_development
from autoformalism.rebuttal.prefit_replay import sealed_read


def outputs(program, parameters, rows, seconds):
    """The program's outputs on these rows, in one call, as their evaluator runs it."""
    reply = run_worker(
        {
            "mode": "evaluate",
            "sources": [program.source],
            "names": [program.name],
            "parameters": list(parameters),
            "inputs": np.asarray(rows, dtype=float).tolist(),
            "cuts": [],
        },
        seconds=seconds,
    )
    return np.asarray(reply["outputs"], dtype=float)


def check(sealed, root, index, seconds):
    """One task's reports, one per target, or why it has none."""
    path = root / "results" / str(index) / "native-selection.json"
    if not path.exists():
        return [{"task": index, "skipped": "no sealed selection"}]
    payload = sealed_read(path)["selection"]["selection_payload"]
    if "programs" not in payload:
        return [{"task": index, "skipped": "the pick is not a program"}]
    row = sealed["rows"][index]
    development, context = load_development(sealed, row)
    values, labels = training_rows(sealed["plan"], row, development.train, context)
    lengths = [len(trajectory.time) for trajectory in development.train.trajectories]
    if sum(lengths) != len(values):
        raise ValueError("the trajectories' rows do not add up to the table")
    starts = np.cumsum([0, *lengths[:-1]])
    reports = []
    for column, target in enumerate(context.targets):
        record = payload["programs"][target]
        program = check_program(record["function"], record["variables"])
        joined = outputs(program, record["parameters"], values, seconds)
        separate = np.concatenate([
            outputs(program, record["parameters"], values[start:start + rows], seconds)
            for start, rows in zip(starts, lengths, strict=True)
        ])
        label = labels[:, column]
        with np.errstate(all="ignore"):
            joined_error = float(np.mean((joined - label) ** 2))
            separate_error = float(np.mean((separate - label) ** 2))
        largest = float(np.max(np.abs(joined)))
        changed = np.abs(separate - joined) > 1e-9 * largest
        reports.append({
            "task": index,
            "benchmark_id": row["benchmark_id"],
            "repetition": row["repetition"],
            "target": target,
            "trajectories": len(lengths),
            "rows": len(values),
            "error_joined": joined_error,
            "error_separate": separate_error,
            "ratio": separate_error / joined_error if joined_error > 0 else None,
            "rows_changed": int(changed.sum()),
            "largest_change_over_largest_output": (
                float(np.max(np.abs(separate - joined))) / largest
                if largest > 0 else None
            ),
        })
    return reports


def main() -> None:
    """Check every sealed pick, or the chosen tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="the LLM-SR run, or a copy of its plan and results")
    parser.add_argument("--release", type=Path,
                        help="this machine's copy of the frozen release")
    parser.add_argument("--task", type=int, action="append",
                        help="a task to check; repeat for more (default: all)")
    parser.add_argument("--seconds", type=float, default=600.0,
                        help="time limit per worker call")
    args = parser.parse_args()
    sealed = sealed_read(args.root / "plan.json")
    if args.release is not None:
        # The receipt and every cell's identity are still checked.
        sealed = {**sealed, "release": str(args.release)}
    for index in range(len(sealed["rows"])):
        if args.task and index not in args.task:
            continue
        for report in check(sealed, args.root, index, args.seconds):
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
