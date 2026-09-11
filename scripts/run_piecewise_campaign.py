#!/usr/bin/env python3
"""Prepare, supervise and summarize the frozen piecewise CPU comparison."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json  # noqa: E402
from autoformalism.rebuttal.piecewise_campaign import (  # noqa: E402
    PiecewisePlan,
    execute,
    prepare,
    summarize,
    verify,
)


def supervised(output: Path, index: int) -> None:
    """Kill the entire worker group on expiry; retain per-arm and poll checkpoints."""
    root = output / f"results/case_{index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    plan = PiecewisePlan.model_validate(verify(output)["plan"])
    seconds = plan.worker_seconds
    with (root / "worker.log").open("a") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--output",
                str(output),
                "--task-index",
                str(index),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            write_json(
                root / "supervisor.json", {"status": "timeout", "resumable": True}
            )
            raise SystemExit(
                "Worker timeout; saved arms and polling state can be resumed"
            ) from None
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        if code:
            write_json(
                root / "supervisor.json", {"status": "worker_failed", "exit_code": code}
            )
            raise SystemExit(code)
    write_json(root / "supervisor.json", {"status": "complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        with (output / "prepare.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = (
                PiecewisePlan.model_validate(read_json(args.config))
                if args.config
                else PiecewisePlan()
            )
            result = prepare(plan, output, args.source)
        print(result["identity"], "paired jobs", len(result["cases"]))
    elif args.stage == "summarize":
        summarize(output)
        print(output / "summary.md")
    else:
        frozen = verify(output)
        if args.task_index is None or not 0 <= args.task_index < len(frozen["cases"]):
            parser.error("run/worker requires a valid task-index")
        root = output / f"results/case_{args.task_index:03d}"
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"{args.stage}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage == "worker":
                print(execute(output, args.task_index)["status"])
            else:
                supervised(output, args.task_index)


if __name__ == "__main__":
    main()
