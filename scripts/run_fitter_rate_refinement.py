#!/usr/bin/env python3
"""Freeze, run and summarize the user-submitted saved-start rate comparison."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json  # noqa: E402
from autoformalism.rebuttal.fitter_rate_refinement import (  # noqa: E402
    RatePlan,
    execute_rate,
    prepare_rate,
    summarize_rate,
    verify_rate,
    verify_rate_result,
)
from autoformalism.rebuttal.fitter_stagnation import checkpoint  # noqa: E402
from autoformalism.staged_topology import content_hash  # noqa: E402


def run_supervised(output: Path, index: int) -> dict:
    """Stop a stalled worker without discarding completed fits or reference files."""
    frozen = verify_rate(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        verify_rate_result(root, existing)
        return existing
    root.mkdir(parents=True, exist_ok=True)
    plan = RatePlan.model_validate(frozen["plan"])
    with (root / "worker.log").open("a") as log:
        worker = subprocess.Popen(
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
            code = worker.wait(timeout=plan.worker_seconds(task["kind"]))
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
            result = {"status": "timeout", "error": "worker wall-clock limit reached"}
        except BaseException:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            raise
        else:
            result = checkpoint(root / "result.json", identity) if code == 0 else None
            if result is not None:
                return result
            result = {
                "status": "worker_failed",
                "error": f"worker exit {code}; see worker.log",
            }
    result.update(
        identity=identity,
        task=task,
        test_data_opened=False,
        private_reference_opened=False,
        llm_calls=0,
    )
    write_json(root / "result.json", result)
    return result


def main() -> None:
    """Keep scheduler supervision and locked task mutation outside the library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        if args.config is None or args.source is None:
            parser.error("prepare requires --config and --source")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "prepare.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = prepare_rate(
                RatePlan.model_validate(read_json(args.config)),
                args.source.resolve(),
                output,
            )
        print(frozen["freeze_sha256"], "tasks", len(frozen["tasks"]))
    elif args.stage == "summarize":
        summarize_rate(output)
        print(output / "summary.md")
    else:
        frozen = verify_rate(output)
        if args.task_index is None or not 0 <= args.task_index < len(frozen["tasks"]):
            parser.error("run/worker requires a valid --task-index")
        root = output / "results" / frozen["tasks"][args.task_index]["name"]
        root.mkdir(parents=True, exist_ok=True)
        lock_name = "supervisor.lock" if args.stage == "run" else "worker.lock"
        with (root / lock_name).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = (run_supervised if args.stage == "run" else execute_rate)(
                output, args.task_index
            )
        print(result["status"])


if __name__ == "__main__":
    main()
