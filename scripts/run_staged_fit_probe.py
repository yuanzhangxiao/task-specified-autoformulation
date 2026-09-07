#!/usr/bin/env python3
"""Prepare or supervise the source-bound staged numerical handoff."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from autoformalism.rebuttal.fitter_diagnostic import (  # noqa: E402
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.staged_fit_probe import (  # noqa: E402
    StagedFitPlan,
    execute_probe,
    prepare_probe,
    read_checkpoint,
    verify_probe,
)


def run_supervised(output: Path) -> dict:
    """Retain terminal outcomes and kill an over-budget worker process group."""
    frozen = verify_probe(output)
    identity = sha256(output / "freeze.json")
    existing = read_checkpoint(output / "result.json", identity)
    if existing is not None:
        return existing
    plan = StagedFitPlan.model_validate(frozen["plan"])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(output),
    ]
    started = monotonic()
    with (output / "worker.log").open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            code = process.wait(timeout=plan.worker_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = {
                "status": "timeout",
                "error": f"worker exceeded {plan.worker_seconds:g}s",
            }
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        else:
            result = (
                read_checkpoint(output / "result.json", identity) if code == 0 else None
            )
            if result is None:
                result = {
                    "status": "worker_failed",
                    "error": f"worker exited {code}; see worker.log",
                }
    result.update(
        freeze_sha256=identity,
        task_seconds=monotonic() - started,
        test_data_opened=False,
        private_reference_opened=False,
        llm_calls=0,
    )
    write_json(output / "result.json", result)
    return result


def main() -> None:
    """Expose preparation, supervised execution and its internal worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--function-plan", type=Path)
    parser.add_argument("--function-results", type=Path)
    parser.add_argument("--public-root", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args.stage == "worker":
        print(execute_probe(output)["status"])
        return
    output.mkdir(parents=True, exist_ok=True)
    # One owner prepares or supervises this single-task output directory.
    with (output / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage == "prepare":
            if any(
                getattr(args, name) is None
                for name in (
                    "config",
                    "function_plan",
                    "function_results",
                    "public_root",
                )
            ):
                parser.error(
                    "prepare requires config, function-plan, function-results "
                    "and public-root"
                )
            result = prepare_probe(
                StagedFitPlan.model_validate(read_json(args.config)),
                args.function_plan,
                args.function_results,
                args.public_root,
                output,
            )
            print(result["protocol"], sha256(output / "freeze.json"))
        else:
            print(run_supervised(output)["status"])


if __name__ == "__main__":
    main()
