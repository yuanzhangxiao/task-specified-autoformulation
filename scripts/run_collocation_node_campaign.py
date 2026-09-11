#!/usr/bin/env python3
"""Prepare, supervise, and summarize the CPU collocation-node comparison."""

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

from autoformalism.rebuttal.collocation_node_campaign import (  # noqa: E402
    NodeCampaignPlan,
    execute_nodes,
    prepare_nodes,
    summarize_nodes,
    verify_nodes,
)
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json  # noqa: E402
from autoformalism.rebuttal.fitter_stagnation import checkpoint  # noqa: E402
from autoformalism.staged_topology import content_hash  # noqa: E402


def supervised(output: Path, index: int) -> dict:
    """Kill a stalled process group and retain an explicit terminal checkpoint."""
    frozen = verify_nodes(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["identity"], task])
    root = output / "results" / task["name"]
    previous = checkpoint(root / "result.json", identity)
    if previous is not None:
        return previous
    root.mkdir(parents=True, exist_ok=True)
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
            code = worker.wait(
                timeout=NodeCampaignPlan.model_validate(frozen["plan"]).worker_seconds
            )
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
    if checkpoint(root / "fit.json", identity) is not None:
        return execute_nodes(output, index)
    result.update(
        identity=identity,
        task=task,
        llm_calls=0,
        test_data_opened=False,
        private_reference_opened=False,
    )
    write_json(root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires source and config")
        with (output / "prepare.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = prepare_nodes(
                NodeCampaignPlan.model_validate(read_json(args.config)),
                args.source,
                output,
            )
        print(frozen["identity"], "tasks", len(frozen["tasks"]))
    elif args.stage == "summarize":
        summarize_nodes(output)
        print(output / "summary.md")
    else:
        frozen = verify_nodes(output)
        if args.task_index is None or not 0 <= args.task_index < len(frozen["tasks"]):
            parser.error("run/worker requires valid task-index")
        root = output / "results" / frozen["tasks"][args.task_index]["name"]
        root.mkdir(parents=True, exist_ok=True)
        name = "worker.lock" if args.stage == "worker" else "supervisor.lock"
        with (root / name).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = (execute_nodes if args.stage == "worker" else supervised)(
                output, args.task_index
            )
        print(result["status"])


if __name__ == "__main__":
    main()
