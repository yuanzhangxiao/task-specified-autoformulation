#!/usr/bin/env python3
"""Prepare, gate, supervise and report the final nine-run fitter campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def supervise(output: Path, action: str, index: int) -> int:
    frozen = json.loads((output / "freeze.json").read_text())
    if action == "run" and not 0 <= index < len(frozen["tasks"]):
        raise ValueError("task index outside frozen matrix")
    root = output / ("gate" if action == "gate" else f"results/task_{index:03d}")
    root.mkdir(parents=True, exist_ok=True)
    limit = 540 if action == "gate" else frozen["plan"]["supervisor_seconds"]
    with (
        (root / "supervisor.lock").open("w") as lock,
        (root / "worker.log").open("a", buffering=1) as log,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                action + "-worker",
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
            code = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            code = 124
        finally:
            # A killed native child must never remain after its controller exits.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        (root / "supervisor.json").write_text(
            json.dumps({"exit_code": code, "limit_seconds": limit})
        )
        return code


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "gate", "gate-worker", "run", "run-worker", "summarize"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    if args.action in ("gate", "run"):
        raise SystemExit(supervise(args.output, args.action, args.task_index))
    if args.action == "summarize":
        from final_fitter_report import report

        print(json.dumps(report(args.output)["counts"]))
        return
    from autoformalism.rebuttal.final_fitter import FinalFitterPlan, execute, prepare

    if args.action == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires --source and --config")
        r = prepare(
            args.source,
            args.output,
            FinalFitterPlan.model_validate_json(args.config.read_text()),
        )
        print(json.dumps({"identity": r["identity"], "tasks": len(r["tasks"])}))
    elif args.action == "gate-worker":
        from autoformalism.rebuttal.final_fitter_smoke import gate

        r = gate(args.output)
        print(json.dumps(r))
        raise SystemExit(0 if r.get("pass") else 1)
    else:
        print(json.dumps({"status": execute(args.output, args.task_index)["status"]}))


if __name__ == "__main__":
    main()
