#!/usr/bin/env python3
"""Prepare, supervise and report the isolated parameter-freedom campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

# Permit standard-library-only reporting through python -S.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def supervise(output: Path, action: str, index: int) -> int:
    """Kill the entire worker group before the allocation deadline."""
    frozen = json.loads((output / "freeze.json").read_text())
    if action != "gate" and not 0 <= index < len(frozen["tasks"]):
        raise ValueError("task index outside the frozen plan")
    root = output / ("gate" if action == "gate" else f"results/task_{index:03d}")
    root.mkdir(parents=True, exist_ok=True)
    plan = frozen["plan"]
    limit = (
        plan["gate_seconds"] + 60 if action == "gate" else plan["supervisor_seconds"]
    )
    with (
        (root / "supervisor.lock").open("w") as lock,
        (root / "worker.log").open("a") as log,
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
        (root / "supervisor.json").write_text(
            json.dumps({"exit_code": code, "limit_seconds": limit})
        )
        return code


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "gate",
            "gate-worker",
            "run",
            "run-worker",
            "summarize",
            "smoke",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    if args.action in ("gate", "run"):
        raise SystemExit(supervise(args.output, args.action, args.task_index))
    if args.action == "summarize":
        from parameter_freedom_report import report

        print(json.dumps(report(args.output)["counts"]))
        return
    if args.action == "smoke":
        from run_attainability_campaign import smoke

        smoke(args.output / "smoke", resolution=True)
        return
    from autoformalism.rebuttal.parameter_freedom import (
        FreedomPlan,
        execute,
        gate,
        prepare,
    )

    if args.action == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires --source and --config")
        result = prepare(
            args.source,
            args.output,
            FreedomPlan.model_validate_json(args.config.read_text()),
        )
        print(
            json.dumps({"identity": result["identity"], "tasks": len(result["tasks"])})
        )
    elif args.action == "gate-worker":
        result = gate(args.output)
        print(json.dumps(result))
        raise SystemExit(0 if result["pass"] else 1)
    else:
        result = execute(args.output, args.task_index)
        print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
