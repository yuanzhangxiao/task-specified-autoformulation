#!/usr/bin/env python3
"""Supervised resolution/sensitivity preflight and bounded reference fits."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from run_attainability_campaign import smoke, summarize, supervised


def profile_supervised(output, index):
    """Terminate the worker process group if the bounded profile does not return."""
    root = output / f"profiles/{index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "supervisor.lock").open("w") as lock,
        (root / "worker.log").open("a") as log,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "profile-worker",
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
            code = child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            code = 124
        if code:
            from autoformalism.rebuttal.attainability_campaign import verify
            from autoformalism.rebuttal.fitter_diagnostic import write_json
            from autoformalism.staged_topology import content_hash

            frozen = verify(output)
            task = frozen["profiles"][index]
            if not (root / "result.json").exists():
                write_json(
                    root / "result.json",
                    {
                        "identity": content_hash([frozen["identity"], "profile", task]),
                        "task": task,
                        "status": "worker_timeout" if code == 124 else "worker_failed",
                        "exit_code": code,
                    },
                    immutable=True,
                )


def report(output):
    """Summarize stored outcomes without importing a numerical runtime."""
    summarize(output)
    frozen = json.loads((output / "freeze.json").read_text())
    rows = []
    for task in frozen["profiles"]:
        path = output / f"profiles/{task['index']:03d}/result.json"
        rows.append(
            json.loads(path.read_text())
            if path.exists()
            else {"task": task, "status": "missing"}
        )
    gate = (
        json.loads((output / "preflight.json").read_text())
        if (output / "preflight.json").exists()
        else {}
    )
    lines = [
        "# Resolution and sensitivity preflight",
        "",
        f"Preflight passed: {gate.get('pass')}",
        "",
        "One CPU per task. Fixed points: no optimization. Dense/sparse use the "
        "same starts, tolerances and budgets.",
        "",
        "| Point | Format | Status | State s | Sensitivity s | Starting train NMSE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        t, state, sens = r["task"], r.get("state", {}), r.get("sensitivity", {})
        lines.append(
            f"| {t['point']} | {t['format']} | {r['status']} | "
            f"{state.get('seconds')} | {sens.get('seconds')} | "
            f"{sens.get('nmse', state.get('nmse'))} |"
        )
    lines += [
        "",
        "Gate checks:",
        "",
        *[f"- {c['check']}: {c['pass']}" for c in gate.get("checks", [])],
        "",
        "Numerical failure, timeout, and blocked optimization are distinct. "
        "Reference information is diagnostic only; no test data or LLM calls.",
        "",
    ]
    (output / "preflight_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "smoke",
            "profile",
            "profile-worker",
            "gate",
            "run",
            "summarize",
        ],
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "summarize":
        report(args.output)
        return
    if args.action == "profile":
        profile_supervised(args.output, args.task_index)
        return
    if args.action == "run":
        supervised(args.output, args.task_index, False)
        return
    from autoformalism.rebuttal.attainability_campaign import AttainabilityPlan
    from autoformalism.rebuttal.resolution_campaign import (
        preflight,
        prepare_resolution,
        run_profile,
    )

    if args.action == "prepare":
        plan = AttainabilityPlan.model_validate_json(args.config.read_text())
        result = prepare_resolution(args.source, args.output, plan)
        print(
            json.dumps(
                {
                    "identity": result["identity"],
                    "profiles": len(result["profiles"]),
                    "tasks": len(result["tasks"]),
                }
            )
        )
    elif args.action == "smoke":
        from autoformalism.rebuttal.attainability_campaign import code_identity
        from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
        from autoformalism.staged_topology import content_hash

        identity = content_hash([code_identity(), "resolution-smoke"])
        done = args.output / "resolution_smoke.json"
        started = args.output / "resolution_smoke_started.json"
        if done.exists():
            if read_json(done).get("identity") != identity:
                raise ValueError("smoke code changed")
            print("Verified smoke checkpoint retained")
            return
        if started.exists():
            raise ValueError("smoke interrupted; no fresh budget in the same output")
        write_json(started, {"identity": identity}, immutable=True)
        smoke(args.output, resolution=True)
        write_json(done, {"identity": identity, "status": "passed"}, immutable=True)
    elif args.action == "profile-worker":
        print(run_profile(args.output, args.task_index)["status"])
    elif args.action == "gate":
        result = preflight(args.output)
        print(json.dumps(result, indent=2))
        if not result["pass"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
