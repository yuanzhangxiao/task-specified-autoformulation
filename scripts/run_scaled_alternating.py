#!/usr/bin/env python3
"""Prepare, gate, run and summarize exactly one joint/alternating pair."""

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


def child(args, log, seconds, *, group=False):
    """The outer supervisor owns the group; stage children inherit that group."""
    process = subprocess.Popen(
        [sys.executable, __file__, *args],
        stdout=log,
        stderr=log,
        start_new_session=group,
    )
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
        return 124
    finally:
        if group:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def supervise(output, action, index):
    root = output / ("gate" if action == "gate" else f"results/task_{index:03d}")
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "supervisor.lock").open("w") as lock,
        (root / "worker.log").open("a", buffering=1) as log,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = child(
            [action + "-worker", "--output", str(output), "--task-index", str(index)],
            log,
            840 if action == "gate" else 6600,
            group=True,
        )
    (root / "supervisor.json").write_text(json.dumps({"exit_code": code}))
    return code


def run(output, index):
    from autoformalism.rebuttal.fitter_diagnostic import write_json
    from autoformalism.rebuttal.scaled_alternating import (
        ScaledAlternatingPlan,
        checked,
        common_data,
        verify,
    )
    from autoformalism.rebuttal.scaled_alternating_stages import (
        best_of,
        finish,
        interrupted_result,
        task_identity,
    )

    frozen = verify(output)
    common_data(output, frozen)
    identity = task_identity(frozen, index)
    plan = ScaledAlternatingPlan.model_validate(frozen["plan"])
    root = output / f"results/task_{index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists():
        return checked(root / "result.json", identity)
    stages = [
        ("initializer", plan.initializer_seconds),
        ("screen", plan.screen_seconds),
        ("refinement", plan.refinement_seconds),
    ]
    stages += [
        (f"replay-{split}-{solver}", plan.replay_seconds)
        for split in ("train", "val")
        for solver in ("Radau", "BDF")
    ]
    for name, seconds in stages:
        if name.startswith("replay-") and not (root / "selected.json").exists():
            screen = checked(root / "screen/result.json", identity)
            refined = checked(root / "refinement/result.json", identity)
            best = best_of(screen, refined)
            selected = {
                "identity": identity,
                "best": best,
                "selection": "minimum full-training rollout cost",
                "source": "refinement"
                if best and best == refined.get("best")
                else "screen",
            }
            write_json(root / "selected.json", selected, immutable=True)
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        if (path / "result.json").exists():
            checked(path / "result.json", identity)
            continue
        if (path / "started.json").exists():
            checked(path / "started.json", identity)
            interrupted_result(root, name, identity, 125)
            continue
        write_json(
            path / "started.json",
            {"identity": identity, "budget_seconds": seconds},
            immutable=True,
        )
        with (path / "worker.log").open("a", buffering=1) as log:
            code = child(
                [
                    "stage-worker",
                    "--output",
                    str(output),
                    "--task-index",
                    str(index),
                    "--stage",
                    name,
                ],
                log,
                seconds,
            )
        if not (path / "result.json").exists():
            interrupted_result(root, name, identity, code)
    return finish(output, index)


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
            "stage-worker",
            "summarize",
            "smoke",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.action in ("gate", "run"):
        if args.task_index not in (0, 1):
            parser.error("task index must be 0 or 1")
        raise SystemExit(supervise(args.output, args.action, args.task_index))
    if args.action == "summarize":
        from scaled_alternating_report import report

        print(json.dumps(report(args.output)["counts"]))
        return
    if args.action == "prepare":
        from autoformalism.rebuttal.scaled_alternating import (
            ScaledAlternatingPlan,
            prepare,
        )

        if args.config is None or args.source is None:
            parser.error("prepare requires --source and --config")
        r = prepare(
            args.source,
            args.output,
            ScaledAlternatingPlan.model_validate_json(args.config.read_text()),
        )
        print(json.dumps({"identity": r["identity"], "tasks": len(r["tasks"])}))
    elif args.action in ("gate-worker", "smoke"):
        from autoformalism.rebuttal.scaled_alternating_gate import gate, smoke

        r = gate(args.output) if args.action == "gate-worker" else smoke(args.output)
        print(json.dumps(r))
        raise SystemExit(0 if r["pass"] else 1)
    elif args.action == "run-worker":
        print(json.dumps({"status": run(args.output, args.task_index)["status"]}))
    else:
        from autoformalism.rebuttal.scaled_alternating import verify
        from autoformalism.rebuttal.scaled_alternating_stages import (
            failed_result,
            stage,
            task_identity,
        )

        try:
            value = stage(args.output, args.task_index, args.stage)
        except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
            frozen = verify(args.output)
            value = failed_result(
                args.output / f"results/task_{args.task_index:03d}",
                args.stage,
                task_identity(frozen, args.task_index),
                1,
                status="timeout"
                if isinstance(error, TimeoutError)
                else "numerical_failed",
                message=f"{type(error).__name__}: {error}",
            )
        print(json.dumps({"status": value["status"]}))


if __name__ == "__main__":
    main()
