#!/usr/bin/env python3
"""Bounded review campaign, reports and held-out evaluation after global freeze."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write


def interrupted(root, plan, task, index, reason):
    """No new numerical allowance after an interrupted external worker."""
    directory = io.round_path(root, task, index)
    old = io.read_round(root, task, index)
    if old:
        return old
    parent = io.read_round(root, task, index - 1) if index else None
    proposal = directory / "proposal.json"
    saved = sealed_read(proposal) if proposal.exists() else {}
    multi = plan["protocol"] == io.MULTI_PROTOCOL
    draft = (
        saved.get("construction_draft")
        or saved.get("bundle")
        or (parent or {}).get("construction_draft")
    )
    return sealed_write(
        directory / "result.json",
        {
            "task": task,
            "round": index,
            "status": "worker_interrupted",
            "error": reason,
            "trial": None,
            "selected": parent["selected"] if parent else None,
            "closed": False if multi else parent is None or parent["selected"] is None,
            "cost": saved.get("cost", {}),
            **(
                {
                    "construction_draft": draft
                    if not (parent or {}).get("selected")
                    else None
                }
                if multi
                else {}
            ),
            "test_data_opened": False,
            "fresh_budget_on_resume": False,
        },
    )


def fit_worker(root: Path, index: int, task_index: int):
    plan = io.verify(root)
    if (
        not 0 <= task_index < len(plan["tasks"])
        or not 0 <= index < plan["config"]["rounds"]
    ):
        raise ValueError("task/round outside frozen matrix")
    task = plan["tasks"][task_index]
    directory = io.round_path(root, task, index)
    with public._lock(directory / "supervisor"):
        if io.read_round(root, task, index):
            return {"status": "already_complete"}
        marker = directory / "worker_started.json"
        if marker.exists():
            return interrupted(
                root, plan, task, index, "earlier worker consumed this attempt"
            )
        # Never consume a numerical attempt merely because its proposal is absent.
        if not (directory / "proposal.json").exists():
            return {"status": "proposal_missing"}
        sealed_write(
            marker, {"plan": plan["artifact_sha256"], "round": index, "task": task}
        )
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "fit-inner",
                    "--root",
                    str(root),
                    "--round",
                    str(index),
                    "--task-index",
                    str(task_index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def terminate(*_):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

            old = {
                s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)
            }
            try:
                try:
                    code = process.wait(timeout=plan["config"]["fit_worker_seconds"])
                except subprocess.TimeoutExpired:
                    terminate()
                    code = 124
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
        if not io.read_round(root, task, index):
            return interrupted(
                root, plan, task, index, f"worker exit {code}; inspect worker.log"
            )
        return {"status": "finished", "exit_code": code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "prepare-continuation",
            "verify",
            "run",
            "fit-inner",
            "fit-task",
            "finish-round",
            "report",
            "export",
            "evaluate",
            "evaluation-report",
            "demo",
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-round", type=int, default=2)
    parser.add_argument("--visits", type=int, default=5)
    parser.add_argument(
        "--continuation-protocol",
        choices=sorted(io.CONTINUATION_PROTOCOLS),
        default=io.CONTINUATION_PROTOCOL,
    )
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument(
        "--round", type=int, default=int(os.environ.get("AF_ROUND", "0"))
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else args.output)
    if root is None:
        parser.error("--root required")
    root = root.resolve()
    command = args.command
    if command == "prepare-continuation":
        from autoformalism.rebuttal.review_continuation import prepare

        if args.source is None:
            parser.error("prepare-continuation requires --source")
        plan = prepare(
            args.source,
            root,
            args.source_round,
            args.visits,
            protocol=args.continuation_protocol,
        )
        value = {
            "identity": plan["artifact_sha256"],
            "tasks": len(plan["tasks"]),
            "source_round": args.source_round,
            "additional_visits": args.visits,
        }
    elif command == "prepare":
        if args.config is None or args.public_root is None:
            parser.error("prepare requires --config and --public-root")
        plan = io.freeze(args.config, args.public_root.resolve(), root)
        value = {
            "identity": plan["artifact_sha256"],
            "tasks": len(plan["tasks"]),
            "rounds": plan["config"]["rounds"],
        }
    elif command == "verify":
        value = {"identity": io.verify(root)["artifact_sha256"]}
    elif command in {"run", "fit-inner", "fit-task", "finish-round"}:
        with io.execution_lease(root):
            if command == "run":
                pipeline.run_proposals(
                    root, args.round, args.base_url, wall_seconds=args.wall_seconds
                )
                value = {"status": "proposal_pass_finished"}
            elif command == "fit-task":
                value = fit_worker(root, args.round, args.task_index)
            elif command == "fit-inner":
                plan = io.verify(root)
                result = pipeline.fit_one(
                    root, plan, plan["tasks"][args.task_index], args.round
                )
                value = {"status": result["status"] if result else "dependency_missing"}
            else:
                plan = io.verify(root)
                if args.round == 0:
                    for task in plan["tasks"]:
                        if task["arm"] == "refit_only":
                            pipeline.fit_one(root, plan, task, 0)
                value = reporting.report(root)
                if plan["protocol"] == io.MULTI_PROTOCOL and any(
                    io.read_round(root, task, args.round) is None
                    for task in plan["tasks"]
                ):
                    raise ValueError(
                        "round incomplete: missing task results; inspect logs"
                    )
    elif command == "report":
        value = reporting.report(root)
    elif command == "export":
        with io.execution_lease(root, exclusive=True):
            value = reporting.export(root, allow_partial=args.allow_partial)
    elif command == "evaluation-report":
        value = reporting.evaluation_report(root)
    elif command == "evaluate":
        if args.public_root is None:
            parser.error("evaluate requires --public-root")
        value = reporting.evaluate(
            root, args.public_root.resolve(), args.shard, args.shards
        )
    else:
        from autoformalism.rebuttal.review_deadline_demo import run

        value = run(root)
    if "rows" in value:
        value = {k: v for k, v in value.items() if k != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
