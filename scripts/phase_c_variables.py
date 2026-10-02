#!/usr/bin/env python3
"""Freeze, run and inspect variable-only Phase C confirmations."""

import argparse
import json
import signal
from pathlib import Path
from time import monotonic

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.research import variable_confirmation as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--review-inventory", action="store_true")
    parser.add_argument("--targeted-usage", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    if (args.review_inventory or args.targeted_usage) and args.command != "prepare":
        parser.error(
            "diagnostic options are frozen by prepare; later commands use the plan"
        )
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("require --root or --plan")
    if args.output and args.output.resolve() != (root / "results").resolve():
        parser.error("output must be the plan's results directory")
    if args.command == "prepare":
        if args.source_plan is None:
            parser.error("prepare requires --source-plan")
        plan = baseline.freeze(
            args.source_plan,
            root,
            review_inventory=args.review_inventory,
            targeted_usage=args.targeted_usage,
        )
        print(
            json.dumps(
                {"identity": plan["artifact_sha256"], "tasks": len(plan["tasks"])}
            )
        )
        return
    plan = baseline.verify(root)
    if args.command == "run":
        if not args.base_url:
            parser.error("run requires --base-url")
        deadline = (
            monotonic() + min(args.wall_seconds, plan["config"]["wall_seconds"]) - 300
        )
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def can_start():
            return not stopping and monotonic() < deadline

        for task in plan["tasks"]:
            if not can_start():
                break
            try:
                result = baseline.propose(
                    root, plan, task, args.base_url, can_start=can_start
                )
                print(
                    json.dumps({"task": task["task_id"], "status": result["status"]}),
                    flush=True,
                )
            except DeferredCall:
                break
    elif args.command == "report":
        result = baseline.report(root, plan)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status_counts",
                        "first_reply_accepted",
                        "first_reply_total",
                        "repair_calls",
                    )
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {"identity": plan["artifact_sha256"], "tasks": len(plan["tasks"])}
            )
        )


if __name__ == "__main__":
    main()
