#!/usr/bin/env python3
"""Prepare, run and inspect a frozen construction study; no functions or fitting."""

import argparse
import json
import signal
from pathlib import Path
from time import monotonic

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.research import construction_comparison as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repair-source", type=Path)
    parser.add_argument("--study", choices=campaign.STUDIES)
    parser.add_argument("--bookkeeping-policy", choices=campaign.bookkeeping.POLICIES)
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("require --root or --plan")
    if args.output and args.output.resolve() != (root / "results").resolve():
        parser.error("output must be the plan's results directory")
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source (the previous topology campaign)")
        plan = campaign.freeze(
            args.source,
            root,
            study=args.study or "comparison",
            bookkeeping_policy=args.bookkeeping_policy or "legacy",
            repair_source=args.repair_source,
        )
    else:
        if args.source or args.study or args.bookkeeping_policy or args.repair_source:
            parser.error("public inputs and study are frozen by prepare")
        plan = campaign.verify(root)
    if args.command == "run":
        if not args.base_url:
            parser.error("run requires --base-url")
        if plan.get("study") != "comparison":
            print(
                json.dumps({"storage_check": campaign.check_storage(root)}), flush=True
            )
        stopping = False
        deadline = (
            monotonic() + min(args.wall_seconds, plan["config"]["wall_seconds"]) - 300
        )

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
                result = campaign.propose(
                    root, plan, task, args.base_url, can_start=can_start
                )
                print(
                    json.dumps({"task": task["task_id"], "status": result["status"]}),
                    flush=True,
                )
            except DeferredCall:
                break
    elif args.command == "report":
        result = campaign.report(root, plan)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    else:
        print(
            json.dumps(
                {"identity": plan["artifact_sha256"], "tasks": len(plan["tasks"])}
            )
        )


if __name__ == "__main__":
    main()
