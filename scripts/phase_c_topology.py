#!/usr/bin/env python3
"""Freeze, continue and inspect Phase C topology from saved inventories."""

import argparse
import json
import signal
from pathlib import Path
from time import monotonic

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.research import topology_confirmation as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--usage-source", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("require --root or --plan")
    if args.output and args.output.resolve() != (root / "results").resolve():
        parser.error("output must be the plan's results directory")
    if args.command == "prepare":
        if args.base_source is None or args.usage_source is None:
            parser.error("prepare requires --base-source and --usage-source")
        plan = campaign.freeze(args.base_source, args.usage_source, root)
    else:
        if args.base_source or args.usage_source:
            parser.error("source imports are frozen by prepare")
        plan = campaign.verify(root)
    if args.command == "run":
        if not args.base_url:
            parser.error("run requires --base-url")
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
