#!/usr/bin/env python3
"""Prepare, construct, fit, independently assess and inspect the Phase C pilot."""

import argparse
import json
import signal
from pathlib import Path
from time import monotonic

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.research import construction_baseline as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "run", "fit", "assess", "report")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=baseline.REPO / "configs/phase_c_construction_v1.json",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("require --root or --plan")
    if args.output and args.output.resolve() != (root / "results").resolve():
        parser.error("output must be the plan's results directory")
    if args.command == "prepare":
        if args.release is None:
            parser.error("prepare requires --release")
        plan = baseline.freeze(args.release, root, args.config)
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
    elif args.command in {"fit", "assess"}:
        if args.index is None or not 0 <= args.index < len(plan["tasks"]):
            parser.error("require a valid --index")
        result = getattr(baseline, args.command)(root, plan, plan["tasks"][args.index])
        print(json.dumps({"status": result["status"]}))
    elif args.command == "report":
        result = baseline.report(root, plan)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("status_counts", "assessment_counts", "aggregate")
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
