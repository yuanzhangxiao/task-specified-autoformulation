#!/usr/bin/env python3
"""Repair saved basin models against public equation witnesses."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import basin_repair_pilot as pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "run", "fit", "report")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("--root or --plan required")
    if args.command == "prepare":
        if args.source is None or args.audit is None:
            parser.error("--source and --audit required")
        plan = pilot.freeze(args.source, args.audit, root)
        result = {"identity": plan["artifact_sha256"], "tasks": len(plan["tasks"])}
    elif args.command == "verify":
        result = {"identity": pilot.verify(root)["artifact_sha256"]}
    elif args.command == "run":
        result = pilot.run_proposals(root, args.base_url, args.wall_seconds)
    elif args.command == "fit":
        result = pilot.fit_task(root, args.index)
    else:
        result = pilot.report(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
