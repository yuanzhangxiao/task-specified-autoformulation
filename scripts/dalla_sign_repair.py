#!/usr/bin/env python3
"""Freeze, review, refit or report a separate public-context sign-repair campaign."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import dalla_sign_repair as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "run", "fit", "report")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--wall-seconds", type=int, default=3000)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root or (args.plan.parent if args.plan else None)
    if root is None:
        parser.error("--root or --plan is required")
    if args.command == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare needs --source and --config")
        plan = campaign.freeze(args.source, args.config, root)
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "tasks": len(plan["rows"]),
            "eligible_models": sum(
                bool(r["review_context"]["eligible_slots"]) for r in plan["rows"]
            ),
        }
    elif args.command == "verify":
        result = {
            "status": "verified",
            "plan_sha256": campaign.verify(root)["artifact_sha256"],
        }
    elif args.command == "run":
        if not args.base_url:
            parser.error("run needs --base-url")
        result = campaign.run_reviews(root, args.base_url, args.wall_seconds)
    elif args.command == "fit":
        index = (
            args.index
            if args.index is not None
            else int(os.environ["SLURM_ARRAY_TASK_ID"])
        )
        result = campaign.fit_one(root, index)
    else:
        result = campaign.report(root)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "run" and any(
        row["review_status"] is None for row in result["rows"]
    ):
        raise SystemExit(3)  # Do not release CPU dependencies after a partial review.


if __name__ == "__main__":
    main()
