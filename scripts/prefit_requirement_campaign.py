#!/usr/bin/env python3
"""Freeze, inspect and repair public model requirements without a fitting stage."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_requirements import (
    freeze,
    replay_saved_repairs,
    run,
    summarize,
    verify,
)


def main() -> None:
    """Expose a resumable ACES worker and readable diagnostics independently."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ("verify", "diagnostics", "summary", "run", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=name != "run")
        if name == "replay":
            command.add_argument("--previous-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--plan", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--base-url", required=True)
            command.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze(args.source, args.config, args.output)
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "source_models": len(plan["cases"]),
            "distinct_source_gaps": sum(c["cohort"] == "repair" for c in plan["cases"]),
            "episodes": len(plan["tasks"]),
        }
    elif args.command == "replay":
        report = replay_saved_repairs(args.root, args.previous_root)
        result = {k: v for k, v in report.items() if k != "rows"}
        result["full_report"] = str(args.root / "replay.json")
    elif args.command == "verify":
        result = {
            "status": "verified",
            "plan_sha256": verify(args.root)["artifact_sha256"],
        }
    elif args.command == "diagnostics":
        plan = verify(args.root)
        result = [
            {
                "source_task": c["bundle"]["source_task"],
                "case_id": c["case_id"],
                "diagnosis": c["baseline"],
                "review_facts": c["review_facts"],
            }
            for c in plan["cases"]
        ]
    else:
        root = args.root
        if args.command == "run":
            root = root or (args.plan.parent if args.plan else None)
            if root is None:
                parser.error("run requires --root or --plan")
            if args.plan and args.plan.resolve() != (root / "plan.json").resolve():
                parser.error("--plan must be ROOT/plan.json")
            if args.output and args.output.resolve() != (root / "results").resolve():
                parser.error("--output must be ROOT/results")
            result = run(root, args.base_url, wall_seconds=args.wall_seconds)
        else:
            result = summarize(root)
        result = {k: v for k, v in result.items() if k != "rows"}
        result["full_report"] = str(root / "summary.json")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
