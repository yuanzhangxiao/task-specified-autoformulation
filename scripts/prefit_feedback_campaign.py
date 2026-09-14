#!/usr/bin/env python3
"""Freeze, run and report matched local feedback; no fitting command exists."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_feedback import ARMS, freeze, run, summarize, verify


def main() -> None:
    """Expose replay, local episodes and readable reports independently."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ("verify", "run", "summary", "cases"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=name != "run")
        if name == "run":
            command.add_argument("--plan", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--base-url", required=True)
            command.add_argument("--arm", choices=ARMS)
            command.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze(args.corpus, args.config, args.output)
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "cases": len(plan["selected"]),
            "episodes": len(plan["tasks"]),
            "maximum_provider_calls": len(plan["tasks"])
            * plan["config"]["model_settings"]["attempts_per_step"],
        }
    elif args.command == "verify":
        result = {
            "status": "verified",
            "plan_sha256": verify(args.root)["artifact_sha256"],
        }
    elif args.command == "cases":
        plan = verify(args.root)
        result = [
            {
                "case_id": e["case"]["case_id"],
                "cohort": e["cohort"],
                "kind": e["case"]["kind"],
                "source_task": e["case"]["task_id"],
                "component": e["case"]["component"],
                "baseline_diagnostics": e["baseline"]["diagnostics"],
            }
            for e in plan["selected"]
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
            result = run(
                root, args.base_url, arm=args.arm, wall_seconds=args.wall_seconds
            )
        else:
            result = summarize(root)
        result = {k: v for k, v in result.items() if k != "rows"}
        result["full_report"] = str(root / "summary.json")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
