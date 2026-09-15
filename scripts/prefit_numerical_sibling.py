#!/usr/bin/env python3
"""Prepare, replay, propose and fit one optional numerical-feedback sibling."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import prefit_numerical_sibling as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    for name in (
        "parent-fit",
        "continuation-fit",
        "source",
        "construction-root",
        "config",
        "root",
    ):
        command.add_argument("--" + name, type=Path, required=True)
    for name in ("replay", "verify", "run", "fit", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=name != "run")
        if name == "run":
            command.add_argument("--plan", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--base-url", required=True)
            command.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = campaign.prepare(
            args.parent_fit,
            args.continuation_fit,
            args.source,
            args.construction_root,
            args.config,
            args.root,
        )
        result = {"status": "prepared", "plan_sha256": plan["artifact_sha256"]}
    elif args.command == "replay":
        replay = campaign.replay(args.root)
        result = {k: replay[k] for k in ("identity", "status")}
        campaign.report(args.root)
    elif args.command == "verify":
        plan, residual = campaign.verify(args.root)
        result = {
            "status": "verified",
            "plan_sha256": plan["artifact_sha256"],
            "replay_status": residual["status"],
        }
    elif args.command == "fit":
        result = campaign.fit_child(args.root)
    elif args.command == "report":
        result = campaign.report(args.root)
    else:
        root = args.root or (args.plan.parent if args.plan else None)
        if root is None:
            parser.error("run requires --root or --plan")
        if args.plan and args.plan.resolve() != (root / "plan.json").resolve():
            parser.error("--plan must be ROOT/plan.json")
        if args.output and args.output.resolve() != (root / "results").resolve():
            parser.error("--output must be ROOT/results")
        result = campaign.run(root, args.base_url, wall_seconds=args.wall_seconds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
