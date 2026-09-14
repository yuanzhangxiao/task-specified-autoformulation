#!/usr/bin/env python3
"""Freeze, construct, fit and summarize the matched pre-fitting pilot."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_construction_campaign import (
    freeze,
    report,
    run,
    summarize,
    verify,
)


def main() -> None:
    """Construction uses an existing local model server; fitting is CPU-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--public-root", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    for name in ("run", "fit", "summary", "report", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=name != "run")
        if name in {"run", "fit"}:
            command.add_argument("--arm", choices=("brief_only", "training_evidence"))
            command.add_argument("--wall-seconds", type=float)
        if name == "run":
            command.add_argument("--plan", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--base-url", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.config, args.public_root, args.output)
        result = {
            "plan_sha256": result["artifact_sha256"],
            "tasks": len(result["tasks"]),
        }
    elif args.command == "run":
        root = args.root or (args.plan.parent if args.plan else None)
        if root is None:
            parser.error("run requires --root or --plan")
        if args.plan and args.plan.resolve() != (root / "plan.json").resolve():
            parser.error("--plan must be ROOT/plan.json")
        if args.output and args.output.resolve() != (root / "results").resolve():
            parser.error("--output must be ROOT/results")
        result = run(
            root,
            "construct",
            args.base_url,
            arm=args.arm,
            wall_seconds=args.wall_seconds,
        )
    elif args.command == "fit":
        result = run(args.root, "fit", arm=args.arm, wall_seconds=args.wall_seconds)
    elif args.command == "summary":
        result = summarize(args.root)
    elif args.command == "report":
        result = report(args.root)
    else:
        plan = verify(args.root)
        result = {"status": "verified", "plan_sha256": plan["artifact_sha256"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
