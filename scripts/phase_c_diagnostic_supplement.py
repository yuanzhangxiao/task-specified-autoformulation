#!/usr/bin/env python3
"""Prepare, audit, run or inspect the M33 fixed-vector diagnostic supplement."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import diagnostic_supplement_campaign as campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--source", type=Path, required=True)
    for name in ("prepare", "verify", "run", "report"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--source", type=Path, required=True)
            child.add_argument("--config", type=Path, required=True)
        if name == "run":
            child.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = campaign.audit(args.source)
    elif args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.source,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {"status": "verified", "tasks": len(plan["tasks"])}
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
