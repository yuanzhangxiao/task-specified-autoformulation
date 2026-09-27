#!/usr/bin/env python3
"""Bounded evaluator-assisted qualification, separate from discovery campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import reference_followup as followup
from autoformalism.benchmarks import reference_qualification as qualification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--public-data-root", type=Path, required=True)
    prepare.add_argument("--private-data-root", type=Path, default=Path("data_raw"))
    followup_prepare = commands.add_parser("followup-prepare")
    followup_prepare.add_argument("--source-root", type=Path, required=True)
    for name in ("run", "report", "followup-report"):
        subparser = commands.add_parser(name)
        if name == "run":
            subparser.add_argument("--index", type=int, required=True)
    for command in commands.choices.values():
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = qualification.prepare(
            args.root, args.public_data_root, args.private_data_root
        )
    elif args.command == "followup-prepare":
        result = followup.prepare(args.root, args.source_root)
    elif args.command == "followup-report":
        result = followup.report(args.root)
    elif args.command == "run":
        result = qualification.run_task(args.root, args.index)
        result = {
            k: v for k, v in result.items() if k not in {"reference", "fitted_replay"}
        }
    else:
        result = qualification.report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
