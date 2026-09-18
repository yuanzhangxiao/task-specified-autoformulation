#!/usr/bin/env python3
"""Assess equations, fitted activity and observed responses without LLMs/refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.mechanism_assessment import prepare, report, run


def main() -> None:
    """Keep export separate; all source bundles are immutable saved models."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("--bundle", type=Path, action="append", required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/mechanism_assessment_v1.json",
    )
    task = commands.add_parser("run")
    task.add_argument("--index", type=int, required=True)
    summary = commands.add_parser("report")
    for command in (freeze, task, summary):
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.bundle, args.public_root, args.root, args.config)
    elif args.command == "run":
        result = run(args.root, args.index)
    else:
        result = report(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "rows",
                    "groups",
                    "config",
                    "fitted_activity",
                    "response_behavior",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
