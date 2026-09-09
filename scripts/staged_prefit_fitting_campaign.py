#!/usr/bin/env python3
"""Freeze, fit, or summarize the exact staged pre-fit candidate handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_prefit_fitting_campaign import (
    freeze_campaign,
    run_task,
    summarize_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_campaign(
            args.config, args.source_root, args.public_root, args.output_root
        )
    elif args.command == "run":
        result = run_task(args.output_root, args.task_index)
    else:
        result = summarize_campaign(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
