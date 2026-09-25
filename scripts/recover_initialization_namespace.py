#!/usr/bin/env python3
"""Recover explicitly selected unstarted round-zero fits in a separate directory."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import initialization_recovery as recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source is None or not args.task:
            parser.error("prepare requires --source and at least one --task")
        value = recovery.prepare(args.source, args.root, args.task)
    elif args.command == "run":
        if args.task_index is None:
            parser.error("run requires --task-index")
        value = recovery.run(args.root, args.task_index)
    else:
        value = recovery.report(args.root)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
