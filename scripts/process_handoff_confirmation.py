#!/usr/bin/env python3
"""Freeze or summarize the fresh matched signed-process confirmation."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import process_handoff_confirmation as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        if args.source is None or args.audit is None:
            parser.error("prepare requires --source and --audit")
        plan = experiment.freeze(args.source, args.audit, root)
        result = {"identity": plan["artifact_sha256"], **plan["handoff_confirmation"]}
    elif args.command == "verify":
        plan = experiment.pilot.verify(root)
        if "handoff_confirmation" not in plan:
            parser.error("not a handoff confirmation")
        result = {"identity": plan["artifact_sha256"]}
    else:
        summary = experiment.report(root)
        result = {
            key: summary[key]
            for key in (
                "historical_status_counts",
                "current_status_counts",
                "historical_accounting",
                "current_accounting",
            )
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
