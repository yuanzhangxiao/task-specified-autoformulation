#!/usr/bin/env python3
"""Freeze, verify or report the fresh basin assembly-policy comparison."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import process_assembly_confirmation as experiment


def main() -> None:
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
        result = {
            "identity": plan["artifact_sha256"],
            "experiment": experiment.PROTOCOL,
            "fresh_constructions": 8,
            "fit_arms": 16,
        }
    elif args.command == "verify":
        plan = experiment.pilot.verify(root)
        if "assembly_confirmation" not in plan:
            parser.error("not an assembly confirmation")
        result = {"identity": plan["artifact_sha256"]}
    else:
        summary = experiment.report(root)
        result = {
            k: v for k, v in summary.items() if k.endswith(("counts", "accounting"))
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
