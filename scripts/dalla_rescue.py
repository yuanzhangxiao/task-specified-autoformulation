#!/usr/bin/env python3
"""Prepare, execute or inspect the bounded CPU demonstration rescue."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import dalla_rescue as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.inputs is None:
            parser.error("--inputs is required")
        result = campaign.freeze(args.inputs, args.root)
        result = {
            "plan_sha256": result["artifact_sha256"],
            "tasks": len(result["rows"]),
        }
    elif args.command == "fit":
        index = args.index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = campaign.run_one(args.root, index)
        result = {"status": result["status"], "task": result["task"]}
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
