#!/usr/bin/env python3
"""Freeze, execute or report the general process-pruning pilot."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import process_pruning as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        result = campaign.freeze(args.source, args.root)
        result = {
            "identity": result["artifact_sha256"],
            "tasks": len(result["rows"]),
            "policy": result["policy"],
        }
    elif args.stage == "fit":
        result = campaign.run_one(args.root, args.index)
        result = {
            "task": result["task"],
            "choice": result["choice"]["status"],
            "selected": result["selection"]["selected"],
        }
    else:
        result = campaign.report(args.root)
        result = {
            "identity": result["identity"],
            "status_counts": result["status_counts"],
            "llm_calls": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
