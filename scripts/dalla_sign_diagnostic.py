#!/usr/bin/env python3
"""Prepare, verify, fit or report the explicitly assisted CPU sign diagnostic."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import dalla_sign_diagnostic as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--site", choices=("aces", "delta"), default="aces")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source is None or args.decisions is None:
            parser.error("prepare requires --source and --decisions")
        plan = io.freeze(args.source, args.decisions, args.root, site=args.site)
        result = {"plan_sha256": plan["artifact_sha256"], "arms": 2, "gpus": 0}
    elif args.command == "verify":
        result = {
            "status": "verified",
            "plan_sha256": io.verify(args.root)["artifact_sha256"],
        }
    elif args.command == "fit":
        index = (
            args.index
            if args.index is not None
            else int(os.environ["SLURM_ARRAY_TASK_ID"])
        )
        result = io.run_one(args.root, index)
    else:
        result = io.report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
