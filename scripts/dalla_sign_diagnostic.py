#!/usr/bin/env python3
"""Prepare, verify, fit or report the explicitly assisted CPU sign diagnostic."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import dalla_sign_diagnostic as io
from autoformalism.rebuttal.prefit_replay import sealed_read


def main():
    campaign = io
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--site", choices=("aces", "delta"), default="aces")
    parser.add_argument("--index", type=int)
    parser.add_argument("--campaign", choices=("r4", "canonical"), default="r4")
    args = parser.parse_args()
    saved = args.root / "plan.json"
    if args.campaign == "canonical" or (
        saved.exists()
        and sealed_read(saved)["protocol"] == "dalla-canonical-sign-rescue-1"
    ):
        from autoformalism.rebuttal import dalla_canonical_rescue as campaign
    if args.command == "prepare":
        if args.source is None or args.decisions is None:
            parser.error("prepare requires --source and --decisions")
        plan = campaign.freeze(args.source, args.decisions, args.root, site=args.site)
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "arms": plan.get("task_count", 2),
            "gpus": 0,
        }
    elif args.command == "verify":
        result = {
            "status": "verified",
            "plan_sha256": campaign.verify(args.root)["artifact_sha256"],
        }
    elif args.command == "fit":
        index = (
            args.index
            if args.index is not None
            else int(os.environ["SLURM_ARRAY_TASK_ID"])
        )
        result = campaign.run_one(args.root, index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
