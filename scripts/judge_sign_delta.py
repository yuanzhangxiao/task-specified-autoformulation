#!/usr/bin/env python3
"""Freeze, run, or inspect a portable Delta sign recheck."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import judge_sign_delta as campaign


def main() -> None:
    """Keep execution resumable without any remote ACES paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--image-sha256")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if args.stage == "freeze":
        if not args.source_plan or not args.image_sha256:
            parser.error("freeze requires --source-plan and --image-sha256")
        campaign.freeze(args.source_plan, args.root, args.image_sha256)
    elif args.stage == "verify":
        print(json.dumps({"identity": campaign.verify(args.root)["artifact_sha256"]}))
        return
    elif args.stage == "run":
        if not args.base_url:
            parser.error("run requires --base-url")
        for index in range(len(campaign.verify(args.root)["reviews"])):
            campaign.run_one(args.root, index, args.base_url)
            campaign.report(args.root)
    value = campaign.report(args.root)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "unique_reviews",
                    "affected_unique_reviews",
                    "status_counts",
                    "cost",
                    "execution",
                    "origin_plan_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
