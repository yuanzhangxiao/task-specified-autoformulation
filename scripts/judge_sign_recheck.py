#!/usr/bin/env python3
"""Audit saved M4 sign evidence and rejudge only affected symbolic pairs."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import judge_sign_recheck as campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        campaign.freeze(args.source, args.root)
        value = campaign.report(args.root)
    elif args.stage == "run":
        if not args.base_url:
            parser.error("run requires --base-url")
        plan = campaign.verify(args.root)
        for index in range(len(plan["reviews"])):
            result = campaign.run_one(args.root, index, args.base_url)
            print(json.dumps({"index": index, "status": result["status"]}), flush=True)
            campaign.report(args.root)
        value = campaign.report(args.root)
    elif args.stage == "report":
        value = campaign.report(args.root)
    else:
        value = campaign.verify(args.root)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "artifact_sha256",
                    "identity",
                    "unique_reviews",
                    "affected_unique_reviews",
                    "status_counts",
                    "cost",
                    "optimizer_calls",
                    "proposer_calls",
                    "model_changes",
                    "selection_changes",
                )
                if k in value
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
