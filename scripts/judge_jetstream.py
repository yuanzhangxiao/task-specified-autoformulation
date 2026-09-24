#!/usr/bin/env python3
"""Freeze, run or inspect one authenticated Jetstream judge compatibility pilot."""

import argparse
import getpass
import json
import os
from pathlib import Path

from autoformalism.rebuttal import judge_jetstream as pilot


def main() -> None:
    """Read credentials only for a live request, never from a command-line argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--review-index", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.source_plan is None:
            parser.error("freeze requires --source-plan")
        value = pilot.freeze(args.source_plan, args.root, args.review_index)
        print(
            json.dumps(
                {
                    k: value[k]
                    for k in (
                        "artifact_sha256",
                        "selection",
                        "placements",
                        "execution",
                        "maximum_paired_reviews",
                        "maximum_physical_requests",
                    )
                },
                indent=2,
            )
        )
        return
    if args.source_plan is not None or args.review_index is not None:
        parser.error("source-plan and review-index apply only to freeze")
    if args.command == "run":

        def key() -> str:
            return os.environ.get("AF_JETSTREAM_API_KEY") or getpass.getpass(
                "Jetstream API key (hidden; not saved): "
            )

        pilot.run(args.root, key)
    if args.command == "verify":
        print(json.dumps({"identity": pilot.verify(args.root)["artifact_sha256"]}))
        return
    value = pilot.report(args.root)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "status",
                    "paired_review_completed",
                    "calibration_established",
                    "cost",
                    "schema_attempt_counts",
                    "review_availability",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
