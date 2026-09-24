#!/usr/bin/env python3
"""Export frozen labeled cases, then run/check portable Jetstream revalidation."""

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

from autoformalism.rebuttal import judge_jetstream_calibration as campaign
from autoformalism.rebuttal.judge_jetstream_calibration_report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--data-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    for command in ("verify", "run", "report"):
        sub.add_parser(command).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        value = campaign.export(args.source, args.data_root, args.output)
        print(
            json.dumps(
                {
                    "export_sha256": value["artifact_sha256"],
                    "pairs": len(value["pairs"]),
                }
            )
        )
        return
    if args.command == "freeze":
        print(json.dumps(campaign.freeze(args.source, args.root), indent=2))
        return
    if args.command == "verify":
        print(
            json.dumps({"identity": campaign.verify(args.root)[0]["artifact_sha256"]})
        )
        return
    if args.command == "run":
        try:
            campaign.run(
                args.root,
                lambda: os.environ.get("AF_JETSTREAM_API_KEY")
                or getpass.getpass("Jetstream API key (hidden, not saved): "),
                lambda text: print(
                    f"[{time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True
                ),
            )
        except KeyboardInterrupt:
            print(
                "Interrupted; saved calls retained. "
                "Rerun resumes unstarted orientations.",
                file=sys.stderr,
            )
    value = report(args.root)
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "status",
                    "status_counts",
                    "known_case_gate_passed",
                    "fresh_holdout_calibration_established",
                )
            }
            | {"cost": {k: v for k, v in value["cost"].items() if k != "calls"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
