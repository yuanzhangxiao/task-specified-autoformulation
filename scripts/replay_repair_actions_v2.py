#!/usr/bin/env python3
"""Replay historical repair actions read-only; no provider, fitting or judge."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.repair_action_replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--arm",
        default="redesigned_runtime",
        choices=("redesigned_runtime", "redesigned_prefit_judge"),
    )
    args = parser.parse_args()
    report = replay(args.source_root, args.output, args.arm)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"rows", "source_file_sha256"}
            },
            indent=2,
        )
    )
    if not report["stored_reply_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
