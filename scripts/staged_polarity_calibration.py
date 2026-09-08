#!/usr/bin/env python3
"""Freeze or run the topology-stage interaction-polarity calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_polarity_calibration import (
    freeze_polarity_calibration,
    run_polarity_calibration,
)


def main() -> None:
    """Dispatch immutable freeze and resumable run operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_polarity_calibration(args.config, args.output)
        print(
            json.dumps(
                {"plan_sha256": result["plan_sha256"], "tasks": len(result["tasks"])}
            )
        )
    else:
        print(
            json.dumps(
                run_polarity_calibration(
                    args.plan,
                    args.output,
                    args.base_url,
                    wall_seconds=args.wall_seconds,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
