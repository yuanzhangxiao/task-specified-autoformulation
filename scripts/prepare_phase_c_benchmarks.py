#!/usr/bin/env python3
"""Prepare/verify the separate Phase C development release without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.phase_c_release import FAMILIES, build, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private-data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    report = (
        build(args.root, args.private_data_root, tuple(args.families))
        if args.command == "build"
        else verify(args.root)
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"cells", "files"}}, indent=2
        )
    )
    if not report["ready_for_development"]:
        raise SystemExit(
            "Some development cells failed qualification; inspect diagnostics."
        )


if __name__ == "__main__":
    main()
