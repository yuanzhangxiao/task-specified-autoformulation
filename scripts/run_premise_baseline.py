"""Run or inspect the bounded continuous premise-baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.research.premise_baseline_campaign import (
    report_campaign,
    run_campaign,
)


def main() -> None:
    """Use an explicit output root and preserve partial-comparison denominators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-checkpoints", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        if args.max_new_checkpoints is not None:
            parser.error("--report-only does not accept --max-new-checkpoints")
        report = report_campaign(args.output_root)
    else:
        report = run_campaign(
            args.output_root, max_new_checkpoints=args.max_new_checkpoints
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
