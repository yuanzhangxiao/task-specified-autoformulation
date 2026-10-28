"""Run or inspect the isolated finite premise reference without a notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.research.premise_reference import (
    ReferenceConfig,
    report_control,
    run_control,
)


def main() -> None:
    """Require an explicit output root and print complete or partial denominators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trials-per-world", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-new-worlds", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        if args.max_new_worlds is not None:
            parser.error("--report-only does not accept --max-new-worlds")
        result = report_control(args.output_root)
    else:
        result = run_control(
            args.output_root,
            ReferenceConfig(args.trials_per_world, args.seed),
            max_new_worlds=args.max_new_worlds,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
