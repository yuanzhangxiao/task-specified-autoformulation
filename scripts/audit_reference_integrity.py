#!/usr/bin/env python3
"""Audit all Phase-B reference generators without overwriting released datasets."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.reference_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--input-contract",
        choices=("legacy-events-1", "continuous-rates-1"),
        default="legacy-events-1",
    )
    parser.add_argument(
        "--suite", type=Path, default=Path("configs/benchmarks/phase_b_suite_v1.json")
    )
    parser.add_argument(
        "--include-test-protocols",
        action="store_true",
        help="Check newly simulated private test schedules; never score models.",
    )
    args = parser.parse_args()
    result = run_audit(
        args.output_root,
        args.data_root,
        args.suite,
        include_test_protocols=args.include_test_protocols,
        input_contract=args.input_contract,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    if not result["numerical_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
