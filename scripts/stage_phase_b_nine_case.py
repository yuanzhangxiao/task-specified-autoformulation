#!/usr/bin/env python3
"""Stage the nine Phase B cases' public inputs, or verify a staged copy.

`stage` copies each cell's training table, validation table and proposer
prompt from the first source directory holding the file the comparison's
campaign recorded, checking every digest, and writes a sealed receipt. No
test data or manifest is staged. `verify` checks a staged copy, for example
after it has been moved to another machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.phase_b_nine_case import stage, verify


def main() -> None:
    """Stage or verify; neither reads test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    copy = sub.add_parser("stage")
    copy.add_argument("--source", type=Path, action="append", required=True)
    copy.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        receipt = stage(args.source, args.output)
        value = {
            "receipt_sha256": verify(args.output.expanduser().resolve()),
            "cells": len(receipt["cells"]),
            "transforms": receipt["transforms"],
        }
    else:
        value = {"receipt_sha256": verify(args.root.expanduser().resolve())}
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
