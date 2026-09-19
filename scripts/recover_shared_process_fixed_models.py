#!/usr/bin/env python3
"""Preserve round zero and evaluate its fixed-model empty-vector failures once."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.shared_process_recovery import recover

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.source, args.output), indent=2))
