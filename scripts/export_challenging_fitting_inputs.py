#!/usr/bin/env python3
"""Export known-equation M6 inputs without regenerating benchmark data."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.challenging_fitting_inputs import export_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_inputs(args.release, args.output), indent=2))
