#!/usr/bin/env python3
"""Freeze M7 training-selected endpoints for the separate M9 assisted diagnostic."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.screening_diagnostic import export_starts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_starts(args.source, args.output), indent=2))
