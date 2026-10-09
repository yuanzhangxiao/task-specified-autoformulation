#!/usr/bin/env python3
"""Export hash-verified M22 warm checkpoints for the separate M23 evaluator."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.conditional_diagnostics import export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
