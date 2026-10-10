#!/usr/bin/env python3
"""Render marked prompt changes and audit saved repair triggers; no model calls."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_deferred_review import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
