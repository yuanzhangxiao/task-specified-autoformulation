#!/usr/bin/env python3
"""Render revised construction requests offline from a saved inspection."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_focus_review import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
