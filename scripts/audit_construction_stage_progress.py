#!/usr/bin/env python3
"""Inspect first replies, stage exits and global repair without any model calls."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_stage_progress import audit, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    write_report(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
