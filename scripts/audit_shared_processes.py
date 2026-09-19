#!/usr/bin/env python3
"""Inventory an existing public model snapshot without optimization or LLM calls."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.shared_process_audit import audit_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_bundle(args.bundle, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
