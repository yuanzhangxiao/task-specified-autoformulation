#!/usr/bin/env python3
"""Export proposer and judge JSON Schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.schemas import export_json_schemas


def main() -> None:
    """Export deterministic schema files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("schemas"))
    args = parser.parse_args()
    for path in export_json_schemas(args.output):
        print(path)


if __name__ == "__main__":
    main()

