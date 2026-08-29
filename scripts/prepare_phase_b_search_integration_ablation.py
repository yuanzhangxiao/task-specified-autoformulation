#!/usr/bin/env python3
"""Freeze the matched judge/no-judge Phase-B search integration plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.search_integration_ablation import (
    freeze_search_integration_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = freeze_search_integration_plan(args.config, args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
