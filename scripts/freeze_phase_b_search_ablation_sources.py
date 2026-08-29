#!/usr/bin/env python3
"""Freeze matched judge/no-judge selections for common final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.search_integration_ablation import (
    freeze_search_ablation_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--hidden-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = freeze_search_ablation_sources(
        args.plan,
        args.search_root,
        args.hidden_audit,
        args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
