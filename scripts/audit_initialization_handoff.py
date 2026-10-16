#!/usr/bin/env python3
"""Qualify initializer information and matched-trajectory checks offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.research.initialization_handoff_audit import audit


def main() -> None:
    """Write a separate immutable audit; neither launch calls nor modify sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial-information",
        choices=("unknown", "causal_runtime"),
        default="unknown",
        help="Explicitly qualify the existing shared, recorded-t0 causal runtime.",
    )
    parser.add_argument("--handoffs", type=Path)
    args = parser.parse_args()
    result = audit(
        args.source, args.root, mode=args.initial_information, handoffs=args.handoffs
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
