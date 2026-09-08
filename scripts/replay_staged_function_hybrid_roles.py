#!/usr/bin/env python3
"""Replay certified outer-gain role repair over a frozen hybrid campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_function_role_replay import (
    replay_hybrid_outer_gain_repairs,
)


def main() -> None:
    """Write one versioned, public-only offline replay report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            replay_hybrid_outer_gain_repairs(args.source_root, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
