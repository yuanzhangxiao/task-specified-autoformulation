#!/usr/bin/env python3
"""Replay frozen multiround replies through the revised parameter-role policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_multiround_role_replay import (
    replay_multiround_role_repairs,
)


def main() -> None:
    """Write one public-only, call-free replay report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            replay_multiround_role_repairs(args.source_root, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
