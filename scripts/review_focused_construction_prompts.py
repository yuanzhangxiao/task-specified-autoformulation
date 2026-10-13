#!/usr/bin/env python3
"""Render revised construction requests offline from a saved inspection."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_focus_review import render
from autoformalism.search import (
    construction_completion,
    construction_coverage,
    construction_decisions,
    construction_focus,
    construction_readable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=(
            construction_focus.POLICY,
            construction_readable.POLICY,
            construction_completion.POLICY,
            construction_coverage.POLICY,
            construction_decisions.POLICY,
        ),
        default=construction_focus.POLICY,
    )
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output, args.policy), indent=2))


if __name__ == "__main__":
    main()
