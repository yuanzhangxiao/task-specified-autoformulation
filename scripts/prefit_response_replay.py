#!/usr/bin/env python3
"""Audit saved construction slots on CPU without fitting or provider calls."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import replay


def main() -> None:
    """Save the full immutable corpus and print only its compact counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.source, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_sha256": result["artifact_sha256"],
                "counts": result["counts"],
                "output": str(args.output),
                "limitation": result["limitation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
