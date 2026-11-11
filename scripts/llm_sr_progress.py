#!/usr/bin/env python3
"""Show how far each task of an LLM-SR campaign has got; never calls a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.llm_sr_progress import campaign_progress


def main() -> None:
    """Print the campaign's progress as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(campaign_progress(args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
