#!/usr/bin/env python3
"""Bind the D3 pilot to the selected GPT-OSS proposer operating point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.baseline_pilot import (
    freeze_baseline_llm_operating_point,
)


def main() -> None:
    """Validate the passing calibration and freeze the D3 LLM settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--proposer-plan", type=Path, required=True)
    parser.add_argument("--proposer-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_baseline_llm_operating_point(
        args.baseline_plan,
        args.proposer_plan,
        args.proposer_analysis,
        args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
