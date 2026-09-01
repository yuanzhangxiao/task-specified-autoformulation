#!/usr/bin/env python3
"""Freeze a selected proposer operating point for cross-cluster confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.proposer_transport_calibration import (
    prepare_selected_proposer_confirmation,
)


def main() -> None:
    """Validate the primary analysis and write a one-budget confirmation plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--primary-platform", required=True)
    parser.add_argument("--confirmation-platform", required=True)
    args = parser.parse_args()
    manifest = prepare_selected_proposer_confirmation(
        args.source_plan,
        args.source_analysis,
        args.output_config,
        primary_platform=args.primary_platform,
        confirmation_platform=args.confirmation_platform,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
