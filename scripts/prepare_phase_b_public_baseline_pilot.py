#!/usr/bin/env python3
"""Freeze the public-only two-cell baseline development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.baseline_pilot import freeze_baseline_pilot


def main() -> None:
    """Validate public inputs and write immutable baseline task ledgers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--prompt-overlay-config", type=Path, required=True)
    parser.add_argument("--proposer-transport-plan", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_baseline_pilot(
        args.config,
        args.output_root,
        public_data_root=args.public_data_root,
        target_contract_root=args.target_contract_root,
        prompt_overlay_config_path=args.prompt_overlay_config,
        proposer_transport_plan_path=args.proposer_transport_plan,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
