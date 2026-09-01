#!/usr/bin/env python3
"""Freeze the matched GPT-OSS proposer token-budget calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.proposer_transport_calibration import (
    freeze_proposer_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_proposer_calibration(
        args.config,
        args.output_root,
        public_data_root=args.public_data_root,
        target_contract_root=args.target_contract_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
