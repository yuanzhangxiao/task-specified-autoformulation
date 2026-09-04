#!/usr/bin/env python3
"""Freeze the public inputs and task ledger for staged search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_search_pilot import (
    freeze_staged_search_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_staged_search_pilot(
        args.config,
        args.output_root,
        public_data_root=args.public_data_root,
        target_contract_root=args.target_contract_root,
        mechanism_spec_root=args.mechanism_spec_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
