#!/usr/bin/env python3
"""Freeze the matched feedback-rich proposer refinement pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.proposer_refinement_pilot import (
    freeze_refinement_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    parser.add_argument("--judge-protocol", type=Path, required=True)
    args = parser.parse_args()

    result = freeze_refinement_pilot(
        args.config,
        args.output_root,
        public_data_root=args.public_data_root,
        target_contract_root=args.target_contract_root,
        mechanism_spec_root=args.mechanism_spec_root,
        judge_protocol_path=args.judge_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
