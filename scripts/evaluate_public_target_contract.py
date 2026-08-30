#!/usr/bin/env python3
"""Evaluate one candidate against a prompt-committed public target contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.schemas import CandidateModel
from autoformalism.targets import PublicTargetContract, evaluate_public_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate = CandidateModel.model_validate_json(
        args.candidate.read_text(encoding="utf-8")
    )
    contract = PublicTargetContract.model_validate_json(
        args.contract.read_text(encoding="utf-8")
    )
    result = evaluate_public_targets(candidate, contract)
    payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
