#!/usr/bin/env python3
"""Export immutable public training evidence without loading held-out data."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_training import export_bundle
from autoformalism.search.topology_compatibility import EvidenceContract


def main() -> None:
    """Require explicit candidate-class hypotheses instead of inferring them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contract = EvidenceContract.model_validate_json(args.runtime_contract.read_text())
    value = export_bundle(args.source_plan, args.public_root, args.output, contract)
    print(
        json.dumps(
            {
                "artifact_sha256": value["artifact_sha256"],
                "cases": len(value["cells"]),
                "test_data_opened": False,
                "llm_calls": 0,
                "optimizer_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
