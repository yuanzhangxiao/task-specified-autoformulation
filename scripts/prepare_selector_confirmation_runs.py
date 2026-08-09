"""Index completed frozen-selector refits for existing evaluation scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.result_root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": "full",
                "benchmark": payload["benchmark_id"],
                "tier": payload["tier"],
                "seed": payload["seed"],
                "status": payload["status"],
                "source": str(path),
                "test_mse": payload.get("test_mse"),
                "validation_mse": payload.get("validation_mse"),
                "artifact_id": payload["artifact_id"],
                "term_count": payload.get("term_count"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
