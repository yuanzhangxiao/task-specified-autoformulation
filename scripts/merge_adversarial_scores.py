"""Merge resumable adversarial-judge shards and verify complete coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ("pair_id", "known_label", "judge_model", "repetition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=336)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.inputs if path.is_file()]
    if not frames:
        raise SystemExit("no score shards were found")
    merged = pd.concat(frames, ignore_index=True)
    conflicts = merged.groupby(list(KEY)).aggregate_score.nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise SystemExit(f"conflicting duplicate score keys: {len(conflicts)}")
    merged = merged.drop_duplicates(list(KEY)).sort_values(list(KEY))
    if len(merged) != args.expected:
        raise SystemExit(
            f"incomplete adversarial scores: {len(merged)}/{args.expected}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"merged {len(merged)} unique scores into {args.output}")


if __name__ == "__main__":
    main()
