"""Merge resumable comparative-judge shards and verify complete coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ("pair_id", "judge_model", "repetition", "order")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first"),
        default="error",
        help="fail on conflicting preferences or keep the first recorded execution",
    )
    args = parser.parse_args()
    frames = []
    for input_index, path in enumerate(args.inputs):
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["_input_index"] = input_index
        frame["_source_row"] = range(len(frame))
        frames.append(frame)
    if not frames:
        raise SystemExit("no comparative score shards were found")
    merged = pd.concat(frames, ignore_index=True)
    conflicts = merged.groupby(list(KEY)).baseline_preference.nunique(dropna=False)
    conflicts = conflicts[conflicts > 1]
    if len(conflicts) and args.duplicate_policy == "error":
        raise SystemExit(f"conflicting duplicate comparison keys: {len(conflicts)}")
    raw_count = len(merged)
    merged = (
        merged.sort_values(["_input_index", "_source_row"], kind="stable")
        .drop_duplicates(list(KEY), keep="first")
        .sort_values(list(KEY))
    )
    if len(merged) != args.expected:
        raise SystemExit(
            f"incomplete comparative scores: {len(merged)}/{args.expected}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.drop(columns=["_input_index", "_source_row"])
    merged.to_csv(args.output, index=False)
    print(
        f"merged {len(merged)} unique comparisons into {args.output}; "
        f"discarded_duplicates={raw_count - len(merged)} "
        f"conflicting_preference_keys={len(conflicts)} "
        f"policy={args.duplicate_policy}"
    )


if __name__ == "__main__":
    main()
