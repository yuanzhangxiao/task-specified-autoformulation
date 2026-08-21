"""Merge resumable hybrid-judge shards and verify complete coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEY = ("pair_id", "judge_model", "repetition", "order")


def _key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row["pair_id"]),
        str(row["judge_model"]),
        int(row["repetition"]),
        str(row["order"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--failure-inputs", nargs="*", type=Path, default=())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first"),
        default="error",
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
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        conflicts = merged.groupby(list(KEY)).baseline_preference.nunique(
            dropna=False
        )
        conflicts = conflicts[conflicts > 1]
        if len(conflicts) and args.duplicate_policy == "error":
            raise SystemExit(f"conflicting duplicate hybrid keys: {len(conflicts)}")
        raw_count = len(merged)
        merged = (
            merged.sort_values(["_input_index", "_source_row"], kind="stable")
            .drop_duplicates(list(KEY), keep="first")
            .sort_values(list(KEY))
        )
    else:
        merged = pd.DataFrame(columns=KEY)
        conflicts = pd.Series(dtype=int)
        raw_count = 0

    failure_rows: list[dict[str, object]] = []
    for path in args.failure_inputs:
        if not path.is_file():
            continue
        failure_rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    failures_by_key: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for row in failure_rows:
        row_key = _key(row)
        if row_key in failures_by_key and failures_by_key[row_key] != row:
            if args.duplicate_policy == "error":
                raise SystemExit(f"conflicting duplicate failure key: {row_key}")
            continue
        failures_by_key.setdefault(row_key, row)
    success_keys = {
        (str(row[0]), str(row[1]), int(row[2]), str(row[3]))
        for row in merged.loc[:, list(KEY)].itertuples(index=False, name=None)
    }
    overlap = success_keys & failures_by_key.keys()
    if overlap:
        raise SystemExit(
            f"keys occur in both success and failure shards: {len(overlap)}"
        )
    covered = len(success_keys) + len(failures_by_key)
    if covered != args.expected:
        raise SystemExit(f"incomplete hybrid outcomes: {covered}/{args.expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(
        columns=["_input_index", "_source_row"], errors="ignore"
    ).to_csv(args.output, index=False)
    failure_output = args.failure_output or args.output.with_name(
        "hybrid_judge_failures.jsonl"
    )
    failure_output.parent.mkdir(parents=True, exist_ok=True)
    failure_output.write_text(
        "".join(
            json.dumps(failures_by_key[key], sort_keys=True) + "\n"
            for key in sorted(failures_by_key)
        ),
        encoding="utf-8",
    )
    print(
        f"merged {len(merged)} successes and {len(failures_by_key)} failures "
        f"into {args.output} and {failure_output}; "
        f"discarded_duplicates={raw_count - len(merged)} "
        f"discarded_failure_duplicates={len(failure_rows) - len(failures_by_key)} "
        f"conflicting_preference_keys={len(conflicts)} "
        f"policy={args.duplicate_policy}"
    )


if __name__ == "__main__":
    main()
