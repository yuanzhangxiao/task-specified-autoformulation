#!/usr/bin/env python3
"""Archive terminal hybrid-judge failures so only failed keys are retried."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

FAILURE_NAME = "hybrid_judge_failures.jsonl"
ARCHIVE_PATTERN = re.compile(r"hybrid_judge_failures\.attempt_(\d{3})\.jsonl")
MANIFEST_NAME = "hybrid_judge_failure_retries.jsonl"
REQUIRED_KEYS = {
    "pair_id",
    "judge_model",
    "repetition",
    "order",
    "error_type",
    "error",
}


def _read_failures(path: Path) -> list[dict[str, Any]]:
    """Load and validate one nonempty append-only failure ledger."""
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int, str]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"failure line {line_number} is not an object: {path}")
        missing = REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(
                f"failure line {line_number} is missing {sorted(missing)}: {path}"
            )
        key = (
            str(value["pair_id"]),
            str(value["judge_model"]),
            int(value["repetition"]),
            str(value["order"]),
        )
        if key in keys:
            raise ValueError(f"duplicate failure key {key}: {path}")
        keys.add(key)
        rows.append(value)
    return rows


def _next_attempt(shard_root: Path) -> int:
    attempts = [
        int(match.group(1))
        for path in shard_root.glob("hybrid_judge_failures.attempt_*.jsonl")
        if (match := ARCHIVE_PATTERN.fullmatch(path.name)) is not None
    ]
    return max(attempts, default=0) + 1


def rotate_failure_ledger(shard_root: Path) -> dict[str, Any] | None:
    """Archive active failures and leave an empty resume ledger."""
    active = shard_root / FAILURE_NAME
    if not active.is_file() or active.stat().st_size == 0:
        return None
    rows = _read_failures(active)
    if not rows:
        return None
    attempt = _next_attempt(shard_root)
    archive = shard_root / f"hybrid_judge_failures.attempt_{attempt:03d}.jsonl"
    if archive.exists():
        raise FileExistsError(f"retry archive already exists: {archive}")
    payload = active.read_bytes()
    record: dict[str, Any] = {
        "schema_version": "hybrid-judge-failure-retry-1",
        "attempt": attempt,
        "archived_ledger": archive.name,
        "archived_ledger_sha256": hashlib.sha256(payload).hexdigest(),
        "failure_count": len(rows),
        "failures_by_stage": dict(
            sorted(
                Counter(
                    str(row.get("failure_stage", "unknown")) for row in rows
                ).items()
            )
        ),
        "failures_by_error_type": dict(
            sorted(Counter(str(row["error_type"]) for row in rows).items())
        ),
    }
    active.replace(archive)
    active.touch()
    manifest = shard_root / MANIFEST_NAME
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    total = 0
    for shard_id in args.shard_ids:
        shard_root = args.root / "shards" / f"shard_{shard_id}"
        if not shard_root.is_dir():
            raise FileNotFoundError(f"missing shard directory: {shard_root}")
        record = rotate_failure_ledger(shard_root)
        if record is None:
            print(f"shard_{shard_id}: no active failures")
            continue
        count = int(record["failure_count"])
        total += count
        print(
            f"shard_{shard_id}: archived {count} failures as "
            f"{record['archived_ledger']}"
        )
    print(f"prepared {total} failed judgments for retry")


if __name__ == "__main__":
    main()
