#!/usr/bin/env python3
"""Inventory every Phase-B campaign or evaluation root in one pass.

Results live in several roots, written by campaigns with different shapes, on
a machine that whoever is reading may not have. This walks a set of roots,
reports coverage before scores, and writes one JSON that can be transferred
and inspected on its own.

It reads only what a campaign already wrote. It never runs a model, opens test
data, or recomputes a metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

#: Statuses that mean the method ran and reached a scientific conclusion,
#: including an honest failure to find a model.
TERMINAL = {
    "complete",
    "inexpressible",
    "rollout_failed",
    "no_candidates",
    "discovery_failed",
    "source_unavailable",
}


def sha256(path: Path) -> str:
    """Identify a file by content so a transferred copy can be checked."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Return parsed JSON, or a dict describing why it could not be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # unreadable artifacts must not stop the sweep
        return {"__unreadable__": f"{type(exc).__name__}: {exc}"}


def roster_median(scored: list[float], planned: int) -> float | None:
    """Median over every planned identity, ranking unscored rows worst.

    Reporting a median over successes alone rewards a method for failing on
    the cells it finds hard, so the denominator is the roster.
    """
    if planned <= 0 or not scored:
        return None
    ranked = sorted(scored)
    index = (planned - 1) // 2
    return ranked[index] if index < len(ranked) else None


def summarize_rows(rows: list[dict]) -> dict:
    """Coverage and scores for one group of result rows."""
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    scored = [
        float(row["normalized_mse"])
        for row in rows
        if isinstance(row.get("normalized_mse"), (int, float))
    ]
    return {
        "rows": len(rows),
        "scored": len(scored),
        "status_counts": counts,
        "benchmarks": len({row.get("benchmark_id") for row in rows} - {None}),
        "median_over_scored": statistics.median(scored) if scored else None,
        "median_over_roster": roster_median(scored, len(rows)),
    }


def group_key(row: dict) -> str:
    """Whatever this campaign calls the method."""
    for field in ("source_kind", "method", "cohort"):
        value = row.get(field)
        if value:
            return str(value)
    return "unknown"


def inventory_root(root: Path) -> dict:
    """Describe one campaign or evaluation directory without interpreting it."""
    record: dict[str, Any] = {"root": str(root), "exists": root.is_dir()}
    if not record["exists"]:
        return record

    for name in ("plan.json", "summary.json", "manifest.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = read_json(path)
        entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if isinstance(payload, dict):
            for field in (
                "protocol", "schema_version", "status", "expected",
                "plan_sha256", "artifact_sha256", "test_data_opened",
                "private_reference_opened", "parameter_refit_applied",
                "reporting_qualifications", "submission_id",
            ):
                if field in payload:
                    entry[field] = payload[field]
            rows = payload.get("rows")
            if isinstance(rows, list) and rows:
                grouped: dict[str, list[dict]] = {}
                for row in rows:
                    grouped.setdefault(group_key(row), []).append(row)
                entry["groups"] = {
                    key: summarize_rows(items) for key, items in sorted(grouped.items())
                }
        record[name] = entry

    # Per-task results, which exist whether or not a summary was written.
    results = root / "results"
    if results.is_dir():
        rows, frozen, sealed = [], 0, 0
        for directory in sorted(results.iterdir()):
            if not directory.is_dir():
                continue
            result = directory / "result.json"
            if result.is_file():
                payload = read_json(result)
                if isinstance(payload, dict):
                    payload.setdefault("index", directory.name)
                    rows.append(payload)
            if (directory / "native-selection.json").is_file():
                frozen += 1
            sealed += 1
        record["results"] = {
            "task_directories": sealed,
            "with_result_json": len(rows),
            "with_frozen_model": frozen,
            **summarize_rows(rows),
        }

    submissions = root / "submissions"
    if submissions.is_dir():
        record["submissions"] = []
        for directory in sorted(submissions.iterdir()):
            manifest = directory / "manifest.json"
            item = {"submission": directory.name}
            if manifest.is_file():
                payload = read_json(manifest)
                if isinstance(payload, dict):
                    item.update(
                        {
                            key: payload.get(key)
                            for key in ("task_count", "batch_size", "batches", "tier")
                        }
                    )
            record["submissions"].append(item)
    return record


def main() -> None:
    """Inventory the given roots, plus anything found under --scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument(
        "--scan",
        type=Path,
        action="append",
        default=[],
        help="Directory whose immediate children are candidate roots",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = list(args.root)
    for directory in args.scan:
        if directory.is_dir():
            roots.extend(
                child
                for child in sorted(directory.iterdir())
                if child.is_dir()
                and any(
                    (child / name).exists()
                    for name in ("plan.json", "summary.json", "results")
                )
            )
    seen: set[Path] = set()
    records = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(inventory_root(resolved))

    value = {
        "schema_version": "phase-b-results-inventory-1",
        "roots": records,
    }
    args.out.write_text(json.dumps(value, indent=2, default=str) + "\n")

    for record in records:
        print(f"\n=== {record['root']}")
        if not record["exists"]:
            print("    (missing)")
            continue
        for name in ("plan.json", "summary.json"):
            entry = record.get(name)
            if not entry:
                continue
            bits = [
                f"{key}={entry[key]}"
                for key in ("protocol", "status", "expected")
                if key in entry
            ]
            print(f"    {name}: {' '.join(bits) or 'present'}")
            for key, group in (entry.get("groups") or {}).items():
                print(
                    f"      {key:<18} rows={group['rows']:<5}"
                    f" scored={group['scored']:<5}"
                    f" cells={group['benchmarks']:<4}"
                    f" median_roster={group['median_over_roster']}"
                )
        got = record.get("results")
        if got:
            print(
                f"    results/: tasks={got['task_directories']}"
                f" results={got['with_result_json']}"
                f" frozen_models={got['with_frozen_model']}"
                f" scored={got['scored']} {got['status_counts']}"
            )
        for item in record.get("submissions", []):
            name, count = item["submission"], item.get("task_count")
            print(f"    submission {name}: {count} tasks")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
