#!/usr/bin/env python3
"""Recompute a frozen evaluation's medians over the full planned roster.

The sealed report states medians conditional on success, which rewards a
method for failing on the cells it finds hardest: a method that scores only
its ten easiest conditions reports a better number than one that scores all
forty. This reranks every planned identity, placing unscored ones worst, and
reports both figures side by side with the denominators behind them.

This is a reporting change, not a re-evaluation. It reads the rows the sealed
run already produced and opens nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

#: Why an identity has no score. Kept apart because a timeout and a refusal
#: are different claims about a method, even though both leave a gap.
MISSING_KINDS = (
    "missing_timed_out",
    "missing_failed",
    "adaptation_failed",
    "evaluator_unsupported",
)


def roster_median(scored: list[float], planned: int) -> float | None:
    """Median over every planned identity, ranking unscored ones worst."""
    if planned <= 0 or not scored:
        return None
    ranked = sorted(scored)
    index = (planned - 1) // 2
    return ranked[index] if index < len(ranked) else None


def score_of(row: dict) -> float | None:
    """The target-prediction score, under whichever name this row uses."""
    for field in (
        "target_nmse",
        "normalized_mse",
        "target_normalized_mse",
        "nmse",
    ):
        value = row.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    endpoint = row.get("target_prediction")
    if isinstance(endpoint, dict):
        value = endpoint.get("normalized_mse")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def method_of(row: dict) -> str:
    for field in ("method_label", "method", "source_kind"):
        value = row.get(field)
        if value:
            return str(value)
    return "unknown"


def collect(rows: list[dict], planned_per_method: dict[str, int]) -> dict:
    """Per method: scored median, roster median, and the counts behind them."""
    scores: dict[str, list[float]] = defaultdict(list)
    statuses: dict[str, dict[str, int]] = defaultdict(dict)
    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        method = method_of(row)
        seen[method] += 1
        status = str(row.get("status", "unknown"))
        statuses[method][status] = statuses[method].get(status, 0) + 1
        value = score_of(row)
        if value is not None:
            scores[method].append(value)

    summary = {}
    for method in sorted(set(seen) | set(planned_per_method)):
        planned = planned_per_method.get(method, seen.get(method, 0))
        values = scores.get(method, [])
        summary[method] = {
            "planned": planned,
            "rows_seen": seen.get(method, 0),
            "scored": len(values),
            "unscored": planned - len(values),
            "median_conditional_on_success": (
                statistics.median(values) if values else None
            ),
            "median_over_roster": roster_median(values, planned),
            "status_counts": statuses.get(method, {}),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, help="external_baseline_report.json, for its rows"
    )
    parser.add_argument(
        "--roster", type=Path, help="external_baseline_roster.jsonl, one row per line"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.report and not args.roster:
        parser.error("supply --report or --roster")

    rows: list[dict] = []
    planned: dict[str, int] = {}
    if args.report:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows") or payload.get("subjects") or [])
        for method, counts in (payload.get("by_method") or {}).items():
            if isinstance(counts, dict) and "planned" in counts:
                planned[method] = int(counts["planned"])
    if args.roster:
        for line in args.roster.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    summary = collect(rows, planned)
    value = {
        "schema_version": "phase-b-roster-median-report-1",
        "median_is_conditional_on_success": False,
        "test_data_opened": False,
        "summary": summary,
    }
    if args.out:
        args.out.write_text(json.dumps(value, indent=2) + "\n")

    def fmt(item: object) -> str:
        return f"{item:.4g}" if isinstance(item, (int, float)) else "n/a"

    header = (
        f"{'method':<28}{'planned':>8}{'scored':>8}"
        f"{'conditional':>14}{'roster':>14}"
    )
    print(header)
    print("-" * len(header))
    for method, stats in summary.items():
        print(
            f"{method:<28}{stats['planned']:>8}{stats['scored']:>8}"
            f"{fmt(stats['median_conditional_on_success']):>14}"
            f"{fmt(stats['median_over_roster']):>14}"
        )
    if not any(stats["scored"] for stats in summary.values()):
        print(
            "\nNo per-row scores were found. The sealed report may aggregate "
            "only; pass --roster with the per-identity rows."
        )


if __name__ == "__main__":
    main()
