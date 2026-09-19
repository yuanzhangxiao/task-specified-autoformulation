#!/usr/bin/env python3
"""Report external-baseline endpoints against the full frozen roster.

The shared assembler counts every non-adapted outcome as a source failure and
its metrics contain only adapted subjects. This report joins the master roster
by explicit identity and keeps four outcome states distinct, so a missing
artifact and an unsupported evaluator are never read as a method's failure.
No mean is pooled across methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from autoformalism.rebuttal.external_baseline_freeze import ExternalBaselineSource
from autoformalism.rebuttal.final_evaluation import FinalEvaluationRecord
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome

STATES = ("evaluated", "adaptation_failed", "missing", "evaluator_unsupported")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    roster = _read(args.roster, ExternalBaselineSource)
    outcomes = _read(args.outcomes, SourceAdapterOutcome)
    records = _read(args.records, FinalEvaluationRecord)
    rows = join(roster, outcomes, records)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0]) if rows else ("request_id",)
    with (output_root / "external_baseline_roster_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows)
    (output_root / "external_baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["by_method"], indent=2, sort_keys=True))


def join(
    roster: tuple[ExternalBaselineSource, ...],
    outcomes: tuple[SourceAdapterOutcome, ...],
    records: tuple[FinalEvaluationRecord, ...],
) -> list[dict[str, object]]:
    """Attach each planned identity to its outcome and evaluated endpoints."""
    by_outcome = {item.request_id: item for item in outcomes}
    if len(by_outcome) != len(outcomes):
        raise ValueError("source outcomes repeat a request identifier")
    expected = {item.request_id for item in roster}
    if set(by_outcome) != expected:
        raise ValueError(
            "outcome identifiers differ from the frozen roster; "
            f"unexpected={sorted(set(by_outcome) - expected)[:5]}, "
            f"absent={sorted(expected - set(by_outcome))[:5]}"
        )
    by_subject = {item.subject_id: item for item in records}
    rows: list[dict[str, object]] = []
    for source in roster:
        outcome = by_outcome[source.request_id]
        record = (
            by_subject.get(outcome.subject_id)
            if outcome.status == "adapted"
            else None
        )
        if outcome.status == "adapted" and record is None:
            raise ValueError(f"adapted outcome has no record: {source.request_id}")
        state = {
            "adapted": "evaluated",
            "failed": "adaptation_failed",
            "missing": "missing",
            "evaluator_unsupported": "evaluator_unsupported",
        }[outcome.status]
        target = record.target_prediction if record is not None else None
        rows.append(
            {
                "request_id": source.request_id,
                "method_id": source.method_id,
                "benchmark_id": source.benchmark_id,
                "tier": source.tier,
                "repetition": source.repetition,
                "state": state,
                "reason": source.reason or outcome.error or "",
                "runtime_valid": (
                    record.runtime.valid if record is not None else None
                ),
                "target_nmse": (
                    target.normalized_mse
                    if target is not None and target.status == "available"
                    else None
                ),
                "target_status": target.status if target is not None else None,
                "evaluation_protocol": (
                    target.evaluation_protocol if target is not None else None
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Report per-method and per-cell states with full expected counts."""
    methods = sorted({str(row["method_id"]) for row in rows})
    by_method = {}
    for method in methods:
        subset = [row for row in rows if row["method_id"] == method]
        scored = [
            float(row["target_nmse"])
            for row in subset
            if row["target_nmse"] is not None
        ]
        by_method[method] = {
            "planned": len(subset),
            **{
                state: sum(row["state"] == state for row in subset)
                for state in STATES
            },
            "scored_count": len(scored),
            "target_nmse_median_conditional_on_success": (
                median(scored) if scored else None
            ),
        }
    by_cell = {}
    for method in methods:
        for cell in sorted(
            {
                (str(row["benchmark_id"]), str(row["tier"]))
                for row in rows
                if row["method_id"] == method
            }
        ):
            subset = [
                row
                for row in rows
                if row["method_id"] == method
                and (row["benchmark_id"], row["tier"]) == cell
            ]
            scored = [
                float(row["target_nmse"])
                for row in subset
                if row["target_nmse"] is not None
            ]
            by_cell[f"{method}|{cell[0]}|{cell[1]}"] = {
                "planned": len(subset),
                "scored_count": len(scored),
                "target_nmse_median_conditional_on_success": (
                    median(scored) if scored else None
                ),
                **{
                    state: sum(row["state"] == state for row in subset)
                    for state in STATES
                },
            }
    return {
        "schema_version": "phase-b-external-baseline-report-1",
        "planned_identity_count": len(rows),
        "median_is_conditional_on_success": True,
        "pooled_cross_method_mean_reported": False,
        "weighted_overall_score_defined": False,
        "by_method": by_method,
        "by_method_and_cell": by_cell,
    }


def _read(path: Path, model):
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    main()
