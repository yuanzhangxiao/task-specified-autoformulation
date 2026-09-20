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


def full_roster_median(scored: list[float], planned: int) -> float | None:
    """Median over every planned identity, ranking unscored rows worst.

    A median needs only an ordering, so an unscored row can be placed beyond
    every observed value without inventing a magnitude for it. Conditioning on
    success instead flatters whichever method fails most often.

    Returned as the lower median, so the statistic is always a value the method
    actually produced. Undefined when at least half the roster is unscored,
    because the middle of the ranking then falls among the failures.
    """
    if planned <= 0:
        return None
    ranked = sorted(scored)
    index = (planned - 1) // 2
    return ranked[index] if index < len(ranked) else None


def _unscored_reason(row: dict[str, object]) -> str | None:
    """Separate a compute limit from a model that could not be integrated."""
    if row["state"] != "evaluated" or row["target_nmse"] is not None:
        return None
    message = str(row.get("target_message") or "")
    return "timeout" if "Timeout" in message else "diverged"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument(
        "--evaluations",
        type=Path,
        help="JSON list of {label, roster, outcomes, records} to combine",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.evaluations is not None:
        rows, sources = load_evaluations(args.evaluations)
    else:
        for name in ("roster", "outcomes", "records"):
            if getattr(args, name) is None:
                parser.error(f"--{name} is required without --evaluations")
        rows = join(
            _read(args.roster, ExternalBaselineSource),
            _read(args.outcomes, SourceAdapterOutcome),
            _read(args.records, FinalEvaluationRecord),
        )
        sources = [{"label": "single", "roster": str(args.roster)}]

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
    report["evaluations"] = sources
    (output_root / "external_baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["by_method"], indent=2, sort_keys=True))


def load_evaluations(manifest: Path) -> tuple[list[dict[str, object]], list[dict]]:
    """Combine separately executed evaluations without merging their identity.

    Methods are evaluated under their own receipts and on different dates. The
    rows are joined for one comparison table, but each keeps the evaluation it
    came from, and a method may not appear in two: that would mean the same
    model was scored twice under different conditions.
    """
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"expected a nonempty JSON list: {manifest}")
    rows: list[dict[str, object]] = []
    sources: list[dict] = []
    seen: dict[str, str] = {}
    for entry in entries:
        label = str(entry["label"])
        joined = join(
            _read(Path(entry["roster"]), ExternalBaselineSource),
            _read(Path(entry["outcomes"]), SourceAdapterOutcome),
            _read(Path(entry["records"]), FinalEvaluationRecord),
            label=label,
        )
        for row in joined:
            method = str(row["method_id"])
            if seen.setdefault(method, label) != label:
                raise ValueError(
                    f"method {method!r} appears in both {seen[method]!r} "
                    f"and {label!r}; each method is evaluated once"
                )
        rows.extend(joined)
        sources.append(
            {
                "label": label,
                "roster": str(entry["roster"]),
                "records": str(entry["records"]),
                "receipt": entry.get("receipt"),
                "evaluated_on": entry.get("evaluated_on"),
                "row_count": len(joined),
            }
        )
    return rows, sources


def join(
    roster: tuple[ExternalBaselineSource, ...],
    outcomes: tuple[SourceAdapterOutcome, ...],
    records: tuple[FinalEvaluationRecord, ...],
    *,
    label: str = "single",
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
                "evaluation": label,
                "request_id": source.request_id,
                "method_id": source.method_id,
                "benchmark_id": source.benchmark_id,
                "tier": source.tier,
                "repetition": source.repetition,
                "state": state,
                "terminal_status": source.terminal_status or "",
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
                "target_message": (
                    (target.message or "") if target is not None else ""
                ),
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
        unscored = [_unscored_reason(row) for row in subset]
        by_method[method] = {
            # which separately executed evaluation produced these rows
            "evaluation": sorted({str(row["evaluation"]) for row in subset}),
            "planned": len(subset),
            **{
                state: sum(row["state"] == state for row in subset)
                for state in STATES
            },
            # A wall-clock timeout is a budget limit, not a method failure.
            "missing_timed_out": sum(
                row["terminal_status"] == "timed_out" for row in subset
            ),
            "missing_failed": sum(
                row["terminal_status"] == "failed" for row in subset
            ),
            "scored_count": len(scored),
            # Headline: every planned identity counted, failures ranked worst.
            "target_nmse_median_full_roster": full_roster_median(
                scored, len(subset)
            ),
            "full_roster_median_defined": (
                full_roster_median(scored, len(subset)) is not None
            ),
            # Diagnostic only: reads better the more a method fails.
            "target_nmse_median_conditional_on_success": (
                median(scored) if scored else None
            ),
            "evaluated_but_unscored": sum(item is not None for item in unscored),
            "unscored_timeout": sum(item == "timeout" for item in unscored),
            "unscored_diverged": sum(item == "diverged" for item in unscored),
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
                "target_nmse_median_full_roster": full_roster_median(
                    scored, len(subset)
                ),
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
        "headline_median": "target_nmse_median_full_roster",
        "full_roster_median_ranks_unscored_worst": True,
        "full_roster_median_is_the_lower_median": True,
        "conditional_median_is_diagnostic_only": True,
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
