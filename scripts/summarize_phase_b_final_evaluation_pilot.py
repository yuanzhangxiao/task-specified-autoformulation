#!/usr/bin/env python3
"""Summarize separate endpoints for a frozen Phase-B final-evaluation pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.final_evaluation import FinalEvaluationRecord
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-outcomes", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        json.loads(args.freeze_manifest.read_text(encoding="utf-8")),
        _read_outcomes(args.source_outcomes),
        _read_records(args.records),
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pilot_endpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "pilot_endpoint_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        f"wrote separate endpoint report for {report['requested_source_count']} "
        f"planned subjects to {output_root}"
    )


def build_report(
    manifest: dict[str, Any],
    outcomes: tuple[SourceAdapterOutcome, ...],
    records: tuple[FinalEvaluationRecord, ...],
) -> dict[str, Any]:
    """Join the frozen source ledger and endpoint records without scalarization."""
    if manifest.get("status") != "frozen_before_test_or_private_evaluation":
        raise ValueError("pilot source manifest is not frozen")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pilot source manifest has no sources")
    expected_count = int(manifest.get("expected_source_count", -1))
    if len(sources) != expected_count:
        raise ValueError("pilot source manifest is incomplete")
    outcome_by_request = {item.request_id: item for item in outcomes}
    if len(outcome_by_request) != len(outcomes):
        raise ValueError("source outcomes contain duplicate request identifiers")
    expected_requests = {str(item["request_id"]) for item in sources}
    if set(outcome_by_request) != expected_requests:
        raise ValueError("source outcomes differ from the frozen pilot requests")
    record_by_subject = {item.subject_id: item for item in records}
    if len(record_by_subject) != len(records):
        raise ValueError("final records contain duplicate subject identifiers")

    rows: list[dict[str, Any]] = []
    for source in sources:
        outcome = outcome_by_request[str(source["request_id"])]
        record = (
            None
            if outcome.subject_id is None
            else record_by_subject.get(outcome.subject_id)
        )
        if outcome.status == "adapted" and record is None:
            raise ValueError(
                f"adapted source has no final record: {outcome.request_id}"
            )
        if record is not None:
            expected_method = (
                "autoformalism"
                if source["method_id"] == "autoformalism"
                else "raw_data_agent:openai:gpt-5.6-sol"
            )
            expected_identity = (
                expected_method,
                source["benchmark_id"],
                source["tier"],
                int(source["repetition"]),
            )
            actual_identity = (
                record.method,
                record.benchmark_id,
                record.tier,
                record.repetition,
            )
            if actual_identity != expected_identity:
                raise ValueError(
                    f"final record identity differs for {outcome.request_id}: "
                    f"expected={expected_identity}, actual={actual_identity}"
                )
        rows.append(_source_row(source, outcome, record))
    used_subjects = {
        str(row["subject_id"]) for row in rows if row["subject_id"] is not None
    }
    if used_subjects != set(record_by_subject):
        raise ValueError("final records contain subjects outside the frozen pilot")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_id"], row["benchmark_id"], row["tier"])].append(row)
    groups = [
        _group_row(key, grouped[key])
        for key in sorted(grouped, key=lambda item: (item[1], item[2], item[0]))
    ]
    return {
        "schema_version": "phase-b-final-evaluation-pilot-report-1",
        "status": "complete",
        "requested_source_count": len(rows),
        "adapted_source_count": sum(row["source_status"] == "adapted" for row in rows),
        "final_record_count": len(records),
        "weighted_overall_score_defined": False,
        "qualitative_llm_requested": False,
        "endpoint_interpretation": (
            "All endpoints are reported separately; conditional means exclude "
            "unavailable endpoints and retain their coverage denominators."
        ),
        "groups": groups,
        "subjects": rows,
    }


def _source_row(
    source: dict[str, Any],
    outcome: SourceAdapterOutcome,
    record: FinalEvaluationRecord | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "request_id": source["request_id"],
        "method_id": source["method_id"],
        "benchmark_id": source["benchmark_id"],
        "tier": source["tier"],
        "repetition": source["repetition"],
        "source_status": outcome.status,
        "source_error_type": outcome.error_type,
        "source_error": outcome.error,
        "subject_id": outcome.subject_id,
    }
    if record is None:
        return base
    mechanism = record.public_mechanism.evaluation
    hidden_available = [
        item for item in record.hidden_mechanisms if item.status == "available"
    ]
    interventions = [
        item for item in record.interventions if item.status == "available"
    ]
    base.update(
        {
            "parameterization_status": record.parameterization.status,
            "runtime_valid": record.runtime.valid,
            "public_mechanism_status": record.public_mechanism.status,
            "mechanism_compliance": (
                None if mechanism is None else mechanism.mechanism_compliance
            ),
            "target_status": record.target_prediction.status,
            "target_test_nmse": record.target_prediction.normalized_mse,
            "hidden_required_count": len(record.hidden_mechanisms),
            "hidden_recovered_count": sum(
                item.recovered for item in record.hidden_mechanisms
            ),
            "hidden_nmse_conditional_mean": _mean(
                [item.aligned_test_nmse for item in hidden_available]
            ),
            "intervention_required_count": len(record.interventions),
            "intervention_available_count": len(interventions),
            "intervention_target_nmse_mean": _mean(
                [item.target_nmse for item in interventions]
            ),
            "intervention_direction_accuracy": _mean(
                [
                    float(item.response_direction_correct)
                    for item in interventions
                    if item.response_direction_correct is not None
                ]
            ),
            "intervention_shape_correlation_mean": _mean(
                [item.response_shape_correlation for item in interventions]
            ),
            "intervention_peak_timing_error_mean": _mean(
                [item.peak_timing_error_fraction for item in interventions]
            ),
            "state_count": record.complexity.state_count,
            "latent_state_count": record.complexity.latent_state_count,
            "process_count": record.complexity.process_count,
            "parameter_count": record.complexity.parameter_count,
            "additive_term_count": record.complexity.additive_term_count,
        }
    )
    return base


def _group_row(
    key: tuple[str, str, str], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    adapted = [row for row in rows if row["source_status"] == "adapted"]
    hidden_required = sum(int(row.get("hidden_required_count", 0)) for row in adapted)
    hidden_recovered = sum(
        int(row.get("hidden_recovered_count", 0)) for row in adapted
    )
    intervention_required = sum(
        int(row.get("intervention_required_count", 0)) for row in adapted
    )
    intervention_available = sum(
        int(row.get("intervention_available_count", 0)) for row in adapted
    )
    return {
        "method_id": key[0],
        "benchmark_id": key[1],
        "tier": key[2],
        "requested_repetitions": len(rows),
        "source_completion_rate": len(adapted) / len(rows),
        "runtime_valid_rate": _rate(adapted, "runtime_valid"),
        "public_mechanism_coverage": _coverage(adapted, "mechanism_compliance"),
        "mechanism_compliance_mean": _mean(
            [row.get("mechanism_compliance") for row in adapted]
        ),
        "target_replay_coverage": _rate_value(adapted, "target_status", "available"),
        "target_test_nmse_mean": _mean(
            [row.get("target_test_nmse") for row in adapted]
        ),
        "target_test_nmse_median": _median(
            [row.get("target_test_nmse") for row in adapted]
        ),
        "hidden_mechanism_recovery_rate": (
            None if not hidden_required else hidden_recovered / hidden_required
        ),
        "hidden_nmse_conditional_mean": _mean(
            [row.get("hidden_nmse_conditional_mean") for row in adapted]
        ),
        "intervention_coverage": (
            None
            if not intervention_required
            else intervention_available / intervention_required
        ),
        "intervention_target_nmse_mean": _mean(
            [row.get("intervention_target_nmse_mean") for row in adapted]
        ),
        "intervention_direction_accuracy": _mean(
            [row.get("intervention_direction_accuracy") for row in adapted]
        ),
        "intervention_shape_correlation_mean": _mean(
            [row.get("intervention_shape_correlation_mean") for row in adapted]
        ),
        "intervention_peak_timing_error_mean": _mean(
            [row.get("intervention_peak_timing_error_mean") for row in adapted]
        ),
        "state_count_mean": _mean([row.get("state_count") for row in adapted]),
        "latent_state_count_mean": _mean(
            [row.get("latent_state_count") for row in adapted]
        ),
        "process_count_mean": _mean([row.get("process_count") for row in adapted]),
        "parameter_count_mean": _mean(
            [row.get("parameter_count") for row in adapted]
        ),
        "additive_term_count_mean": _mean(
            [row.get("additive_term_count") for row in adapted]
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the non-scalar group endpoint table."""
    lines = [
        "# Phase-B two-cell final-evaluation pilot",
        "",
        "All endpoints are reported separately; no weighted overall score is defined.",
        "Conditional means exclude unavailable endpoints, whose coverage is shown.",
        "",
        "| Benchmark | Method | Source | Runtime | Mechanism | Target replay | "
        "Target NMSE | Hidden recovery | Hidden NMSE | Intervention | "
        "Intervention NMSE | States | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        lines.append(
            "| {benchmark_id} ({tier}) | {method_id} | {source_completion_rate} | "
            "{runtime_valid_rate} | {mechanism_compliance_mean} | "
            "{target_replay_coverage} | {target_test_nmse_mean} | "
            "{hidden_mechanism_recovery_rate} | {hidden_nmse_conditional_mean} | "
            "{intervention_coverage} | {intervention_target_nmse_mean} | "
            "{state_count_mean} | {parameter_count_mean} |".format(
                **{key: _format(value) for key, value in group.items()}
            )
        )
    lines.extend(
        [
            "",
            "The JSON companion retains every repetition plus direction, shape, "
            "timing, and complexity endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_outcomes(path: Path) -> tuple[SourceAdapterOutcome, ...]:
    return tuple(
        SourceAdapterOutcome.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _read_records(path: Path) -> tuple[FinalEvaluationRecord, ...]:
    return tuple(
        FinalEvaluationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _numbers(values: list[object]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _mean(values: list[object]) -> float | None:
    numeric = _numbers(values)
    return None if not numeric else statistics.fmean(numeric)


def _median(values: list[object]) -> float | None:
    numeric = _numbers(values)
    return None if not numeric else statistics.median(numeric)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return None if not rows else sum(bool(row.get(key)) for row in rows) / len(rows)


def _rate_value(rows: list[dict[str, Any]], key: str, value: object) -> float | None:
    return None if not rows else sum(row.get(key) == value for row in rows) / len(rows)


def _coverage(rows: list[dict[str, Any]], key: str) -> float | None:
    return (
        None
        if not rows
        else sum(row.get(key) is not None for row in rows) / len(rows)
    )


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
