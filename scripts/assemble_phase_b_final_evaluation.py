#!/usr/bin/env python3
"""Assemble separate final endpoints for frozen Phase-B model subjects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.final_evaluation import (
    FinalEvaluationRecord,
    FrozenEvaluationSubject,
    evaluate_frozen_subject,
    evaluation_summary,
)
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec


def _read_subjects(path: Path) -> tuple[FrozenEvaluationSubject, ...]:
    subjects = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    identifiers = [item.subject_id for item in subjects]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("frozen evaluation subject identifiers must be unique")
    return subjects


def _read_specs(root: Path) -> dict[tuple[str, str], MechanismEvaluationSpec]:
    result: dict[tuple[str, str], MechanismEvaluationSpec] = {}
    for path in sorted(root.glob("*.json")):
        spec = MechanismEvaluationSpec.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        key = (spec.benchmark_id, spec.tier)
        if key in result:
            raise ValueError(f"duplicate mechanism specification: {key}")
        result[key] = spec
    return result


def _read_outcomes(path: Path) -> tuple[SourceAdapterOutcome, ...]:
    outcomes = tuple(
        SourceAdapterOutcome.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    request_ids = [item.request_id for item in outcomes]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("source adapter outcome request identifiers must be unique")
    return outcomes


def _validate_outcomes(
    subjects: tuple[FrozenEvaluationSubject, ...],
    outcomes: tuple[SourceAdapterOutcome, ...],
) -> None:
    expected = {item.subject_id for item in subjects}
    adapted = {item.subject_id for item in outcomes if item.status == "adapted"}
    if adapted != expected:
        raise ValueError(
            "adapted source outcomes differ from frozen subjects; "
            f"missing={sorted(expected - adapted)}, extra={sorted(adapted - expected)}"
        )


def assemble(
    subjects: tuple[FrozenEvaluationSubject, ...],
    specs: dict[tuple[str, str], MechanismEvaluationSpec],
    *,
    allow_missing_specs: bool,
) -> tuple[FinalEvaluationRecord, ...]:
    """Evaluate public endpoints and retain post-freeze private endpoints."""
    records = []
    for subject in subjects:
        key = (subject.benchmark_id, subject.tier)
        spec = specs.get(key)
        if (
            subject.public_mechanism_applicable
            and spec is None
            and not allow_missing_specs
        ):
            raise ValueError(f"missing public mechanism specification: {key}")
        records.append(
            evaluate_frozen_subject(
                subject,
                spec,
                mechanism_spec_not_applicable=(not subject.public_mechanism_applicable),
            )
        )
    return tuple(records)


def _flat_row(record: FinalEvaluationRecord) -> dict[str, Any]:
    mechanism = record.public_mechanism.evaluation
    hidden = [item for item in record.hidden_mechanisms if item.status == "available"]
    intervention = [item for item in record.interventions if item.status == "available"]
    llm = record.qualitative_llm
    return {
        "subject_id": record.subject_id,
        "method": record.method,
        "benchmark_id": record.benchmark_id,
        "tier": record.tier,
        "repetition": record.repetition,
        "source_adapter": record.source_provenance.adapter,
        "source_sha256": record.source_provenance.source_sha256,
        "candidate_sha256": record.source_provenance.candidate_sha256,
        "parameterization_status": record.parameterization.status,
        "runtime_valid": record.runtime.valid,
        "runtime_failure_count": len(record.runtime.failures),
        "runtime_warning_count": len(record.runtime.warnings),
        "public_mechanism_status": record.public_mechanism.status,
        "mechanism_specification_source": (
            record.public_mechanism.specification_source or ""
        ),
        "public_prompt_sha256": record.public_mechanism.public_prompt_sha256 or "",
        "mechanism_coverage": (
            "" if mechanism is None else mechanism.mechanism_coverage
        ),
        "graph_mechanism_compliance": (
            "" if mechanism is None else mechanism.graph_mechanism_compliance
        ),
        "graph_mechanism_compliance_complete": (
            ""
            if mechanism is None
            else mechanism.graph_mechanism_compliance_complete
        ),
        "mechanism_annotation_compliance": (
            ""
            if mechanism is None
            else mechanism.mechanism_annotation_compliance
        ),
        "mechanism_annotation_compliance_complete": (
            ""
            if mechanism is None
            else mechanism.mechanism_annotation_compliance_complete
        ),
        # Retained as graph-compliance aliases for existing consumers.
        "mechanism_compliance": (
            "" if mechanism is None else mechanism.mechanism_compliance
        ),
        "mechanism_compliance_complete": (
            "" if mechanism is None else mechanism.mechanism_compliance_complete
        ),
        "target_prediction_status": record.target_prediction.status,
        "target_evaluation_protocol": record.target_prediction.evaluation_protocol,
        "target_trajectory_count": record.target_prediction.trajectory_count,
        "target_successful_trajectory_count": (
            record.target_prediction.successful_trajectory_count
        ),
        "target_test_nmse": (
            ""
            if record.target_prediction.normalized_mse is None
            else record.target_prediction.normalized_mse
        ),
        "hidden_mechanism_required_count": len(record.hidden_mechanisms),
        "hidden_mechanism_recovered_count": sum(
            item.recovered for item in record.hidden_mechanisms
        ),
        "hidden_nmse_conditional_mean": (
            ""
            if not hidden
            else sum(item.aligned_test_nmse or 0.0 for item in hidden) / len(hidden)
        ),
        "intervention_required_count": len(record.interventions),
        "intervention_available_count": len(intervention),
        "intervention_target_nmse_mean": (
            ""
            if not intervention
            else sum(item.target_nmse or 0.0 for item in intervention)
            / len(intervention)
        ),
        "state_count": record.complexity.state_count,
        "latent_state_count": record.complexity.latent_state_count,
        "process_count": record.complexity.process_count,
        "parameter_count": record.complexity.parameter_count,
        "additive_term_count": record.complexity.additive_term_count,
        "qualitative_llm_status": "not_requested" if llm is None else llm.status,
        "qualitative_llm_requested_calls": 0 if llm is None else llm.requested_calls,
        "qualitative_llm_successful_calls": (
            0 if llm is None else llm.successful_calls
        ),
    }


def _write_outputs(
    output_root: Path,
    records: tuple[FinalEvaluationRecord, ...],
    *,
    subjects_sha256: str,
    outcomes: tuple[SourceAdapterOutcome, ...] | None = None,
    outcomes_sha256: str | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "final_evaluation_records.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in records),
        encoding="utf-8",
    )
    rows = [_flat_row(item) for item in records]
    with (output_root / "final_evaluation_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = tuple(rows[0]) if rows else ("subject_id",)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = evaluation_summary(records)
    if outcomes is not None:
        adapted_count = sum(item.status == "adapted" for item in outcomes)
        summary.update(
            {
                "source_request_count": len(outcomes),
                "source_adapted_count": adapted_count,
                "source_failed_count": len(outcomes) - adapted_count,
                "source_completion_rate": (
                    adapted_count / len(outcomes) if outcomes else None
                ),
            }
        )
    (output_root / "final_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "phase-b-final-evaluation-run-1",
        "subjects_sha256": subjects_sha256,
        "source_outcomes_sha256": outcomes_sha256,
        "record_count": len(records),
        "selection_frozen_required": True,
        "private_metrics_post_freeze_required": True,
        "weighted_overall_score_defined": False,
    }
    (output_root / "final_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "final_evaluation_summary.md", summary)


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    labels = (
        ("Frozen records", "record_count"),
        ("Runtime-valid rate", "runtime_valid_rate"),
        ("Replay-complete rate", "replay_complete_rate"),
        ("Source completion rate", "source_completion_rate"),
        ("Target-test coverage", "target_prediction_coverage"),
        ("Target-trajectory success rate", "target_trajectory_success_rate"),
        ("Mean target test NMSE", "mean_target_test_nmse"),
        ("Public-mechanism coverage", "public_mechanism_coverage"),
        (
            "Mean public graph-mechanism compliance",
            "mean_public_graph_mechanism_compliance",
        ),
        (
            "Graph-mechanism complete-assessment rate",
            "public_graph_mechanism_complete_assessment_rate",
        ),
        (
            "Mean mechanism-annotation compliance",
            "mean_public_mechanism_annotation_compliance",
        ),
        (
            "Mechanism-annotation complete-assessment rate",
            "public_mechanism_annotation_complete_assessment_rate",
        ),
        ("Hidden mechanism recovery rate", "hidden_mechanism_recovery_rate"),
        (
            "Mean hidden NMSE conditional on recovery",
            "mean_hidden_nmse_conditional_on_recovery",
        ),
        ("Mean intervention target NMSE", "mean_intervention_target_nmse"),
        ("Qualitative LLM requested calls", "qualitative_llm_requested_calls"),
        ("Qualitative LLM response rate", "qualitative_llm_response_rate"),
    )
    lines = [
        "# Phase-B final evaluation",
        "",
        "Endpoints are reported separately; no weighted overall score is defined.",
        "",
        "| Endpoint | Value |",
        "|---|---:|",
    ]
    for label, key in labels:
        value = summary.get(key)
        rendered = "N/A" if value is None else str(value)
        lines.append(f"| {label} | {rendered} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--mechanism-config-root", type=Path, required=True)
    parser.add_argument("--source-outcomes", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-missing-mechanism-specs", action="store_true")
    args = parser.parse_args()
    raw = args.subjects.read_bytes()
    subjects = _read_subjects(args.subjects)
    outcomes = (
        None if args.source_outcomes is None else _read_outcomes(args.source_outcomes)
    )
    if outcomes is not None:
        _validate_outcomes(subjects, outcomes)
    records = assemble(
        subjects,
        _read_specs(args.mechanism_config_root),
        allow_missing_specs=args.allow_missing_mechanism_specs,
    )
    _write_outputs(
        args.output_root,
        records,
        subjects_sha256=hashlib.sha256(raw).hexdigest(),
        outcomes=outcomes,
        outcomes_sha256=(
            None
            if args.source_outcomes is None
            else hashlib.sha256(args.source_outcomes.read_bytes()).hexdigest()
        ),
    )
    print(f"wrote {len(records)} frozen evaluation records to {args.output_root}")


if __name__ == "__main__":
    main()
