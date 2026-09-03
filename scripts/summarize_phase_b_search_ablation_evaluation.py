#!/usr/bin/env python3
"""Report sealed Phase-B endpoints separately for each frozen search arm."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.final_evaluation import FinalEvaluationRecord
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from autoformalism.rebuttal.search_integration_ablation import (
    FrozenSearchAblationSource,
)
from scripts.summarize_phase_b_final_evaluation_pilot import (
    _format,
    _group_row,
    _source_row,
)


def build_report(
    sources: tuple[FrozenSearchAblationSource, ...],
    outcomes: tuple[SourceAdapterOutcome, ...],
    records: tuple[FinalEvaluationRecord, ...],
) -> dict[str, Any]:
    """Join frozen arm sources to final records without scalarizing endpoints."""
    if not sources:
        raise ValueError("frozen search source ledger is empty")
    source_by_request = {item.request_id: item for item in sources}
    if len(source_by_request) != len(sources):
        raise ValueError("frozen search source ledger has duplicate request IDs")
    outcome_by_request = {item.request_id: item for item in outcomes}
    if len(outcome_by_request) != len(outcomes):
        raise ValueError("source outcomes contain duplicate request IDs")
    if set(outcome_by_request) != set(source_by_request):
        raise ValueError("source outcomes differ from the frozen search ledger")
    record_by_subject = {item.subject_id: item for item in records}
    if len(record_by_subject) != len(records):
        raise ValueError("final records contain duplicate subject IDs")

    rows: list[dict[str, Any]] = []
    for request_id, source in sorted(source_by_request.items()):
        outcome = outcome_by_request[request_id]
        record = (
            None
            if outcome.subject_id is None
            else record_by_subject.get(outcome.subject_id)
        )
        if outcome.status == "adapted" and record is None:
            raise ValueError(f"adapted source has no final record: {request_id}")
        if record is not None:
            expected = (
                "autoformalism",
                source.benchmark_id,
                source.tier,
                source.repetition,
            )
            actual = (
                record.method,
                record.benchmark_id,
                record.tier,
                record.repetition,
            )
            if actual != expected:
                raise ValueError(
                    f"final record identity differs for {request_id}: "
                    f"expected={expected}, actual={actual}"
                )
        row = _source_row(
            {
                "request_id": source.request_id,
                "method_id": source.arm_id,
                "benchmark_id": source.benchmark_id,
                "tier": source.tier,
                "repetition": source.repetition,
            },
            outcome,
            record,
        )
        row["arm_id"] = source.arm_id
        row["frozen_artifact_status"] = source.artifact_status
        rows.append(row)

    used_subjects = {
        str(row["subject_id"]) for row in rows if row.get("subject_id") is not None
    }
    if used_subjects != set(record_by_subject):
        raise ValueError("final records contain subjects outside the frozen ledger")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm_id"], row["benchmark_id"], row["tier"])].append(row)
        paired[(row["benchmark_id"], row["tier"], row["repetition"])].append(row)
    groups = [
        _group_row(key, grouped[key])
        for key in sorted(grouped, key=lambda item: (item[1], item[2], item[0]))
    ]
    paired_trials = [_paired_trial_row(key, paired[key]) for key in sorted(paired)]
    return {
        "schema_version": "phase-b-search-ablation-endpoint-report-1",
        "status": "complete",
        "test_data_opened_after_selection_freeze": True,
        "private_reference_opened_after_selection_freeze": True,
        "weighted_overall_score_defined": False,
        "endpoint_interpretation": (
            "Endpoints are reported separately by arm, benchmark, tier, and "
            "repetition. Conditional endpoint means retain their coverage "
            "denominators; no scalar winner is inferred."
        ),
        "requested_source_count": len(rows),
        "adapted_source_count": sum(row["source_status"] == "adapted" for row in rows),
        "final_record_count": len(records),
        "groups": groups,
        "paired_trials": paired_trials,
        "subjects": rows,
    }


def _paired_trial_row(
    key: tuple[str, str, int], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Retain matched-arm availability without turning endpoints into a score."""
    by_arm = {str(item["arm_id"]): item for item in rows}
    expected = {"paired_question_consensus", "no_judge"}
    if set(by_arm) != expected:
        raise ValueError(f"paired trial has missing or duplicate arms: {key}")
    judge = by_arm["paired_question_consensus"]
    no_judge = by_arm["no_judge"]
    return {
        "benchmark_id": key[0],
        "tier": key[1],
        "repetition": key[2],
        "both_sources_adapted": (
            judge["source_status"] == "adapted"
            and no_judge["source_status"] == "adapted"
        ),
        "judge_source_status": judge["source_status"],
        "no_judge_source_status": no_judge["source_status"],
        "judge_runtime_valid": judge.get("runtime_valid"),
        "no_judge_runtime_valid": no_judge.get("runtime_valid"),
        "judge_target_test_nmse": judge.get("target_test_nmse"),
        "no_judge_target_test_nmse": no_judge.get("target_test_nmse"),
        "judge_mechanism_compliance": judge.get("mechanism_compliance"),
        "no_judge_mechanism_compliance": no_judge.get("mechanism_compliance"),
        "judge_graph_mechanism_compliance": judge.get(
            "graph_mechanism_compliance"
        ),
        "no_judge_graph_mechanism_compliance": no_judge.get(
            "graph_mechanism_compliance"
        ),
        "judge_mechanism_annotation_compliance": judge.get(
            "mechanism_annotation_compliance"
        ),
        "no_judge_mechanism_annotation_compliance": no_judge.get(
            "mechanism_annotation_compliance"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render arm-level endpoint groups and paired-source coverage."""
    lines = [
        "# Search-ablation sealed endpoint report",
        "",
        "All endpoints remain separate; no weighted overall score or inferred "
        "arm winner is defined.",
        "",
        (
            "| Benchmark | Arm | Source | Runtime | Graph mechanism | "
            "Annotation metadata | Target replay | Target NMSE | "
            "Hidden recovery | Hidden NMSE | Intervention | States | Parameters |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        formatted = {key: _format(value) for key, value in group.items()}
        lines.append(
            "| {benchmark_id} ({tier}) | {method_id} | {source_completion_rate} | "
            "{runtime_valid_rate} | {graph_mechanism_compliance_mean} | "
            "{mechanism_annotation_compliance_mean} | {target_replay_coverage} | "
            "{target_test_nmse_mean} | "
            "{hidden_mechanism_recovery_rate} | {hidden_nmse_conditional_mean} | "
            "{intervention_coverage} | {state_count_mean} | "
            "{parameter_count_mean} |".format(**formatted)
        )
    lines.extend(
        [
            "",
            (
                "| Benchmark | Repetition | Both adapted | Judge source | "
                "No-judge source |"
            ),
            "|---|---:|:---:|---|---|",
        ]
    )
    for trial in report["paired_trials"]:
        lines.append(
            "| {benchmark_id} ({tier}) | {repetition} | {both_sources_adapted} | "
            "{judge_source_status} | {no_judge_source_status} |".format(
                **{key: _format(value) for key, value in trial.items()}
            )
        )
    lines.extend(
        [
            "",
            "The JSON and CSV companions retain every repetition and all "
            "intervention, hidden-subspace, and complexity endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_root: Path) -> None:
    """Write one JSON, Markdown, and subject-level CSV report bundle."""
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "search_ablation_endpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "search_ablation_endpoint_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    rows = report["subjects"]
    assert isinstance(rows, list)
    with (root / "search_ablation_endpoint_subjects.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(rows[0]) if rows else ["arm_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_lines(path: Path, model: type[Any]) -> tuple[Any, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-outcomes", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        _read_lines(args.sources, FrozenSearchAblationSource),
        _read_lines(args.source_outcomes, SourceAdapterOutcome),
        _read_lines(args.records, FinalEvaluationRecord),
    )
    write_report(report, args.output_root)
    print(
        f"wrote arm-level endpoint report for {report['requested_source_count']} "
        f"planned sources to {args.output_root}"
    )


if __name__ == "__main__":
    main()
