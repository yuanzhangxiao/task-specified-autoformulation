"""Tests for stored hybrid-judge rationale auditing."""

from __future__ import annotations

import json

import pytest

from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from scripts.audit_hybrid_judge_evidence import audit_evidence, render_markdown


def _labels() -> HybridCalibrationLabels:
    return HybridCalibrationLabels.model_validate(
        {
            "pair_id": "pair_1",
            "overall_preference": "baseline",
            "absolute_labels": [
                {
                    "criterion": "semantic_fluxes_not_duplicated",
                    "subject_id": "candidate",
                    "baseline": "pass",
                    "mutated": "fail",
                    "rationale": "the mutation duplicates one physical flux",
                    "label_source": "mutation_contract:duplicated_gp_flux",
                }
            ],
            "comparative_labels": [
                {
                    "criterion": "fewer_unsupported_assumptions",
                    "preference": "baseline",
                    "rationale": "only the mutation assumes a duplicate flux",
                    "label_source": "mutation_contract:duplicated_gp_flux",
                }
            ],
        }
    )


def _row(repetition: int, order: str) -> dict[str, object]:
    position = "A" if order == "baseline_a" else "B"
    baseline = {"verdict": "pass", "evidence": "one balance term"}
    mutated = {"verdict": "pass", "evidence": "two terms are distinct"}
    left, right = (baseline, mutated) if position == "A" else (mutated, baseline)
    return {
        "pair_id": "pair_1",
        "judge_model": "vllm:test",
        "mutation_type": "duplicated_gp_flux",
        "repetition": repetition,
        "order": order,
        "baseline_position": position,
        "absolute_assessments": json.dumps(
            [
                {
                    "criterion": "semantic_fluxes_not_duplicated",
                    "subject_id": "candidate",
                    "candidate_a": left,
                    "candidate_b": right,
                }
            ]
        ),
        "comparative_assessments": json.dumps(
            [
                {
                    "criterion": "fewer_unsupported_assumptions",
                    "verdict": "tie",
                    "evidence": "both candidates use the same assumptions",
                }
            ]
        ),
    }


def test_audit_records_normalized_absolute_and_comparative_evidence() -> None:
    rows = [_row(0, "baseline_a"), _row(0, "baseline_b")]
    payload = audit_evidence(
        rows,
        {"pair_1": _labels()},
        {"pair_1": ("duplicated_gp_flux", "structure_1")},
    )

    assert payload["score_row_count"] == 2
    assert payload["error_count"] == 4
    absolute = [item for item in payload["errors"] if item["kind"] == "absolute"]
    comparative = [
        item for item in payload["errors"] if item["kind"] == "comparative"
    ]
    assert len(absolute) == 2
    assert len(comparative) == 2
    assert all(item["actual_baseline"] == "pass" for item in absolute)
    assert all(item["actual_mutated"] == "pass" for item in absolute)
    assert all(item["incorrect_sides"] == ["mutated"] for item in absolute)
    assert {item["baseline_evidence"] for item in absolute} == {
        "one balance term"
    }
    assert {item["mutated_evidence"] for item in absolute} == {
        "two terms are distinct"
    }

    performance = {
        (item["kind"], item["criterion"]): item
        for item in payload["certified_performance"]
    }
    assert performance[("absolute", "semantic_fluxes_not_duplicated")][
        "accuracy"
    ] == 0.5
    assert performance[("comparative", "fewer_unsupported_assumptions")][
        "accuracy"
    ] == 0.0
    markdown = render_markdown(payload, examples_per_group=1)
    assert "two terms are distinct" in markdown
    assert "baseline_a=1, baseline_b=1" in markdown


def test_audit_rejects_duplicate_outcome_key() -> None:
    row = _row(0, "baseline_a")

    with pytest.raises(ValueError, match="duplicate score outcome"):
        audit_evidence(
            [row, row],
            {"pair_1": _labels()},
            {"pair_1": ("duplicated_gp_flux", "structure_1")},
        )
