"""Tests for offline hybrid-judge failure attribution."""

from __future__ import annotations

import json

import pytest

from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from scripts.analyze_hybrid_diagnostics import (
    PairMetadata,
    analyze_diagnostics,
    render_markdown,
)


def _label(pair_id: str) -> HybridCalibrationLabels:
    return HybridCalibrationLabels.model_validate(
        {
            "pair_id": pair_id,
            "overall_preference": "baseline",
            "absolute_labels": [
                {
                    "criterion": "semantic_fluxes_not_duplicated",
                    "subject_id": "candidate",
                    "baseline": "pass",
                    "mutated": "fail",
                    "rationale": "controlled duplicate",
                    "label_source": "mutation_contract:duplicated_gp_flux",
                }
            ],
            "comparative_labels": [
                {
                    "criterion": "parsimony_while_task_sufficient",
                    "preference": "baseline",
                    "rationale": "baseline omits the duplicate",
                    "label_source": "mutation_contract:duplicated_gp_flux",
                }
            ],
        }
    )


def _absolute(position: str) -> str:
    baseline = {"verdict": "pass", "evidence": "baseline evidence"}
    mutated = {"verdict": "fail", "evidence": "mutated evidence"}
    left, right = (baseline, mutated) if position == "A" else (mutated, baseline)
    return json.dumps(
        [
            {
                "criterion": "semantic_fluxes_not_duplicated",
                "subject_id": "candidate",
                "candidate_a": left,
                "candidate_b": right,
            }
        ]
    )


def _comparative(position: str) -> str:
    verdict = "candidate_a" if position == "A" else "candidate_b"
    return json.dumps(
        [
            {
                "criterion": "parsimony_while_task_sufficient",
                "verdict": verdict,
                "evidence": "baseline is more parsimonious",
            }
        ]
    )


def _row(
    pair_id: str,
    repetition: int,
    order: str,
    *,
    absolute_delta: float,
    relative_delta: float,
) -> dict[str, object]:
    position = "A" if order == "baseline_a" else "B"
    baseline_score = 0.5 + absolute_delta / 2
    mutated_score = 0.5 - absolute_delta / 2
    left, right = (
        (baseline_score, mutated_score)
        if position == "A"
        else (mutated_score, baseline_score)
    )
    decision = absolute_delta + 0.25 * relative_delta
    return {
        "pair_id": pair_id,
        "judge_model": "vllm:test",
        "mutation_type": "duplicated_gp_flux",
        "repetition": repetition,
        "order": order,
        "baseline_position": position,
        "baseline_preference": "baseline" if decision > 0.05 else "mutated",
        "baseline_decision_value": decision,
        "baseline_relative_preference": (relative_delta + 1.0) / 2.0,
        "candidate_a_score": left,
        "candidate_b_score": right,
        "candidate_a_hard_status": "",
        "candidate_b_hard_status": "",
        "deterministic_assessments": "[]",
        "absolute_assessments": _absolute(position),
        "comparative_assessments": _comparative(position),
    }


def _fixture() -> tuple[
    list[dict[str, object]],
    dict[str, HybridCalibrationLabels],
    dict[str, PairMetadata],
]:
    rows = []
    for repetition in range(2):
        for order in ("baseline_a", "baseline_b"):
            rows.append(
                _row(
                    "pair_1",
                    repetition,
                    order,
                    absolute_delta=0.2,
                    relative_delta=0.5,
                )
            )
            rows.append(
                _row(
                    "pair_2",
                    repetition,
                    order,
                    absolute_delta=0.1,
                    relative_delta=-1.0,
                )
            )
    labels = {pair_id: _label(pair_id) for pair_id in ("pair_1", "pair_2")}
    metadata = {
        "pair_1": PairMetadata("duplicated_gp_flux", "structure_1"),
        "pair_2": PairMetadata("duplicated_gp_flux", "structure_2"),
    }
    return rows, labels, metadata


def test_diagnostics_attribute_pairs_questions_and_weight_sensitivity() -> None:
    rows, labels, metadata = _fixture()

    payload = analyze_diagnostics(
        rows,
        [],
        labels,
        metadata,
        comparative_weights=(0.0, 0.25),
        tie_thresholds=(0.05,),
    )
    model = payload["models"]["vllm:test"]
    pairs = {item["pair_id"]: item for item in model["pair_diagnostics"]}
    assert pairs["pair_1"]["correct"] is True
    assert pairs["pair_1"]["order_consistency"] == 1.0
    assert pairs["pair_2"]["correct"] is False
    assert pairs["pair_2"]["decision_mean"] == pytest.approx(-0.15)

    current = next(
        item for item in model["aggregation_sensitivity"] if item["is_current"]
    )
    no_comparative = next(
        item
        for item in model["aggregation_sensitivity"]
        if item["comparative_weight"] == 0.0
    )
    assert current["pair_accuracy"] == 0.5
    assert no_comparative["pair_accuracy"] == 1.0
    assert model["leave_one_structure_out"]["structure_count"] == 2

    questions = model["question_performance"]
    assert {item["layer"] for item in questions} == {
        "llm_semantic_absolute",
        "llm_comparative",
    }
    assert all(item["accuracy"] == 1.0 for item in questions)
    markdown = render_markdown(payload)
    assert "pair_2" in markdown
    assert "exploratory" in markdown


def test_diagnostics_reject_changed_frozen_scoring_contract() -> None:
    rows, labels, metadata = _fixture()
    rows[0]["baseline_decision_value"] = 0.0

    with pytest.raises(ValueError, match="does not match reconstructed"):
        analyze_diagnostics(rows, [], labels, metadata)


def test_diagnostics_count_provider_failure_end_to_end() -> None:
    rows, labels, metadata = _fixture()
    rows = [row for row in rows if not (
        row["pair_id"] == "pair_1"
        and row["repetition"] == 0
        and row["order"] == "baseline_a"
    )]
    failure = {
        "pair_id": "pair_1",
        "judge_model": "vllm:test",
        "repetition": 0,
        "order": "baseline_a",
    }

    payload = analyze_diagnostics(
        rows,
        [failure],
        labels,
        metadata,
        comparative_weights=(0.25,),
        tie_thresholds=(0.05,),
    )
    model = payload["models"]["vllm:test"]
    pair = next(
        item for item in model["pair_diagnostics"] if item["pair_id"] == "pair_1"
    )
    assert model["response_success_rate"] == pytest.approx(7 / 8)
    assert pair["response_success_rate"] == 0.75
    assert pair["end_to_end_call_accuracy"] == 0.75
