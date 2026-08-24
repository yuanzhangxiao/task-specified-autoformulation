"""Tests for frozen comparative-criterion and score-constant ablations."""

from __future__ import annotations

import json

import pytest

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedComparativeLabel,
    HybridCalibrationLabels,
)
from autoformalism.schemas import RelativeCriterion
from scripts.analyze_hybrid_comparative_ablation import (
    _ABLATIONS,
    analyze,
    evaluate_row,
)


def _labels() -> HybridCalibrationLabels:
    return HybridCalibrationLabels(
        pair_id="pair_1",
        overall_preference="baseline",
        absolute_labels=(),
        comparative_labels=(
            ExpectedComparativeLabel(
                criterion=RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
                preference="baseline",
                rationale="The mutation is unnecessary.",
                label_source="mutation_contract:test",
            ),
            ExpectedComparativeLabel(
                criterion=RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
                preference="unlabeled",
                rationale="Not certified.",
                label_source="not_scored_by_mutation_contract",
            ),
            ExpectedComparativeLabel(
                criterion=RelativeCriterion.MECHANISTIC_INTERPRETABILITY,
                preference="unlabeled",
                rationale="Not certified.",
                label_source="not_scored_by_mutation_contract",
            ),
        ),
    )


def _row(order: str) -> dict[str, str]:
    baseline_a = order == "baseline_a"
    assessments = []
    for criterion, baseline_verdict in (
        (RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT, "baseline"),
        (RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS, "tie"),
        (RelativeCriterion.MECHANISTIC_INTERPRETABILITY, "mutated"),
    ):
        verdict = baseline_verdict
        if baseline_verdict == "baseline":
            verdict = "candidate_a" if baseline_a else "candidate_b"
        elif baseline_verdict == "mutated":
            verdict = "candidate_b" if baseline_a else "candidate_a"
        assessments.append(
            {
                "criterion": criterion.value,
                "verdict": verdict,
                "evidence": "Frozen comparative answer.",
            }
        )
    return {
        "pair_id": "pair_1",
        "mutation_type": "test_mutation",
        "judge_model": "vllm:test",
        "repetition": "0",
        "order": order,
        "baseline_position": "A" if baseline_a else "B",
        "candidate_a_score": "0.5",
        "candidate_b_score": "0.5",
        "candidate_a_hard_status": "",
        "candidate_b_hard_status": "",
        "comparative_assessments": json.dumps(assessments),
    }


def test_criterion_subsets_change_deterministic_decision() -> None:
    row = _row("baseline_a")
    labels = _labels()
    definitions = {item.name: item for item in _ABLATIONS}

    all_three = evaluate_row(
        row,
        labels=labels,
        definition=definitions["all_three"],
        comparative_weight=0.25,
        tie_threshold=0.05,
    )
    parsimony = evaluate_row(
        row,
        labels=labels,
        definition=definitions["parsimony_only"],
        comparative_weight=0.25,
        tie_threshold=0.05,
    )
    labeled = evaluate_row(
        row,
        labels=labels,
        definition=definitions["mutation_contract_labeled_only"],
        comparative_weight=0.25,
        tie_threshold=0.05,
    )

    assert all_three["comparative_preference_for_baseline"] == 0.5
    assert all_three["predicted"] == "tie"
    assert parsimony["decision_for_baseline"] == 0.25
    assert parsimony["predicted"] == "baseline"
    assert labeled["selected_criteria"] == [
        RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT.value
    ]
    assert labeled["predicted"] == "baseline"


def test_analysis_preserves_order_symmetry_and_counts_failures() -> None:
    labels = {"pair_1": _labels()}
    metrics = analyze(
        [_row("baseline_a"), _row("baseline_b")],
        [{"judge_model": "vllm:test", "failure_stage": "atomic_evidence"}],
        labels,
        comparative_weight=0.25,
        tie_threshold=0.05,
        lambdas=(0.0, 0.25),
        thresholds=(0.0, 0.05),
    )

    model = metrics["models"]["vllm:test"]
    parsimony = model["fixed_ablation"]["parsimony_only"]
    assert parsimony["conditional_accuracy"] == 1.0
    assert parsimony["end_to_end_accuracy"] == pytest.approx(2.0 / 3.0)
    assert parsimony["order_consistency"] == 1.0
    assert len(model["sensitivity"]["all_three"]) == 4
