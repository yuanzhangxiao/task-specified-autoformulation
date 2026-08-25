"""Tests for offline symmetry-preserving hybrid aggregation."""

from __future__ import annotations

import json

import pytest

from autoformalism.judging import HybridScoringConfig, score_hybrid_pair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
)
from scripts.analyze_hybrid_symmetric_aggregation import (
    RULE_FINAL_MEAN,
    RULE_QUESTION_CONSENSUS,
    RULE_UNCERTAINTY_ABSTENTION,
    _uncertainty_direction,
    aggregate_trial,
    analyze,
)


def _paired(
    criterion: AbsoluteCriterion,
    first: AbsoluteVerdict,
    second: AbsoluteVerdict,
) -> PairedAbsoluteAssessment:
    return PairedAbsoluteAssessment(
        criterion=criterion,
        subject_id="candidate",
        candidate_a=CandidateAbsoluteAssessment(
            verdict=first,
            evidence="First candidate evidence.",
        ),
        candidate_b=CandidateAbsoluteAssessment(
            verdict=second,
            evidence="Second candidate evidence.",
        ),
    )


def _result(
    *,
    source: tuple[AbsoluteVerdict, AbsoluteVerdict],
    comparative: dict[RelativeCriterion, RelativeVerdict],
) -> HybridJudgeResult:
    return HybridJudgeResult(
        absolute_assessments=(
            _paired(AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, *source),
            _paired(
                AbsoluteCriterion.SINK_ROLES_CONSISTENT,
                AbsoluteVerdict.PASS,
                AbsoluteVerdict.PASS,
            ),
            _paired(
                AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
                AbsoluteVerdict.PASS,
                AbsoluteVerdict.PASS,
            ),
        ),
        comparative_assessments=tuple(
            RelativeAssessment(
                criterion=criterion,
                verdict=comparative[criterion],
                evidence="Comparative test evidence.",
            )
            for criterion in RelativeCriterion
        ),
    )


def _row(
    *,
    pair_id: str,
    mutation_type: str,
    order: str,
    result: HybridJudgeResult,
    deterministic: tuple[PairedAbsoluteAssessment, ...],
    config: HybridScoringConfig,
) -> dict[str, str]:
    score = score_hybrid_pair(
        result,
        deterministic,
        RequirementRegistry(),
        config,
    )
    baseline_position = "A" if order == "baseline_a" else "B"
    normalized = (
        score.decision_value
        if baseline_position == "A"
        else -score.decision_value
        if score.decision_value is not None
        else None
    )
    return {
        "pair_id": pair_id,
        "judge_model": "vllm:test",
        "mutation_type": mutation_type,
        "repetition": "0",
        "order": order,
        "baseline_position": baseline_position,
        "baseline_decision_value": str(normalized),
        "requirements": RequirementRegistry().model_dump_json(),
        "deterministic_assessments": json.dumps(
            [item.model_dump(mode="json") for item in deterministic]
        ),
        "absolute_assessments": json.dumps(
            [item.model_dump(mode="json") for item in result.absolute_assessments]
        ),
        "comparative_assessments": json.dumps(
            [
                item.model_dump(mode="json")
                for item in result.comparative_assessments
            ]
        ),
    }


def _fixture() -> tuple[
    list[dict[str, str]],
    dict[str, HybridCalibrationLabels],
    HybridScoringConfig,
]:
    config = HybridScoringConfig()
    deterministic_a = (
        _paired(
            AbsoluteCriterion.TASK_INPUTS_REACH_TARGETS,
            AbsoluteVerdict.PASS,
            AbsoluteVerdict.PASS,
        ),
    )
    deterministic_b = deterministic_a
    always_first_a = {
        RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT: (
            RelativeVerdict.CANDIDATE_A
        ),
        RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS: (
            RelativeVerdict.CANDIDATE_A
        ),
        RelativeCriterion.MECHANISTIC_INTERPRETABILITY: (
            RelativeVerdict.CANDIDATE_A
        ),
    }
    position_biased_b = {
        RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT: (
            RelativeVerdict.CANDIDATE_B
        ),
        RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS: (
            RelativeVerdict.CANDIDATE_B
        ),
        RelativeCriterion.MECHANISTIC_INTERPRETABILITY: (
            RelativeVerdict.CANDIDATE_A
        ),
    }
    tradeoff_a = _result(
        source=(AbsoluteVerdict.PASS, AbsoluteVerdict.FAIL),
        comparative=always_first_a,
    )
    tradeoff_b = _result(
        source=(AbsoluteVerdict.FAIL, AbsoluteVerdict.PASS),
        comparative=position_biased_b,
    )
    tie = dict.fromkeys(RelativeCriterion, RelativeVerdict.TIE)
    equivalent_a = _result(
        source=(AbsoluteVerdict.PASS, AbsoluteVerdict.PASS),
        comparative=tie,
    )
    equivalent_b = equivalent_a
    rows = [
        _row(
            pair_id="tradeoff",
            mutation_type="tradeoff_wrong_sink_vs_duplicate",
            order="baseline_a",
            result=tradeoff_a,
            deterministic=deterministic_a,
            config=config,
        ),
        _row(
            pair_id="tradeoff",
            mutation_type="tradeoff_wrong_sink_vs_duplicate",
            order="baseline_b",
            result=tradeoff_b,
            deterministic=deterministic_b,
            config=config,
        ),
        _row(
            pair_id="equivalent",
            mutation_type="algebraic_reordering_equivalent",
            order="baseline_a",
            result=equivalent_a,
            deterministic=deterministic_a,
            config=config,
        ),
        _row(
            pair_id="equivalent",
            mutation_type="algebraic_reordering_equivalent",
            order="baseline_b",
            result=equivalent_b,
            deterministic=deterministic_b,
            config=config,
        ),
    ]
    labels = {
        "tradeoff": HybridCalibrationLabels(
            pair_id="tradeoff",
            overall_preference=ExpectedPairPreference.UNLABELED,
            absolute_labels=(),
            comparative_labels=(),
        ),
        "equivalent": HybridCalibrationLabels(
            pair_id="equivalent",
            overall_preference=ExpectedPairPreference.TIE,
            absolute_labels=(),
            comparative_labels=(),
        ),
    }
    return rows, labels, config


def test_question_consensus_withholds_position_biased_criterion() -> None:
    rows, _labels, config = _fixture()

    trial = aggregate_trial(rows[0], rows[1], config=config)

    assert trial["comparative_disagreements"] == [
        RelativeCriterion.MECHANISTIC_INTERPRETABILITY.value
    ]
    assert trial["rules"][RULE_QUESTION_CONSENSUS]["preference"] == "baseline"
    assert trial["orientation_half_gap"] > 0.0


def test_uncertainty_rule_abstains_when_orientation_interval_crosses_tie() -> None:
    assert _uncertainty_direction(0.06, 0.04, 0.05) == "indeterminate"
    assert _uncertainty_direction(0.20, 0.04, 0.05) == "baseline"
    assert _uncertainty_direction(0.00, 0.04, 0.05) == "tie"


def test_analysis_scores_only_labeled_equivalence_winners() -> None:
    rows, labels, config = _fixture()

    result = analyze(rows, [], labels, config=config)

    assert result["attempted_paired_trials"] == 2
    assert result["complete_paired_trials"] == 2
    assert result["paired_response_coverage"] == 1.0
    for rule in (
        RULE_FINAL_MEAN,
        RULE_QUESTION_CONSENSUS,
        RULE_UNCERTAINTY_ABSTENTION,
    ):
        metrics = result["rules"][rule]
        assert metrics["labeled_accuracy_conditional"] == 1.0
        assert metrics["equivalence_tie_rate"] == 1.0
    assert result["comparative_disagreements"] == [
        {
            "mutation_type": "tradeoff_wrong_sink_vs_duplicate",
            "criterion": "mechanistic_interpretability",
            "count": 1,
        }
    ]


def test_aggregate_trial_rejects_mismatched_identity() -> None:
    rows, _labels, config = _fixture()
    mismatched = {**rows[1], "pair_id": "other"}

    with pytest.raises(ValueError, match="same trial"):
        aggregate_trial(rows[0], mismatched, config=config)
