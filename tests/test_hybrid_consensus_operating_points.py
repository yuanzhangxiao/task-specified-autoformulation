"""Tests for frozen paired-question-consensus call-budget analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from scripts.analyze_hybrid_consensus_operating_points import (
    analyze_operating_points,
)
from scripts.analyze_hybrid_symmetric_aggregation import (
    RULE_QUESTION_CONSENSUS,
)

CONFIG = Path("configs/hybrid_judge_consensus_operating_point_v1.json")


def _label(
    pair_id: str,
    preference: ExpectedPairPreference,
) -> HybridCalibrationLabels:
    return HybridCalibrationLabels(
        pair_id=pair_id,
        overall_preference=preference,
        absolute_labels=(),
        comparative_labels=(),
    )


def _trial(pair_id: str, repetition: int, decision: float) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "repetition": repetition,
        "rules": {
            RULE_QUESTION_CONSENSUS: {
                "decision": decision,
                "preference": (
                    "baseline"
                    if decision > 0.05
                    else "mutated"
                    if decision < -0.05
                    else "tie"
                ),
            }
        },
    }


def _protocol() -> dict[str, object]:
    return {
        "source_primary_rule": RULE_QUESTION_CONSENSUS,
        "new_llm_calls": 0,
        "available_seed_ids": [0, 1],
        "scoring": {"tie_threshold": 0.05},
        "candidate_grid": {
            "target_complete_paired_seeds": [1, 2],
            "maximum_distinct_seed_attempts": [1, 2],
        },
        "cost_accounting": {
            "judge_operations_per_paired_seed_attempt": 2,
            "logical_llm_stages_per_judge_operation": 2,
        },
        "selection_gate": {
            "minimum_decision_coverage": 1.0,
            "minimum_labeled_decision_coverage": 1.0,
            "minimum_labeled_accuracy": 1.0,
            "minimum_equivalence_tie_accuracy": 1.0,
            "minimum_known_dominance_accuracy": 1.0,
            "minimum_known_dominance_pair_accuracy": 1.0,
            "minimum_modal_preference_consistency": 1.0,
            "maximum_mean_repeat_decision_sd": 0.0,
        },
        "interpretation_boundary": "Tradeoffs are unscored.",
    }


def test_analyzer_selects_one_complete_seed_with_one_response_retry() -> None:
    labels = {
        "dominance": _label(
            "dominance",
            ExpectedPairPreference.BASELINE,
        ),
        "equivalence": _label(
            "equivalence",
            ExpectedPairPreference.TIE,
        ),
        "tradeoff": _label(
            "tradeoff",
            ExpectedPairPreference.UNLABELED,
        ),
    }
    symmetric = {
        "schema_version": "hybrid-symmetric-aggregation-analysis-1",
        "trials": [
            _trial("dominance", 1, 0.2),
            _trial("equivalence", 0, 0.0),
            _trial("equivalence", 1, 0.0),
            _trial("tradeoff", 0, 0.2),
            _trial("tradeoff", 1, 0.2),
        ],
    }

    result = analyze_operating_points(
        symmetric,
        labels,
        _protocol(),
        bootstrap_samples=50,
    )

    selected = result["selected_operating_point"]
    assert selected["configuration"] == "target_1_complete_paired_seed_within_2"
    assert selected["decision_coverage"] == 1.0
    assert selected["labeled_accuracy_conditional"] == 1.0
    assert selected["known_dominance_pair_accuracy"] == 1.0
    assert selected["expected_paired_seed_attempts"] == pytest.approx(7 / 6)
    assert selected["expected_judge_operations_per_pair"] == pytest.approx(7 / 3)
    assert selected["unlabeled_tradeoff_preference_counts"] == {"baseline": 2}
    assert result["source_complete_paired_trials"] == 5
    assert result["expected_paired_trials"] == 6


def test_analyzer_rejects_duplicate_symmetric_trial() -> None:
    trial = _trial("pair", 0, 0.2)

    with pytest.raises(ValueError, match="duplicate symmetric trial"):
        analyze_operating_points(
            {
                "schema_version": "hybrid-symmetric-aggregation-analysis-1",
                "trials": [trial, trial],
            },
            {"pair": _label("pair", ExpectedPairPreference.BASELINE)},
            _protocol(),
            bootstrap_samples=10,
        )


def test_frozen_operating_point_config_keeps_science_and_cost_separate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen_before_offline_operating_point_rescoring"
    assert payload["new_llm_calls"] == 0
    assert payload["source_primary_rule"] == RULE_QUESTION_CONSENSUS
    assert payload["candidate_grid"]["scientific_disagreement_retry_policy"].startswith(
        "none"
    )
    assert payload["cost_accounting"] == {
        "judge_operations_per_paired_seed_attempt": 2,
        "logical_llm_stages_per_judge_operation": 2,
    }
    assert "minimum_labeled_accuracy" in payload["selection_gate"]
    assert payload["interpretation_boundary"].startswith("unlabeled")
