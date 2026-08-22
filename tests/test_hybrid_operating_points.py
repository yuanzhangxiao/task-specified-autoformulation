"""Tests for offline hybrid-judge call-budget analysis."""

from __future__ import annotations

import pytest

from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from scripts.analyze_hybrid_operating_points import analyze_operating_points


def _labels(pair_id: str) -> HybridCalibrationLabels:
    return HybridCalibrationLabels.model_validate(
        {
            "pair_id": pair_id,
            "overall_preference": "baseline",
            "absolute_labels": [],
            "comparative_labels": [],
        }
    )


def _row(
    pair_id: str,
    repetition: int,
    order: str,
    decision: float,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "judge_model": "ollama:test",
        "repetition": repetition,
        "order": order,
        "baseline_decision_value": decision,
    }


def _failure(pair_id: str, repetition: int, order: str) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "judge_model": "ollama:test",
        "repetition": repetition,
        "order": order,
    }


def test_operating_points_measure_order_bias_and_provider_failures() -> None:
    rows = [
        _row("pair_1", 0, "baseline_a", 0.5),
        _row("pair_1", 0, "baseline_b", 0.5),
        _row("pair_1", 1, "baseline_a", 0.5),
        _row("pair_2", 0, "baseline_a", 0.5),
        _row("pair_2", 0, "baseline_b", -0.5),
        _row("pair_2", 1, "baseline_a", 0.5),
        _row("pair_2", 1, "baseline_b", -0.5),
    ]
    failures = [_failure("pair_1", 1, "baseline_b")]
    labels = {pair_id: _labels(pair_id) for pair_id in ("pair_1", "pair_2")}

    metrics = analyze_operating_points(
        rows,
        failures,
        labels,
        bootstrap_samples=100,
    )["ollama:test"]
    by_name = {
        item["configuration"]: item for item in metrics["operating_points"]
    }

    order_a = by_name["one_order_a_one_call"]
    assert order_a["calls_per_pair"] == 1
    assert order_a["response_success_rate"] == 1.0
    assert order_a["pair_accuracy_conditional_on_decision"] == 1.0
    assert order_a["strict_end_to_end_pair_accuracy"] == 1.0

    order_b = by_name["one_order_b_one_call"]
    assert order_b["response_success_rate"] == 0.75
    assert order_b["pair_decision_coverage"] == 0.75
    assert order_b["pair_accuracy_conditional_on_decision"] == pytest.approx(1 / 3)
    assert order_b["strict_end_to_end_pair_accuracy"] == 0.25

    both_once = by_name["both_orders_1_repetition"]
    assert both_once["calls_per_pair"] == 2
    assert both_once["response_success_rate"] == 0.875
    assert both_once["pair_decision_coverage"] == 1.0
    assert both_once["pair_accuracy_conditional_on_decision"] == 0.5
    assert both_once["strict_end_to_end_pair_accuracy"] == 0.25
    assert both_once["order_consistency_rate"] == pytest.approx(1 / 3)

    both_twice = by_name["both_orders_2_repetitions"]
    assert both_twice["calls_per_pair"] == 4
    assert both_twice["complete_response_rate"] == 0.5
    assert both_twice["pair_accuracy_conditional_ci95"] is not None
    assert both_twice["strict_end_to_end_pair_accuracy_ci95"] is not None

    adaptive = {
        item["max_attempts_per_orientation"]: item
        for item in metrics["adaptive_operating_points"]
    }
    no_retry = adaptive[1]
    assert no_retry["expected_calls_per_pair"] == 2.0
    assert no_retry["paired_response_coverage"] == 0.75
    assert no_retry["pair_accuracy_conditional_on_paired_response"] == pytest.approx(
        1 / 3
    )
    assert no_retry["end_to_end_pair_accuracy"] == 0.25

    retry_once = adaptive[2]
    assert retry_once["expected_calls_per_pair"] == 2.25
    assert retry_once["retry_activation_rate"] == 0.25
    assert retry_once["paired_response_coverage"] == 1.0
    assert retry_once["pair_accuracy_conditional_on_paired_response"] == 0.5
    assert retry_once["end_to_end_pair_accuracy"] == 0.5
    assert retry_once["order_consistency_rate"] == 0.5


def test_operating_points_reject_success_failure_overlap() -> None:
    row = _row("pair_1", 0, "baseline_a", 0.5)

    with pytest.raises(ValueError, match="both successes and failures"):
        analyze_operating_points(
            [row],
            [_failure("pair_1", 0, "baseline_a")],
            {"pair_1": _labels("pair_1")},
            bootstrap_samples=10,
        )


def test_operating_points_reject_unrecorded_order() -> None:
    with pytest.raises(ValueError, match="unrecorded"):
        analyze_operating_points(
            [_row("pair_1", 0, "baseline_a", 0.5)],
            [],
            {"pair_1": _labels("pair_1")},
            bootstrap_samples=10,
        )
