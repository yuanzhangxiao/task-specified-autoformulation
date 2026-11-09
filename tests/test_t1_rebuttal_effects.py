"""Regression checks for matched, coverage-preserving intervention contrasts."""

import pytest

from scripts.compare_t1_rebuttal_effects import (
    check_record,
    effect_metrics,
    summarize_model,
)


def record(observed, predicted):
    return {
        "time": [0, 1, 2],
        "trajectory_id": "probe",
        "observed": observed,
        "predicted": predicted,
        "success": predicted is not None,
    }


def test_signed_peak_and_latency_use_each_models_own_control():
    base = record([5, 5, 5], [8, 7, 6])
    probe = record([5, 3, 4], [8, 6, 4])
    result = effect_metrics(probe, base)
    assert result["effect_rse"] == pytest.approx(2 / 5)
    assert result["correct_peak_direction"]
    assert result["reference_signed_peak"] == result["predicted_signed_peak"] == -2
    assert result["absolute_peak_timing_error_min"] == 1


def test_no_effect_abstains_on_peak_time_and_missing_control_propagates():
    base = record([5, 5, 5], [8, 7, 6])
    probe = record([5, 3, 4], [8, 7, 6])
    result = effect_metrics(probe, base)
    assert result["effect_rse"] == 1
    assert result["predicted_peak_min"] is None
    assert result["absolute_peak_timing_error_min"] is None
    assert not result["correct_peak_direction"]
    assert effect_metrics(probe, record([5, 5, 5], None))["status"] == "rollout_failed"
    assert effect_metrics(base, base)["status"] == "negligible_reference_effect"


def test_mismatched_grids_and_references_rejected():
    probe = record([5, 3, 4], [8, 7, 6])
    base = dict(probe, time=[0, 1, 3])
    with pytest.raises(ValueError, match="grids"):
        effect_metrics(probe, base)
    with pytest.raises(ValueError, match="reference"):
        check_record(probe, dict(base, targets={"v01": [5, 3, 4]}))


def test_unavailable_endpoint_keeps_every_planned_contrast():
    rows = summarize_model({"model_id": "missing"}, {}, {})
    assert len(rows) == len({row["contrast"] for row in rows}) == 8
    assert all(row["status"] == "model_unavailable" for row in rows)
