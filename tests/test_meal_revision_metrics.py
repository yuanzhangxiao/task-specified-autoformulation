"""Measured meal timing, conditional insulin, and train-only paired comparisons."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.research import meal_revision_metrics as metrics
from autoformalism.schemas.public_fitting import PublicSplit


def training(*, confound=None, omit=None, shifted_mismatch=False):
    """Shift identical 90 g pulses and give all three an observed insulin delay."""
    rows = []
    schedules = [(k, v, 90) for k, v in metrics.SHIFTED.items()]
    schedules += [(f"dose_{n}", 0, n) for n in (30, 60, 120, 150)]
    schedules += [("control0", 0, 0), ("control1", 0, 0), ("plus_infusion", 0, 90)]
    for tid, onset, dose in schedules:
        if tid == omit:
            continue
        t = np.arange(230.0)
        if tid == "control1":
            t = t[::3]
        center = onset + 5 + int(shifted_mismatch and tid == "train_006")
        meal = (dose / 5) * np.maximum(0, 1 - np.abs(t - center) / 5)
        infusion = np.zeros(len(t))
        if tid in (confound, "control1", "plus_infusion"):
            infusion[1] = 1
        rows.append(
            {
                "trajectory_id": tid,
                "time": t.tolist(),
                "targets": {
                    "I": (22 + 90 * np.exp(-(((t - onset - 101) / 35) ** 2))).tolist(),
                    "Gp": (0.4 * t + 16).tolist(),
                    "U": np.ones(len(t)).tolist(),
                },
                "external_inputs": {
                    metrics.MEAL: meal.tolist(),
                    metrics.INFUSION: infusion.tolist(),
                },
            }
        )
    return PublicSplit.model_validate(
        {"name": "train", "fingerprint": "sample", "rows": rows}
    )


def evaluate(data, *, early=91, plateau=False, glucose_gain=None, clearance=None):
    # Deliberately permute channel order and vary residual scale to catch unpack errors.
    blocks = []
    for j, r in enumerate(data.rows):
        t = np.asarray(r.time)
        peak = t[np.argmax(r.targets["I"])] - early
        fit = 100 - (t - peak) ** 2 / 1000
        if plateau and r.trajectory_id == "train_007":
            fit = 0.0 * t + 10.0
        blocks.append(
            np.column_stack((0.0 * t + j, 0.0 * t - j, (fit - r.targets["I"]) / 2))
        )
    return metrics.summarize(
        data,
        ("U", "Gp", "I"),
        {"U": 1.0, "Gp": 4.0, "I": 2.0},
        np.concatenate(blocks).ravel(),
        glucose_gain=glucose_gain,
        clearance=clearance,
    )


def test_meal_groups_global_weighting_and_paired_report():
    data = training()
    parent = evaluate(data, glucose_gain=2.0, clearance=0)
    child = evaluate(data, early=1, glucose_gain=2.0, clearance=0)
    pair = metrics.compare(parent, child)
    assert parent["all_five_unconfounded_doses_at_zero"]
    groups = {r["dose_g"]: r for r in parent["single_meal_by_dose"]}
    assert set(groups) == set(metrics.DOSES)
    assert groups[90]["unconfounded_at_zero"]["trajectories"] == ["train_003"]
    assert len(groups[90]["all"]["trajectories"]) == 4
    assert parent["no_meal_controls"]["zero_external_insulin"]["trajectories"] == [
        "control0"
    ]
    assert parent["no_meal_controls"]["with_external_insulin"]["trajectories"] == [
        "control1"
    ]
    assert parent["shifted_90g"]["absolute_peak_time_error_min"] == {
        "count": 3,
        "median": 91,
        "mad": 0,
    }
    assert parent["shifted_90g"]["absolute_peak_height_error"][
        "median"
    ] == pytest.approx(12)
    assert pair["routing"]["status"] == "early_peak_criterion_cleared"
    assert not pair["routing"]["eligible_for_proposer"]
    assert [r["grid_tolerance_min"] for r in pair["routing"]["witnesses"]] == [1.0] * 3
    assert pair["target_training_nmse_delta"]["U"] == 0
    assert pair["no_meal_target_nmse_delta"]["Gp"] == {
        "count": 2,
        "median": 0,
        "mad": 0,
    }
    expected = np.average(
        np.arange(len(data.rows)) ** 2, weights=[len(r.time) for r in data.rows]
    )
    assert parent["target_training_nmse"]["Gp"] == pytest.approx(expected)
    assert parent["target_training_nmse"]["U"] == pytest.approx(expected)
    assert parent["training_nmse"] == pytest.approx(
        np.mean(list(parent["target_training_nmse"].values()))
    )
    assert sum(parent["normalized_sse_share"].values()) == pytest.approx(1)
    markdown = metrics.render_comparison(parent, child)
    assert "30" in markdown and "150" in markdown and "plus_infusion" in markdown
    assert "No meal, with infusion" in markdown and "Parent time (height)" in markdown
    assert "Conditional insulin" in markdown
    for w in parent["conditional_postmeal"]["rows"]:
        start, end = w["meal_end_min"], w["observed_peak_min"]
        expected = w["observed_I_at_meal_end"] + 2 * (
            (end**2 - start**2) * 0.2 + 16 * (end - start)
        )
        assert w["conditional_I_at_peak"] == pytest.approx(expected)


def test_unique_persistent_peaks_use_interval_threshold():
    data = training()
    parent, child = evaluate(data), evaluate(data, early=2)
    pair = metrics.compare(parent, child)
    assert pair["routing"]["eligible_for_proposer"]
    assert pair["routing"]["status"] == "persistent_early_peak"
    assert [r["peak_time_error_min"] for r in pair["routing"]["witnesses"]] == [-2] * 3


def test_numerical_metrics_and_comparison_roundtrip_as_finite_json():
    score = evaluate(
        training(), glucose_gain=np.float64(2.0), clearance=np.float64(0.0)
    )
    for record in (score, metrics.compare(score, score)):
        assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize(
    "kwargs",
    [{"omit": "train_007"}, {"confound": "train_003"}, {"shifted_mismatch": True}],
)
def test_missing_shifted_witness_defers(kwargs):
    data = training(**kwargs)
    scores = evaluate(data)
    assert (
        metrics.compare(scores, scores)["routing"]["status"] == "insufficient_evidence"
    )


def test_flat_fitted_peak_defers_even_if_first_maximum_is_early():
    row = evaluate(training(), plateau=True)
    assert not metrics.compare(row, row)["routing"]["eligible_for_proposer"]
    assert row["shifted_90g"]["absolute_peak_time_error_min"]["count"] == 2


def test_conditioning_handles_nonzero_decay_and_time_varying_glucose():
    data = training()
    a, decay = 1.5, 0.02
    parent = evaluate(data, glucose_gain=a, clearance=decay)
    for row in parent["conditional_postmeal"]["rows"]:
        t0, t1 = row["meal_end_min"], row["observed_peak_min"]

        def particular(t):
            return a * ((16 + 0.4 * t) / decay - 0.4 / decay**2)

        expected = particular(t1) + np.exp(-decay * (t1 - t0)) * (
            row["observed_I_at_meal_end"] - particular(t0)
        )
        assert row["conditional_I_at_peak"] == pytest.approx(expected)


def test_incompatible_data_residual_and_scale_fail_closed():
    data = training()
    with pytest.raises(ValueError, match="public training"):
        evaluate(data.model_copy(update={"name": "val"}))
    with pytest.raises(ValueError, match="complete, flat and finite"):
        metrics.summarize(
            data,
            metrics.CHANNELS,
            dict.fromkeys(metrics.CHANNELS, 1.0),
            np.array([np.nan]),
        )
    parent = evaluate(data)
    child = deepcopy(parent)
    child["training_scales"]["U"] *= 2
    with pytest.raises(ValueError, match="data or scales differ"):
        metrics.compare(parent, child)
    child = deepcopy(parent)
    child["training_sha256"] = "changed"
    with pytest.raises(ValueError, match="data or scales differ"):
        metrics.compare(parent, child)
    child = deepcopy(parent)
    child["rows"].reverse()
    with pytest.raises(ValueError, match="pairing differs"):
        metrics.compare(parent, child)
