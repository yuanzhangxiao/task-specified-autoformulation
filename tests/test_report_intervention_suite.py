"""Event metrics and exposure counts read from a scored suite root."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_anchor_fidelity as anchor
from scripts import report_intervention_suite as suite

CSTR = anchor.fits.CSTR
TIME = np.round(np.arange(0.0, 30.05, 0.1), 10)


def _lagged_pulse(start, width, lag):
    """A first-order response to a unit pulse, delayed by ``lag``."""
    on = ((start + lag <= TIME) & (start + width + lag > TIME)).astype(float)
    out = np.zeros_like(TIME)
    for i in range(1, len(TIME)):
        out[i] = out[i - 1] + 0.1 * (on[i - 1] - out[i - 1]) / 0.5
    return out


def test_a_missing_lag_shows_as_an_early_peak():
    control = np.full(len(TIME), 350.0)
    truth = control + 4 * _lagged_pulse(20.0, 1.0, 0.3)
    early = control + 4 * _lagged_pulse(20.0, 1.0, 0.0)
    result = suite.event_metrics(
        TIME, truth, control, early, control, [20.0, 30.0], True
    )
    assert result["direction_correct"] is True
    assert result["peak_delay"] == pytest.approx(-0.3)
    assert result["event_rse"] > 0.1
    exact = suite.event_metrics(
        TIME, truth, control, truth, control, [20.0, 30.0], True
    )
    assert exact["peak_delay"] == 0.0 and exact["event_rse"] == 0.0


def test_a_wrong_or_missing_effect_gets_the_direction_wrong():
    control = np.full(len(TIME), 350.0)
    truth = control + 4 * _lagged_pulse(20.0, 1.0, 0.3)
    flipped = control - 4 * _lagged_pulse(20.0, 1.0, 0.3)
    for predicted in (flipped, control):
        result = suite.event_metrics(
            TIME, truth, control, predicted, control, [20.0, 30.0], True
        )
        assert result["direction_correct"] is False
    assert (
        suite.event_metrics(TIME, truth, control, control, control, [20.0, 30.0], True)[
            "peak_delay"
        ]
        is None
    )
    none = suite.event_metrics(
        TIME, control, control, truth, control, [20.0, 30.0], True
    )
    assert none == {"status": "negligible_reference_effect"}


def _lagged(level):
    out = np.zeros_like(TIME)
    for i in range(1, len(TIME)):
        out[i] = out[i - 1] + 0.1 * (level[i - 1] - out[i - 1]) / 0.5
    return out


def test_an_event_is_the_change_since_its_window_opens():
    control = np.full(len(TIME), 350.0)
    level = np.where(TIME >= 8.0, 2.0, np.where(TIME >= 2.0, 1.0, 0.0))
    truth = control + 10 * _lagged(level)
    # Right up to the second level, then a fall where the truth rises.
    predicted = control + 10 * _lagged(np.where(TIME >= 8.0, 0.5, level))
    first = suite.event_metrics(
        TIME, truth, control, predicted, control, [2.0, 8.0], False
    )
    second = suite.event_metrics(
        TIME, truth, control, predicted, control, [8.0, 14.0], False
    )
    assert first["direction_correct"] is True and first["peak_delay"] is None
    assert first["event_rse"] == pytest.approx(0.0)
    assert second["true_peak"] == pytest.approx(10.0, rel=1e-3)
    assert second["direction_correct"] is False


def _root(tmp_path):
    """A scored toy root: one probe, one exact model and one failed rollout."""
    events = [{"name": "pulse", "window": [20.0, 30.0], "timing": True}]
    schedules = {
        CSTR: {
            "basal": {"role": "control", "control": None, "window": None},
            "probe": {
                "role": "contrast",
                "control": "basal",
                "window": [20.0, 30.0],
                "group": "B_time_scale",
                "events": events,
            },
            "dropped_probe": {
                "role": "contrast",
                "control": "basal",
                "window": [20.0, 30.0],
                "group": "A_range",
                "events": [],
            },
        }
    }
    subjects = [
        {"method": m, "repetition": 0, "source_id": f"{m}:0"}
        for m in ("exact", "broken")
    ]
    plan = sealed_write(
        tmp_path / "plan.json",
        {
            "schedules": schedules,
            "validity_rule": {"toy": True},
            "rows": subjects,
            "study_rows": [],
        },
    )
    digest = plan["artifact_sha256"]
    sealed_write(
        tmp_path / "validity.json", {"plan": digest, "dropped": ["dropped_probe"]}
    )
    control = np.full(len(TIME), 350.0)
    truth = control + 4 * _lagged_pulse(20.0, 1.0, 0.3)
    for name, curve in (("basal", control), ("probe", truth)):
        sealed_write(
            tmp_path / "references" / CSTR / f"{name}.json",
            {"plan": digest, "reference": {"time": TIME.tolist(), "T": curve.tolist()}},
        )
        sealed_write(
            tmp_path / "subjects" / "0000" / f"{name}.json",
            {"plan": digest, "curves": {"T": curve.tolist()}},
        )
    base = {"readout": "target", "quantity": "T", "case": "probe"}
    sealed_write(
        tmp_path / "scores.json",
        {
            "rows": [
                {
                    **base,
                    "subject": 0,
                    "status": "complete",
                    "effect_rse": 0.0,
                    "absolute_nmse": 0.0,
                },
                {**base, "subject": 1, "status": "rollout_unavailable"},
            ]
        },
    )
    return tmp_path


def test_failures_and_wrong_models_are_counted_as_exposed(tmp_path):
    evaluation = suite.evaluate(_root(tmp_path))
    assert evaluation["kept"] == ["probe"]
    rows = {r["method"]: r for r in evaluation["rows"]}
    assert rows["exact"]["exposed"] is False
    assert rows["exact"]["events"][0]["peak_delay"] == 0.0
    assert rows["broken"]["exposed"] is True
    assert rows["broken"]["reasons"] == ["failed: rollout_unavailable"]
    summary = {(s["method"], s["group"]): s for s in suite.summarize(evaluation)}
    assert summary[("broken", "all")]["median_response_rse"] == float("inf")
    assert summary[("exact", "B_time_scale")]["exposed"] == 0
    assert summary[("exact", "all")]["median_abs_peak_delay"] == {"pulse": 0.0}
