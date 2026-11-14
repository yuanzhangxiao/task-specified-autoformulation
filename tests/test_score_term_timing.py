"""Event timing of declared mechanisms, measured from the sample before a pulse."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import report_intervention_suite as suite
from scripts import score_term_timing as timing

TIME = np.round(np.arange(0.0, 30.05, 0.1), 10)
WINDOW = [20.0, 30.0]


def _filled(start, width, tau):
    """A state filled by a unit pulse through a first-order lag."""
    on = ((start <= TIME) & (start + width > TIME)).astype(float)
    out = np.zeros_like(TIME)
    for i in range(1, len(TIME)):
        out[i] = out[i - 1] + 0.1 * (on[i - 1] - out[i - 1]) / tau
    return out


def test_a_term_reading_the_input_is_measured_from_before_the_pulse():
    control = np.zeros(len(TIME))
    truth = 5 * _filled(20.0, 0.3, 0.3)
    box = 5 * ((TIME >= 20.0) & (TIME < 20.3)).astype(float)
    result = timing.event_timing(TIME, truth, control, box, control, WINDOW)
    assert result["direction_correct"]
    assert result["predicted_peak_time"] < 20.3
    # Measured from the window's first sample, the jump vanishes and the drop
    # at the pulse's end reads as a response of the wrong sign.
    legacy = suite.event_metrics(TIME, truth, control, box, control, WINDOW, True)
    assert not legacy["direction_correct"]


def test_a_lagged_term_follows_the_truth_more_closely_than_a_box():
    control = np.zeros(len(TIME))
    truth = 5 * _filled(20.0, 0.3, 0.3)
    lagged = 5 * _filled(20.0, 0.3, 0.25)
    box = 5 * ((TIME >= 20.0) & (TIME < 20.3)).astype(float)
    close = timing.event_timing(TIME, truth, control, lagged, control, WINDOW)
    instant = timing.event_timing(TIME, truth, control, box, control, WINDOW)
    assert close["event_rse"] < 0.1 < instant["event_rse"]
    assert close["peak_error"] == pytest.approx(0.0)


def test_the_shape_rse_ignores_size_but_not_timing():
    control = np.zeros(len(TIME))
    truth = 5 * _filled(20.0, 0.3, 0.3)
    half = 0.5 * truth
    result = timing.event_timing(TIME, truth, control, half, control, WINDOW)
    assert result["event_rse"] == pytest.approx(0.25)
    assert result["best_scale"] == pytest.approx(2.0)
    assert result["shape_rse"] == pytest.approx(0.0, abs=1e-12)
    box = 5 * ((TIME >= 20.0) & (TIME < 20.3)).astype(float)
    instant = timing.event_timing(TIME, truth, control, box, control, WINDOW)
    assert instant["shape_rse"] > 0.1
    silent = timing.event_timing(TIME, truth, control, control, control, WINDOW)
    assert silent["shape_rse"] == pytest.approx(1.0)
    assert silent["best_scale"] == 0.0


def test_a_continuous_response_peaks_at_the_same_time_under_both_rules():
    control = np.full(len(TIME), 350.0)
    truth = control + 3 * _filled(20.0, 0.3, 0.4)
    predicted = control + 3 * _filled(20.0, 0.3, 0.1)
    result = timing.event_timing(TIME, truth, control, predicted, control, WINDOW)
    legacy = suite.event_metrics(TIME, truth, control, predicted, control, WINDOW, True)
    assert result["peak_error"] == pytest.approx(legacy["peak_delay"])


def test_a_window_needs_a_sample_before_it():
    flat = np.zeros(len(TIME))
    with pytest.raises(ValueError, match="sample before"):
        timing.event_timing(TIME, flat, flat, flat, flat, [0.0, 5.0])
