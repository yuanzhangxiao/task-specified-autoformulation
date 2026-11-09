"""Checks for the conditional glucose rate and its memory-path intervention."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.schemas import CandidateModel
from scripts import evaluate_t1_meal_appearance as probe


def test_clamping_isolates_memory_and_fast_path(monkeypatch):
    monkeypatch.setattr(probe, "END", 12)
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "meal_filter",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "M", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "2*M + 3*meal_event_g - Gp"},
                {"state": "M", "rhs": "meal_event_g - M"},
            ],
            "observation_mappings": [{"channel": "Gp", "expression": "Gp"}],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "expression": "Gp"},
                {"state": "M", "scope": "global", "expression": "0"},
            ],
        }
    )
    model = compile_candidate(
        candidate, ValidationContext(targets=("Gp",), external_inputs=("meal_event_g",))
    )
    basal = {"Gp": 100, "Gt": 1, "EGP": 1, "Uii": 1, "E": 0}
    case, control = (
        probe.public_clamp("canonical_named_easy", basal, meals)
        for meals in (((4, 2),), ())
    )
    fast = probe.clamped_states(model, control, {}, {}, "Radau")
    fed = probe.clamped_states(model, case, {}, {}, "Radau")
    np.testing.assert_allclose(fed[0], 100)
    reference = probe.glucose_rate(model, control, {}, fast)
    native = probe.glucose_rate(model, case, {}, fed) - reference
    disabled = probe.glucose_rate(model, case, {}, fast) - reference
    np.testing.assert_allclose(
        native, 2 * fed[1] + 3 * case.external_inputs["meal_event_g"]
    )
    np.testing.assert_allclose(disabled, 3 * case.external_inputs["meal_event_g"])
    assert native[6] > 0 and disabled[6] == 0


def test_physical_units_and_absent_peak():
    time = np.arange(3, dtype=float)
    truth = np.array([0, 1, 0])
    score = probe.flux_scores(time, np.zeros(3), truth)
    assert score["response_rse"] == 1 and score["area_ratio"] == 0
    assert score["peak_time_error_min"] is None
    score = probe.flux_scores(time, truth * 2, truth)
    assert score["area_ratio"] == 2 and score["peak_time_error_min"] == 0
    with pytest.raises(ValueError, match="positive energy"):
        probe.flux_scores(time, truth, truth * 0)
    with pytest.raises(ValueError, match="aligned finite"):
        probe.flux_scores(time, np.full(3, np.nan), truth)


def test_reference_clamp_identifies_appearance_from_the_signed_glucose_rate():
    dm = probe.dalla_man
    parameters = dm.DallaManParameters()
    basal = dm.compute_dalla_man_basal(parameters)
    gut_filled = basal.initial_state.copy()
    gut_filled[dm.STATE_INDEX["Qgut"]] += 600
    control, _ = dm._rhs_and_derived(
        basal.initial_state, parameters, basal, 0, "original"
    )
    fed, q = dm._rhs_and_derived(gut_filled, parameters, basal, 0, "original")
    assert fed[dm.STATE_INDEX["Gp"]] - control[dm.STATE_INDEX["Gp"]] == pytest.approx(
        q["Ra"]
    )
    assert q["Ra"] == pytest.approx(
        parameters.f * parameters.kabs * 600 / parameters.BW
    )
