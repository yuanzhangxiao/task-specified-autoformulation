"""Data-free tests for the two-compartment diagnostic and term accounting."""

import numpy as np
import pytest

from scripts import explain_t2_initial_disposal as attribution


def test_joint_compartment_closure_only_uses_zero_intake_prefix():
    row = {
        "time": [0, 1, 2, 3],
        "inputs": {"meal_rate_g_per_min": [0, 0, 0, 1]},
        "targets": {"Gp": [2, 3, 4, 999], "U": [1, 1, 1, 10]},
        "auxiliaries": {
            "Gt": [5, 4.5, 4, 999],
            "EGP": [2, 2, 2, 9],
            "E": [0.5] * 4,
        },
    }
    result = attribution.observed_storage_balance(row)
    assert result["status"] == "pre_ingestion"
    assert result["last_minute"] == 2
    assert result["max_abs_plasma_error"] == 1
    assert result["max_abs_total_error"] == 0
    row["inputs"]["meal_rate_g_per_min"][0] = 1
    assert attribution.observed_storage_balance(row)["samples"] == 0


def test_flow_errors_can_cancel_exactly():
    time = np.arange(4)
    gp = [2.0, 3.0, 4.0, 5.0]
    b = attribution.glucose_budget(time, 3, [1.5] * 4, gp, gp, [0.5] * 4, [1.0] * 4)
    assert b["source_area_error"] == b["disposal_area_error"] == -1.5
    with pytest.raises(ValueError, match="recorded sample"):
        attribution.glucose_budget(time, 5, [1.5] * 4, gp, gp, [0.5] * 4, [1.0] * 4)


def test_disposal_term_closure_and_mismatch(monkeypatch):
    t = np.arange(3, dtype=float)
    monkeypatch.setattr(
        attribution.fitting,
        "_scalar_insulin",
        lambda row, knots, law, time: (np.ones(3), time.copy()),
    )
    p = {
        "afcoef9": 1.0,
        "afcoef10": 2.0,
        "afcoef11": 1.0,
        "afcoef12": 0.25,
        "afcoef13": 2.0,
        "afcoef14": 1.0,
        "afcoef15": 0.0,
        "afcoef16": 0.0,
        "memory_rate": 0.2,
        "disp_tissue": 0.0,
        "init_tissue_gain": 3.0,
    }
    model = {
        "formula": {"parameters": p},
        "centers": {"Gp": 0, "I": 0, "Gt_ref": 100},
        "scales": {"I": 1, "Gp": 1},
        "knots": {},
        "scalar_model": {},
    }
    row = {"time": t, "auxiliaries": {"Gt": [110, 102, 100], "Uii": [1, 1, 1]}}
    u = 2 + 6 * t + 0.6 * np.exp(-0.2 * t)
    pred = {"Gp": [4.0, 4.0, 4.0], "I": [1, 1, 1], "U": u}
    parts = attribution.disposal_terms(model, row, pred)
    np.testing.assert_allclose(parts["action_initial_tissue"], 0.6 * np.exp(-0.2 * t))
    np.testing.assert_allclose(parts["action_insulin_drive"], 4 * t)
    pred["U"] = u + 1
    with pytest.raises(ValueError, match="sum differs"):
        attribution.disposal_terms(model, row, pred)
