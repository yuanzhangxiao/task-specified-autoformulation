"""Exploratory interventions must preserve physical and scoring semantics."""

import numpy as np
import pytest

from scripts import probe_t1_interventions as probe


def test_suite_includes_controls_both_directions_and_fixed_meal_mass():
    cases = {case["id"]: case for case in probe.cases()}
    assert len(cases) == 7
    for case in cases.values():
        if case["paired_control"]:
            assert case["paired_control"] in cases
    assert cases["fasting_gt80"]["initial_gt_multiplier"] == 0.8
    assert cases["fasting_gt120"]["initial_gt_multiplier"] == 1.2
    assert sum(m[1] for m in cases["meal_gt100"]["meals"]) == sum(
        m[1] for m in cases["split_meal_gt100"]["meals"]
    )


def test_effect_error_distinguishes_correct_zero_and_failed_predictions():
    control = {"success": True, "observed": [3, 3], "predicted": [5, 5]}
    zero = {"success": True, "observed": [3, 7], "predicted": [5, 5]}
    correct = {"success": True, "observed": [3, 7], "predicted": [5, 9]}
    assert probe.effect_error(zero, control)["relative_squared_error"] == 1
    assert probe.effect_error(correct, control)["relative_squared_error"] == 0
    assert probe.effect_error(control, control)["relative_squared_error"] is None
    assert probe.effect_error({"success": False}, control)["status"] == "rollout_failed"


def test_gt_preparation_preserves_other_initials_and_reference_equations():
    p = probe.reference.DallaManParameters()
    basal = probe.reference.compute_dalla_man_basal(p)
    changed = basal.initial_state.copy()
    i = probe.reference.STATE_INDEX["Gt"]
    changed[i] *= 1.2
    old_rhs = probe.reference._rhs_and_derived(
        basal.initial_state, p, basal, 0, "original"
    )[0]
    new_rhs = probe.reference._rhs_and_derived(changed, p, basal, 0, "original")[0]
    assert new_rhs[0] - old_rhs[0] == pytest.approx(
        p.k2 * (changed[i] - basal.initial_state[i])
    )
    plan = {
        "reference_parameters": probe.asdict(p),
        "reference_rtol": 1e-10,
        "reference_atol": 1e-12,
        "reference_max_step": 0.5,
        "duration": 5,
        "dt": 1,
    }
    row = probe.generate_reference(probe.cases()[2], plan, "Radau")
    assert row["targets"]["v01"][0] == pytest.approx(basal.initial_state[0])
    assert row["auxiliaries"]["v05"][0] == pytest.approx(changed[i])
    assert row["auxiliaries"]["v02"][0] == pytest.approx(basal.EGPb)
    assert np.isfinite(row["targets"]["v01"]).all()
    assert row["targets"]["v01"][-1] > row["targets"]["v01"][0]
