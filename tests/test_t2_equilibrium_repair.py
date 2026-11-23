"""Exercise the certificate-scoped repair, physical sign and response metrics."""

import casadi as ca
import pytest

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import t2_equilibrium_repair as exp


def inputs(slot):
    intercept, slope, drug, raw = exp.LAWS[slot]
    parameters = {intercept: 2.0, slope: 7.0, "other": 4.0}
    if drug:
        parameters[drug] = 1.5
    original = {
        "equations": {"I": raw, "Gp": "other*Gt"},
        "parameters": parameters,
        "initializers": {},
    }
    bounded = exp.source.boundary(original, "I", 22.0)
    ref = {"Gt": 130.0, "I": 25.5, exp.DRUG: 0.0}
    return original, bounded, parameters, ref


def rate(form, gt, insulin, dose, *, rho=None):
    parsed = RestrictedParser().parse(form["equations"]["I"], location="I")
    values = {**form["parameters"], "Gt": gt, "I": insulin, exp.DRUG: dose}
    if rho is not None:
        values[exp.RHO] = rho
    return float(_translate(ca, parsed, values))


@pytest.mark.parametrize("slot", [2, 15])
def test_equilibrium_and_restoring_signs_with_shared_fasting_reference(slot):
    original, bounded, parameters, ref = inputs(slot)
    intercept = exp.LAWS[slot][0]
    models = exp.candidates(slot, original, bounded, parameters, 22.0, ref)
    assert models["bounded"] == bounded
    for arm in ("equilibrium", "restoring"):
        f = models[arm]
        assert f["equations"]["Gp"] == original["equations"]["Gp"]
        assert intercept not in f["parameters"]
        assert rate(f, 130.0, 25.5, 0.0) == pytest.approx(0.0)
        assert rate(f, 100.0, 0.0, 0.0) >= 0
    eq, restoring = models["equilibrium"], models["restoring"]
    # At zero forcing and the reference Gt, equilibrium alone cannot change I.
    assert rate(eq, 130.0, 30.0, 0.0) == 0
    for rho in [-0.2, 0.2]:
        assert rate(restoring, 130.0, 30.0, 0.0, rho=rho) == pytest.approx(
            -0.04 * (30.0 - 25.5)
        )
        assert rate(restoring, 130.0, 10.0, 0.0, rho=rho) == pytest.approx(
            0.04 * (25.5 - 10.0)
        )
    # The nested baseline gives the same predictions even above the reference Gt.
    assert rate(eq, 145.0, 29.0, 0.1) == pytest.approx(
        rate(restoring, 145.0, 29.0, 0.1, rho=0.0)
    )
    assert exp.RHO not in original["parameters"]


def test_rejects_unqualified_expression_and_different_bounded_parent():
    orig, b, p, ref = inputs(2)
    with pytest.raises(ValueError, match="qualified source"):
        exp.candidates(
            2, orig | {"equations": {"I": "0", "Gp": "other*Gt"}}, b, p, 22.0, ref
        )
    with pytest.raises(ValueError, match="qualified source"):
        exp.candidates(2, orig, exp.source.boundary(orig, "I", 10), p, 22.0, ref)
    with pytest.raises(KeyError):
        exp.candidates(9, orig, b, p, 22.0, ref)


def test_peak_timing_tail_error_and_no_threshold_for_nonfasting_variant():
    rows = [
        {
            "trajectory_id": "a",
            "inputs": {exp.MEAL: [0, 0, 0, 0], exp.DRUG: [0, 0, 0, 0]},
        },
        {
            "trajectory_id": "b",
            "inputs": {exp.MEAL: [0, 1, 3, 0], exp.DRUG: [0, 0, 0, 0]},
        },
    ]
    curves = [
        {
            "trajectory_id": "a",
            "time": [0, 20, 120, 180],
            "observed": {"I": [2] * 4},
            "predicted": {"I": [2, 2, 3, 3]},
        },
        {
            "trajectory_id": "b",
            "time": [0, 20, 120, 180],
            "observed": {"I": [2, 10, 4, 2]},
            "predicted": {"I": [2, 3, 9, 4]},
        },
    ]
    result = exp.behavior(curves, rows, "a")
    assert result["events"][0]["peak_time_error_min"] is None
    assert result["median_abs_peak_time_error_min"] == 100.0
    assert result["maximum_unforced_drift_pmol_L"] == 1.0
    assert result["reference_event_prediction_drift_pmol_L"] == 1.0
    assert result["events"][1]["last_hour_mae_pmol_L"] == 3.5


def test_training_freeze_required_before_validation(tmp_path):
    root = tmp_path / "experiment"
    sealed_write(root / "plan.json", {"registered": True})
    with pytest.raises(FileNotFoundError, match=r"ranking\.json"):
        exp.evaluate(root, validation=tmp_path / "validation.json")
