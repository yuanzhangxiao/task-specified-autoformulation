"""Equation-derived affine propagation, initial sensitivities and rejection gates."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import larger_coupled_inputs as controls
from autoformalism.fitting.affine_propagation import AffineRollouts, verify_outputs


@pytest.fixture(scope="module")
def linear_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("affine") / "inputs.json"
    controls.export_inputs(path)
    return read_seal(path)


def problem(data, case="linear3", *, reference=False):
    base = data["commons"][case + "_s0"]
    return checks.TrainingProblem.model_validate(
        {
            "request": base["request"],
            "coordinates": base["coordinates"],
            "start": base["start"],
            "training": data["cases"][case]["training"],
            "incumbent": data["cases"][case]["reference_parameters"]
            if reference
            else base["start"],
        }
    )


@pytest.mark.parametrize("case", list(controls.CASES))
def test_affine_outputs_and_initial_derivatives(linear_data, case):
    p = problem(linear_data, case)
    oracle = AffineRollouts(p, checks.ConfidencePolicy())
    x = oracle.vector(p.incumbent)
    r, j = oracle.evaluate(x, monotonic() + 30)
    other, _ = oracle.evaluate(x, monotonic() + 30, jacobian=False, method="DOP853")
    assert r == pytest.approx(other, abs=2e-8)
    direction = np.arange(1, len(x) + 1) / len(x)
    h = 1e-5
    plus, _ = oracle.evaluate(x + h * direction, monotonic() + 10, jacobian=False)
    minus, _ = oracle.evaluate(x - h * direction, monotonic() + 10, jacobian=False)
    assert j @ direction == pytest.approx((plus - minus) / (2 * h), abs=1e-7)
    assert np.linalg.norm(j[:, -1]) > 0


@pytest.mark.parametrize("extra", ["+x0*x1", "+u*x1", "+u*u", "+t*x0"])
def test_nonconstant_or_nonlinear_dynamics_rejected(linear_data, extra):
    p = problem(linear_data).model_dump(mode="json")
    p["request"]["base_candidate"]["state_equations"][0]["rhs"] += extra
    with pytest.raises(ValueError, match="autonomous jointly affine"):
        AffineRollouts(
            checks.TrainingProblem.model_validate(p), checks.ConfidencePolicy()
        )


def test_verification_skips_sensitivities_and_obeys_deadline(linear_data, monkeypatch):
    p = problem(linear_data, reference=True)
    oracle = AffineRollouts(p, checks.ConfidencePolicy())
    actual = oracle.evaluate
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["jacobian"] is False
        return actual(*args, **kwargs)

    monkeypatch.setattr(oracle, "evaluate", evaluate)
    checked = verify_outputs(oracle, p.incumbent, monotonic() + 30, lambda _: None)
    assert len(calls) == 2 and checked["training_nmse"] < 1e-25
    with pytest.raises(TimeoutError):
        actual(oracle.vector(p.incumbent), monotonic() - 1)
    bad = deepcopy(p.incumbent)
    bad["a0"] = -1
    with pytest.raises(ValueError, match="outside declared domain"):
        actual(oracle.vector(bad), monotonic() + 1)
