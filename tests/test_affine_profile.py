"""Matrix basis projection agrees with full rollouts and outer finite differences."""

from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import larger_coupled_inputs as controls
from autoformalism.fitting.affine_profile import AffineProfile
from autoformalism.fitting.affine_propagation import AffineRollouts


@pytest.fixture(scope="module")
def linear_inputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("profile-matrix") / "inputs.json"
    controls.export_inputs(path)
    return read_seal(path)


def make_problem(data, case="linear3"):
    b = data["commons"][case + "_s0"]
    return checks.TrainingProblem.model_validate(
        {
            "request": b["request"],
            "training": data["cases"][case]["training"],
            "coordinates": b["coordinates"],
            "start": b["start"],
            "incumbent": b["start"],
        }
    )


@pytest.mark.parametrize("case", list(controls.CASES))
def test_projection_full_rollout_and_derivatives(linear_inputs, case):
    problem = make_problem(linear_inputs, case)
    oracle = AffineRollouts(problem, checks.ConfidencePolicy())
    profile = AffineProfile(oracle)
    vector = oracle.vector(problem.start)
    fitted, r, j, audit = profile.evaluate(vector, monotonic() + 30)
    full, _ = oracle.evaluate(fitted, monotonic() + 30, jacobian=False)
    assert r == pytest.approx(full, abs=2e-12)
    assert (
        len(profile.inner)
        == len(profile.outer)
        == linear_inputs["cases"][case]["states"]
    )
    assert np.all(fitted >= oracle.lower) and np.all(fitted <= oracle.upper)
    step = np.zeros(len(vector))
    step[profile.outer] = np.arange(1, len(profile.outer) + 1) / 10
    h = 1e-5
    rp = profile.evaluate(vector + h * step, monotonic() + 30)[1]
    rm = profile.evaluate(vector - h * step, monotonic() + 30)[1]
    assert j @ step[profile.outer] == pytest.approx((rp - rm) / (2 * h), abs=2e-6)
    assert audit["derivative"] == "fixed-active-set exact residual derivative"
    with pytest.raises(TimeoutError):
        profile.evaluate(vector, monotonic() - 1)
