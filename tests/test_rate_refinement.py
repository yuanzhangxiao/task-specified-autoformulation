"""Direct rate derivatives, domain preservation and the extreme-tau regression."""

from time import monotonic

import numpy as np
import pytest
from scipy.optimize import least_squares

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.rate_refinement import RateCoordinates
from autoformalism.fitting.sensitivity_probe import (
    SymbolicODE,
    SymbolicOracle,
    symbolic_rollout,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from autoformalism.schemas import CandidateModel
from tests.test_sensitivity_probe import settings, trajectory


def test_direct_rates_preserve_rollout_and_have_boundary_derivative(tmp_path):
    physical = compile_candidate(recovery_candidate(), CONTEXT)
    mapped = RateCoordinates(physical)
    params = dict.fromkeys(physical.parameter_names, 1.0)
    params.update(c=-0.8, tau_p=np.finfo(float).max)
    beta = mapped.start(params)
    row = trajectory(True)
    system = SymbolicODE(mapped.model)
    x = np.array([beta[n] for n in system.names])
    y, jac, _, _ = symbolic_rollout(
        system, row, x, settings(), None, sensitivities=True
    )
    old = symbolic_rollout(
        SymbolicODE(physical),
        row,
        np.array([params[n] for n in physical.parameter_names]),
        settings(),
        None,
    )[0]
    np.testing.assert_allclose(y, old, atol=1e-7)
    col = system.names.index("rate_tau_p")
    shifted = x.copy()
    shifted[col] += 1e-5
    fd = (symbolic_rollout(system, row, shifted, settings(), None)[0] - y) / 1e-5
    assert np.max(abs(jac[:, 0, col])) > 0.01
    np.testing.assert_allclose(jac[:, :, col], fd, atol=2e-4, rtol=2e-4)
    view = mapped.physical_view(beta, 24)
    assert view["parameters"]["tau_p"] is None
    assert view["parameters"]["c"] == -0.8
    train = DatasetSplit(SplitName.TRAIN, (row,), "boundary")
    layout = RolloutOracle(physical, train, {"v01": 1}, settings(), tmp_path, None)
    low, high = mapped.bounds(layout.lower, layout.upper)
    assert 0 < low[col] < 1e-300
    assert low[system.names.index("c")] == -np.inf
    assert high[col] == 1e12


def test_rate_optimizer_escapes_saved_extreme_time_constant(tmp_path):
    payload = {
        "candidate_id": "rate_regression",
        "parent_candidate_id": None,
        "states": [{"name": "x", "kind": "latent"}],
        "state_equations": [{"state": "x", "rhs": "-x/tau"}],
        "observation_mappings": [{"channel": "v01", "expression": "x+c"}],
        "parameters": [
            {"name": "tau", "role": "time_constant", "scope": "global"},
            {"name": "c", "role": "offset", "scope": "global"},
        ],
        "initial_conditions": [{"state": "x", "fixed_value": 2, "scope": "global"}],
    }
    physical = compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
    mapped = RateCoordinates(physical)
    row = trajectory()
    row = Trajectory(
        row.trajectory_id,
        row.time,
        {"v01": 2 * np.exp(-0.6 * row.time) - 0.4},
        {},
        {},
        {},
        {},
    )
    train = DatasetSplit(SplitName.TRAIN, (row,), "decay")
    oracle = SymbolicOracle(
        SymbolicODE(mapped.model),
        train,
        1,
        settings(),
        tmp_path / "calls",
        monotonic() + 20,
        sensitivities=True,
    )
    layout = RolloutOracle(
        physical, train, {"v01": 1}, settings(), tmp_path / "layout", None
    )
    oracle.lower, oracle.upper = mapped.bounds(layout.lower, layout.upper)

    def optimizer(fun, x, **kwargs):
        kwargs["jac"] = oracle.jacobian
        return least_squares(fun, x, **kwargs)

    report = instrumented_fit(
        oracle,
        mapped.start({"tau": np.finfo(float).max, "c": 1.5}),
        diff_step=None,
        max_nfev=50,
        optimizer=optimizer,
    )
    assert report["cost"] < 1e-12, report
    assert report["parameters"]["rate_tau"] == pytest.approx(0.6, abs=1e-5)
    assert report["actual_residual_calls"] > 2


def test_rewrite_preserves_guard_and_rejects_other_uses():
    payload = recovery_candidate().model_dump(mode="json")
    model = compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
    mapped = RateCoordinates(model)
    params = dict.fromkeys(model.parameter_names, 1.0)
    params["tau"] = 1e-16
    assert mapped.start(params)["rate_tau"] == 1e12
    payload["state_equations"][0]["rhs"] = "-m*tau+k_u*u01"
    with pytest.raises(ValueError, match="simple denominator"):
        RateCoordinates(
            compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
        )
    with pytest.raises(ValueError, match="invalid physical"):
        mapped.start({**params, "tau": 0})
    with pytest.raises(ValueError, match="positive and finite"):
        mapped.physical_view({"rate_tau": 0}, 24)


def test_trusted_bounds_and_signed_offsets_survive_mapping(tmp_path):
    payload = recovery_candidate().model_dump(mode="json")
    for p in payload["parameters"]:
        if p["name"] == "tau":
            p["bounds"] = {"lower": 0.5, "upper": 8}
        elif p["name"] == "k_p":
            p["bounds"] = {"lower": 0.2, "upper": 3}
    physical = compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
    mapped = RateCoordinates(physical)
    train = DatasetSplit(SplitName.TRAIN, (trajectory(True),), "bounds")
    layout = RolloutOracle(physical, train, {"v01": 1}, settings(), tmp_path, None)
    low, high = mapped.bounds(layout.lower, layout.upper)
    assert low[mapped.names.index("rate_tau")] == 0.125
    assert high[mapped.names.index("rate_tau")] == 2
    assert low[mapped.names.index("k_p")] == 0.2
    assert high[mapped.names.index("k_p")] == 3
    assert low[mapped.names.index("c")] == -np.inf


def test_internal_rate_name_collision_fails_closed():
    payload = recovery_candidate().model_dump(mode="json")
    payload["parameters"].append(
        {"name": "rate_tau", "role": "offset", "scope": "global"}
    )
    payload["processes"][0]["expression"] += "+rate_tau"
    with pytest.raises(ValueError, match="symbol collision"):
        RateCoordinates(
            compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
        )
