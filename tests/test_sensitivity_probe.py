"""Analytic parameter derivatives, hidden paths and equivalent rollout checks."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from autoformalism.schemas import CandidateModel


def decay_model():
    return compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "decay_sensitivity",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [{"state": "x", "rhs": "-a*x"}],
                "observation_mappings": [{"channel": "v01", "expression": "x"}],
                "parameters": [{"name": "a", "role": "rate", "scope": "global"}],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 2.0}
                ],
            }
        ),
        ValidationContext(targets=("v01",)),
    )


def trajectory(inputs=False):
    time = np.linspace(0, 4, 21)
    return Trajectory(
        "training",
        time,
        {"v01": np.zeros(len(time))},
        {},
        {"u01": np.where(time < 1, 0.0, 0.7)} if inputs else {},
        {},
        {},
    )


def settings():
    return FitConfig(
        integration_method="Radau", relative_tolerance=1e-9, absolute_tolerance=1e-11
    )


def test_affine_rhs_has_nonlinear_rollout_sensitivity():
    system = SymbolicODE(decay_model())
    data = trajectory()
    theta = np.array([0.6])
    predicted, jac, _, _ = symbolic_rollout(
        system, data, theta, settings(), None, sensitivities=True
    )
    assert system.affine and system.rhs_affine
    expected = 2 * np.exp(-0.6 * data.time)
    assert np.max(abs(predicted[:, 0] - expected)) < 1e-8
    assert np.max(abs(jac[:, 0, 0] + data.time * expected)) < 1e-8
    fx, fp, _, _ = system.local(0, [2], theta, [])
    assert float(fx) == -0.6 and float(fp) == -2


def test_sensitivity_includes_hidden_states_and_algebraic_feedback():
    model = compile_candidate(recovery_candidate(), CONTEXT)
    system = SymbolicODE(model)
    theta = np.array([1.0] * 7)
    data = trajectory(True)
    assert not system.rhs_affine
    value, jac, _, _ = symbolic_rollout(
        system, data, theta, settings(), None, sensitivities=True
    )
    params = dict(zip(system.names, theta, strict=True))
    production = simulate_trajectory(
        model, data, params, {}, settings(), reset_observed_states=False
    )
    assert production.success
    assert np.max(abs(value[:, 0] - production.predictions["v01"])) < 1e-7
    for col in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[col] += 1e-4
        minus[col] -= 1e-4
        high = symbolic_rollout(system, data, plus, settings(), None)[0][:, 0]
        low = symbolic_rollout(system, data, minus, settings(), None)[0][:, 0]
        assert np.max(abs(jac[:, 0, col] - (high - low) / 2e-4)) < 1e-5


def test_unsupported_initial_state_and_nonsmooth_rhs_fail_closed():
    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["initial_conditions"][0].update(fixed_value=None, expression="v01")
    with pytest.raises(ValueError, match="fixed numeric"):
        SymbolicODE(
            compile_candidate(
                CandidateModel.model_validate(payload),
                ValidationContext(targets=("v01",)),
            )
        )
    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "-a*abs(x)"
    with pytest.raises(ValueError, match="nonsmooth"):
        SymbolicODE(
            compile_candidate(
                CandidateModel.model_validate(payload),
                ValidationContext(targets=("v01",)),
            )
        )


def test_sensitivity_deadline_is_enforced():
    with pytest.raises(TimeoutError):
        symbolic_rollout(
            SymbolicODE(decay_model()),
            trajectory(),
            np.array([0.6]),
            settings(),
            0,
            sensitivities=True,
        )
