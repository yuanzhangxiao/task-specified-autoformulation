"""Analytic parameter derivatives, hidden paths and equivalent rollout checks."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.sensitivity_probe import (
    SensitivityContractError,
    SymbolicODE,
    symbolic_rollout,
)
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


def test_explosive_rollout_retains_factual_evidence_and_rejects_false_success(tmp_path):
    from time import monotonic

    from scipy.optimize import least_squares

    from autoformalism.data import DatasetSplit, SplitName
    from autoformalism.fitting.sensitivity_probe import SymbolicOracle
    from autoformalism.fitting.stagnation import instrumented_fit

    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "a*x**2"
    system = SymbolicODE(
        compile_candidate(
            CandidateModel.model_validate(payload), ValidationContext(targets=("v01",))
        )
    )
    training = DatasetSplit(SplitName.TRAIN, (trajectory(),), "synthetic-explosion")
    oracle = SymbolicOracle(
        system,
        training,
        1.0,
        settings(),
        tmp_path,
        monotonic() + 20,
        sensitivities=True,
    )

    def optimizer(fun, x, **kwargs):
        kwargs["jac"] = oracle.jacobian
        return least_squares(fun, x, **kwargs)

    report = instrumented_fit(
        oracle, {"a": 0.6}, diff_step=None, max_nfev=5, optimizer=optimizer
    )
    assert report["optimizer_native_success"] and not report["optimizer_success"]
    assert report["parameters"] is None
    assert report["valid_residual_evaluations"] == 0
    evidence = report["failure_evidence"][0]
    assert evidence["trajectory_id"] == "training"
    detail = evidence["diagnostic"]
    assert detail["integration_interval"] == [0.0, 4.0]
    assert not detail["attempt_is_accepted_solver_state"]
    assert 0.7 < detail["last_rhs_attempt"]["time"] < 0.9
    assert detail["last_rhs_attempt"]["state_rhs"]["x"] > 0
    assert "cause" not in detail


def test_rollout_timeout_keeps_deadline_exception_and_context():
    from time import monotonic

    from autoformalism.fitting.sensitivity_probe import SymbolicRolloutTimeout

    with pytest.raises(SymbolicRolloutTimeout) as caught:
        symbolic_rollout(
            SymbolicODE(decay_model()),
            trajectory(),
            np.array([0.6]),
            settings(),
            monotonic() - 1,
        )
    assert isinstance(caught.value, TimeoutError)
    assert caught.value.diagnostic["trajectory_id"] == "training"


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


def test_direct_observation_initial_state_is_trajectory_specific():
    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["initial_conditions"][0].update(fixed_value=None, expression="v01")
    system = SymbolicODE(
        compile_candidate(
            CandidateModel.model_validate(payload),
            ValidationContext(targets=("v01",)),
        )
    )
    source = trajectory()
    data = Trajectory(
        source.trajectory_id,
        source.time,
        {"v01": np.concatenate(([3.5], source.targets["v01"][1:]))},
        source.auxiliaries,
        source.external_inputs,
        source.fixed_covariates,
        source.derivatives,
    )
    predicted, jac, _, _ = symbolic_rollout(
        system, data, np.array([0.6]), settings(), None, sensitivities=True
    )
    expected = 3.5 * np.exp(-0.6 * data.time)
    assert system.audit["trajectory_specific_initialization"]
    assert system.audit["initial_parameter_sensitivity_zero_certified"]
    assert np.max(abs(predicted[:, 0] - expected)) < 1e-8
    assert np.max(abs(jac[:, 0, 0] + data.time * expected)) < 1e-8


def test_fitted_initial_state_and_nonsmooth_rhs_fail_closed():
    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["initial_conditions"][0].update(
        fixed_value=None,
        initialization_range={"lower": 0.0, "upper": 4.0},
    )
    with pytest.raises(SensitivityContractError, match="fitted initial"):
        SymbolicODE(
            compile_candidate(
                CandidateModel.model_validate(payload),
                ValidationContext(targets=("v01",)),
            )
        )
    payload = decay_model().validated.candidate.model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "-a*abs(x)"
    with pytest.raises(SensitivityContractError, match="nonsmooth"):
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
