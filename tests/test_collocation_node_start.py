"""Collocation can optimize finite guesses even when the starting ODE blows up."""

from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import matching_probe as probe
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas import CandidateModel


def problem(observed=True):
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "initial_rollout_blowup",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": "a*x**2"}],
            "observation_mappings": [
                {"channel": "v01", "expression": "x" if observed else "2*x"}
            ],
            "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
            "initial_conditions": [
                {"state": "x", "scope": "global", "fixed_value": 1.0}
            ],
        }
    )
    model = compile_candidate(candidate, ValidationContext(targets=("v01",)))
    splits = []
    for name, end in ((SplitName.TRAIN, 2.0), (SplitName.VALIDATION, 2.3)):
        time = np.linspace(0, end, 41)
        y = (1.0 if observed else 2.0) / (1 - 0.1 * time)
        row = Trajectory(name.value, time, {"v01": y}, {}, {}, {}, {})
        splits.append(DatasetSplit(name, (row,), name.value))
    return model, *splits


@pytest.mark.parametrize("observed", [True, False])
def test_fallback_recovers_from_actual_blowup(observed, tmp_path):
    model, training, validation = problem(observed)
    common = {
        "initializer_seconds": 20,
        "refinement_seconds": 15,
        "maximum_function_evaluations": 30,
    }
    old = fit_collocation_forward_sensitivity(
        model,
        training,
        validation,
        CollocationSensitivityConfig(**common),
        tmp_path / "old",
        initial_parameters={"a": 1.0},
    )
    assert old["status"] == "fit_failed"
    assert not old["initializer"]["collocation_optimizer_started"]
    result = fit_collocation_forward_sensitivity(
        model,
        training,
        validation,
        CollocationSensitivityConfig(
            **common, collocation_node_start="rollout_or_observed"
        ),
        tmp_path / "fallback",
        initial_parameters={"a": 1.0},
    )
    assert result["status"] == "complete", result
    assert result["initializer"]["success"]
    assert result["initializer"]["collocation_optimizer_started"]
    assert (
        result["initializer"]["node_initialization"][0]["source"]
        == "observed_and_fixed_initials"
    )
    assert result["parameters"]["a"] == pytest.approx(0.1, abs=1e-6)
    assert result["validation"]["normalized_mse"] < 1e-9
    assert not result["initializer"]["initial_conditions_optimized"]


def test_optional_rollout_keeps_successful_guess_and_does_not_read_validation(
    monkeypatch,
):
    model, training, _ = problem()
    system = SymbolicODE(model)
    config = CollocationSensitivityConfig().fit_config()
    guess, record = probe.collocation_node_guess(
        system,
        training.trajectories[0],
        np.array([0.1]),
        config,
        "rollout_or_observed",
        monotonic() + 5,
    )
    assert record["source"] == "rollout"
    assert guess[-1, 0] == pytest.approx(1.25, rel=1e-6)


def test_shared_warmup_expiry_uses_finite_guesses_without_ode(monkeypatch):
    model, training, _ = problem(False)
    system = SymbolicODE(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm-up must not exceed its shared allowance")

    monkeypatch.setattr(probe, "symbolic_rollout", forbidden)
    guess, record = probe.collocation_node_guess(
        system,
        training.trajectories[0],
        np.array([1.0]),
        CollocationSensitivityConfig().fit_config(),
        "rollout_or_observed",
        monotonic() - 1,
    )
    assert np.all(guess == 1.0)
    assert record["source"] == "observed_and_fixed_initials"


def test_fixed_initial_overrides_first_observed_guess():
    model, training, _ = problem()
    row = training.trajectories[0]
    observed = row.targets["v01"].copy()
    observed[0] = 1.4
    row = Trajectory(row.trajectory_id, row.time, {"v01": observed}, {}, {}, {}, {})
    guess = probe.observed_node_guess(SymbolicODE(model), row)
    assert guess[0, 0] == 1.0
    assert guess[1, 0] == row.targets["v01"][1]


def test_nonfinite_local_equations_fail_without_rollout():
    model, _, _ = problem()
    with pytest.raises(ValueError, match="nonfinite"):
        probe.check_node_guess(
            SymbolicODE(model), 0.0, np.array([1e200]), np.array([1.0]), []
        )
