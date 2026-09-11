"""Exact threshold values, branch selections, polling and mesh recovery."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.directional_poll import poll_fit
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.rebuttal.fitter_diagnostic import read_json
from autoformalism.schemas import CandidateModel


def model(rhs="a", observation="max(x-10,0)"):
    return compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "piecewise_control",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [{"state": "x", "rhs": rhs}],
                "observation_mappings": [{"channel": "v01", "expression": observation}],
                "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 9}
                ],
            }
        ),
        ValidationContext(targets=("v01",)),
    )


def data(name=SplitName.TRAIN, end=4):
    t = np.linspace(0, end, 21)
    row = Trajectory(name.value, t, {"v01": np.maximum(t - 1, 0)}, {}, {}, {}, {})
    return DatasetSplit(name, (row,), name.value + str(end))


@pytest.mark.parametrize(
    "expression,slopes",
    [
        ("max(x-10,0)", [0, 0.5, 1]),
        ("min(x-10,0)", [1, 0.5, 0]),
        ("abs(x-10)", [-1, 0, 1]),
        ("min(max(x-10,0),2)", [0, 0.5, 1]),
    ],
)
def test_branch_values_and_documented_ties(expression, slopes):
    system = SymbolicODE(model(observation=expression), allow_piecewise=True)
    for x, slope in zip((9.0, 10.0, 11.0), slopes, strict=True):
        assert float(system.local(0, [x], [1], [])[2]) == slope
    assert system.has_piecewise
    assert not system.audit["classical_sensitivity_claimed"]
    assert system.audit["collocation_hessian"] == "limited-memory"


def test_piecewise_rhs_crossing_branch_sensitivity_matches_perturbed_rollouts():
    compiled = model("a - 0.2*max(x-10,0)", "x")
    system = SymbolicODE(compiled, allow_piecewise=True)
    assert system.augmented_jacobian is None
    row = data().trajectories[0]
    settings = FitConfig(
        integration_method="Radau", relative_tolerance=1e-10, absolute_tolerance=1e-12
    )
    values, jac, states, _ = symbolic_rollout(
        system,
        row,
        np.array([1.0]),
        settings,
        None,
        sensitivities=True,
    )
    assert states[0, 0] < 10 < states[-1, 0]
    for step in (1e-3, 1e-4):
        ys = []
        for a in (1 + step, 1 - step):
            result = simulate_trajectory(
                compiled, row, {"a": a}, {}, settings, reset_observed_states=False
            )
            assert result.success
            ys.append(result.predictions["v01"])
        difference = (ys[0] - ys[1]) / (2 * step)
        crossing = np.isclose(row.time, 1.0)
        np.testing.assert_allclose(
            jac[~crossing, 0, 0], difference[~crossing], atol=2e-5
        )
        # At the crossing the flow is C1, not C2: central error is O(h).
        np.testing.assert_allclose(jac[crossing, 0, 0], 1.0, atol=1e-8)
        assert np.max(abs(difference[crossing] - 1)) < 0.06 * step + 1e-6
    production = simulate_trajectory(
        compiled, row, {"a": 1.0}, {}, settings, reset_observed_states=False
    )
    np.testing.assert_allclose(values[:, 0], production.predictions["v01"], atol=1e-7)


class ToyOracle:
    names = ("x",)
    lower, upper = np.array([-20.0]), np.array([20.0])

    def __init__(self, interrupt=None, fail=False):
        self.failures = SimpleNamespace(count=0)
        self.calls = 0
        self.interrupt, self.fail = interrupt, fail

    def vector(self, start):
        return np.array([start["x"]])

    def __call__(self, x):
        self.calls += 1
        if self.calls == self.interrupt:
            raise KeyboardInterrupt()
        if self.fail:
            self.failures.count += 1
            return np.array([1.0])  # A finite penalty must never be selected.
        return np.array([max(x[0] - 10, 0) - 1])


def run_poll(path, oracle, identity="test"):
    return poll_fit(
        oracle,
        [{"x": 10.0}],
        scales=np.array([1.0]),
        max_calls=90,
        seconds=10,
        checkpoint=path,
        identity=identity,
    )


def test_poll_escapes_tie_and_resume_preserves_exact_point_sequence(tmp_path):
    baseline = run_poll(tmp_path / "baseline.json", ToyOracle())
    with pytest.raises(KeyboardInterrupt):
        run_poll(tmp_path / "resume.json", ToyOracle(interrupt=7))
    resumed = run_poll(tmp_path / "resume.json", ToyOracle())
    assert resumed["parameters"] == baseline["parameters"] == {"x": 11.0}
    assert resumed["cost"] == 0
    assert not resumed["optimizer_success"] and not resumed["stationarity_claimed"]
    assert (
        read_json(tmp_path / "resume.json")["history"]
        == read_json(tmp_path / "baseline.json")["history"]
    )
    with pytest.raises(ValueError, match="identity"):
        run_poll(tmp_path / "resume.json", ToyOracle(), "changed")
    complete = ToyOracle()
    run_poll(tmp_path / "resume.json", complete)
    assert complete.calls == 0


def test_poll_all_failure_penalties_do_not_become_fitted_parameters(tmp_path):
    result = run_poll(tmp_path / "failed.json", ToyOracle(fail=True))
    assert result["parameters"] is None and result["cost"] is None
    assert result["valid_residual_evaluations"] == 0


@pytest.mark.parametrize("mesh", [1, 2])
def test_exact_threshold_fit_recovers_from_inactive_start_and_refines_mesh(
    tmp_path, mesh
):
    result = fit_collocation_forward_sensitivity(
        model(),
        data(),
        data(SplitName.VALIDATION, 5),
        CollocationSensitivityConfig(
            initializer_seconds=15,
            refinement_seconds=20,
            maximum_function_evaluations=160,
            collocation_mesh_substeps=mesh,
            collocation_node_start="rollout_or_observed",
        ),
        tmp_path,
        initial_parameters={"a": 0.1},
    )
    assert result["status"] == "complete", result
    assert result["initializer"]["collocation_optimizer_started"], result
    assert result["initializer"]["latent_decision_variables"] == 40 * mesh
    assert result["validation"]["normalized_mse"] < 1e-7, result
    assert not result["forward_sensitivity_jacobian_used"]
    assert result["refinement"]["production_training_rollout_verified"]
    assert not result["validation_used_for_fitting"]
