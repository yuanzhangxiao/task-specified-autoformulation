"""Independent residual checks, analytic sensitivities and logged optimizer behavior."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.stagnation import (
    RolloutOracle,
    inspect_steps,
    instrumented_fit,
)
from autoformalism.rebuttal.fitter_diagnostic import read_json
from tests.test_fitter_diagnostic import _candidate


@pytest.fixture
def problem(tmp_path):
    time = np.linspace(0, 3, 31)
    target = np.exp(-0.72 * time)
    model = compile_candidate(_candidate(("v01",)), ValidationContext(targets=("v01",)))
    trajectory = Trajectory("training", time, {"v01": target}, {}, {}, {}, {})
    training = DatasetSplit(SplitName.TRAIN, (trajectory,), "synthetic")
    settings = FitConfig(
        allow_derivative_regression=False,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
    )
    scale = float(np.std(target))
    oracle = RolloutOracle(
        model, training, {"v01": scale}, settings, tmp_path / "trace", None
    )
    return oracle, model, training, settings, scale


def test_oracle_matches_independent_rollout_and_counts_duplicate_calls(problem):
    oracle, model, training, settings, scale = problem
    point = oracle.vector({"decay": 1.0})
    simulation = simulate_trajectory(
        model, training.trajectories[0], {"decay": 1}, {}, settings
    )
    expected = (
        simulation.predictions["v01"] - training.trajectories[0].targets["v01"]
    ) / scale
    np.testing.assert_array_equal(oracle(point), expected)
    np.testing.assert_array_equal(oracle(point), expected)
    assert oracle.calls == 2
    assert len(list(oracle.directory.glob("[0-9]*.json"))) == 2
    assert read_json(oracle.directory / "000001.json")["integration_failures"] == 0


def test_profile_agrees_with_analytic_derivative_and_caches_each_repeat(
    problem, tmp_path
):
    oracle, _, training, _, scale = problem
    cache = tmp_path / "profile"
    report = inspect_steps(oracle, {"decay": 1.0}, (1e-4, 1e-3), cache, "identity")
    time = training.trajectories[0].time
    derivative = -time * np.exp(-time) / scale
    assert report["steps"][0]["central"]["column_norms"][0] == pytest.approx(
        np.linalg.norm(derivative), rel=1e-6
    )
    assert report["exact_repeat_residual_difference"] == 0
    assert oracle.calls == 6  # Two real repeats and both sides of each of two steps.
    assert (
        inspect_steps(oracle, {"decay": 1.0}, (1e-4, 1e-3), cache, "identity") == report
    )
    assert oracle.calls == 6
    with pytest.raises(ValueError, match="profile cache differs"):
        inspect_steps(oracle, {"decay": 1.0}, (1e-4, 1e-3), cache, "different")


def test_optimizer_recovers_decay_and_exposes_jacobian_calls(problem):
    oracle, *_ = problem
    report = instrumented_fit(oracle, {"decay": 1.0}, diff_step=1e-4, max_nfev=30)
    assert report["optimizer_success"]
    assert report["parameters"]["decay"] == pytest.approx(0.72, abs=1e-5)
    assert report["actual_residual_calls"] > report["nfev"]
    assert report["njev"] > 0
    assert report["optimality"] < 1e-6
    assert report["iterations"][-1]["distance_from_start"] > 0.2
    assert report["selection"] == "optimizer_return"


def test_timeout_preserves_best_evaluated_vector_without_claiming_convergence(problem):
    oracle, *_ = problem

    def timed_out(fun, start, **kwargs):
        fun(start)
        fun(np.array([0.72]))
        raise TimeoutError("synthetic timeout")

    report = instrumented_fit(
        oracle, {"decay": 1.0}, diff_step=None, max_nfev=30, optimizer=timed_out
    )
    assert not report["optimizer_success"]
    assert report["selection"] == "best_finite_evaluation_after_timeout"
    assert report["parameters"] == {"decay": 0.72}
    assert report["actual_residual_calls"] == 2
    assert report["nfev"] is None


def test_integration_failure_cannot_be_retained_as_best(problem):
    oracle, *_ = problem

    def failed(point):
        oracle.failures.record("synthetic integration failure")
        return np.ones(3)

    oracle.raw = failed
    oracle(np.array([1.0]))
    assert oracle.best is None
    assert read_json(oracle.directory / "000001.json")["integration_failures"] == 1


def test_invalid_profile_domain_fails_before_symmetric_probe(problem, tmp_path):
    oracle, *_ = problem
    with pytest.raises(ValueError, match="leaves frozen parameter domain"):
        inspect_steps(oracle, {"decay": 1e-9}, (1e-4, 1e-3), tmp_path / "cache", "id")
    with pytest.raises(ValueError, match="parameter vector differs"):
        oracle.vector({"wrong": 1.0})
