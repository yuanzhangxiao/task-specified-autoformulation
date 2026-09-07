"""Numerical step domains, real recovery, timeout preservation and timing parity."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.fitting import fit_candidate, fitter, simulate_trajectory
from autoformalism.fitting.numerical import (
    ScaledJacobian,
    TrackedResidual,
    bounded_step,
)
from autoformalism.fitting.simulation import RolloutProfile
from tests.test_stagnation_objective import problem  # noqa: F401


def test_scaled_jacobian_handles_small_parameters_and_reuses_base():
    function = TrackedResidual(
        lambda x: np.array([3 * x[0] + 2 * x[1], x[1] ** 2]), lambda: 0
    )
    x = np.array([0.000245, 0.5])
    function(x)
    jac = ScaledJacobian(function, np.zeros(2), np.ones(2), 1e-4, 1.0)
    result = jac(x)
    np.testing.assert_allclose(result, [[3, 2], [0, 1.0001]], atol=1e-9)
    assert function.calls == 3
    assert jac.last_steps[0] == pytest.approx(1e-4)


@pytest.mark.parametrize(
    "x,lo,hi,h,expected",
    [
        (0.0, 0.0, 1.0, 1e-4, 1e-4),
        (1.0, 0.0, 1.0, 1e-4, -1e-4),
        (0.00001, 0.0, 1.0, 1e-4, 1e-4),
        (0.5, 0.0, 1.0, 2.0, 0.5),
    ],
)
def test_probes_never_leave_parameter_domain(x, lo, hi, h, expected):
    step = bounded_step(x, lo, hi, h)
    assert step == pytest.approx(expected)
    assert lo <= x + step <= hi


@pytest.mark.parametrize(
    "value,lo,hi,h", [(2, 0, 1, 0.1), (0, 0, 0, 0.1), (0, 0, 1, 0), (0, 0, 1, np.nan)]
)
def test_invalid_probe_contract_fails(value, lo, hi, h):
    with pytest.raises(ValueError):
        bounded_step(value, lo, hi, h)


def test_scaled_production_fit_recovers_known_parameter(problem):  # noqa: F811
    _, model, training, settings, _ = problem
    validation = DatasetSplit(SplitName.VALIDATION, training.trajectories, "validation")
    config = settings.model_copy(
        update={
            "finite_difference_policy": "scaled",
            "finite_difference_step": 1e-4,
        }
    )
    result = fit_candidate(
        model, training, validation, config, initial_global_parameters={"decay": 1.0}
    )
    assert result.success
    assert result.global_parameters["decay"] == pytest.approx(0.72, abs=1e-5)
    assert result.diagnostics[0].actual_residual_evaluations > 0


def test_production_timeout_retains_real_improvement_but_remains_failed(
    problem,  # noqa: F811
    monkeypatch,
):
    _, model, training, settings, _ = problem
    validation = DatasetSplit(SplitName.VALIDATION, training.trajectories, "validation")

    def interrupted(fun, x, **kwargs):
        fun(x)
        fun(np.array([0.72]))
        raise TimeoutError("wall-clock limit reached")

    monkeypatch.setattr(fitter, "least_squares", interrupted)
    result = fit_candidate(
        model, training, validation, settings, initial_global_parameters={"decay": 1.0}
    )
    assert not result.success
    assert result.global_parameters["decay"] == 0.72
    assert result.diagnostics[0].retained_best_on_timeout
    assert result.diagnostics[0].actual_residual_evaluations == 2
    assert result.diagnostics[0].status == -2


@pytest.mark.parametrize("method", ["RK45", "DOP853", "Radau"])
def test_profile_preserves_exact_rollout_and_counts_solver_work(problem, method):  # noqa: F811
    _, model, training, settings, _ = problem
    settings = settings.model_copy(update={"integration_method": method})
    arguments = (model, training.trajectories[0], {"decay": 1.0}, {}, settings)
    plain = simulate_trajectory(*arguments)
    profile = RolloutProfile()
    measured = simulate_trajectory(*arguments, profile=profile)
    np.testing.assert_array_equal(plain.states, measured.states)
    np.testing.assert_array_equal(plain.predictions["v01"], measured.predictions["v01"])
    assert profile.rhs_calls >= profile.solver_nfev > 0
    assert (
        profile.total_seconds >= profile.integration_seconds >= profile.rhs_seconds > 0
    )
    assert profile.observation_seconds >= profile.observation_expression_seconds > 0
    assert all(v >= 0 for v in asdict(profile).values())


def test_profile_preserves_partial_rhs_work_on_timeout(problem):  # noqa: F811
    _, model, training, settings, _ = problem
    from time import monotonic

    profile = RolloutProfile()
    with pytest.raises(TimeoutError):
        simulate_trajectory(
            model,
            training.trajectories[0],
            {"decay": 1.0},
            {},
            settings,
            deadline=monotonic() - 1,
            profile=profile,
        )
    assert profile.total_seconds > 0


def test_failed_residual_penalty_is_not_best_trial():
    failures = [0]

    def function(x):
        if x[0] == 0:
            failures[0] += 1
            return np.zeros(1)
        return x

    tracked = TrackedResidual(function, lambda: failures[0])
    tracked(np.ones(1))
    tracked(np.zeros(1))
    np.testing.assert_array_equal(tracked.best_x, np.ones(1))


def test_nonfinite_parameter_vector_cannot_be_retained():
    tracked = TrackedResidual(lambda x: np.zeros(1), lambda: 0)
    tracked(np.array([np.nan]))
    assert tracked.best_x is None


def test_profile_residual_is_identical_to_production_objective(problem):  # noqa: F811
    from autoformalism.fitting.runtime_probe import measured_residual

    oracle, model, training, settings, scale = problem
    profiles = []
    actual = measured_residual(
        model, training, {"decay": 1.0}, {"v01": scale}, settings, None, profiles
    )
    np.testing.assert_array_equal(actual, oracle(np.ones(1)))
    assert len(profiles) == 1
    assert profiles[0]["rhs_calls"] > 0


def test_later_timeout_preserves_progress_without_displacing_completed_fit(
    problem,  # noqa: F811
    monkeypatch,
):
    from scipy.optimize import OptimizeResult

    _, model, training, settings, _ = problem
    validation = DatasetSplit(SplitName.VALIDATION, training.trajectories, "validation")
    calls = []

    def optimizer(fun, start, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            residual = fun(np.array([0.8]))
            return OptimizeResult(
                x=np.array([0.8]),
                success=True,
                status=1,
                message="completed",
                cost=float(0.5 * residual @ residual),
                nfev=1,
            )
        fun(np.array([0.72]))
        raise TimeoutError("wall-clock limit reached")

    monkeypatch.setattr(fitter, "least_squares", optimizer)
    result = fit_candidate(
        model,
        training,
        validation,
        settings.model_copy(update={"number_of_starts": 2}),
        initial_global_parameters={"decay": 1.0},
    )
    assert result.success
    assert result.global_parameters["decay"] == 0.8
    assert result.best_start_index == 0
    unfinished = result.diagnostics[1]
    assert unfinished.retained_best_on_timeout
    assert unfinished.retained_variables == (("parameter:decay", 0.72),)
    assert unfinished.cost < result.diagnostics[0].cost
