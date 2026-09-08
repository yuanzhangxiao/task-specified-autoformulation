"""Analytic input-response checks independent of the adaptive solver."""

from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import RolloutProfile, forcing_segment_indices
from autoformalism.schemas import CandidateModel


@pytest.fixture
def pulse_problem():
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "forced_memory",
            "parent_candidate_id": None,
            "change_summary": "Analytic memory with an independent fast state.",
            "states": [
                {"name": name, "kind": "latent", "unit": "unit", "description": name}
                for name in ("m", "z")
            ],
            "state_equations": [
                {"state": "m", "rhs": "-m/tau + gain*u + auxiliary"},
                {"state": "z", "rhs": "offset-z/0.05"},
            ],
            "observation_mappings": [
                {"channel": "y", "expression": "m+z", "unit": "unit"}
            ],
            "parameters": [
                {"name": name, "scope": "global", "role": "coefficient"}
                for name in ("tau", "gain", "offset")
            ],
            "initial_conditions": [
                {"state": name, "scope": "global", "fixed_value": 0}
                for name in ("m", "z")
            ],
        }
    )
    context = ValidationContext(
        targets=("y",), external_inputs=("u",), auxiliaries=("auxiliary",)
    )
    model = compile_candidate(candidate, context)
    time = np.linspace(0, 30, 301)
    signal = np.where((time >= 8) & (time < 12), -1.0, 0.0)
    trajectory = Trajectory(
        "pulse",
        time,
        {"y": np.zeros_like(time)},
        {"auxiliary": np.zeros_like(time)},
        {"u": signal},
        {},
        {},
    )
    return model, trajectory, {"tau": 0.11, "gain": 0.125, "offset": 0.027}


def analytic_memory(time, forcing, tau, gain):
    """Exact exponential convolution of each affine input interval."""
    values = np.zeros(len(time))
    for i, h in enumerate(np.diff(time)):
        one_minus = -np.expm1(-h / tau)
        slope = (forcing[i + 1] - forcing[i]) / h
        values[i + 1] = np.exp(-h / tau) * values[i] + gain * (
            tau * one_minus * forcing[i] + tau * (h - tau * one_minus) * slope
        )
    return values


@pytest.mark.parametrize("method", ["RK45", "DOP853", "Radau"])
@pytest.mark.parametrize("rtol", [1e-7, 1e-9, 1e-10])
def test_delayed_pulse_is_integrated_and_memory_survives_segments(
    pulse_problem, method, rtol
):
    model, trajectory, parameters = pulse_problem
    profile = RolloutProfile()
    settings = FitConfig(
        integration_method=method,
        relative_tolerance=rtol,
        absolute_tolerance=rtol / 100,
    )
    result = simulate_trajectory(
        model, trajectory, parameters, {}, settings, profile=profile
    )
    assert result.success, result.message
    exact = analytic_memory(
        trajectory.time,
        trajectory.external_inputs["u"],
        parameters["tau"],
        parameters["gain"],
    )
    np.testing.assert_allclose(result.states[0], exact, atol=rtol, rtol=2e-5)
    assert np.min(result.states[0]) < -0.013
    assert result.states[0, 120] < -0.005  # No state reset when the pulse ends.
    assert profile.integration_segments == 5


def test_unrelated_parameter_cannot_change_memory(pulse_problem):
    model, trajectory, parameters = pulse_problem
    settings = FitConfig(
        integration_method="Radau", relative_tolerance=1e-9, absolute_tolerance=1e-11
    )
    first = simulate_trajectory(model, trajectory, parameters, {}, settings)
    second = simulate_trajectory(
        model,
        trajectory,
        {**parameters, "offset": parameters["offset"] + 1e-4},
        {},
        settings,
    )
    assert first.success and second.success
    np.testing.assert_allclose(first.states[0], second.states[0], atol=1e-9, rtol=0)
    assert not np.array_equal(first.states[1], second.states[1])


def test_all_declared_input_and_auxiliary_breakpoints_are_kept(pulse_problem):
    model, trajectory, _ = pulse_problem
    auxiliary = np.zeros_like(trajectory.time)
    auxiliary[201] = 1e-20  # Small genuine changes must not be smoothed away.
    modified = replace(trajectory, auxiliaries={"auxiliary": auxiliary})
    assert forcing_segment_indices(model, modified).tolist() == [
        0,
        79,
        80,
        119,
        120,
        200,
        201,
        202,
        300,
    ]


def test_zero_and_exactly_linear_forcing_do_not_restart_unnecessarily(pulse_problem):
    model, trajectory, _ = pulse_problem
    time = np.arange(301, dtype=float)
    for signal in (np.zeros_like(time), 2 * time):
        modified = replace(trajectory, time=time, external_inputs={"u": signal})
        assert forcing_segment_indices(model, modified).tolist() == [0, 300]


def test_irregular_sampling_and_single_sample_pulse(pulse_problem):
    model, trajectory, parameters = pulse_problem
    time = np.array([0.0, 2.0, 2.01, 2.02, 6.0, 20.0])
    signal = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    trajectory = replace(
        trajectory,
        time=time,
        targets={"y": time * 0},
        external_inputs={"u": signal},
        auxiliaries={"auxiliary": time * 0},
    )
    result = simulate_trajectory(
        model,
        trajectory,
        parameters,
        {},
        FitConfig(
            integration_method="Radau",
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
        ),
    )
    assert result.success
    exact = analytic_memory(time, signal, parameters["tau"], parameters["gain"])
    np.testing.assert_allclose(result.states[0], exact, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_maximum_integration_step_requires_finite_positive_value(value):
    with pytest.raises(ValueError):
        FitConfig(maximum_integration_step=value)


def test_refining_step_cap_preserves_pulse_response(pulse_problem):
    model, trajectory, parameters = pulse_problem
    results = [
        simulate_trajectory(
            model,
            trajectory,
            parameters,
            {},
            FitConfig(
                integration_method="Radau",
                relative_tolerance=1e-9,
                absolute_tolerance=1e-11,
                maximum_integration_step=cap,
            ),
        )
        for cap in (None, 0.1, 0.05)
    ]
    assert all(r.success for r in results)
    for result in results[1:]:
        np.testing.assert_allclose(result.states, results[0].states, atol=1e-9, rtol=0)
