"""Analytical and physical regressions for trusted reference event handling."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.benchmarks.phase_b_generation import (
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.rebuttal.dalla_man import (
    STATE_INDEX,
    DallaManExternalForcing,
    DallaManParameters,
    _gastric_emptying_rate,
    simulate_dalla_man,
)
from autoformalism.reference_integration import (
    ReferenceSolver,
    integrate_reference,
    reference_time_grid,
)


@pytest.mark.parametrize("method", ["LSODA", "Radau", "DOP853"])
@pytest.mark.parametrize("start,end", [(10.25, 15.75), (0.0, 0.25), (29.75, 30.0)])
def test_pulse_with_off_grid_boundaries_matches_analytical_solution(method, start, end):
    time = np.arange(31, dtype=float)

    def rhs(t, state):
        return np.array([float(start <= t < end) - state[0]])

    values = integrate_reference(
        rhs,
        time,
        np.zeros(1),
        boundaries=[start, end],
        solver=ReferenceSolver(method=method, max_step=100, rtol=1e-10, atol=1e-12),
    )[:, 0]
    expected = (1 - np.exp(-np.clip(time - start, 0, end - start))) * np.exp(
        -np.maximum(time - end, 0)
    )
    np.testing.assert_allclose(values, expected, atol=2e-9, rtol=1e-8)


@pytest.mark.parametrize("duration,dt", [(1, 0.3), (1, 0), (np.nan, 1), (1, np.inf)])
def test_invalid_output_grid_is_rejected(duration, dt):
    with pytest.raises(ValueError):
        reference_time_grid(duration, dt)


def test_invalid_reference_boundaries_and_states_fail_closed():
    for boundaries in ([np.nan], [-1], [2]):
        with pytest.raises(ValueError, match="boundar"):
            integrate_reference(
                lambda t, x: x, np.array([0, 1]), np.zeros(1), boundaries=boundaries
            )
    with pytest.raises(ValueError, match="initial state"):
        integrate_reference(lambda t, x: x, np.array([0, 1]), np.array([np.nan]))


def test_cstr_feed_temperature_pulse_has_physical_dip_and_solver_agreement():
    protocol = next(
        p
        for p in phase_b_protocols("cstr")
        if p.protocol_id == "validation_tf_mid_pulse"
    )
    primary = simulate_phase_b(protocol)
    check = simulate_phase_b(
        protocol,
        solver=ReferenceSolver(method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.025),
    )
    np.testing.assert_allclose(primary.states, check.states, atol=2e-5, rtol=1e-7)
    temperature = primary.states[:, 1]
    assert temperature[0] - temperature.min() == pytest.approx(4.358115911, abs=2e-5)
    assert primary.inputs[100, 1] == 345
    assert primary.inputs[150, 1] == 350
    assert primary.derivatives["T"][100] == pytest.approx(-5, abs=1e-7)
    refined = simulate_phase_b(protocol.model_copy(update={"dt": 0.05}))
    np.testing.assert_allclose(
        primary.states, refined.states[::2], atol=1e-9, rtol=1e-10
    )


def test_alien_delayed_pulses_agree_with_independent_solver():
    protocol = next(
        p
        for p in phase_b_protocols("alien_device")
        if p.protocol_id == "train_multi_pulse_a"
    )
    primary = simulate_phase_b(protocol)
    check = simulate_phase_b(
        protocol,
        solver=ReferenceSolver(method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.05),
    )
    np.testing.assert_allclose(primary.states, check.states, atol=2e-6, rtol=1e-6)
    assert np.ptp(primary.states[:, -1]) > 0.01


def test_dalla_off_grid_forcing_boundaries_carry_true_endpoint():
    arguments = {
        "meals": (),
        "duration": 5,
        "variant": "original",
        "external_forcing": DallaManExternalForcing(
            insulin_pmol_per_kg_min=((0.25, 0.75, 1.0),)
        ),
    }
    coarse = simulate_dalla_man(**arguments, dt=1)
    fine = simulate_dalla_man(**arguments, dt=0.25)
    np.testing.assert_allclose(coarse.states, fine.states[::4], rtol=1e-9, atol=1e-9)


def test_dalla_meal_samples_include_full_jump_at_zero_interior_and_end():
    result = simulate_dalla_man(
        meals=((0, 30), (10, 20), (10, 10), (20, 15)),
        duration=20,
        dt=1,
        variant="original",
    )
    p = DallaManParameters()
    expected = sum(
        1000
        * grams
        * np.exp(-p.kgri * np.maximum(result.time - t, 0))
        * (result.time >= t)
        for t, grams in ((0, 30), (10, 30), (20, 15))
    )
    np.testing.assert_allclose(
        result.states[:, STATE_INDEX["Qsto1"]], expected, rtol=2e-7, atol=1e-5
    )
    assert result.meal_event_g.sum() == 75
    assert result.meal_event_g[-1] == 15


def test_dalla_multimeal_derivatives_use_same_gastric_reference_as_integration():
    result = simulate_dalla_man(
        meals=((0, 90), (10, 30)), duration=20, dt=0.1, variant="original"
    )
    p = DallaManParameters()
    q1, q2 = STATE_INDEX["Qsto1"], STATE_INDEX["Qsto2"]
    reference = float(result.states[100, q1] + result.states[100, q2])
    for i in (100, 150, 199):
        a, b = result.states[i, [q1, q2]]
        expected = p.kgri * a - _gastric_emptying_rate(a + b, reference, p) * b
        assert result.derivatives["Qsto2"][i] == pytest.approx(expected, abs=1e-10)
    numerical = (result.states[151, q2] - result.states[149, q2]) / 0.2
    assert numerical == pytest.approx(result.derivatives["Qsto2"][150], rel=1e-3)


@pytest.mark.parametrize("meals", [((0.5, 30),), ((np.nan, 30),), ((0, np.nan),)])
def test_unrepresentable_or_nonfinite_meals_are_rejected(meals):
    with pytest.raises(ValueError, match="meal"):
        simulate_dalla_man(meals=meals, duration=10, dt=1, variant="original")
