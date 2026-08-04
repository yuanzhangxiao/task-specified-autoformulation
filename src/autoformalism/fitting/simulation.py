"""Numerically integrate compiled candidates over one measured trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import Trajectory
from autoformalism.expressions import (
    CompiledModel,
    PiecewiseLinearForcing,
    RuntimeExpressionError,
)
from autoformalism.fitting.models import FitConfig, SimulationResult


def simulate_trajectory(
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, float],
    initial_conditions: Mapping[str, float],
    config: FitConfig,
    *,
    deadline: float | None = None,
    reset_observed_states: bool | None = None,
) -> SimulationResult:
    """Roll out a compiled ODE, converting numerical exceptions into failures.

    ``reset_observed_states=False`` is an evaluation-only free-rollout mode. It
    is rejected when a measured target is used as an exogenous forcing symbol,
    because that would reveal the held-out trajectory during rollout.
    """
    time = np.asarray(trajectory.time, dtype=float).copy()
    try:
        if len(time) < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError("simulation needs at least two increasing time points")
        resettable = set(model.observed_state_channels)
        known_initial_values = _known_initial_values(model, trajectory)
        derived_initials = {
            name: model.initial_condition_value(name, known_initial_values)
            for name in model.state_names
            if name not in resettable
        }
        missing_initials = sorted(
            name
            for name in model.state_names
            if name not in resettable
            and derived_initials[name] is None
            and name not in initial_conditions
            and name not in model.validated.causal_derivative_initials
        )
        extra_initials = sorted(set(initial_conditions) - set(model.state_names))
        if missing_initials or extra_initials:
            raise ValueError(
                f"initial-condition mismatch: missing={missing_initials}, "
                f"extra={extra_initials}"
            )
        initial_state = np.asarray(
            [
                _observed_state_value(model, trajectory, name, 0)
                if name in resettable
                else (
                    0.0
                    if name in model.validated.causal_derivative_initials
                    else (
                        derived_initials[name]
                        if derived_initials[name] is not None
                        else initial_conditions[name]
                    )
                )
                for name in model.state_names
            ],
            dtype=float,
        )
        if not np.isfinite(initial_state).all():
            raise ValueError("initial conditions contain nonfinite values")
        _check_deadline(deadline)
        use_resets = (
            bool(model.validated.context.lagged_targets)
            if reset_observed_states is None
            else reset_observed_states
        )
        leaked_targets = (
            model.validated.forcing_symbols
            & frozenset(model.validated.context.lagged_targets)
        )
        if not use_resets and leaked_targets:
            raise ValueError(
                "free rollout cannot use measured target forcing: "
                f"{sorted(leaked_targets)}"
            )
        if use_resets:
            state_values = _simulate_one_step_intervals(
                model, trajectory, parameters, initial_state, config, deadline
            )
        elif config.integration_backend == "fixed_rk4":
            state_values = _simulate_fixed_intervals(
                model, trajectory, parameters, initial_state, config, deadline
            )
        else:
            forcing = trajectory_forcing(model, trajectory)
            solution = solve_ivp(
                lambda current_time, state: _deadline_rhs(
                    model,
                    current_time,
                    state,
                    parameters,
                    forcing,
                    deadline,
                ),
                (float(time[0]), float(time[-1])),
                initial_state,
                t_eval=time,
                method=config.integration_method,
                rtol=config.relative_tolerance,
                atol=config.absolute_tolerance,
            )
            if not solution.success:
                raise RuntimeError(f"integration failed: {solution.message}")
            if solution.y.shape != (len(model.state_names), len(time)):
                raise RuntimeError(
                    f"unexpected integration shape: {solution.y.shape}"
                )
            state_values = solution.y
        if not np.isfinite(state_values).all():
            raise RuntimeError("integration produced a nonfinite trajectory")
        for index in range(state_values.shape[1]):
            model.validate_state_constraints(state_values[:, index])

        predictions: dict[str, np.ndarray] = {
            channel: np.empty(len(time), dtype=float)
            for channel in model.validated.observation_expressions
        }
        for index, current_time in enumerate(time):
            forcing = trajectory_forcing(
                model,
                trajectory,
                causal_index=max(0, index - 1),
            )
            observed = model.observe(
                float(current_time),
                state_values[:, index],
                parameters,
                forcing,
            )
            for channel, value in observed.items():
                predictions[channel][index] = value
        if any(not np.isfinite(values).all() for values in predictions.values()):
            raise RuntimeError("observation mapping produced nonfinite values")
        return SimulationResult(True, time, state_values.copy(), predictions)
    except (
        ArithmeticError,
        RuntimeError,
        RuntimeExpressionError,
        TypeError,
        ValueError,
    ) as exc:
        return SimulationResult(False, time, None, {}, str(exc))


def trajectory_forcing(
    model: CompiledModel,
    trajectory: Trajectory,
    *,
    causal_index: int | None = None,
) -> PiecewiseLinearForcing:
    """Build the strict supplied-channel interpolator for a trajectory."""
    channels: dict[str, np.ndarray] = {}
    for name in model.validated.forcing_symbols:
        if name in trajectory.auxiliaries:
            channels[name] = trajectory.auxiliaries[name]
        elif name in trajectory.external_inputs:
            channels[name] = trajectory.external_inputs[name]
        elif name in trajectory.fixed_covariates:
            try:
                value = float(trajectory.fixed_covariates[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"fixed covariate {name} must be numeric") from exc
            channels[name] = np.full(len(trajectory.time), value)
        elif (
            name in trajectory.targets
            and name in model.validated.context.lagged_targets
        ):
            channels[name] = trajectory.targets[name]
        else:
            raise ValueError(f"trajectory is missing forcing channel {name}")
    return PiecewiseLinearForcing(
        trajectory.time,
        channels,
        allowed_channels=model.validated.forcing_symbols,
        causal_step_channels=(
            model.validated.forcing_symbols
            & frozenset(model.validated.context.lagged_targets)
        ),
        causal_index=causal_index,
    )


def _simulate_one_step_intervals(
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, float],
    initial_state: np.ndarray,
    config: FitConfig,
    deadline: float | None,
) -> np.ndarray:
    """Integrate each slot using targets observed at that slot's start only."""
    states = np.empty((len(model.state_names), len(trajectory.time)), dtype=float)
    states[:, 0] = initial_state
    for index in range(len(trajectory.time) - 1):
        _check_deadline(deadline)
        forcing = trajectory_forcing(model, trajectory, causal_index=index)
        interval_initial = causal_interval_state(
            model, trajectory, states[:, index], index
        )
        start = float(trajectory.time[index])
        end = float(trajectory.time[index + 1])
        if config.integration_backend == "fixed_rk4":
            states[:, index + 1] = _fixed_rk4_interval(
                model,
                start,
                end,
                interval_initial,
                parameters,
                forcing,
                config.fixed_step_substeps,
                deadline,
            )
        else:
            solution = solve_ivp(
                lambda current_time, state, interval_forcing=forcing: (
                    _deadline_rhs(
                        model,
                        current_time,
                        state,
                        parameters,
                        interval_forcing,
                        deadline,
                    )
                ),
                (start, end),
                interval_initial,
                t_eval=[end],
                method=config.integration_method,
                rtol=config.relative_tolerance,
                atol=config.absolute_tolerance,
            )
            if not solution.success:
                raise RuntimeError(f"integration failed: {solution.message}")
            states[:, index + 1] = solution.y[:, -1]
    return states


def _simulate_fixed_intervals(
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, float],
    initial_state: np.ndarray,
    config: FitConfig,
    deadline: float | None,
) -> np.ndarray:
    """Integrate an open-loop trajectory with fixed RK4 measurement steps."""
    states = np.empty((len(model.state_names), len(trajectory.time)), dtype=float)
    states[:, 0] = initial_state
    forcing = trajectory_forcing(model, trajectory)
    for index in range(len(trajectory.time) - 1):
        states[:, index + 1] = _fixed_rk4_interval(
            model,
            float(trajectory.time[index]),
            float(trajectory.time[index + 1]),
            states[:, index],
            parameters,
            forcing,
            config.fixed_step_substeps,
            deadline,
        )
    return states


def _fixed_rk4_interval(
    model: CompiledModel,
    start: float,
    end: float,
    initial_state: np.ndarray,
    parameters: Mapping[str, float],
    forcing: PiecewiseLinearForcing,
    substeps: int,
    deadline: float | None,
) -> np.ndarray:
    """Advance one sampling interval with deterministic fixed-step RK4."""
    state = np.asarray(initial_state, dtype=float).copy()
    step = (end - start) / substeps
    for index in range(substeps):
        _check_deadline(deadline)
        time = start + index * step
        k1 = model.rhs(time, state, parameters, forcing)
        k2 = model.rhs(time + step / 2.0, state + step * k1 / 2.0, parameters, forcing)
        k3 = model.rhs(time + step / 2.0, state + step * k2 / 2.0, parameters, forcing)
        k4 = model.rhs(time + step, state + step * k3, parameters, forcing)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state


def _deadline_rhs(
    model: CompiledModel,
    time: float,
    state: np.ndarray,
    parameters: Mapping[str, float],
    forcing: PiecewiseLinearForcing,
    deadline: float | None,
) -> np.ndarray:
    _check_deadline(deadline)
    return model.rhs(time, state, parameters, forcing)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("fitting wall-clock limit reached")


def _observed_state_value(
    model: CompiledModel,
    trajectory: Trajectory,
    state_name: str,
    index: int,
) -> float:
    """Read one causally available value for a directly observed state."""
    channel = model.observed_state_channels[state_name]
    if channel in trajectory.targets:
        return float(trajectory.targets[channel][index])
    if channel in trajectory.auxiliaries:
        return float(trajectory.auxiliaries[channel][index])
    raise ValueError(
        f"trajectory is missing observed state channel {channel} for {state_name}"
    )


def _known_initial_values(
    model: CompiledModel,
    trajectory: Trajectory,
) -> dict[str, float]:
    """Collect only public values available at the first prediction boundary."""
    values = {
        model.validated.context.time_symbol: float(trajectory.time[0]),
        **{name: float(data[0]) for name, data in trajectory.targets.items()},
        **{name: float(data[0]) for name, data in trajectory.auxiliaries.items()},
    }
    for name, data in trajectory.external_inputs.items():
        try:
            value = float(data[0])
        except (TypeError, ValueError):
            # Structured event schedules are public task metadata, not scalar
            # initialization symbols.  Their numeric forcing representations,
            # when present, are exposed under separate declared channel names.
            continue
        if np.isfinite(value):
            values[name] = value
    for name, value in trajectory.fixed_covariates.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            # Structured metadata (for example, a JSON event schedule) can be
            # public task context without being a scalar model input.  Omit it
            # from the numeric initialization environment; an expression that
            # actually references it will then fail closed as a missing symbol.
            continue
        if np.isfinite(numeric_value):
            values[name] = numeric_value
    return values


def causal_interval_state(
    model: CompiledModel,
    trajectory: Trajectory,
    propagated_state: np.ndarray,
    index: int,
) -> np.ndarray:
    """Reset directly observed components and preserve every latent component."""
    interval_state = np.asarray(propagated_state, dtype=float).copy()
    for state_index, state_name in enumerate(model.state_names):
        if state_name in model.observed_state_channels:
            interval_state[state_index] = _observed_state_value(
                model, trajectory, state_name, index
            )
        elif state_name in model.validated.causal_derivative_initials:
            base_state = model.validated.causal_derivative_initials[state_name]
            if index == 0:
                interval_state[state_index] = 0.0
            else:
                current = _observed_state_value(
                    model, trajectory, base_state, index
                )
                previous = _observed_state_value(
                    model, trajectory, base_state, index - 1
                )
                interval_state[state_index] = (current - previous) / float(
                    trajectory.time[index] - trajectory.time[index - 1]
                )
    return interval_state
