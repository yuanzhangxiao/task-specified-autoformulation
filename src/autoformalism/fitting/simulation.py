"""Numerically integrate compiled candidates over one measured trajectory."""

from __future__ import annotations

from collections.abc import Mapping

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
) -> SimulationResult:
    """Roll out a compiled ODE, converting numerical exceptions into failures."""
    time = np.asarray(trajectory.time, dtype=float).copy()
    try:
        if len(time) < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError("simulation needs at least two increasing time points")
        missing_initials = sorted(set(model.state_names) - set(initial_conditions))
        extra_initials = sorted(set(initial_conditions) - set(model.state_names))
        if missing_initials or extra_initials:
            raise ValueError(
                f"initial-condition mismatch: missing={missing_initials}, "
                f"extra={extra_initials}"
            )
        initial_state = np.asarray(
            [initial_conditions[name] for name in model.state_names], dtype=float
        )
        if not np.isfinite(initial_state).all():
            raise ValueError("initial conditions contain nonfinite values")
        forcing = trajectory_forcing(model, trajectory)
        solution = solve_ivp(
            lambda current_time, state: model.rhs(
                current_time, state, parameters, forcing
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
            raise RuntimeError(f"unexpected integration shape: {solution.y.shape}")
        if not np.isfinite(solution.y).all():
            raise RuntimeError("integration produced a nonfinite trajectory")

        predictions: dict[str, np.ndarray] = {
            channel: np.empty(len(time), dtype=float)
            for channel in model.validated.observation_expressions
        }
        for index, current_time in enumerate(time):
            observed = model.observe(
                float(current_time),
                solution.y[:, index],
                parameters,
                forcing,
            )
            for channel, value in observed.items():
                predictions[channel][index] = value
        if any(not np.isfinite(values).all() for values in predictions.values()):
            raise RuntimeError("observation mapping produced nonfinite values")
        return SimulationResult(True, time, solution.y.copy(), predictions)
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
        else:
            raise ValueError(f"trajectory is missing forcing channel {name}")
    return PiecewiseLinearForcing(
        trajectory.time,
        channels,
        allowed_channels=model.validated.forcing_symbols,
    )
