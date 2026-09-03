"""Optional CasADi multiple-shooting initialization for nonlinear fits.

This module is imported only when the corresponding fitting policy is selected.
It translates the already validated restricted expression tree; proposer text is
never evaluated as Python code.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from time import monotonic
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import CompiledModel
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.fitting.models import FitConfig, InitializationDiagnostic
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.schemas import (
    ConstraintEnforcement,
    ConstraintKind,
    ConstraintSource,
    ParameterDomain,
    ParameterScope,
)

_DIVISION_EPSILON = 1e-12
_TRUSTED_CONSTRAINT_SOURCES = frozenset(
    {
        ConstraintSource.UNSPECIFIED,
        ConstraintSource.BENCHMARK,
        ConstraintSource.RUNTIME,
        ConstraintSource.DETERMINISTIC,
    }
)


class OptionalFittingDependencyError(RuntimeError):
    """An explicitly requested optional numerical backend is unavailable."""


@dataclass(frozen=True)
class CasadiInitializationResult:
    """Parameter starting point and auditable initializer diagnostic."""

    parameters: Mapping[str, float]
    diagnostic: InitializationDiagnostic


def initialize_parameters_with_multiple_shooting(
    model: CompiledModel,
    training: DatasetSplit,
    config: FitConfig,
    target_scales: Mapping[str, float],
    *,
    preferred_parameters: Mapping[str, float] | None = None,
) -> CasadiInitializationResult:
    """Estimate global parameters from training trajectories using CasADi.

    The objective is the same normalized observed-target rollout error used by
    the normal fitter.  Latent shooting states are optimized internally; no
    measured latent values or latent derivatives are supplied.
    """
    try:
        ca = import_module("casadi")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OptionalFittingDependencyError(
            "casadi_multiple_shooting requires the optional 'casadi' extra"
        ) from exc

    started = monotonic()
    try:
        parameters, objective, iterations = _solve(
            ca,
            model,
            training,
            config,
            target_scales,
            preferred_parameters or {},
        )
    except (RuntimeError, ValueError) as exc:
        return CasadiInitializationResult(
            parameters={},
            diagnostic=InitializationDiagnostic(
                backend="casadi_multiple_shooting",
                success=False,
                status="failed",
                message=str(exc),
                objective=None,
                iterations=None,
                wall_seconds=monotonic() - started,
            ),
        )
    return CasadiInitializationResult(
        parameters=parameters,
        diagnostic=InitializationDiagnostic(
            backend="casadi_multiple_shooting",
            success=True,
            status="complete",
            message="CasADi/IPOPT multiple-shooting initialization completed",
            objective=objective,
            iterations=iterations,
            wall_seconds=monotonic() - started,
            parameter_estimates=tuple(
                f"{name}={value:.17g}" for name, value in parameters.items()
            ),
        ),
    )


def _solve(
    ca: Any,
    model: CompiledModel,
    training: DatasetSplit,
    config: FitConfig,
    target_scales: Mapping[str, float],
    preferred_parameters: Mapping[str, float],
) -> tuple[dict[str, float], float, int | None]:
    candidate = model.validated.candidate
    trajectory_specific = sorted(
        item.name
        for item in candidate.parameters
        if item.scope is ParameterScope.TRAJECTORY_SPECIFIC
    )
    if trajectory_specific:
        raise ValueError(
            "CasADi initializer currently supports global parameters only; "
            f"trajectory_specific={trajectory_specific}"
        )

    opti = ca.Opti()
    parameter_symbols: dict[str, Any] = {}
    for spec in candidate.parameters:
        symbol = opti.variable()
        parameter_symbols[spec.name] = symbol
        start = float(preferred_parameters.get(spec.name, 1.0))
        if spec.domain is ParameterDomain.POSITIVE:
            start = max(start, 1.0)
            opti.subject_to(symbol >= np.finfo(float).eps)
        elif spec.domain is ParameterDomain.NONNEGATIVE:
            start = max(start, 0.0)
            opti.subject_to(symbol >= 0.0)
        _apply_trusted_parameter_constraints(opti, symbol, spec.name, model)
        opti.set_initial(symbol, start)

    objective: Any = 0.0
    residual_count = 0
    for trajectory in training.trajectories:
        trajectory_objective, trajectory_count = _trajectory_objective(
            ca,
            opti,
            model,
            trajectory,
            parameter_symbols,
            target_scales,
            config,
        )
        objective += trajectory_objective
        residual_count += trajectory_count
    if residual_count == 0:
        raise ValueError("CasADi initializer received no observed target rows")
    if parameter_symbols:
        objective += 1e-12 * ca.sumsqr(
            ca.vertcat(*parameter_symbols.values())
        )
    opti.minimize(objective / residual_count)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {
            "max_iter": config.casadi_maximum_iterations,
            "max_cpu_time": config.casadi_maximum_wall_time_seconds,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-8,
        },
    )
    solution = opti.solve()
    estimates = {
        name: float(solution.value(symbol))
        for name, symbol in parameter_symbols.items()
    }
    if not all(np.isfinite(value) for value in estimates.values()):
        raise RuntimeError("CasADi initializer returned nonfinite parameters")
    stats = solution.stats()
    raw_iterations = stats.get("iter_count")
    iterations = int(raw_iterations) if raw_iterations is not None else None
    return estimates, float(solution.value(objective / residual_count)), iterations


def _trajectory_objective(
    ca: Any,
    opti: Any,
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, Any],
    target_scales: Mapping[str, float],
    config: FitConfig,
) -> tuple[Any, int]:
    state_names = model.state_names
    state_index = {name: index for index, name in enumerate(state_names)}
    sample_indices = _sample_indices(
        trajectory.number_of_rows,
        config.casadi_maximum_intervals_per_trajectory,
    )
    current = opti.variable(len(state_names))
    _initialize_state_node(opti, current, model, trajectory, 0)
    _constrain_initial_state(ca, opti, current, model, trajectory, parameters)

    objective: Any = 0.0
    residual_count = 0
    start_observation = _symbolic_observation(
        ca,
        model,
        current,
        parameters,
        trajectory,
        sample_indices[0],
        float(trajectory.time[sample_indices[0]]),
    )
    for channel, predicted in start_observation.items():
        observed = float(trajectory.targets[channel][sample_indices[0]])
        objective += ((predicted - observed) / target_scales[channel]) ** 2
        residual_count += 1

    for interval_number, (left, right) in enumerate(
        pairwise(sample_indices),
        start=1,
    ):
        interval_state = _reset_observed_states(
            ca,
            current,
            model,
            trajectory,
            left,
            state_index,
        )
        propagated = _rk4_interval(
            ca,
            model,
            interval_state,
            parameters,
            trajectory,
            left,
            right,
            config.fixed_step_substeps,
        )
        if (
            interval_number % config.casadi_shooting_interval_count == 0
            and right != sample_indices[-1]
        ):
            node = opti.variable(len(state_names))
            _initialize_state_node(opti, node, model, trajectory, right)
            opti.subject_to(node == propagated)
            current = node
        else:
            current = propagated
        observation = _symbolic_observation(
            ca,
            model,
            current,
            parameters,
            trajectory,
            max(0, right - 1),
            float(trajectory.time[right]),
        )
        for channel, predicted in observation.items():
            observed = float(trajectory.targets[channel][right])
            objective += ((predicted - observed) / target_scales[channel]) ** 2
            residual_count += 1
    return objective, residual_count


def _sample_indices(number_of_rows: int, maximum_intervals: int) -> tuple[int, ...]:
    if number_of_rows < 2:
        raise ValueError("CasADi initializer needs at least two trajectory rows")
    interval_count = number_of_rows - 1
    if interval_count <= maximum_intervals:
        return tuple(range(number_of_rows))
    indices = np.linspace(0, interval_count, maximum_intervals + 1, dtype=int)
    return tuple(int(index) for index in np.unique(indices))


def _initialize_state_node(
    opti: Any,
    node: Any,
    model: CompiledModel,
    trajectory: Trajectory,
    index: int,
) -> None:
    observed = model.direct_state_observation_channels
    for state_index, state_name in enumerate(model.state_names):
        channel = observed.get(state_name)
        start = _channel_value(trajectory, channel, index) if channel else 0.0
        opti.set_initial(node[state_index], start)


def _constrain_initial_state(
    ca: Any,
    opti: Any,
    node: Any,
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, Any],
) -> None:
    time = float(trajectory.time[0])
    known = _known_initial_values(trajectory, model)
    known.update(parameters)
    for index, state_name in enumerate(model.state_names):
        spec = next(
            (
                item
                for item in model.validated.candidate.initial_conditions
                if item.state == state_name
            ),
            None,
        )
        if spec is None:
            if state_name in model.validated.causal_derivative_initials:
                opti.subject_to(node[index] == 0.0)
            continue
        if spec.fixed_value is not None:
            opti.subject_to(node[index] == float(spec.fixed_value))
        elif spec.expression is not None:
            expression = model.validated.initial_condition_expressions[state_name]
            environment = {model.validated.context.time_symbol: time, **known}
            opti.subject_to(
                node[index] == _translate(ca, expression, environment)
            )
        elif spec.initialization_range is not None:
            opti.subject_to(node[index] >= spec.initialization_range.lower)
            opti.subject_to(node[index] <= spec.initialization_range.upper)


def _known_initial_values(
    trajectory: Trajectory,
    model: CompiledModel,
) -> dict[str, float]:
    values = {
        model.validated.context.time_symbol: float(trajectory.time[0]),
        **{name: float(data[0]) for name, data in trajectory.targets.items()},
        **{name: float(data[0]) for name, data in trajectory.auxiliaries.items()},
    }
    for name, data in trajectory.external_inputs.items():
        try:
            numeric = float(data[0])
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values[name] = numeric
    for name, value in trajectory.fixed_covariates.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values[name] = numeric
    return values


def _reset_observed_states(
    ca: Any,
    state: Any,
    model: CompiledModel,
    trajectory: Trajectory,
    index: int,
    state_index: Mapping[str, int],
) -> Any:
    resettable = model.observed_state_channels
    if not resettable:
        return state
    values = [state[position] for position in range(len(model.state_names))]
    for state_name, channel in resettable.items():
        values[state_index[state_name]] = _channel_value(trajectory, channel, index)
    return ca.vertcat(*values)


def _rk4_interval(
    ca: Any,
    model: CompiledModel,
    state: Any,
    parameters: Mapping[str, Any],
    trajectory: Trajectory,
    causal_index: int,
    right_index: int,
    substeps: int,
) -> Any:
    start = float(trajectory.time[causal_index])
    end = float(trajectory.time[right_index])
    step = (end - start) / substeps
    current = state
    for substep in range(substeps):
        time = start + substep * step
        k1 = _symbolic_rhs(
            ca, model, current, parameters, trajectory, causal_index, time
        )
        k2 = _symbolic_rhs(
            ca,
            model,
            current + step * k1 / 2.0,
            parameters,
            trajectory,
            causal_index,
            time + step / 2.0,
        )
        k3 = _symbolic_rhs(
            ca,
            model,
            current + step * k2 / 2.0,
            parameters,
            trajectory,
            causal_index,
            time + step / 2.0,
        )
        k4 = _symbolic_rhs(
            ca,
            model,
            current + step * k3,
            parameters,
            trajectory,
            causal_index,
            time + step,
        )
        current = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return current


def _symbolic_rhs(
    ca: Any,
    model: CompiledModel,
    state: Any,
    parameters: Mapping[str, Any],
    trajectory: Trajectory,
    causal_index: int,
    time: float,
) -> Any:
    environment = _environment(
        ca, model, state, parameters, trajectory, causal_index, time
    )
    return ca.vertcat(
        *(
            _translate(
                ca,
                model.validated.equation_expressions[state_name],
                environment,
            )
            for state_name in model.state_names
        )
    )


def _symbolic_observation(
    ca: Any,
    model: CompiledModel,
    state: Any,
    parameters: Mapping[str, Any],
    trajectory: Trajectory,
    causal_index: int,
    time: float,
) -> dict[str, Any]:
    environment = _environment(
        ca, model, state, parameters, trajectory, causal_index, time
    )
    return {
        channel: _translate(ca, expression, environment)
        for channel, expression in model.validated.observation_expressions.items()
    }


def _environment(
    ca: Any,
    model: CompiledModel,
    state: Any,
    parameters: Mapping[str, Any],
    trajectory: Trajectory,
    causal_index: int,
    time: float,
) -> dict[str, Any]:
    forcing = trajectory_forcing(model, trajectory, causal_index=causal_index)
    environment: dict[str, Any] = {
        model.validated.context.time_symbol: time,
        **parameters,
        **{
            state_name: state[index]
            for index, state_name in enumerate(model.state_names)
        },
    }
    for channel in model.validated.forcing_symbols:
        environment[channel] = forcing.value(channel, time)
    for process_name in model.validated.process_order:
        environment[process_name] = _translate(
            ca,
            model.validated.process_expressions[process_name],
            environment,
        )
    return environment


def _translate(
    ca: Any,
    expression: ParsedExpression,
    environment: Mapping[str, Any],
) -> Any:
    return _walk(ca, expression.tree, environment)


def _walk(ca: Any, node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _walk(ca, node.body, environment)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _walk(ca, node.operand, environment)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _walk(ca, node.left, environment)
        right = _walk(ca, node.right, environment)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / _guard_denominator(ca, right)
        if isinstance(node.op, ast.Pow):
            return ca.power(left, right)
        raise AssertionError("restricted parser admitted an unsupported operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_walk(ca, argument, environment) for argument in node.args]
        return _call(ca, node.func.id, arguments)
    raise AssertionError(
        f"restricted parser admitted unsupported node {type(node).__name__}"
    )


def _guard_denominator(ca: Any, value: Any) -> Any:
    return ca.if_else(
        ca.fabs(value) >= _DIVISION_EPSILON,
        value,
        ca.if_else(value >= 0.0, _DIVISION_EPSILON, -_DIVISION_EPSILON),
    )


def _call(ca: Any, name: str, arguments: list[Any]) -> Any:
    argument = arguments[0]
    if name == "exp":
        return ca.exp(argument)
    if name == "log":
        return ca.log(argument)
    if name == "tanh":
        return ca.tanh(argument)
    if name == "sqrt":
        return ca.sqrt(argument)
    if name == "abs":
        return ca.fabs(argument)
    if name == "min":
        result = argument
        for other in arguments[1:]:
            result = ca.fmin(result, other)
        return result
    if name == "max":
        result = argument
        for other in arguments[1:]:
            result = ca.fmax(result, other)
        return result
    if name == "sigmoid":
        return 1.0 / (1.0 + ca.exp(-argument))
    if name == "softplus":
        return ca.fmax(argument, 0.0) + ca.log(1.0 + ca.exp(-ca.fabs(argument)))
    raise AssertionError(f"restricted parser admitted unsupported function {name}")


def _apply_trusted_parameter_constraints(
    opti: Any,
    symbol: Any,
    parameter_name: str,
    model: CompiledModel,
) -> None:
    for constraint in model.validated.candidate.constraints:
        if (
            constraint.subject != parameter_name
            or constraint.enforcement is not ConstraintEnforcement.HARD
            or constraint.source not in _TRUSTED_CONSTRAINT_SOURCES
        ):
            continue
        if constraint.kind is ConstraintKind.NONNEGATIVE:
            opti.subject_to(symbol >= 0.0)
        elif constraint.kind is ConstraintKind.POSITIVE:
            opti.subject_to(symbol >= np.finfo(float).eps)
        if constraint.bounds is not None:
            opti.subject_to(symbol >= constraint.bounds.lower)
            opti.subject_to(symbol <= constraint.bounds.upper)


def _channel_value(
    trajectory: Trajectory,
    channel: str | None,
    index: int,
) -> float:
    if channel is None:
        raise ValueError("missing channel")
    if channel in trajectory.targets:
        return float(trajectory.targets[channel][index])
    if channel in trajectory.auxiliaries:
        return float(trajectory.auxiliaries[channel][index])
    if channel in trajectory.external_inputs:
        return float(trajectory.external_inputs[channel][index])
    if channel in trajectory.fixed_covariates:
        return float(trajectory.fixed_covariates[channel])
    raise ValueError(f"trajectory is missing channel {channel}")
