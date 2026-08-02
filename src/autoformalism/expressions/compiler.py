"""Safe numerical interpreter, ODE compiler, and forcing interpolation."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autoformalism.expressions.diagnostics import RuntimeExpressionError
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.expressions.validation import (
    CandidateValidator,
    ValidatedCandidate,
    ValidationContext,
)
from autoformalism.schemas import CandidateModel, ConstraintKind

SAFE_DIVISION_EPSILON = 1e-12


class Forcing(Protocol):
    """Time-indexed supplied auxiliary/input interface."""

    def value(self, channel: str, time: float) -> float:
        """Return one finite channel value at a supported time."""
        ...


class PiecewiseLinearForcing:
    """Strict piecewise-linear interpolation without extrapolation."""

    def __init__(
        self,
        time: ArrayLike,
        channels: Mapping[str, ArrayLike],
        *,
        allowed_channels: frozenset[str],
        causal_step_channels: frozenset[str] = frozenset(),
        causal_index: int | None = None,
    ) -> None:
        time_values = np.asarray(time, dtype=float)
        if time_values.ndim != 1 or len(time_values) == 0:
            raise ValueError("forcing time must be a nonempty one-dimensional array")
        if not np.isfinite(time_values).all():
            raise ValueError("forcing time contains nonfinite values")
        if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
            raise ValueError("forcing time must be strictly increasing")
        unknown = sorted(set(channels) - set(allowed_channels))
        if unknown:
            raise ValueError(f"forcing contains unavailable channels: {unknown}")

        numeric_channels: dict[str, NDArray[np.float64]] = {}
        for name, raw_values in channels.items():
            try:
                values = np.asarray(raw_values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"forcing channel {name} must be numeric") from exc
            if values.ndim != 1 or len(values) != len(time_values):
                raise ValueError(
                    f"forcing channel {name} must align one-to-one with time"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"forcing channel {name} contains nonfinite values")
            values = values.copy()
            values.setflags(write=False)
            numeric_channels[name] = values
        self._time = time_values.copy()
        self._time.setflags(write=False)
        self._channels = numeric_channels
        self._allowed_channels = allowed_channels
        self._causal_step_channels = causal_step_channels
        self._causal_index = causal_index
        if causal_step_channels - allowed_channels:
            raise ValueError("causal step channels must be allowed forcing channels")
        if causal_step_channels and causal_index is None:
            raise ValueError("causal step forcing requires a causal index")
        if causal_index is not None and not 0 <= causal_index < len(time_values):
            raise ValueError("causal forcing index is outside the trajectory")

    def value(self, channel: str, time: float) -> float:
        """Interpolate a channel inside its closed time support."""
        if channel not in self._allowed_channels:
            raise RuntimeExpressionError(
                f"forcing channel is not allowed by the benchmark: {channel}"
            )
        if channel not in self._channels:
            raise RuntimeExpressionError(f"forcing channel is missing: {channel}")
        if not math.isfinite(time):
            raise RuntimeExpressionError("forcing query time must be finite")
        lower = float(self._time[0])
        upper = float(self._time[-1])
        tolerance = max(1e-12, (upper - lower) * 1e-12)
        if time < lower - tolerance or time > upper + tolerance:
            raise RuntimeExpressionError(
                f"forcing query {time} is outside [{lower}, {upper}]"
            )
        bounded_time = min(max(time, lower), upper)
        if channel in self._causal_step_channels:
            assert self._causal_index is not None
            return float(self._channels[channel][self._causal_index])
        return float(np.interp(bounded_time, self._time, self._channels[channel]))


@dataclass(frozen=True)
class CompiledModel:
    """Numerical ODE right-hand side and observation mappings."""

    validated: ValidatedCandidate

    @property
    def state_names(self) -> tuple[str, ...]:
        """Return state vector order."""
        return tuple(item.name for item in self.validated.candidate.states)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return declared parameter order."""
        return tuple(item.name for item in self.validated.candidate.parameters)

    @property
    def observed_state_channels(self) -> Mapping[str, str]:
        """Map resettable model states to directly observed data channels."""
        if not self.validated.context.lagged_targets:
            return {}
        available = set(self.validated.context.targets) | set(
            self.validated.context.auxiliaries
        )
        mapping = {
            state: state for state in self.state_names if state in available
        }
        for channel, expression in self.validated.observation_expressions.items():
            body = expression.tree.body
            if (
                channel in available
                and isinstance(body, ast.Name)
                and body.id in self.state_names
            ):
                existing = mapping.get(body.id)
                if existing is not None and existing != channel:
                    raise RuntimeExpressionError(
                        f"state {body.id} has conflicting observed channels: "
                        f"{existing}, {channel}"
                    )
                mapping[body.id] = channel
        return mapping

    def rhs(
        self,
        time: float,
        state: ArrayLike,
        parameters: Mapping[str, float],
        forcing: Forcing,
    ) -> NDArray[np.float64]:
        """Evaluate the compiled state derivatives."""
        environment = self._environment(time, state, parameters, forcing)
        values = np.asarray(
            [
                _evaluate(
                    self.validated.equation_expressions[state_name],
                    environment,
                )
                for state_name in self.state_names
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise RuntimeExpressionError("RHS produced nonfinite values")
        return values

    def observe(
        self,
        time: float,
        state: ArrayLike,
        parameters: Mapping[str, float],
        forcing: Forcing,
    ) -> Mapping[str, float]:
        """Evaluate all target observation mappings."""
        environment = self._environment(time, state, parameters, forcing)
        return {
            channel: _evaluate(expression, environment)
            for channel, expression in self.validated.observation_expressions.items()
        }

    def evaluate_expression(
        self,
        expression: ParsedExpression,
        time: float,
        state: ArrayLike,
        parameters: Mapping[str, float],
        forcing: Forcing,
    ) -> float:
        """Evaluate an already restricted expression in the model environment."""
        return _evaluate(
            expression,
            self._environment(time, state, parameters, forcing),
        )

    def initial_condition_value(
        self,
        state_name: str,
        known_initial_values: Mapping[str, float],
    ) -> float | None:
        """Evaluate a fixed or safely parsed analytic initialization rule."""
        spec = next(
            (
                item
                for item in self.validated.candidate.initial_conditions
                if item.state == state_name
            ),
            None,
        )
        if spec is None:
            return None
        if spec.fixed_value is not None:
            return float(spec.fixed_value)
        expression = self.validated.initial_condition_expressions.get(state_name)
        if expression is not None:
            return _evaluate(expression, known_initial_values)
        return None

    def _environment(
        self,
        time: float,
        state: ArrayLike,
        parameters: Mapping[str, float],
        forcing: Forcing,
    ) -> dict[str, float]:
        if not math.isfinite(time):
            raise RuntimeExpressionError("model time must be finite")
        state_values = np.asarray(state, dtype=float)
        if state_values.shape != (len(self.state_names),):
            raise RuntimeExpressionError(
                f"state shape must be {(len(self.state_names),)}, "
                f"got {state_values.shape}"
            )
        if not np.isfinite(state_values).all():
            raise RuntimeExpressionError("state contains nonfinite values")
        expected_parameters = set(self.parameter_names)
        supplied_parameters = set(parameters)
        if supplied_parameters != expected_parameters:
            missing = sorted(expected_parameters - supplied_parameters)
            extra = sorted(supplied_parameters - expected_parameters)
            raise RuntimeExpressionError(
                f"parameter mismatch: missing={missing}, extra={extra}"
            )

        parameter_values: dict[str, float] = {}
        for parameter in self.validated.candidate.parameters:
            value = _finite_float(
                parameters[parameter.name],
                f"parameter {parameter.name}",
            )
            lower = parameter.bounds.lower
            upper = parameter.bounds.upper
            width = upper - lower
            scale = max(abs(lower), abs(upper), abs(value), np.finfo(float).tiny)
            tolerance = max(
                width * 1e-12,
                scale * np.finfo(float).eps * 32.0,
            )
            if value < lower - tolerance or value > upper + tolerance:
                raise RuntimeExpressionError(
                    f"parameter {parameter.name}={value} is outside "
                    f"[{lower}, {upper}]"
                )
            # SciPy finite differences can land a few ULPs outside an active
            # bound. Admit only that numerical fuzz and evaluate at the bound.
            parameter_values[parameter.name] = min(max(value, lower), upper)

        environment = {
            self.validated.context.time_symbol: float(time),
            **parameter_values,
            **{
                name: float(value)
                for name, value in zip(
                    self.state_names,
                    state_values,
                    strict=True,
                )
            },
        }
        for channel in sorted(self.validated.forcing_symbols):
            environment[channel] = _finite_float(
                forcing.value(channel, time),
                f"forcing {channel}",
            )
        for process_name in self.validated.process_order:
            environment[process_name] = _evaluate(
                self.validated.process_expressions[process_name],
                environment,
            )
        return environment

    def validate_state_constraints(
        self,
        state: ArrayLike,
    ) -> None:
        """Check constraints on an accepted state, outside solver trial steps."""
        state_values = np.asarray(state, dtype=float)
        if state_values.shape != (len(self.state_names),):
            raise RuntimeExpressionError(
                f"state shape must be {(len(self.state_names),)}, "
                f"got {state_values.shape}"
            )
        values = dict(zip(self.state_names, state_values, strict=True))
        for constraint in self.validated.candidate.constraints:
            if constraint.subject not in values:
                continue
            value = float(values[constraint.subject])
            invalid = False
            if constraint.kind is ConstraintKind.NONNEGATIVE:
                invalid = value < 0.0
            elif constraint.kind is ConstraintKind.POSITIVE:
                invalid = value <= 0.0
            if constraint.bounds:
                invalid = invalid or not (
                    constraint.bounds.lower <= value <= constraint.bounds.upper
                )
            if invalid:
                bounds = (
                    ""
                    if constraint.bounds is None
                    else " with bounds "
                    f"[{constraint.bounds.lower}, {constraint.bounds.upper}]"
                )
                raise RuntimeExpressionError(
                    f"state {constraint.subject}={value} violates "
                    f"{constraint.kind.value} constraint{bounds}"
                )


def compile_candidate(
    candidate: CandidateModel,
    context: ValidationContext,
    *,
    validator: CandidateValidator | None = None,
) -> CompiledModel:
    """Validate once and return safe numerical callables."""
    validated = (validator or CandidateValidator()).validate(candidate, context)
    return CompiledModel(validated)


def _evaluate(expression: ParsedExpression, environment: Mapping[str, float]) -> float:
    try:
        value = _walk(expression.tree, environment)
    except RuntimeExpressionError:
        raise
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise RuntimeExpressionError(
            f"numerical failure in {expression.source!r}: {exc}"
        ) from exc
    return _finite_float(value, f"expression {expression.source!r}")


def _walk(node: ast.AST, environment: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _walk(node.body, environment)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        try:
            return environment[node.id]
        except KeyError as exc:  # pragma: no cover - semantic validator owns this
            raise RuntimeExpressionError(
                f"missing symbol at runtime: {node.id}"
            ) from exc
    if isinstance(node, ast.UnaryOp):
        operand = _walk(node.operand, environment)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _walk(node.left, environment)
        right = _walk(node.right, environment)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / _guard_denominator(right)
        if isinstance(node.op, ast.Pow):
            exponent = int(right)
            base = _guard_denominator(left) if exponent < 0 else left
            return base**exponent
        raise AssertionError("parser admitted an unsupported binary operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_walk(argument, environment) for argument in node.args]
        return _call_function(node.func.id, arguments)
    raise AssertionError(f"parser admitted unsupported node {type(node).__name__}")


def _guard_denominator(value: float) -> float:
    """Keep division finite near zero without changing denominator sign."""
    if abs(value) >= SAFE_DIVISION_EPSILON:
        return value
    return SAFE_DIVISION_EPSILON if value >= 0.0 else -SAFE_DIVISION_EPSILON


def _call_function(name: str, arguments: list[float]) -> float:
    argument = arguments[0]
    if name == "exp":
        return math.exp(argument)
    if name == "log":
        return math.log(argument)
    if name == "tanh":
        return math.tanh(argument)
    if name == "sqrt":
        return math.sqrt(argument)
    if name == "abs":
        return abs(argument)
    if name == "min":
        return min(arguments)
    if name == "max":
        return max(arguments)
    if name == "sigmoid":
        if argument >= 0.0:
            return 1.0 / (1.0 + math.exp(-argument))
        exp_value = math.exp(argument)
        return exp_value / (1.0 + exp_value)
    if name == "softplus":
        return max(argument, 0.0) + math.log1p(math.exp(-abs(argument)))
    raise AssertionError(f"parser admitted unsupported function {name}")


def _finite_float(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeExpressionError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric):
        raise RuntimeExpressionError(f"{label} must be finite")
    return numeric
