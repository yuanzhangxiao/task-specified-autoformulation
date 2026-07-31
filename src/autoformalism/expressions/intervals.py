"""Conservative interval analysis for hazardous expression domains."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from autoformalism.expressions.diagnostics import ValidationDiagnostic


@dataclass(frozen=True)
class Interval:
    """Conservative closed interval."""

    lower: float
    upper: float

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper


UNKNOWN_INTERVAL = Interval(-math.inf, math.inf)


def analyze_interval(
    node: ast.AST,
    intervals: dict[str, Interval],
    *,
    location: str,
    diagnostics: list[ValidationDiagnostic],
) -> Interval:
    """Infer an interval and append domain-risk diagnostics."""
    if isinstance(node, ast.Expression):
        return analyze_interval(
            node.body, intervals, location=location, diagnostics=diagnostics
        )
    if isinstance(node, ast.Constant):
        value = float(node.value)
        return Interval(value, value)
    if isinstance(node, ast.Name):
        return intervals.get(node.id, UNKNOWN_INTERVAL)
    if isinstance(node, ast.UnaryOp):
        operand = analyze_interval(
            node.operand, intervals, location=location, diagnostics=diagnostics
        )
        if isinstance(node.op, ast.USub):
            return Interval(-operand.upper, -operand.lower)
        return operand
    if isinstance(node, ast.BinOp):
        left = analyze_interval(
            node.left, intervals, location=location, diagnostics=diagnostics
        )
        right = analyze_interval(
            node.right, intervals, location=location, diagnostics=diagnostics
        )
        if isinstance(node.op, ast.Add):
            return Interval(left.lower + right.lower, left.upper + right.upper)
        if isinstance(node.op, ast.Sub):
            return Interval(left.lower - right.upper, left.upper - right.lower)
        if isinstance(node.op, ast.Mult):
            return _multiply(left, right)
        if isinstance(node.op, ast.Div):
            if right.contains_zero():
                diagnostics.append(
                    ValidationDiagnostic(
                        "DOMAIN_DIVISION_ZERO",
                        location,
                        "denominator may be zero",
                    )
                )
                return UNKNOWN_INTERVAL
            reciprocal = Interval(1.0 / right.upper, 1.0 / right.lower)
            return _multiply(left, reciprocal)
        if isinstance(node.op, ast.Pow):
            exponent = int(_signed_literal(node.right))
            if exponent < 0 and left.contains_zero():
                diagnostics.append(
                    ValidationDiagnostic(
                        "DOMAIN_DIVISION_ZERO",
                        location,
                        "negative power base may be zero",
                    )
                )
                return UNKNOWN_INTERVAL
            return _integer_power(left, exponent)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [
            analyze_interval(
                argument,
                intervals,
                location=location,
                diagnostics=diagnostics,
            )
            for argument in node.args
        ]
        function = node.func.id
        argument = arguments[0]
        if function == "exp":
            return Interval(_safe_exp(argument.lower), _safe_exp(argument.upper))
        if function == "log":
            if argument.lower <= 0.0:
                diagnostics.append(
                    ValidationDiagnostic(
                        "DOMAIN_LOG_NONPOSITIVE",
                        location,
                        "logarithm argument may be nonpositive",
                    )
                )
                return UNKNOWN_INTERVAL
            return Interval(math.log(argument.lower), math.log(argument.upper))
        if function == "sqrt":
            if argument.lower < 0.0:
                diagnostics.append(
                    ValidationDiagnostic(
                        "DOMAIN_SQRT_NEGATIVE",
                        location,
                        "square-root argument may be negative",
                    )
                )
                return UNKNOWN_INTERVAL
            return Interval(math.sqrt(argument.lower), math.sqrt(argument.upper))
        if function == "abs":
            if argument.contains_zero():
                return Interval(0.0, max(abs(argument.lower), abs(argument.upper)))
            values = (abs(argument.lower), abs(argument.upper))
            return Interval(min(values), max(values))
        if function == "tanh":
            return Interval(math.tanh(argument.lower), math.tanh(argument.upper))
        if function == "sigmoid":
            return Interval(_sigmoid(argument.lower), _sigmoid(argument.upper))
        if function == "softplus":
            return Interval(_softplus(argument.lower), _softplus(argument.upper))
        if function == "min":
            return Interval(
                min(item.lower for item in arguments),
                min(item.upper for item in arguments),
            )
        if function == "max":
            return Interval(
                max(item.lower for item in arguments),
                max(item.upper for item in arguments),
            )
    return UNKNOWN_INTERVAL


def _multiply(left: Interval, right: Interval) -> Interval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    if any(math.isnan(item) for item in products):
        return UNKNOWN_INTERVAL
    return Interval(min(products), max(products))


def _integer_power(base: Interval, exponent: int) -> Interval:
    if exponent == 0:
        return Interval(1.0, 1.0)
    if exponent < 0:
        positive = _integer_power(base, -exponent)
        return Interval(1.0 / positive.upper, 1.0 / positive.lower)
    if exponent % 2 == 0:
        values = (base.lower**exponent, base.upper**exponent)
        lower = 0.0 if base.contains_zero() else min(values)
        return Interval(lower, max(values))
    return Interval(base.lower**exponent, base.upper**exponent)


def _signed_literal(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    raise AssertionError("parser admitted a nonliteral power")


def _safe_exp(value: float) -> float:
    if value == -math.inf:
        return 0.0
    if value == math.inf or value > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.inf
    return math.exp(value)


def _sigmoid(value: float) -> float:
    if value == math.inf:
        return 1.0
    if value == -math.inf:
        return 0.0
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softplus(value: float) -> float:
    if value == math.inf:
        return math.inf
    if value == -math.inf:
        return 0.0
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

