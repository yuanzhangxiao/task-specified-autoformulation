"""Small deterministic SINDy implementation using supplied derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SINDyFit:
    """Sparse coefficients and executable expression strings."""

    equations: dict[str, str]
    coefficients: NDArray[np.float64]


def library(
    values: NDArray[np.float64],
    names: Sequence[str],
    *,
    polynomial_feature_count: int | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Return the base polynomial/tanh library plus optional linear features."""
    base_count = (
        len(names)
        if polynomial_feature_count is None
        else polynomial_feature_count
    )
    if not 0 <= base_count <= len(names):
        raise ValueError("polynomial_feature_count is outside the feature range")
    columns = [np.ones(len(values))]
    expressions = ["1"]
    for index, name in enumerate(names[:base_count]):
        columns.append(values[:, index])
        expressions.append(name)
    for left in range(base_count):
        for right in range(left, base_count):
            columns.append(values[:, left] * values[:, right])
            expressions.append(f"{names[left]} * {names[right]}")
    for index, name in enumerate(names[:base_count]):
        columns.append(np.tanh(values[:, index]))
        expressions.append(f"tanh({name})")
    for index, name in enumerate(names[base_count:], start=base_count):
        columns.append(values[:, index])
        expressions.append(name)
    return np.column_stack(columns), tuple(expressions)


def fit_sindy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    names: Sequence[str],
    targets: Sequence[str],
    *,
    threshold: float,
    iterations: int = 20,
    polynomial_feature_count: int | None = None,
) -> SINDyFit:
    """Fit sequentially thresholded least squares with normalized columns."""
    theta, expressions = library(
        x, names, polynomial_feature_count=polynomial_feature_count
    )
    norms = np.maximum(np.linalg.norm(theta, axis=0), 1e-12)
    normalized = theta / norms
    coefficients = np.linalg.lstsq(normalized, y, rcond=None)[0]
    for _ in range(iterations):
        previous = coefficients.copy()
        coefficients[np.abs(coefficients) < threshold] = 0.0
        for output in range(y.shape[1]):
            active = coefficients[:, output] != 0.0
            if np.any(active):
                coefficients[active, output] = np.linalg.lstsq(
                    normalized[:, active], y[:, output], rcond=None
                )[0]
        if np.array_equal(previous == 0.0, coefficients == 0.0):
            break
    coefficients = coefficients / norms[:, None]
    equations = {
        target: _expression(coefficients[:, output], expressions)
        for output, target in enumerate(targets)
    }
    return SINDyFit(equations, coefficients)


def _expression(coefficients: NDArray[np.float64], terms: Sequence[str]) -> str:
    pieces = [
        f"({float(value):.17g}) * ({term})"
        for value, term in zip(coefficients, terms, strict=True)
        if value != 0.0
    ]
    return " + ".join(pieces) if pieces else "0"
