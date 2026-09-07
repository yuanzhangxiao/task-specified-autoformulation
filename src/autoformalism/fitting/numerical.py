"""Bound-aware finite differences and preservation of evaluated fit progress."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from autoformalism.fitting.models import FitConfig


class TrackedResidual:
    """Count physical calls and retain only finite, integration-valid trials."""

    def __init__(self, function: Callable, failure_count: Callable[[], int]) -> None:
        self.function = function
        self.failure_count = failure_count
        self.calls = 0
        self.best_x: np.ndarray | None = None
        self.best_cost = np.inf
        self.last_x: np.ndarray | None = None
        self.last_residual: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        before = self.failure_count()
        residual = np.asarray(self.function(x))
        self.last_x = np.asarray(x).copy()
        self.last_residual = residual.copy()
        cost = float(0.5 * residual @ residual)
        if (
            self.failure_count() == before
            and np.isfinite(cost)
            and np.isfinite(self.last_x).all()
            and cost < self.best_cost
        ):
            self.best_x = self.last_x.copy()
            self.best_cost = cost
        return residual

    def at(self, x: np.ndarray) -> np.ndarray:
        """Reuse the optimizer's base residual only for its Jacobian calculation."""
        if self.last_x is not None and np.array_equal(x, self.last_x):
            assert self.last_residual is not None
            return self.last_residual.copy()
        return self(x)


def bounded_step(value: float, lower: float, upper: float, amount: float) -> float:
    """Prefer a forward probe, switching direction or shortening at hard bounds."""
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError("finite-difference base is outside its domain")
    if not np.isfinite(amount) or amount <= 0 or lower >= upper:
        raise ValueError("finite differences require positive steps and domain width")
    forward, backward = upper - value, value - lower
    if forward >= amount:
        point = min(upper, value + amount)
    elif backward >= amount:
        point = max(lower, value - amount)
    elif forward >= backward:
        point = upper
    else:
        point = lower
    actual = point - value
    if actual == 0 or not np.isfinite(actual):
        raise ValueError("finite-difference step is not representable")
    return float(actual)


class ScaledJacobian:
    """One-sided differences with an explicit parameter-scale floor, not new bounds."""

    def __init__(
        self,
        residual: TrackedResidual,
        lower: np.ndarray,
        upper: np.ndarray,
        step: float,
        scale_floor: float,
    ) -> None:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("finite-difference step must be positive and finite")
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("finite-difference scale must be positive and finite")
        self.residual = residual
        self.lower = lower
        self.upper = upper
        self.step = step
        self.scale_floor = scale_floor
        self.last_steps: list[float] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        base = self.residual.at(x)
        columns = []
        self.last_steps = []
        for index, value in enumerate(x):
            step = bounded_step(
                value,
                self.lower[index],
                self.upper[index],
                self.step * max(abs(value), self.scale_floor),
            )
            point = x.copy()
            point[index] = np.clip(value + step, self.lower[index], self.upper[index])
            step = point[index] - value
            if step == 0:
                raise ValueError(
                    "bounded finite-difference probe has zero displacement"
                )
            columns.append((self.residual(point) - base) / step)
            self.last_steps.append(step)
        return np.column_stack(columns)


def jacobian_options(
    residual: TrackedResidual, lower: np.ndarray, upper: np.ndarray, config: FitConfig
) -> dict:
    """Keep SciPy defaults intact unless an explicit finite-difference policy is set."""
    if config.finite_difference_policy == "scaled":
        return {
            "jac": ScaledJacobian(
                residual,
                lower,
                upper,
                config.finite_difference_step or 1e-4,
                config.finite_difference_scale_floor,
            )
        }
    if config.finite_difference_step is not None:
        return {"diff_step": config.finite_difference_step}
    return {}
