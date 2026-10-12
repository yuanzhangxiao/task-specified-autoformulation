"""Conservative stopping evidence for inaccurate, stationary local searches."""

import numpy as np
from pydantic import Field, model_validator

from autoformalism.schemas.base import StrictSchema


class Policy(StrictSchema):
    """Opt-in thresholds in fixed optimizer coordinates; no accuracy relaxation."""

    minimum_nmse: float = Field(default=1e-6, ge=1e-6, le=1)
    gradient_tolerance: float = Field(default=1e-8, gt=0, le=1e-6)
    correlation_tolerance: float = Field(default=1e-8, gt=0, le=1e-6)
    loss_tolerance: float = Field(default=1e-10, gt=0, le=1e-6)
    step_tolerance: float = Field(default=1e-6, gt=0, le=1e-4)
    window: int = Field(default=5, ge=3, le=10)

    @model_validator(mode="after")
    def finite(self):
        if not np.isfinite(list(self.model_dump().values())).all():
            raise ValueError("stationarity thresholds must be finite")
        return self


def witness(history, policy: Policy, target: float) -> dict | None:
    """Require poor loss, bound-aware gradient AND correlation, flat loss/steps.

    Zero Jacobian at one point permits leaving that start, never removing its
    coordinates or certifying a minimum. Distinct starts remain mandatory.
    """
    last = history[-1]
    if last["loss"] <= max(policy.minimum_nmse, 100 * target):
        return None
    if len(history) == 1 and last["zero_jacobian"]:
        return {"reason": "locally_flat_start", "global_minimum_certified": False}
    if (
        len(history) == 1
        and last["gradient"] <= 64 * np.finfo(float).eps
        and last["correlation"] <= 64 * np.finfo(float).eps
    ):
        return {
            "reason": "first_order_stationary_start",
            "projected_gradient_inf": last["gradient"],
            "maximum_residual_column_correlation": last["correlation"],
            "global_minimum_certified": False,
            "progress_history_available": False,
        }
    if len(history) < policy.window:
        return None
    recent = history[-policy.window :]
    losses = [v["loss"] for v in recent]
    steps = [np.linalg.norm(v["q"] - last["q"], ord=np.inf) for v in recent]
    if (
        last["gradient"] <= policy.gradient_tolerance
        and last["correlation"] <= policy.correlation_tolerance
        and max(losses) - min(losses) <= policy.loss_tolerance * max(1, last["loss"])
        and max(steps) <= policy.step_tolerance
    ):
        return {
            "reason": "small_projected_gradient_and_stable_local_search",
            "projected_gradient_inf": last["gradient"],
            "maximum_residual_column_correlation": last["correlation"],
            "loss_range": max(losses) - min(losses),
            "coordinate_step_inf": max(steps),
            "evaluations": len(recent),
            "global_minimum_certified": False,
        }
    return None


def sample(q, vector, residual, jacobian, lower, upper):
    """Use unamplified MSE and column correlations to protect weak directions."""
    product = jacobian.T @ residual
    tolerance = 1e-10 * np.maximum(1, np.abs(vector))
    active = ((vector - lower <= tolerance) & (product > 0)) | (
        (upper - vector <= tolerance) & (product < 0)
    )
    product[active] = 0
    denominator = np.linalg.norm(jacobian, axis=0) * np.linalg.norm(residual)
    correlation = np.divide(
        np.abs(product), denominator, out=np.zeros_like(product), where=denominator > 0
    )
    return {
        "q": q.copy(),
        "loss": float(np.mean(residual**2)),
        "gradient": float(np.max(2 * np.abs(product) / len(residual), initial=0)),
        "correlation": float(np.max(correlation, initial=0)),
        "zero_jacobian": not np.any(jacobian),
    }
