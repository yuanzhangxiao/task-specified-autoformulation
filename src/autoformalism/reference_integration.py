"""Trusted reference integration with explicit forcing boundaries.

This module is for benchmark generators, not discovery-model execution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import solve_ivp

REFERENCE_PROTOCOL = "reference-events-2"
Vector = NDArray[np.float64]
ReferenceRHS = Callable[[float, Vector], Vector]


class ReferenceSolver(BaseModel):
    """Recorded numerical settings, independent of the requested output grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    method: Literal["LSODA", "Radau", "DOP853"] = "LSODA"
    rtol: float = Field(default=1e-10, gt=0, allow_inf_nan=False)
    atol: float = Field(default=1e-12, gt=0, allow_inf_nan=False)
    max_step: float = Field(default=0.5, gt=0, allow_inf_nan=False)


def reference_time_grid(duration: float, dt: float) -> Vector:
    """Require a finite, uniform grid that includes the true horizon exactly."""
    if not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    count = round(duration / dt)
    if count < 1 or not np.isclose(count * dt, duration, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer multiple of dt")
    return np.linspace(0.0, duration, count + 1)


def integrate_reference(
    rhs: ReferenceRHS,
    time: Vector,
    initial: Vector,
    *,
    boundaries: Sequence[float] = (),
    solver: ReferenceSolver | None = None,
) -> Vector:
    """Restart at every forcing discontinuity, preserving continuous states.

    Evaluate each segment's right endpoint from the left. Returned states are
    continuous there; exported inputs/derivatives may use right-hand values.
    Always solve to the boundary, even when it is absent from ``time``.
    """
    settings = solver or ReferenceSolver()
    time = np.asarray(time, dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.isfinite(time).all()
        or not np.all(np.diff(time) > 0)
        or state.ndim != 1
        or not len(state)
        or not np.isfinite(state).all()
    ):
        raise ValueError("invalid reference time grid or initial state")
    knots = np.asarray(boundaries, dtype=float)
    if knots.ndim != 1 or not np.isfinite(knots).all():
        raise ValueError("reference boundaries must be finite")
    if np.any(knots < time[0]) or np.any(knots > time[-1]):
        raise ValueError("reference boundary outside integration horizon")
    knots = np.unique(np.r_[time[0], knots, time[-1]])
    output = np.empty((len(time), len(state)))
    output[0] = state
    for left, right in pairwise(knots):
        indices = np.flatnonzero((time > left) & (time <= right))
        evaluation = np.unique(np.r_[time[indices], right])

        def interval_rhs(
            t: float, values: Vector, end: float = right, start: float = left
        ) -> Vector:
            return rhs(min(t, float(np.nextafter(end, start))), values)

        solution = solve_ivp(
            interval_rhs,
            (left, right),
            state,
            t_eval=evaluation,
            **settings.model_dump(),
        )
        if (
            not solution.success
            or np.shape(solution.y) != (len(state), len(evaluation))
            or not np.isfinite(solution.y).all()
            or not np.array_equal(solution.t, evaluation)
        ):
            raise RuntimeError(
                f"private reference simulation failed: {solution.message}"
            )
        output[indices] = solution.y[:, : len(indices)].T
        state = solution.y[:, -1]
    return output
