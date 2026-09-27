"""Explicit continuous inputs shared by reference generation and public rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from autoformalism.reference_integration import reference_time_grid

InputContract = Literal["legacy-events-1", "continuous-rates-1"]
CONTINUOUS_INPUT_CONTRACT = "continuous-rates-1"
DEFAULT_INGESTION_MINUTES = 10.0


@dataclass(frozen=True)
class LinearInputSchedule:
    """A finite table defines the input, rather than approximating another input."""

    time: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float).copy()
        values = np.asarray(self.values, dtype=float).copy()
        if (
            time.ndim != 1
            or len(time) < 2
            or not np.isfinite(time).all()
            or not np.all(np.diff(time) > 0)
            or values.ndim != 2
            or values.shape[0] != len(time)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid continuous input table")
        time.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)

    def at(self, time: float) -> np.ndarray:
        """Use the same linear convention as PiecewiseLinearForcing."""
        if not np.isfinite(time) or not self.time[0] <= time <= self.time[-1]:
            raise ValueError("input query outside schedule")
        return np.array([np.interp(time, self.time, v) for v in self.values.T])

    @property
    def boundaries(self) -> np.ndarray:
        """Return every slope change, including both endpoints."""
        slopes = np.diff(self.values, axis=0) / np.diff(self.time)[:, None]
        changed = np.any(slopes[1:] != slopes[:-1], axis=1)
        return self.time[np.r_[True, changed, True]]


def merge_input_times(samples: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Prefer declared knots over numerically equivalent observation times.

    For example, 0.3 and three 0.1 intervals must not create a near-zero solver
    segment. Distinct declared knots below floating-point resolution fail closed.
    """
    knots = np.unique(knots)
    samples = np.asarray(samples, dtype=float)
    tolerance = 16 * np.finfo(float).eps * max(1, float(np.max(np.abs(samples))))
    if len(knots) > 1 and np.any(np.diff(knots) <= tolerance):
        raise ValueError("distinct input knots are below numerical time resolution")
    if len(knots):
        index = np.searchsorted(knots, samples)
        left = knots[np.clip(index - 1, 0, len(knots) - 1)]
        right = knots[np.clip(index, 0, len(knots) - 1)]
        samples = samples[
            np.minimum(abs(samples - left), abs(samples - right)) > tolerance
        ]
    return np.unique(np.r_[samples, knots])


def input_time_grid(duration: float, dt: float, knots=()) -> np.ndarray:
    """Retain explicit input knots even between ordinary observation samples."""
    knots = np.asarray(knots, dtype=float)
    if (
        knots.ndim != 1
        or not np.isfinite(knots).all()
        or np.any(knots < 0)
        or np.any(knots > duration)
    ):
        raise ValueError("input knots must lie within the horizon")
    return merge_input_times(reference_time_grid(duration, dt), knots)


def meal_rate_schedule(
    meals: tuple[tuple[float, float], ...],
    duration: float,
    dt: float,
    ingestion_minutes: float = DEFAULT_INGESTION_MINUTES,
    *,
    extra_knots=(),
) -> LinearInputSchedule:
    """Triangular ingestion rates; each complete pulse integrates to its grams.

    Pulses start at the supplied time, peak halfway, and finish after the fixed
    ingestion duration. Simultaneous/overlapping meals add. Truncation is an
    error, never a silent loss of meal mass. There are no initial-state jumps.
    """
    if not np.isfinite(ingestion_minutes) or ingestion_minutes <= 0:
        raise ValueError("ingestion duration must be finite and positive")
    knots = list(extra_knots)
    for start, grams in meals:
        if (
            not np.isfinite([start, grams]).all()
            or start < 0
            or grams <= 0
            or start + ingestion_minutes > duration
        ):
            raise ValueError("complete positive meal must fit within the horizon")
        knots.extend([start, start + ingestion_minutes / 2, start + ingestion_minutes])
    time = input_time_grid(duration, dt, knots)
    rate = np.zeros(len(time))
    for start, grams in meals:
        phase = (time - start) / ingestion_minutes
        rate += 2 * grams / ingestion_minutes * np.maximum(0, 1 - abs(2 * phase - 1))
    return LinearInputSchedule(time, rate[:, None])
