"""Run LLM-ODE's search on Phase-B cells at a pinned upstream revision.

The search is not reimplemented. Upstream's per-variable searcher receives our
trajectories, and upstream's own `System` evaluates the assembled candidates.
This module supplies data, applies the declared prompt adaptation, reproduces
upstream's selection rule, and seals the result.

Three accommodations are forced by the benchmark and are declared in the plan
rather than hidden here:

* Upstream's driver builds its data by integrating a ground-truth equation it
  is given. We cannot do that, so the per-variable searcher is constructed
  directly from our observed trajectories.
* Upstream assumes one trajectory per problem. A Phase-B cell has several, so
  derivatives are estimated within each trajectory and the rows stacked. The
  regression is pointwise, so stacking is sound; differentiating across a
  join would not be.
* Upstream's selection hands its evaluator the sealed test trajectory. Ours
  cannot see it, so selection uses the same rule -- product of the per-variable
  Pareto frontiers, ranked by rolled-out error -- over development data only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from autoformalism.data import DatasetSplit

#: Upstream differentiates with findiff at fourth order; keep that choice.
DERIVATIVE_ACCURACY = 4


class SearcherFactory(Protocol):
    """Constructs one upstream per-variable searcher, injected for testing."""

    def __call__(
        self,
        *,
        time: NDArray[np.float64],
        states: NDArray[np.float64],
        derivative: NDArray[np.float64],
        variable_index: int,
    ) -> object: ...


@dataclass(frozen=True)
class CellArrays:
    """Development data in the shape upstream's searcher expects."""

    time: NDArray[np.float64]
    states: NDArray[np.float64]
    derivatives: NDArray[np.float64]
    channels: tuple[str, ...]
    trajectory_bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.states.shape != self.derivatives.shape:
            raise ValueError("states and derivatives must have the same shape")
        if self.states.shape[1] != len(self.channels):
            raise ValueError("state columns must match the observed channels")
        if not self.trajectory_bounds:
            raise ValueError("require at least one trajectory")


def observed_channels(split: DatasetSplit) -> tuple[str, ...]:
    """Name the channels upstream will treat as system variables."""
    first = split.trajectories[0]
    channels = (*first.targets, *first.auxiliaries)
    for trajectory in split.trajectories:
        if (*trajectory.targets, *trajectory.auxiliaries) != channels:
            raise ValueError("channel identities differ across trajectories")
    return channels


def finite_difference(
    values: NDArray[np.float64], step: float
) -> NDArray[np.float64]:
    """Fourth-order central differences, as upstream's findiff call computes.

    Implemented directly so the campaign does not depend on findiff being
    installed to prepare data; the coefficients are the standard ones upstream
    requests with ``acc=4`` and are checked against an exact polynomial.
    """
    if values.shape[0] < 5:
        raise ValueError("fourth-order differences need at least five samples")
    result = np.empty_like(values, dtype=float)
    interior = slice(2, -2)
    result[interior] = (
        values[:-4] - 8.0 * values[1:-3] + 8.0 * values[3:-1] - values[4:]
    ) / (12.0 * step)
    # One-sided fourth-order stencils keep the endpoints at the same order.
    forward = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / (12.0 * step)
    backward = -forward[::-1]
    count = values.shape[0]
    for index in (0, 1):
        result[index] = np.tensordot(forward, values[index : index + 5], axes=1)
    for index in (count - 2, count - 1):
        result[index] = np.tensordot(backward, values[index - 4 : index + 1], axes=1)
    return result


def cell_arrays(split: DatasetSplit) -> CellArrays:
    """Stack every trajectory after differentiating within each one."""
    channels = observed_channels(split)
    times: list[NDArray[np.float64]] = []
    states: list[NDArray[np.float64]] = []
    derivatives: list[NDArray[np.float64]] = []
    bounds: list[tuple[int, int]] = []
    start = 0
    for trajectory in split.trajectories:
        measured = {**trajectory.targets, **trajectory.auxiliaries}
        block = np.column_stack(
            [np.asarray(measured[name], float) for name in channels]
        )
        time = np.asarray(trajectory.time, dtype=float)
        steps = np.diff(time)
        if not np.allclose(steps, steps[0]):
            raise ValueError("fourth-order differences require a uniform grid")
        derivatives.append(finite_difference(block, float(steps[0])))
        states.append(block)
        times.append(time)
        bounds.append((start, start + len(time)))
        start += len(time)
    return CellArrays(
        time=np.concatenate(times),
        states=np.vstack(states),
        derivatives=np.vstack(derivatives),
        channels=channels,
        trajectory_bounds=tuple(bounds),
    )


def specification_block(
    prompt: str, channels: tuple[str, ...], variable_index: int
) -> str:
    """Append the public task specification to an upstream prompt.

    A declared adaptation: upstream withholds variable meanings to mitigate
    memorisation, which this benchmark controls for with its named and
    obfuscated pairing. The wording is fixed and adds no private information,
    no reference knowledge and nothing about our own pipeline.
    """
    named = ", ".join(f"x{index} = {name}" for index, name in enumerate(channels))
    return (
        "\n\nTask specification (public):\n"
        f"{prompt.strip()}\n\n"
        f"Variables: {named}.\n"
        f"You are proposing the time derivative of x{variable_index} "
        f"({channels[variable_index]})."
    )


def select_system(
    frontiers: tuple[tuple[str, ...], ...],
    score: Callable[[tuple[str, ...]], float | None],
    *,
    maximum_combinations: int = 10_000,
) -> tuple[str, ...] | None:
    """Rank the product of per-variable frontiers, as upstream's driver does.

    Upstream sorts assembled systems by their rolled-out error and keeps the
    best. The cap exists because the product grows multiplicatively and a
    truncated search would otherwise be reported as a completed one.
    """
    if not frontiers or any(not item for item in frontiers):
        return None
    total = 1
    for item in frontiers:
        total *= len(item)
    if total > maximum_combinations:
        raise ValueError(
            f"{total} assembled systems exceed the {maximum_combinations} cap; "
            "record the truncation rather than silently ranking a subset"
        )
    best: tuple[str, ...] | None = None
    best_score = np.inf
    for combination in product(*frontiers):
        value = score(combination)
        if value is None or not np.isfinite(value):
            continue
        if value < best_score:
            best_score, best = float(value), combination
    return best
