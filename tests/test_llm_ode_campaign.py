"""Supplying Phase-B data to LLM-ODE's search without altering it.

Upstream builds its own data by integrating a ground-truth equation, assumes
one trajectory, and hands its selector the sealed test trajectory. None of
those hold here, so this module's job is to accommodate the benchmark while
leaving the search, the derivative order and the selection rule alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.rebuttal.llm_ode_campaign import (
    cell_arrays,
    finite_difference,
    observed_channels,
    select_system,
    specification_block,
)


def _trajectory(identifier: str, count: int = 12, *, step: float = 0.1) -> Trajectory:
    time = np.arange(count) * step
    return Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"y": time**2},
        auxiliaries={"u": 2.0 * time},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )


def _split(*trajectories: Trajectory) -> DatasetSplit:
    return DatasetSplit(SplitName.TRAIN, trajectories, "train")


def test_the_difference_is_fourth_order_as_upstream_requests() -> None:
    """Upstream asks findiff for acc=4, which is exact on a quartic."""
    time = np.linspace(0.0, 2.0, 40)
    values = np.column_stack([time**4 - 3 * time**2, 2 * time**3])
    exact = np.column_stack([4 * time**3 - 6 * time, 6 * time**2])
    got = finite_difference(values, float(time[1] - time[0]))
    assert np.max(np.abs(got - exact)) < 1e-9
    # a second-order rule would not be exact here, so this distinguishes them
    second_order = np.gradient(values, float(time[1] - time[0]), axis=0)
    assert np.max(np.abs(second_order - exact)) > 1e-4


def test_each_trajectory_is_differentiated_before_the_rows_are_stacked() -> None:
    """Differentiating across a join would invent a derivative at the seam."""
    split = _split(_trajectory("a"), _trajectory("b"))
    arrays = cell_arrays(split)
    assert arrays.states.shape == (24, 2)
    assert arrays.derivatives.shape == (24, 2)
    assert arrays.trajectory_bounds == ((0, 12), (12, 24))
    assert arrays.channels == ("y", "u")
    # the seam carries each trajectory's own endpoint derivative, not a blend
    single = cell_arrays(_split(_trajectory("a")))
    assert np.allclose(arrays.derivatives[:12], single.derivatives)
    assert np.allclose(arrays.derivatives[12:], single.derivatives)


def test_a_nonuniform_grid_is_refused() -> None:
    """The stencil assumes a fixed step; silently applying it would be wrong."""
    time = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    trajectory = Trajectory(
        trajectory_id="a",
        time=time,
        targets={"y": time**2},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    with pytest.raises(ValueError, match="uniform grid"):
        cell_arrays(_split(trajectory))


def test_channel_identities_must_agree_across_trajectories() -> None:
    other = Trajectory(
        trajectory_id="b",
        time=np.arange(12) * 0.1,
        targets={"z": np.arange(12, dtype=float)},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    with pytest.raises(ValueError, match="channel identities differ"):
        observed_channels(_split(_trajectory("a"), other))


def test_the_specification_names_the_variables_and_the_derivative() -> None:
    """The adaptation must be unambiguous about what is being asked."""
    block = specification_block("Recover the glucose flux.", ("y", "u"), 1)
    assert "Recover the glucose flux." in block
    assert "x0 = y" in block and "x1 = u" in block
    assert "derivative of x1 (u)" in block
    # nothing about our own pipeline leaks into their prompt
    for forbidden in ("Autoformalism", "judge", "schema", "proposer"):
        assert forbidden.lower() not in block.lower()


def test_selection_ranks_the_product_of_the_per_variable_frontiers() -> None:
    """Upstream assembles systems from the frontiers and keeps the best."""
    frontiers = (("a1", "a2"), ("b1", "b2"))
    scores = {("a1", "b1"): 5.0, ("a1", "b2"): 2.0, ("a2", "b1"): 9.0}
    assert select_system(frontiers, lambda item: scores.get(item)) == ("a1", "b2")
    # a candidate that will not integrate is skipped, not ranked as zero
    assert select_system(frontiers, lambda item: None) is None


def test_an_exploding_product_is_refused_rather_than_truncated() -> None:
    """Ranking a subset and reporting it as the search would be a false claim."""
    frontiers = tuple(tuple(f"v{index}_{n}" for n in range(30)) for index in range(4))
    with pytest.raises(ValueError, match="exceed the"):
        select_system(frontiers, lambda item: 1.0, maximum_combinations=1000)
