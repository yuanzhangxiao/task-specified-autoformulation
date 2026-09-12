"""Deterministic state meshes independent of observation sampling.

Every forcing interpolation corner is retained. The variable target is soft
when those mandatory corners alone exceed it; input data are never decimated.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting.simulation import trajectory_forcing


@dataclass
class StateMesh:
    """Finite-element boundaries plus exact association of all observations."""

    time: np.ndarray
    observation_intervals: np.ndarray
    observation_fractions: np.ndarray
    mandatory_points: int
    forcing_interpolation_error: float


def basis(fraction):
    """Lagrange polynomials through the Radau nodes 0, 1/3 and 1."""
    return (
        3 * fraction * fraction - 4 * fraction + 1,
        4.5 * fraction * (1 - fraction),
        1.5 * fraction * fraction - 0.5 * fraction,
    )


def forcing_values(system, data: Trajectory, time: np.ndarray) -> np.ndarray:
    forcing = trajectory_forcing(system.model, data)
    return np.array(
        [[forcing.value(name, t) for t in time] for name in system.inputs]
    ).reshape(len(system.inputs), len(time))


def _mandatory(time: np.ndarray, inputs: np.ndarray) -> set[int]:
    points = {0, len(time) - 1}
    if inputs.size and len(time) > 2:
        slopes = np.diff(inputs, axis=1) / np.diff(time)
        scale = np.maximum(np.max(np.abs(slopes), axis=1), np.finfo(float).tiny)
        corners = np.any(
            np.abs(np.diff(slopes, axis=1)) > 1e-12 * scale[:, None], axis=0
        )
        points.update((np.flatnonzero(corners) + 1).tolist())
    return points


def plan_meshes(
    system, training: DatasetSplit, target_variables: int | None, substeps: int = 1
) -> tuple[list[StateMesh], dict]:
    """Allocate a global state-variable target, retaining forcing corners exactly.

    Bisect the largest relative-time gap first, with deterministic index ties.
    The observation objective always retains every original sample.
    """
    if training.name is not SplitName.TRAIN:
        raise ValueError("collocation mesh requires training data")
    if target_variables is not None and substeps != 1:
        raise ValueError("choose either a variable target or uniform substeps")
    rows = training.trajectories
    inputs = [forcing_values(system, row, row.time) for row in rows]
    mandatory = [_mandatory(row.time, u) for row, u in zip(rows, inputs, strict=True)]
    selected = [set(x) for x in mandatory]
    full_intervals = sum(len(row.time) - 1 for row in rows)
    floor = sum(len(x) - 1 for x in selected)
    desired = (
        full_intervals
        if target_variables is None
        else max(
            floor, (target_variables - len(system.names)) // (2 * system.state_count)
        )
    )
    desired = min(full_intervals, desired)
    queue = []

    def add_gap(j, left, right):
        if right - left > 1:
            t = rows[j].time
            heapq.heappush(
                queue, (-(t[right] - t[left]) / (t[-1] - t[0]), j, left, right)
            )

    for j, points in enumerate(selected):
        ordered = sorted(points)
        for left, right in pairwise(ordered):
            add_gap(j, left, right)
    intervals = floor
    while queue and intervals < desired:
        _, j, left, right = heapq.heappop(queue)
        t = rows[j].time
        middle = 0.5 * (t[left] + t[right])
        index = int(np.clip(np.searchsorted(t, middle), left + 1, right - 1))
        if index > left + 1 and abs(t[index - 1] - middle) <= abs(t[index] - middle):
            index -= 1
        selected[j].add(index)
        intervals += 1
        add_gap(j, left, index)
        add_gap(j, index, right)

    meshes = []
    for row, u, points, required in zip(rows, inputs, selected, mandatory, strict=True):
        time = row.time[sorted(points)]
        if substeps > 1:
            time = np.concatenate(
                [np.linspace(a, b, substeps + 1)[:-1] for a, b in pairwise(time)]
                + [time[-1:]]
            )
        reconstructed = np.array(
            [
                np.interp(row.time, time, np.interp(time, row.time, values))
                for values in u
            ]
        )
        error = float(np.max(np.abs(reconstructed - u), initial=0.0)) if u.size else 0.0
        scale = max(np.finfo(float).tiny, float(np.max(np.abs(u), initial=0.0)))
        if error > 1e-10 * scale:
            # Tiny slope differences can accumulate over a long horizon. In that
            # case retain the original mesh rather than changing the forcing.
            time = row.time.copy()
            error = 0.0
        indices = np.clip(
            np.searchsorted(time, row.time, side="left") - 1, 0, len(time) - 2
        )
        fractions = (row.time - time[indices]) / (time[indices + 1] - time[indices])
        meshes.append(StateMesh(time, indices, fractions, len(required), error))
    count = len(system.names) + 2 * system.state_count * sum(
        len(m.time) - 1 for m in meshes
    )
    return meshes, {
        "target_variables": target_variables,
        "actual_variables": count,
        "target_exceeded_for_input_fidelity": target_variables is not None
        and count > target_variables,
        "all_observations_retained": True,
        "input_interpolation_preserved": True,
        "trajectories": [
            {
                "trajectory_id": row.trajectory_id,
                "observations": len(row.time),
                "intervals": len(mesh.time) - 1,
                "mandatory_input_points": mesh.mandatory_points,
                "forcing_interpolation_error": mesh.forcing_interpolation_error,
                "boundaries": mesh.time.tolist(),
            }
            for row, mesh in zip(rows, meshes, strict=True)
        ],
    }
