"""Training-only observation anchors on top of input-preserving state meshes."""

from __future__ import annotations

import numpy as np

from autoformalism.fitting import mesh_refinement_worker as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicFitRequest


def observation_anchors(time, targets: dict, count: int) -> list[int]:
    """Bounded, separated high-slope/curvature locations; not a noise certificate.

    All observations remain in the objective. These anchors only allocate state
    nodes. Neighbours bracket the selected changes rather than dropping samples.
    """
    time = np.asarray(time, float)
    scores = np.zeros(len(time))
    for values in targets.values():
        y = np.asarray(values, float)
        if y.shape != time.shape or not np.isfinite(y).all():
            raise ValueError("invalid observed mesh signal")
        span = float(np.ptp(y))
        if span <= 1e-12 or len(time) < 3:
            continue
        speed = np.abs(np.diff(y) / np.diff(time)) / span
        bend = np.abs(np.diff(np.diff(y) / np.diff(time))) / span
        scores[1:] = np.maximum(scores[1:], speed / max(float(speed.max()), 1e-12))
        scores[1:-1] = np.maximum(scores[1:-1], bend / max(float(bend.max()), 1e-12))
    selected = []
    for i in sorted(range(1, len(time) - 1), key=lambda i: (-scores[i], i)):
        if len(selected) >= count or scores[i] < 0.1:
            break
        if all(abs(i - j) >= 3 for j in selected):
            selected.append(i)
    return sorted({j for i in selected for j in (i - 1, i, i + 1)})


def grids(base: dict, target: int | None, minimum: int, anchors: int):
    """Add a fixed training-derived anchor set to every nested ladder level."""
    meshes, audit = mesh.grids(base, target, minimum)
    train = public.unpack_split(TrainingOnlySplit.model_validate(base["training"]))
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(base["request"]))[0],
        allow_piecewise=True,
    )
    for row, detail in zip(train.trajectories, audit["trajectories"], strict=True):
        points = observation_anchors(row.time, row.targets, anchors)
        old = meshes[row.trajectory_id]
        new = sorted(set(old).union(row.time[points].tolist()))
        meshes[row.trajectory_id] = new
        detail.update(
            observation_anchor_indices=points,
            observation_nodes_added=len(new) - len(old),
            boundaries=new,
            intervals=len(new) - 1,
            maximum_interval=max(np.diff(new)),
        )
    audit.update(
        observation_anchor_policy="bounded-slope-curvature-1",
        observation_anchor_centres_per_trajectory=anchors,
        actual_variables=len(system.names)
        + 2 * system.state_count * sum(len(m) - 1 for m in meshes.values()),
    )
    return meshes, audit


def initial_guesses(base: dict, meshes: dict) -> dict:
    """Interpolate frozen generic guesses, never a fitted/reference trajectory."""
    train = public.unpack_split(TrainingOnlySplit.model_validate(base["training"]))
    nodes, inner = {}, {}
    for row in train.trajectories:
        key = row.trajectory_id
        old = np.asarray(base["nodes"][key], float)
        time = np.asarray(meshes[key], float)
        if old.ndim != 2 or len(old) != len(row.time) or not np.isfinite(old).all():
            raise ValueError("invalid frozen generic nodes")
        nodes[key] = np.column_stack(
            [np.interp(time, row.time, x) for x in old.T]
        ).tolist()
        middle = time[:-1] + np.diff(time) / 3
        inner[key] = np.column_stack(
            [np.interp(middle, row.time, x) for x in old.T]
        ).tolist()
    return {"nodes": nodes, "inner_nodes": inner}
