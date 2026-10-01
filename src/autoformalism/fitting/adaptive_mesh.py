"""Training-only mesh construction and Radau dense output for opt-in fitting.

Shooting boundaries are optimization variables, not input interpolation knots.
Every supplied knot remains an integration boundary even when it is not a free
shooting node. Collocation starts at every sample and only subdivides intervals.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def polynomial(start, inner, end, fraction):
    """Quadratic through Radau nodes 0, 1/3 and 1; symbolic or numerical values."""
    linear = (9 * inner - 8 * start - end) / 2
    quadratic = (3 * end + 6 * start - 9 * inner) / 2
    return start + fraction * linear + fraction**2 * quadratic


def polynomial_derivative(start, inner, end, fraction):
    """Derivative with respect to the unit interval coordinate, not physical time."""
    return (9 * inner - 8 * start - end) / 2 + fraction * (
        3 * end + 6 * start - 9 * inner
    )


def validate_mesh(time, mesh) -> np.ndarray:
    """Reject malformed meshes before creating optimizer variables."""
    t, m = np.asarray(time, float), np.asarray(mesh, float)
    if (
        t.ndim != 1
        or m.ndim != 1
        or len(t) < 2
        or len(m) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(m).all()
        or np.any(np.diff(t) <= 0)
        or np.any(np.diff(m) <= 0)
        or m[0] != t[0]
        or m[-1] != t[-1]
    ):
        raise ValueError("invalid time/optimization mesh")
    return m


def initial_mesh(row, method: str) -> list[float]:
    """Frozen observed-response heuristic; no hidden labels or reference values."""
    t = np.asarray(row.time)
    if method == "collocation":
        return t.tolist()
    if method != "shooting":
        raise ValueError("unknown mesh method")
    # Up to twelve uniform windows plus the strongest observable/input corners.
    selected = {0, len(t) - 1, *np.linspace(0, len(t) - 1, 13).astype(int)}
    for values in (*row.targets.values(), *row.external_inputs.values()):
        slope = np.diff(values) / np.diff(t)
        turn = abs(np.diff(slope))
        if len(turn) and np.max(turn) > 0:
            candidates = np.flatnonzero(turn > max(np.max(turn) * 0.25, 1e-12))
            ranked = sorted(candidates, key=lambda i: (-turn[i], i))[:8]
            selected.update(int(i + 1) for i in ranked)
    return t[sorted(selected)].tolist()


def integration_times(time, left: float, right: float) -> np.ndarray:
    """Retain all forcing/sample knots within a shooting interval, without resets."""
    t = np.asarray(time)
    return np.unique(np.r_[left, t[(t > left) & (t < right)], right])


def refine_mesh(mesh, scores, *, threshold: float, maximum_intervals: int) -> list:
    """Bisect the worst quarter of intervals above threshold, with a frozen cap."""
    m = validate_mesh(mesh, mesh)
    s = np.asarray(scores, float)
    if s.shape != (len(m) - 1,) or np.isnan(s).any() or np.any(s < 0):
        raise ValueError("invalid mesh indicators")
    count = min(maximum_intervals - len(s), max(1, int(np.ceil(len(s) / 4))))
    if count <= 0:
        return m.tolist()
    eligible = [i for i, v in enumerate(s) if v > threshold]
    chosen = sorted(eligible, key=lambda i: (-s[i], i))[:count]
    return sorted([*m.tolist(), *[(m[i] + m[i + 1]) / 2 for i in chosen]])


def interpolate_nodes(old_time, old_nodes, new_time) -> np.ndarray:
    """Transfer physical guesses between meshes; never constrain new nodes to data."""
    values = np.asarray(old_nodes, float)
    if (
        values.ndim != 2
        or len(values) != len(old_time)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid warm-start trajectory")
    return np.column_stack(
        [np.interp(new_time, old_time, values[:, j]) for j in range(values.shape[1])]
    )


def collocation_indicators(system, row, theta, nodes, inner, mesh, units) -> list:
    """Off-node ODE defects, not the equations already enforced by collocation."""
    from autoformalism.fitting.simulation import trajectory_forcing

    forcing = trajectory_forcing(system.model, row)
    scores = []
    for i, (left, right) in enumerate(pairwise(mesh)):
        worst = 0.0
        for fraction in (0.15, 0.55, 0.85):
            t = left + (right - left) * fraction
            x = polynomial(nodes[i], inner[i], nodes[i + 1], fraction)
            slope = polynomial_derivative(nodes[i], inner[i], nodes[i + 1], fraction)
            u = [forcing.value(n, t) for n in system.inputs]
            f = np.asarray(system.rhs(t, x, theta, u)).ravel()
            defect = (slope - (right - left) * f) / units
            value = float(np.max(abs(defect)))
            worst = max(worst, value) if np.isfinite(value) else float("inf")
        scores.append(worst)
    return scores
