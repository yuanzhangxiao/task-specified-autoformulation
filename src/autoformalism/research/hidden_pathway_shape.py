"""Screen z'=-b z+g(u), y'=-a y+k z with finite-level public observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog

from autoformalism.research.trajectory_function_fitting import _rows

PREVIOUS = "previous_complete_knots"
LINEAR = "linear_complete_knots"
# Three Gauss points integrate a quartic weight times a linear interpolant exactly.
_POINTS, _WEIGHTS = np.polynomial.legendre.leggauss(3)
_POINTS, _WEIGHTS = (_POINTS + 1) / 2, _WEIGHTS / 2


def _window(t: np.ndarray, y: np.ndarray, u: np.ndarray) -> tuple:
    """Return moments for the linear interpolant of y and the held input levels."""
    width = t[-1] - t[0]
    dt = np.diff(t)
    r = ((t[:-1] - t[0])[:, None] + dt[:, None] * _POINTS) / width
    dr = dt[:, None] * _WEIGHTS / width
    measured = y[:-1, None] + np.diff(y)[:, None] * _POINTS
    # w(t)=30 r²(1-r)² / width, with integral 1 and w=w'=0 at both ends.
    w = 30 * r**2 * (1 - r) ** 2
    dw = 60 * r * (1 - r) * (1 - 2 * r) / width
    ddw = 60 * (1 - 6 * r + 6 * r**2) / width**2
    levels, codes = np.unique(u[:-1], return_inverse=True)
    weights = np.bincount(codes, weights=np.sum(dr * w, axis=1))
    return (
        float(np.sum(dr * dw * measured)),
        -float(np.sum(dr * w * measured)),
        dict(zip(levels.tolist(), weights.tolist(), strict=True)),
        float(np.sum(dr * ddw * measured)),
        width,
    )


def _moments(rows, target, source, interpolation, steps):
    moments = []
    for row in rows:
        y = np.asarray(row.targets[target])
        channels = row.inputs | row.auxiliaries | row.targets
        u = np.asarray(channels[source])
        t = np.asarray(row.time)
        # A linear ramp visits a continuum of g(u) values. Restrict the calculation
        # to constant runs; do not silently treat ramp samples as step-held.
        breaks = np.flatnonzero(np.diff(u) != 0) + 1 if interpolation == LINEAR else []
        for indices in np.split(np.arange(len(t)), breaks):
            if len(indices) <= steps:
                continue
            last = int(indices[-1]) - steps
            starts = sorted(
                {*range(int(indices[0]), last + 1, max(1, steps // 2)), last}
            )
            for start in starts:
                window = slice(start, start + steps + 1)
                moments.append(_window(t[window], y[window], u[window]))
    return moments


def _families(levels):
    """Compare nested degrees and one explicit saturation, in normalized units."""
    x = np.asarray(levels)
    x = (x - (x.min() + x.max()) / 2) / ((x.max() - x.min()) / 2)
    for degree, name in enumerate(("constant", "affine", "quadratic", "cubic")):
        yield name, np.vander(x, degree + 1, increasing=True)
    yield "fixed_tanh", np.column_stack([np.ones(len(x)), np.tanh(x)])


def _poles(s: float, p: float) -> bool:
    return s > 0 and p > 0 and s * s >= 4 * p


def _screen(matrix, rhs, widths, basis, output_error, max_condition):
    design = np.column_stack([matrix[:, :2], matrix[:, 2:] @ basis])
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0] = 1
    scaled = design / scales
    solution, _, rank, singular = np.linalg.lstsq(scaled, rhs, rcond=None)
    theta = solution / scales
    s, p = map(float, theta[:2])
    condition = float(singular[0] / singular[-1]) if rank == design.shape[1] else None
    residual = design @ theta - rhs
    values = {
        "rank": int(rank),
        "columns": design.shape[1],
        "condition": condition,
        "weak_rmse": float(np.sqrt(np.mean(residual**2))),
        "rate_sum": s,
        "rate_product": p,
        "q": (basis @ theta[2:]).tolist(),
        "positive_real_rates": _poles(s, p),
    }
    if condition is None or condition > max_condition:
        return {**values, "status": "abstain", "reason": "rank_or_condition"}
    if output_error is None:
        if not _poles(s, p):
            return {**values, "status": "abstain", "reason": "nominal_poles"}
        return {**values, "status": "estimated", "reason": "no_error_bound"}
    # For any true stable model satisfying the continuous output error contract,
    # |A theta - rhs| <= eps (||w''||₁ + s ||w'||₁ + p). Integrals of absolute
    # derivatives are analytic. Dropping the real-roots condition relaxes feasibility.
    constant_bound = output_error * (40 / np.sqrt(3)) / widths**2
    s_bound = output_error * 3.75 / widths
    allowance = constant_bound + s_bound * s + output_error * p
    scale = max(1.0, float(np.max(np.abs(rhs))), float(np.max(np.abs(design @ theta))))
    slack = 1e-9 * scale
    if _poles(s, p) and np.all(abs(residual) <= allowance + slack):
        return {**values, "status": "estimated", "reason": "within_moment_bounds"}
    rows = np.vstack([design, -design])
    rows[:, 0] -= np.tile(s_bound, 2)
    rows[:, 1] -= output_error
    limits = np.r_[rhs + constant_bound, -rhs + constant_bound] + slack
    feasible = linprog(
        np.zeros(len(theta)),
        A_ub=rows / scales,
        b_ub=limits,
        bounds=[(0, None)] * 2 + [(None, None)] * (len(theta) - 2),
        method="highs",
    )
    if feasible.status == 2:
        return {**values, "status": "excluded", "reason": "moment_bounds_infeasible"}
    # A feasible relaxed system may have nonphysical poles or require measurement
    # errors that disagree between windows. It does not prove an admissible ODE.
    return {**values, "status": "abstain", "reason": "no_verified_physical_estimate"}


def fit_hidden_pathway(
    rows: Sequence[dict],
    *,
    target: str,
    source: str,
    interpolation: str,
    output_error: float | None = None,
    window_steps: int = 32,
    max_condition: float = 1e7,
) -> dict:
    """Estimate q=k g on sampled levels under a two-decay pathway hypothesis.

    output_error bounds |true y - linear interpolant of observations| throughout
    the windows, in output units. It includes noise and interpolation error.
    Positive real poles and a match to the integral moments are necessary checks.
    Only s=a+b, p=ab and q can be inferred by this linear calculation.
    """
    data = _rows(rows, "train")
    if (
        target == source
        or isinstance(window_steps, bool)
        or not isinstance(window_steps, (int, np.integer))
        or window_steps < 4
        or not np.isfinite(max_condition)
        or max_condition < 1
        or (
            output_error is not None
            and (not np.isfinite(output_error) or output_error < 0)
        )
    ):
        raise ValueError("Invalid pathway, window width, conditioning or error bound")
    if target not in data[0].targets or source not in (
        data[0].inputs | data[0].auxiliaries | data[0].targets
    ):
        return {"status": "abstain", "reason": "missing_observations"}
    if interpolation not in {PREVIOUS, LINEAR}:
        return {"status": "abstain", "reason": "unknown_interpolation"}
    moments = _moments(data, target, source, interpolation, window_steps)
    levels = sorted({level for row in moments for level in row[2]})
    info = {
        "levels": levels,
        "windows": len(moments),
        "window_steps": window_steps,
        "output_error": output_error,
        "max_condition": max_condition,
        "fit_events": sorted(r.trajectory_id for r in data),
    }
    if not moments or len(levels) < 2:
        return {**info, "status": "abstain", "reason": "insufficient_level_windows"}
    # Keep the bounded experiment small; no binning of continuous data.
    if len(levels) > 64:
        return {**info, "status": "abstain", "reason": "too_many_levels"}
    matrix = np.asarray(
        [[r[0], r[1], *(r[2].get(v, 0) for v in levels)] for r in moments]
    )
    rhs = np.array([r[3] for r in moments])
    widths = np.array([r[4] for r in moments])

    def fit(basis):
        return _screen(matrix, rhs, widths, basis, output_error, max_condition)

    unconstrained = fit(np.eye(len(levels)))
    return {
        **info,
        **unconstrained,
        "family_screen": {name: fit(basis) for name, basis in _families(levels)},
    }


def forecast_hidden_pathway(
    time: Sequence[float],
    inputs: Sequence[float],
    *,
    y0: float,
    dy0: float,
    rate_sum: float,
    rate_product: float,
    levels: Sequence[float],
    q: Sequence[float],
    interpolation: str,
) -> np.ndarray:
    """Simulate a finite-level fitted law given a separately authorized y'(0)."""
    t, u = np.asarray(time, dtype=float), np.asarray(inputs, dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or u.shape != t.shape
        or not np.isfinite(t).all()
        or not np.all(np.diff(t) > 0)
        or not np.isfinite(u).all()
        or interpolation not in {LINEAR, PREVIOUS}
        or not np.isfinite([y0, dy0, rate_sum, rate_product]).all()
        or not _poles(rate_sum, rate_product)
        or len(levels) != len(q)
        or len(set(levels)) != len(levels)
        or not np.isfinite(levels).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Invalid schedule, rates, boundary values or level estimates")
    if interpolation == LINEAR and np.any(np.diff(u) != 0):
        raise ValueError("An estimated level table cannot forecast a continuous ramp")
    table = dict(zip(levels, q, strict=True))
    if not set(u[:-1]) <= table.keys():
        raise ValueError("Forecast visits an unestimated input level")
    operator = np.array([[0, 1, 0], [-rate_product, -rate_sum, 1], [0, 0, 0]])
    state = np.array([y0, dy0, 0.0])
    transitions = {}
    output = [y0]
    for dt, level in zip(np.diff(t), u[:-1], strict=True):
        if dt not in transitions:
            transitions[dt] = expm(dt * operator)
        state[2] = table[level]
        state = transitions[dt] @ state
        output.append(state[0])
    return np.array(output)
