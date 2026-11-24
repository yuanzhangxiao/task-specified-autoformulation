"""Time-zero capacity of Orion's two frozen affine disposal readouts on training."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isclose, isfinite

from autoformalism.research import trajectory_function_fitting as fitting
from autoformalism.research.affine_direction_certificate import _outward
from autoformalism.staged_topology import content_hash
from scripts.fit_t2_disposal_constrained import DISPOSAL_BUFFER, GRAPHS


def _lower_bound(points: list) -> tuple[Fraction, dict | None]:
    """A shared affine map preserves equal points and convex combinations."""
    best, witness = Fraction(0), None
    for (xa, ya, a), (xb, yb, b) in combinations(points, 2):
        if xa == xb and abs(ya - yb) / 2 > best:
            best = abs(ya - yb) / 2
            witness = {"same_initial_features": [a, b]}
    # Sorting by feature vector puts the middle point between the two ends on
    # any line. Each of the predicted errors contributes at most its weight.
    for (xa, ya, a), (xb, yb, b), (xc, yc, c) in combinations(sorted(points), 3):
        if xa == xb or xb == xc:
            continue
        i = next(i for i, (a0, c0) in enumerate(zip(xa, xc, strict=True)) if a0 != c0)
        w = (xb[i] - xa[i]) / (xc[i] - xa[i])
        if not 0 < w < 1 or any(
            mid != (1 - w) * left + w * right
            for left, mid, right in zip(xa, xb, xc, strict=True)
        ):
            continue
        gap = yb - (1 - w) * ya - w * yc
        if abs(gap) / 2 > best:
            best = abs(gap) / 2
            witness = {
                "endpoints": [a, c],
                "middle_event": b,
                "weight_on_second_endpoint": str(w),
                "observed_middle_minus_interpolation": float(gap),
            }
    return best, witness


def _tissue_line(points: list) -> tuple[Fraction, Fraction]:
    """Minimize maximum initial error in a subset of the family, c + s Gt."""
    xy = sorted({(x[1], y) for x, y, _ in points})
    # For a fixed slope, centring the residual range gives the best intercept.
    # The range is piecewise linear with knots at slopes joining two readings.
    slopes = {Fraction(0)} | {
        (v - y) / (z - x) for (x, y), (z, v) in combinations(xy, 2) if x != z
    }
    choices = []
    for slope in slopes:
        residuals = [y - slope * x for x, y in xy]
        low, high = min(residuals), max(residuals)
        choices.append((high - low, abs(slope), slope, (high + low) / 2))
    _, _, slope, intercept = min(choices)
    return slope, intercept


def assess_family(
    model: dict, rows: list[dict], *, tolerance: float | None = None
) -> dict:
    """Bracket attainable maximum initial error; leave later predictions unassessed."""
    if tolerance is not None and (not isfinite(tolerance) or tolerance < 0):
        raise ValueError("A declared absolute tolerance must be finite and nonnegative")
    form = model["formula"]
    identity = content_hash([model["graph"], form["equations"], form["initializers"]])
    if identity not in GRAPHS:
        raise ValueError("Expected a qualified immediate or delayed disposal model")
    points = []
    for row in fitting._rows(rows, "train"):
        if row.time[0] != 0 or row.auxiliaries["Uii"][0] != 1:
            raise ValueError("Expected time-zero observations with initial Uii = 1")
        x = tuple(
            Fraction(v[0])
            for v in (
                row.targets["Gp"],
                row.auxiliaries["Gt"],
                row.targets["I"],
                row.auxiliaries["Uii"],
            )
        )
        points.append((x, Fraction(row.targets["U"][0]), row.trajectory_id))
    lower, certificate = _lower_bound(points)
    slope, intercept = _tissue_line(points)
    p = form["parameters"].copy()
    if p["afcoef13"] <= 0 or p["afcoef10"] <= 0:
        raise ValueError("Expected positive action gains")
    fixed_offset = Fraction(p["afcoef13"]) * Fraction(p["afcoef15"])
    p.update(
        disp_tissue=float(slope * 100),
        afcoef16=0.0,
        afcoef12=0.0,
        afcoef11=float(intercept + slope * 100 - fixed_offset),
    )
    events = fitting.initial_target_events(
        model["graph"], form | {"parameters": p}, rows, ["U"]
    )
    predictions = {v["event"]: v["predicted"]["U"][0] for v in events}
    upper, minimum_margin = Fraction(0), None
    for x, y, event in points:
        pred = (
            Fraction(p["afcoef11"])
            + fixed_offset
            + Fraction(p["disp_tissue"]) * (x[1] - 100) / 100
        )
        actual = predictions[event]
        if not isclose(float(pred), actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ArithmeticError("Compiled readout differs from the witness")
        upper = max(upper, abs(pred - y), abs(Fraction(actual) - y))
        margin = min(pred, Fraction(actual)) - x[3]
        minimum_margin = (
            margin if minimum_margin is None else min(minimum_margin, margin)
        )
    feasible_initial_domain = minimum_margin >= Fraction(DISPOSAL_BUFFER)
    if tolerance is None:
        status = "no_absolute_tolerance_declared"
    elif Fraction(tolerance) < lower:
        status = "family_conflict_at_initial_time"
    elif upper <= Fraction(tolerance) and feasible_initial_domain:
        status = "initial_tolerance_attainable"
    else:
        status = "unresolved"
    return {
        "scope": "Training time zero only for the two registered structures. "
        "All coefficients remain shared across events. Later dynamics are unassessed.",
        "status": status,
        "tolerance": tolerance,
        "lower_bound": _outward(lower, upper=False),
        "lower_bound_exact": str(lower),
        "lower_bound_witness": certificate,
        "upper_bound": _outward(upper, upper=True) if feasible_initial_domain else None,
        "witness_max_initial_error": _outward(upper, upper=True),
        "witness_satisfies_initial_disposal_buffer": feasible_initial_domain,
        "witness_parameters": p,
        "witness_predictions": predictions,
        "tissue_slope": float(slope),
        "tissue_intercept": float(intercept),
    }
