"""A training-only order certificate for a scalar ODE with a shared driver gain."""

from __future__ import annotations

import ast
import itertools
import sys
from fractions import Fraction
from math import isfinite

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.affine_direction_certificate import _outward, _values
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)

_SIGNS = ("reject_nonnegative", "reject_nonpositive")


def _form(node, active, signals):
    """Classify a subtree as parameter-only, matched forcing, affine, or unsupported."""
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & signals
    if not used & active:
        return "forcing" if used else "constant"
    if isinstance(node, ast.Name):
        return "affine"
    if isinstance(node, ast.UnaryOp):
        return _form(node.operand, active, signals)
    if isinstance(node, ast.BinOp):
        a = _form(node.left, active, signals)
        b = _form(node.right, active, signals)
        if "unsupported" in (a, b):
            return "unsupported"
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return "affine"
        if isinstance(node.op, ast.Mult) and "constant" in (a, b):
            return "affine"
        if isinstance(node.op, ast.Div) and b == "constant":
            return "affine"
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        ):
            return "affine"
    return "unsupported"


def _lower(x: Fraction) -> float:
    """Conservative finite display, even if a difference of two floats overflows."""
    if x > Fraction(sys.float_info.max):
        return sys.float_info.max
    return _outward(x, upper=False)


def _ordered_pair(high, low, driver, same, interpolation, best):
    """Search every endpoint preceded by ordered driver and identical other knots."""
    u, v = high["inputs"][driver], low["inputs"][driver]
    valid = np.logical_and.accumulate(same & (u >= v))
    varied = np.logical_or.accumulate(u > v)
    if interpolation == "previous_complete_knots":
        # A step taken at the endpoint has had zero time to affect the state.
        varied = np.r_[False, varied[:-1]]
    endpoints = np.flatnonzero(valid & varied)
    for i in endpoints:
        if i == 0:
            continue
        diff = high["y"][i] - low["y"][i]
        if not diff:
            continue
        key = _SIGNS[0] if diff < 0 else _SIGNS[1]
        gap = abs(diff)
        if key not in best or gap > best[key][0]:
            best[key] = gap, high, low, int(i)


def _witness(record, driver, channels):
    gap, high, low, i = record
    u, v = high["inputs"][driver], low["inputs"][driver]
    first = int(np.flatnonzero(u[: i + 1] > v[: i + 1])[0])
    return {
        "high_event": high["id"],
        "low_event": low["id"],
        "index": i,
        "time": high["time"][i],
        "initial_target": float(high["y"][0]),
        "observed_high": float(high["y"][i]),
        "observed_low": float(low["y"][i]),
        "driver_high": float(u[i]),
        "driver_low": float(v[i]),
        "first_strict_driver_knot": high["time"][first],
        "exact_observed_gap": str(gap),
        "observed_gap_lower_bound": _lower(gap),
        "matching_other_channels": [name for name in channels if name != driver],
    }


def certify(
    graph: dict,
    formula: dict,
    target: str,
    evidence: TrainingCompatibilityEvidence,
    *,
    tolerance: float = 1.0,
) -> dict:
    """Test y' = a*y + b*u + q(other supplied channels) against ordered training pairs.

    The difference for a matched pair is b times the integral of its input
    difference against exp(a*(t-s)). The kernel is positive for every real a.
    A rise and a fall relative to the lower-input event refute both signs of b.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    validate_evidence(evidence)
    if evidence.split != "train" or any(
        not r.trajectory_id.startswith("train_") for r in evidence.trajectories
    ):
        raise ValueError("Only training events are permitted")
    ctx, c = evidence.context, evidence.contract
    out = {
        "status": "unverified",
        "target": target,
        "tolerance": tolerance,
        "evidence_sha256": evidence.sha256,
    }
    initial = c.initial_rules.get(target)
    if (
        target not in ctx.targets
        or graph["variable_types"].get(target) != "differential"
        or graph["initializers"].get(target)
        != {"mode": "observed_initial", "channel": target}
        or initial is None
        or initial.mode != "observed_initial"
        or initial.arguments
    ):
        return out | {
            "reason": "Requires a differential target with its observed initial value."
        }
    if not all(
        v is True
        for v in (
            c.dependencies_exhaustive,
            c.shared_parameters,
            c.reset_impulse_free,
            c.open_loop,
        )
    ):
        return out | {
            "reason": "Requires shared, exhaustive, open-loop dynamics without resets."
        }
    noise = c.absolute_error_bounds.get(target)
    if noise != 0:
        return out | {
            "reason": "The evidence does not guarantee equal exact initial values."
        }
    value = validate_formula(graph, formula)
    expr = RestrictedParser().parse(value["equations"][target], location=target)
    signals = expr.symbols - value["parameters"].keys()
    channels = sorted(signals - {target})
    supplied = set(ctx.external_inputs + ctx.auxiliaries + ctx.fixed_covariates)
    if (
        not channels
        or set(channels) - supplied
        or set(channels) & graph["variable_types"].keys()
    ):
        return out | {
            "reason": "Requires exact supplied forcing and no other generated states."
        }
    if any(c.absolute_error_bounds.get(k) != 0 for k in channels):
        return out | {
            "reason": "A supplied channel lacks an exact observation contract."
        }
    if any(
        k not in ctx.fixed_covariates
        and c.interpolation.get(k)
        not in {"linear_complete_knots", "previous_complete_knots"}
        for k in channels
    ):
        return out | {
            "reason": "Requires complete previous-value or linear forcing knots."
        }
    drivers = [
        k for k in channels if _form(expr.tree.body, {target, k}, signals) == "affine"
    ]
    if not drivers:
        return out | {
            "reason": "No supplied driver satisfies the required affine form."
        }
    events = [
        {
            "id": r.trajectory_id,
            "time": r.time,
            "y": [Fraction(v) for v in r.targets[target]],
            "inputs": {k: _values(r, k) for k in channels},
        }
        for r in sorted(evidence.trajectories, key=lambda r: r.trajectory_id)
    ]
    bounds = {k: {} for k in drivers}
    coverage = {
        "different_grids": 0,
        "nonzero_time_origin": 0,
        "unequal_initials": 0,
        "matched_pairs": 0,
    }
    for x, y in itertools.combinations(events, 2):
        if x["time"] != y["time"]:
            coverage["different_grids"] += 1
            continue
        if x["time"][0] != 0:
            coverage["nonzero_time_origin"] += 1
            continue
        if x["y"][0] != y["y"][0]:
            coverage["unequal_initials"] += 1
            continue
        coverage["matched_pairs"] += 1
        equal = {k: x["inputs"][k] == y["inputs"][k] for k in channels}
        for k in drivers:
            same = np.ones(len(x["time"]), dtype=bool)
            for other in channels:
                if k != other:
                    same &= equal[other]
            for high, low in ((x, y), (y, x)):
                _ordered_pair(high, low, k, same, c.interpolation.get(k), bounds[k])
    out.update(drivers_checked=drivers, pair_counts=coverage)
    candidates = [
        (min(best[k][0] for k in _SIGNS) / 2 - Fraction(noise), driver)
        for driver, best in bounds.items()
        if all(k in best for k in _SIGNS)
    ]
    lower, driver = max(candidates, default=(Fraction(0), ""))
    if lower <= Fraction(tolerance):
        return out | {
            "reason": "No paired sign witnesses both exceeded the error allowance."
        }
    return out | {
        "status": "conflict",
        "driver": driver,
        "exact_lower_bound": str(lower),
        "worst_sampled_absolute_error_lower_bound": _lower(lower),
        "observation_uncertainty": noise,
        "witnesses": {k: _witness(bounds[driver][k], driver, channels) for k in _SIGNS},
        "scope": (
            "For any shared parameter vector where the ODE is defined, at least one "
            "listed target sample incurs the reported absolute error or greater. "
            "The pooled NMSE remains unknown."
        ),
    }
