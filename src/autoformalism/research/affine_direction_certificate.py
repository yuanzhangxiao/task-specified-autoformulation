"""Certify incompatible response directions for a restricted scalar ODE family."""

from __future__ import annotations

import ast
from fractions import Fraction
from math import inf, isfinite, nextafter

import numpy as np

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)


def _unverified(reason: str) -> dict:
    return {"status": "unverified", "reason": reason}


def _flux_and_threshold(node: ast.expr, target: str) -> tuple[ast.expr, float | None]:
    """Recognize the exact implemented depletion taper; otherwise keep the full RHS."""
    # Extract possible F and scale, then check every node against the complete
    # production template. Any other use of the target remains a state input.
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.left, ast.Call)
        and node.left.args
    ):
        base = node.left.args[0]
        raw = ast.unparse(base)
        identity = ast.dump(node)
        for child in ast.walk(node):
            if not isinstance(child, ast.Constant) or not isinstance(
                child.value, (int, float)
            ):
                continue
            scale = child.value
            if scale < SAFE_DIVISION_EPSILON:
                continue
            gate = f"min(1,max(0,{target}/({scale!r})))"
            expected = ast.parse(
                f"max(({raw}),0)+min(({raw}),0)*{gate}", mode="eval"
            ).body
            if ast.dump(expected) == identity:
                return base, float(scale)
    return node, None


def _degree(node: ast.AST, driver: str) -> int:
    """Upper-bound polynomial degree in one driver; all other symbols stay fixed."""
    if not any(isinstance(n, ast.Name) and n.id == driver for n in ast.walk(node)):
        return 0
    if isinstance(node, ast.Name):
        return 1
    if isinstance(node, ast.UnaryOp):
        return _degree(node.operand, driver)
    if isinstance(node, ast.BinOp):
        a, b = _degree(node.left, driver), _degree(node.right, driver)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return max(a, b)
        if isinstance(node.op, ast.Mult):
            return min(2, a + b)
        if isinstance(node.op, ast.Div) and b == 0:
            return a
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        ):
            return a
    return 2


def _signals(expression, parameters):
    parsed = RestrictedParser().parse(
        ast.unparse(expression), location="underlying_RHS"
    )
    return sorted(parsed.symbols - parameters.keys())


def _values(row, name):
    if name in row.covariates:
        return np.full(len(row.time), row.covariates[name])
    if name in row.auxiliaries:
        return np.asarray(row.auxiliaries[name])
    return np.asarray(row.inputs[name])


def _outward(value: Fraction, *, upper: bool) -> float:
    """Round a witness endpoint to float in the conservative direction."""
    result = float(value)
    relative = Fraction(result) - value
    if (upper and relative < 0) or (not upper and relative > 0):
        return nextafter(result, inf if upper else -inf)
    return result


def _reference(row, target, channels, delta, initial_error, threshold):
    """Bound the constant raw rate, including a partially open depletion gate."""
    t, y = row.time, row.targets[target]
    if len(t) < 2:
        return None
    ref = {}
    for name in channels:
        v = _values(row, name)
        if np.any(v != v[0]):
            return None
        ref[name] = float(v[0])
    start, end = map(Fraction, (y[0], y[-1]))
    dt = Fraction(t[-1]) - Fraction(t[0])
    err = delta + Fraction(initial_error)
    lower, upper = (end - start - err) / dt, (end - start + err) / dt
    first, last = start - Fraction(initial_error), end - delta
    m = Fraction(1)
    if threshold is not None:
        # At constant F the scalar trajectory is monotone, even under the taper.
        # Its minimum is bounded below by the smaller permitted endpoint.
        m = min(m, min(first, last) / Fraction(threshold))
        if m <= 0:
            return None
    # If F>=0, mean rate = F. If F<0, F <= mean rate <= m*F.
    lo = lower / m if lower < 0 else lower
    hi = upper
    try:
        raw_display = [_outward(lo, upper=False), _outward(hi, upper=True)]
        endpoint_display = [_outward(first, upper=False), _outward(last, upper=False)]
        delta_display = _outward(delta, upper=True)
        if not all(map(isfinite, raw_display + endpoint_display + [delta_display])):
            return None
    except OverflowError:
        return None
    assumptions = ["The raw RHS is constant over the complete reference interval."]
    if threshold is None:
        assumptions.append(
            "The target has no depletion gate; its sign is unrestricted."
        )
    else:
        assumptions.append(
            "A nonnegative raw rate is unchanged. A negative raw rate is multiplied "
            "by min(1,max(0,target/scale)). The target is monotone at constant forcing."
        )
    return (
        {
            "event": row.trajectory_id,
            "time": [t[0], t[-1]],
            "forcing_values": ref,
            "initial_target": y[0],
            "observed_final": y[-1],
            "initial_error": initial_error,
            "target_error": delta_display,
            "gate_threshold": threshold,
            "minimum_permitted_endpoint": min(endpoint_display),
            "endpoint_lower_bounds": endpoint_display,
            "minimum_gate_multiplier": _outward(m, upper=False),
            "derivative_bound": raw_display,
            "exact_bounds": {
                "endpoint_lower": [str(first), str(last)],
                "mean_derivative": [str(lower), str(upper)],
                "minimum_gate_multiplier": str(m),
                "raw_derivative": [str(lo), str(hi)],
            },
            "assumptions": assumptions,
        },
        lo,
        hi,
    )


def _best_window(row, target, values, live, sign, rate, delta):
    """Find maximal excess rise or fall in consecutive admissible forcing knots."""
    t = np.asarray(row.time)
    y = np.asarray(row.targets[target])
    candidate = sign * y - float(rate) * (t - t[0])
    first = None
    best = None
    margin = 0.0
    for i, valid in enumerate(live):
        if not valid:
            first = None
            continue
        if first is None or candidate[first] >= candidate[i]:
            first = i
            continue
        change = float(candidate[i] - candidate[first] - 2 * delta)
        if change > margin:
            first_i = first
            best = first_i, i
            margin = change
    if best is None:
        return None
    a, b = best
    # Recheck the selected strict inequality exactly for the recorded floats.
    duration = Fraction(float(t[b])) - Fraction(float(t[a]))
    increase = sign * (Fraction(float(y[b])) - Fraction(float(y[a])))
    required = increase - 2 * Fraction(delta)
    permitted = duration * rate
    if required <= permitted:
        return None
    return {
        "event": row.trajectory_id,
        "indices": [a, b],
        "time": [float(t[a]), float(t[b])],
        "observed": [float(y[a]), float(y[b])],
        "driver_range": [
            float(np.min(values[a : b + 1])),
            float(np.max(values[a : b + 1])),
        ],
        "required_change": float(required),
        "maximum_permitted_change": float(permitted),
        "excess": float(required - permitted),
        "exact_change": {
            "required": str(required),
            "maximum_permitted": str(permitted),
        },
    }


def _windows(rows, target, driver, reference, lo, hi, delta, side):
    peaks = [None, None]
    for row in rows:
        x = _values(row, driver)
        ref = reference["forcing_values"]
        mask = x >= ref[driver] if side == 1 else x <= ref[driver]
        for other, value in ref.items():
            if other != driver:
                mask &= _values(row, other) == value
        for j, (sgn, rate) in enumerate(((1, max(hi, 0)), (-1, max(-lo, 0)))):
            w = _best_window(row, target, x, mask, sgn, rate, delta)
            if w and (peaks[j] is None or w["excess"] > peaks[j]["excess"]):
                peaks[j] = w
    return peaks


def certify(
    graph: dict,
    formula: dict,
    target: str,
    evidence: TrainingCompatibilityEvidence,
    *,
    tolerance: float = 1.0,
) -> dict:
    """Seek an opposing-direction witness, conditional on the public training contract.

    Values must match within the given absolute tolerance. A known observation
    error bound widens it conservatively. Uncertain drivers, unobserved
    initializers and general state feedback are outside this certificate.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    validate_evidence(evidence)
    if evidence.split != "train" or any(
        not r.trajectory_id.startswith("train_") for r in evidence.trajectories
    ):
        raise ValueError("Training trajectories are required")
    ctx, contract = evidence.context, evidence.contract
    rule = graph["initializers"].get(target)
    init = contract.initial_rules.get(target)
    if (
        target not in ctx.targets
        or graph["variable_types"].get(target) != "differential"
        or rule != {"mode": "observed_initial", "channel": target}
        or init is None
        or init.mode != "observed_initial"
    ):
        return _unverified(
            "Requires a differential target initialized from its observed value."
        )
    if not all(
        v is True
        for v in (
            contract.dependencies_exhaustive,
            contract.shared_parameters,
            contract.open_loop,
            contract.reset_impulse_free,
        )
    ):
        return _unverified(
            "Requires exhaustive shared dynamics with open-loop inputs and no resets."
        )
    value = validate_formula(graph, formula)
    tree = (
        RestrictedParser().parse(value["equations"][target], location=target).tree.body
    )
    base, threshold = _flux_and_threshold(tree, target)
    channels = _signals(base, value["parameters"])
    supplied = set(ctx.external_inputs + ctx.auxiliaries + ctx.fixed_covariates)
    if (
        not channels
        or set(channels) - supplied
        or set(channels) & set(graph["variable_types"])
    ):
        return _unverified(
            "Raw derivative has a generated state, explicit time or no supplied driver."
        )
    measurement = contract.absolute_error_bounds.get(target)
    if measurement is None or any(
        contract.absolute_error_bounds.get(k) != 0 for k in channels
    ):
        return _unverified(
            "Requires known target observation error and exact supplied driver values."
        )
    if any(
        k not in ctx.fixed_covariates
        and contract.interpolation.get(k) != "linear_complete_knots"
        for k in channels
    ):
        return _unverified("Requires complete piecewise-linear forcing knots.")
    delta = Fraction(tolerance) + Fraction(measurement)
    drivers = [name for name in channels if _degree(base, name) == 1]
    if not drivers:
        return _unverified("No certified affine driver in the raw derivative.")
    for row in evidence.trajectories:
        anchor = _reference(row, target, channels, delta, measurement, threshold)
        if anchor is None:
            continue
        ref, lo, hi = anchor
        for driver in drivers:
            for side in (1, -1):
                rise, fall = _windows(
                    evidence.trajectories, target, driver, ref, lo, hi, delta, side
                )
                if rise and fall:
                    return {
                        "status": "conflict",
                        "target": target,
                        "driver": driver,
                        "side": "at_or_above" if side == 1 else "at_or_below",
                        "slope_test": "Classify slopes by their sign times the side.",
                        "raw_rhs": ast.unparse(base),
                        "gate_threshold": threshold,
                        "tolerance": tolerance,
                        "observation_uncertainty": measurement,
                        "reference": ref,
                        "rise": rise,
                        "fall": fall,
                        "checks": [
                            *(
                                [
                                    {
                                        "quantity": f"reference {position} lower bound",
                                        "value": endpoint,
                                        "relation": ">=",
                                        "bound": min(
                                            threshold, ref["minimum_permitted_endpoint"]
                                        ),
                                    }
                                    for position, endpoint in zip(
                                        ("initial", "final"),
                                        ref["endpoint_lower_bounds"],
                                        strict=True,
                                    )
                                ]
                                if threshold is not None
                                else []
                            ),
                            {
                                "quantity": "required rise for slope times side <= 0",
                                "value": rise["required_change"],
                                "relation": ">",
                                "bound": rise["maximum_permitted_change"],
                            },
                            {
                                "quantity": "required fall for slope times side >= 0",
                                "value": fall["required_change"],
                                "relation": ">",
                                "bound": fall["maximum_permitted_change"],
                            },
                        ],
                        "scope": "No shared parameters reproduce the listed samples "
                        "within the specified pointwise tolerance where the RHS is "
                        "defined. The pooled NMSE optimum remains unknown.",
                    }
    return _unverified(
        "No common-side pair of opposing directional witnesses exceeded tolerance."
    )
