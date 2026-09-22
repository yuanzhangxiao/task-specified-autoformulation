"""Conservative partial probes without assigning values to unknown parameters."""

from __future__ import annotations

import ast
import math

from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.rebuttal import basin_equation_checks as legacy

POLICY = "basin-public-equation-witnesses-2"


def partial_value(node, environment, parameters):
    """Return (known number, finite) with narrowly justified zero products.

    Unknown parameters are finite scalar leaves, not guessed values. An unknown
    compound expression is not presumed finite: 0*exp(k) and 0*log(k) therefore
    remain unknown. Use the production restricted interpreter for closed nodes.
    """
    if isinstance(node, ast.Name):
        if node.id in environment:
            value = float(environment[node.id])
            return (value, True) if math.isfinite(value) else (None, False)
        return None, node.id in parameters
    if isinstance(node, ast.Constant):
        return float(node.value), math.isfinite(node.value)
    if isinstance(node, ast.UnaryOp):
        value, finite = partial_value(node.operand, environment, parameters)
        return (
            (-value if isinstance(node.op, ast.USub) else value)
            if value is not None
            else None,
            finite,
        )
    children = (
        [node.left, node.right]
        if isinstance(node, ast.BinOp)
        else node.args
        if isinstance(node, ast.Call)
        else []
    )
    values = [partial_value(child, environment, parameters) for child in children]
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and (
            (values[0][0] == 0 and values[1][1]) or (values[1][0] == 0 and values[0][1])
        )
    ):
        return 0.0, True
    if not values or any(v is None or not finite for v, finite in values):
        return None, False
    constants = [ast.Constant(v) for v, _ in values]
    closed = (
        ast.BinOp(constants[0], node.op, constants[1])
        if isinstance(node, ast.BinOp)
        else ast.Call(node.func, constants, [])
    )
    try:
        value = legacy._value(closed, {})
        return (value, True) if math.isfinite(value) else (None, False)
    except legacy.UNVERIFIED_ERRORS:
        return None, False


def assess(candidate, context, case, surveys, parameters=None):
    """Extend static threshold evidence while retaining historical fitted checks."""
    result = legacy.assess(candidate, context, case, surveys, parameters)
    result["policy"] = POLICY
    if parameters is not None:
        return result
    down_name = result["coordinate_bindings"]["downstream_depth"]
    up_name = result["coordinate_bindings"]["upstream_depth"]
    if not down_name or not any(
        c["code"] == "free_outlet_threshold_probes" for c in result["checks"]
    ):
        return result
    rhs = next(e.rhs for e in candidate.state_equations if e.state == down_name)
    node = algebra.expanded(rhs, {p.name: p.expression for p in candidate.processes})
    probes, bad = [], []
    for survey in surveys:
        crest = survey["crest_down"]
        levels = [0.0, crest / 2, crest, crest * 1.25 + 0.01]
        env = {**survey, "inflow_up": 0.0, "inflow_down": 0.0, "t": 0.0}
        if up_name:
            env[up_name] = 0.0
        values = [
            partial_value(
                node, {**env, down_name: h}, {p.name for p in candidate.parameters}
            )[0]
            for h in levels
        ]
        row = {
            "survey": survey,
            "downstream_depths": levels,
            "depth_derivatives_m_per_min": values,
            "upstream_depth": 0.0 if up_name else None,
            "inflow_up": 0.0,
            "inflow_down": 0.0,
        }
        probes.append(row)
        if any(v is not None and abs(v) > legacy.TOLERANCE for v in values[:3]) or (
            values[3] is not None and (values[3] == 0 or values[3] > legacy.TOLERANCE)
        ):
            bad.append(row)
    complete = bool(probes) and all(
        all(v is not None for v in p["depth_derivatives_m_per_min"])
        and p["depth_derivatives_m_per_min"][3] < -legacy.TOLERANCE
        for p in probes
    )
    check = legacy.finding(
        "free_outlet_threshold_probes",
        "fail" if bad else "pass" if complete else "unverified",
        "Public zero-inflow probes with finite unknown parameters. Only provable "
        "parameter-independent values are reported; null remains unknown. "
        "Passing sampled depths is not a global threshold proof.",
        probes=probes,
        counterexamples=bad,
        parameters_source="none_partial_evaluation",
        time_series_used=False,
    )
    result["checks"] = [
        check if c["code"] == check["code"] else c for c in result["checks"]
    ]
    return result
