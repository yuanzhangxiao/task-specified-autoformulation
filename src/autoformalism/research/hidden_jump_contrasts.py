"""Necessary input-jump constraints on the sampled values of an additive RHS term."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import inf, isfinite, nextafter

from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.search.topology_compatibility import CompleteTrajectory

STEP = "previous_complete_knots"
LINEAR = "linear_complete_knots"


@dataclass(frozen=True)
class Edge:
    """Potential at finish minus potential at start is at most bound."""

    start: float
    finish: float
    bound: Fraction
    witness: int


def _outward(value: Fraction, *, lower: bool) -> float:
    """Round a rational lower or upper endpoint outward for its JSON receipt."""
    rounded = float(value)
    if (lower and rounded > value) or (not lower and rounded < value):
        return nextafter(rounded, -inf if lower else inf)
    return rounded


def _relax(
    levels: set[float], edges: list[Edge], anchor: float | None
) -> tuple[dict[float, Fraction], list[Edge]]:
    """Bellman-Ford with exact comparisons; recover a violating cycle if present."""
    distances = {k: Fraction(0) for k in (levels if anchor is None else [anchor])}
    parent: dict[float, Edge] = {}
    for _ in range(len(levels)):
        changed = None
        for edge in edges:
            if edge.start not in distances:
                continue
            bound = distances[edge.start] + edge.bound
            if edge.finish not in distances or bound < distances[edge.finish]:
                distances[edge.finish] = bound
                parent[edge.finish] = edge
                changed = edge.finish
        if changed is None:
            return distances, []
    # Following this many predecessors guarantees that we enter the cycle.
    node = changed
    for _ in levels:
        node = parent[node].start
    cycle = [parent[node]]
    while cycle[-1].start != node:
        cycle.append(parent[cycle[-1].start])
    return distances, list(reversed(cycle))


def _components(levels: set[float], edges: list[Edge]) -> list[dict]:
    """Tight marginal bounds after fixing one independent offset per component."""
    components = []
    remaining = levels.copy()
    reverse = [Edge(e.finish, e.start, e.bound, e.witness) for e in edges]
    while remaining:
        anchor = min(remaining)
        up, _ = _relax(levels, edges, anchor)
        down, _ = _relax(levels, reverse, anchor)
        levels_in_component = []
        for level in sorted(up):
            lo, hi = -down[level], up[level]
            levels_in_component.append(
                {
                    "level": level,
                    "lower": _outward(lo, lower=True),
                    "upper": _outward(hi, lower=False),
                    "exact_relative_value": str(lo) if lo == hi else None,
                }
            )
        components.append({"reference_level": anchor, "levels": levels_in_component})
        remaining.difference_update(up)
    return components


def _affine(edges: list[Edge]) -> dict:
    """Intersect slope bounds for an affine sampled response across all edges."""
    lower, upper = [], []
    for e in edges:
        delta = Fraction(e.finish) - Fraction(e.start)
        (upper if delta > 0 else lower).append(e.bound / delta)
    lo, hi = max(lower), min(upper)
    if lo > hi:
        return {"status": "excluded"}
    return {
        "status": "compatible",
        "slope_lower": _outward(lo, lower=True),
        "slope_upper": _outward(hi, lower=False),
    }


def _interpolant_degree(values: list[dict]) -> int:
    """Degree of the unique polynomial of degree <= n-1 through anchored levels."""
    levels = [Fraction(row["level"]) for row in values]
    differences = [Fraction(row["exact_relative_value"]) for row in values]
    degree = 0
    for order in range(1, len(levels)):
        differences = [
            (right - left) / (levels[j + order] - levels[j])
            for j, (left, right) in enumerate(pairwise(differences))
        ]
        if differences[0]:
            degree = order
    return degree


def screen_input_jumps(
    training_rows: Sequence[dict | CompleteTrajectory],
    *,
    target: str,
    source: str,
    interpolation: Mapping[str, str],
    continuous_nuisance: bool,
    observation_error: float | None,
    curvature_bound: float | None,
    reset_knots: Mapping[str, Sequence[int]] | None = None,
) -> dict:
    """Test y'=A+g(u) conditional on continuous nuisance A and isolated forcing.

    Each record supplies observations of differential y and external u, with no
    derivative labels. The caller specifies a uniform absolute error per y sample
    and an upper bound on |y''| separately inside each smooth sampling interval.
    All nonsource inputs/auxiliaries must have complete interpolation metadata.
    Known reset indices denote boundaries at which y or its nuisance can jump.
    The sampled-input levels and times are assumed exact.
    """
    rows = sorted(
        _rows(training_rows, "train", min_samples=1), key=lambda row: row.trajectory_id
    )
    known = {row.trajectory_id: len(row.time) for row in rows}
    resets = reset_knots or {}
    for name, knots in resets.items():
        if name not in known or any(
            type(i) is not int or not 0 <= i < known[name] for i in knots
        ):
            raise ValueError("Unknown reset event or invalid knot index")
    if source not in rows[0].inputs or target not in rows[0].targets:
        raise ValueError("Expected a supplied input and measured differential target")
    for value in (observation_error, curvature_bound):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError("Uncertainty bounds must be finite and nonnegative")
    result: dict = {
        "target": target,
        "source": source,
        "scope": "conditional_on_additive_input_and_continuous_nuisance",
        "observation_error": observation_error,
        "curvature_bound": curvature_bound,
        "witnesses": [],
    }
    if interpolation.get(source) != STEP:
        return {**result, "status": "abstained", "reason": "no_level_jump_contract"}
    if not continuous_nuisance:
        return {**result, "status": "abstained", "reason": "unverified_continuity"}
    if observation_error is None or curvature_bound is None:
        return {**result, "status": "abstained", "reason": "unbounded_rate_error"}
    forcing = rows[0].inputs.keys() | rows[0].auxiliaries.keys()
    if any(interpolation.get(s) not in (STEP, LINEAR) for s in forcing):
        return {**result, "status": "abstained", "reason": "unknown_forcing_schedule"}
    noise, curvature = Fraction(observation_error), Fraction(curvature_bound)
    levels = {v for row in rows for v in row.inputs[source]}
    edges: list[Edge] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        u = row.inputs[source]
        others = {**row.inputs, **row.auxiliaries}
        t, y = list(map(Fraction, row.time)), list(map(Fraction, row.targets[target]))
        for i in range(1, len(t)):
            a, b = u[i - 1], u[i]
            if a == b:
                continue
            if i + 1 == len(t):
                skipped["no_right_sample"] += 1
                continue
            # Left and right secants both include their endpoint observations.
            if any(j in resets.get(row.trajectory_id, ()) for j in (i - 1, i, i + 1)):
                skipped["possible_reset"] += 1
                continue
            if any(
                name != source and interpolation[name] == STEP and v[i - 1] != v[i]
                for name, v in others.items()
            ):
                skipped["simultaneous_jump"] += 1
                continue
            h, k = t[i] - t[i - 1], t[i + 1] - t[i]
            contrast = (y[i + 1] - y[i]) / k - (y[i] - y[i - 1]) / h
            # Each secant is within M * duration / 2 of its one-sided derivative.
            radius = curvature * (h + k) / 2 + 2 * noise * (1 / h + 1 / k)
            lo, hi = contrast - radius, contrast + radius
            index = len(result["witnesses"])
            result["witnesses"].append(
                {
                    "trajectory": row.trajectory_id,
                    "knot": i,
                    "levels": [a, b],
                    "sample_times": [float(v) for v in t[i - 1 : i + 2]],
                    "contrast": float(contrast),
                    "lower": _outward(lo, lower=True),
                    "upper": _outward(hi, lower=False),
                }
            )
            edges.extend((Edge(a, b, hi, index), Edge(b, a, -lo, index)))
    result["skipped"] = dict(skipped)
    if not edges:
        return {**result, "status": "abstained", "reason": "no_isolated_jump_witness"}
    _, cycle = _relax(levels, edges, None)
    if cycle:
        return {
            **result,
            "status": "incompatible",
            "reason": "negative_cycle_in_relative_value_bounds",
            "cycle": [
                {"from": e.start, "to": e.finish, "witness": e.witness} for e in cycle
            ],
            "minimum_cycle_violation": _outward(
                -sum(e.bound for e in cycle), lower=True
            ),
        }
    components = _components(levels, edges)
    if len(components) > 1:
        status = "disconnected_levels"
    elif all(v["exact_relative_value"] is not None for v in components[0]["levels"]):
        status = "identified_on_levels"
        result["anchored_polynomial_interpolant"] = {
            "degree": _interpolant_degree(components[0]["levels"]),
            "unique_within_degree_at_most": len(levels) - 1,
        }
    else:
        status = "bounded_on_levels"
    return {
        **result,
        "status": status,
        "components": components,
        "affine": _affine(edges),
    }
