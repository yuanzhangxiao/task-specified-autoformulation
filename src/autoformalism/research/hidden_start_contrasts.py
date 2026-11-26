"""Conditional shape evidence from held-fixed RHS arguments at the initial time."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations
from math import comb, inf, isfinite, nextafter

from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.search.topology_compatibility import CompleteTrajectory

Interval = tuple[float, float]


def _initial_tokens(graph: dict, row: CompleteTrajectory, time_symbol: str) -> dict:
    """Equal tokens imply equal values; unequal unknown-map tokens prove nothing."""
    recorded = {
        name: ("number", values[0])
        for group in (row.inputs, row.auxiliaries, row.targets)
        for name, values in group.items()
    }
    recorded.update({name: ("number", v) for name, v in row.covariates.items()})
    recorded[time_symbol] = ("number", row.time[0])
    tokens = {}
    active = set()

    def resolve(name):
        if name in tokens:
            return tokens[name]
        if name in active:
            raise ValueError(f"Cyclic initial dependence at {name}")
        active.add(name)
        state_type = graph["variable_types"].get(name)
        if state_type == "differential":
            rule = graph["initializers"][name]
            mode = rule["mode"]
            if mode == "observed_initial":
                value = recorded[rule["channel"]]
            elif mode == "map":
                value = (
                    "map",
                    name,
                    tuple(recorded[s] for s in rule["public_initials"]),
                    tuple(resolve(s) for s in rule["state_initials"]),
                )
            elif mode == "value" and rule["scope"] == "shared_training":
                value = ("shared", name)
            elif mode == "value" and rule["scope"] == "trajectory":
                value = ("trajectory", name, row.trajectory_id)
            else:
                raise ValueError(f"Unsupported initial rule for {name}: {rule}")
        elif state_type == "algebraic":
            # A measured algebraic source may condition the comparison only under
            # the assumption that its own equation reproduces that measurement.
            value = (
                recorded[name]
                if name in recorded
                else (
                    "equation",
                    name,
                    tuple(resolve(s) for s in graph["sources"][name]),
                )
            )
        else:
            value = recorded[name]
        active.remove(name)
        tokens[name] = value
        return value

    for sources in graph["sources"].values():
        for source in sources:
            resolve(source)
    return tokens


def _affine_witness(levels: dict, numeric_slack: float) -> dict | None:
    """An affine map fixes its middle value by linear interpolation."""
    bounds = {}
    for x, observations in sorted(levels.items()):
        known = [obs["interval"] for obs in observations if obs["interval"] is not None]
        if not known:
            continue
        lo, hi = max(v[0] for v in known), min(v[1] for v in known)
        if Fraction(lo) > Fraction(hi) + Fraction(numeric_slack):
            return {
                "status": "inconsistent_repeated_level",
                "source_level": x,
                "trajectory_ids": [o["id"] for o in observations],
                "intervals": known,
            }
        bounds[x] = (min(lo, hi), max(lo, hi))
    for xa, xb, xc in combinations(bounds, 3):
        weight = (Fraction(xb) - Fraction(xa)) / (Fraction(xc) - Fraction(xa))
        lower, upper = [
            (1 - weight) * Fraction(bounds[xa][i]) + weight * Fraction(bounds[xc][i])
            for i in range(2)
        ]
        middle = bounds[xb]
        slack = Fraction(numeric_slack)
        if Fraction(middle[0]) > upper + slack or Fraction(middle[1]) < lower - slack:
            return {
                "status": "affine_excluded_on_slice",
                "levels": [xa, xb, xc],
                "response_intervals": [bounds[x] for x in (xa, xb, xc)],
                "affine_interval_at_middle": [float(lower), float(upper)],
                "trajectory_ids": [
                    [o["id"] for o in levels[x] if o["interval"] is not None]
                    for x in (xa, xb, xc)
                ],
            }
    return None


def _contrast(
    source: str,
    other: list[str],
    tokens: list[dict],
    rows: list,
    response: list,
    exact_conditioning: bool,
    numeric_slack: float,
) -> dict:
    slices = defaultdict(lambda: defaultdict(list))
    for row, key, observation in zip(rows, tokens, response, strict=True):
        if key[source][0] == "number":
            context = (row.time[0], *(key[s] for s in other))
            slices[context][key[source][1]].append(observation)
    groups = list(slices.values())
    isolated_pairs = sum(
        len(a) * len(b) for group in groups for a, b in combinations(group.values(), 2)
    )
    best = max(groups, key=len, default={})
    evidence = [
        witness
        for group in groups
        if exact_conditioning and (witness := _affine_witness(group, numeric_slack))
    ]
    contradictions = [
        w for w in evidence if w["status"] == "inconsistent_repeated_level"
    ]
    if contradictions:
        status, witness = "inconsistent_repeated_level", contradictions[0]
    elif evidence:
        status, witness = "affine_excluded_on_slice", evidence[0]
    elif len(best) >= 3:
        status, witness = "three_or_more_levels_without_affine_exclusion", None
    elif isolated_pairs:
        status, witness = "two_levels_only", None
    else:
        status, witness = "no_isolated_variation", None
    return {
        "source": source,
        "isolated_pairs": isolated_pairs,
        "maximum_levels": len(best),
        "status": status,
        "witness": witness,
        "largest_slice": [
            {"level": level, "observations": observations}
            for level, observations in sorted(best.items())
        ]
        if len(best) >= 2
        else [],
    }


def _motifs(graph, target, hidden, rows, interpolation, time_symbol):
    """Count isolated forcing changes and strict one-hidden-state cascade candidates."""
    args = graph["sources"][target]
    types = graph["variable_types"]
    forcings = set(rows[0].inputs) | set(rows[0].auxiliaries)
    jumps = []
    cascades = []
    if types[target] != "differential":
        return jumps, cascades
    for source in sorted(forcings.intersection(args)):
        changes = 0
        if interpolation.get(source) == "previous_complete_knots" and all(
            types.get(s) == "differential"
            or s in rows[0].covariates
            or s == time_symbol
            or interpolation.get(s)
            in {"previous_complete_knots", "linear_complete_knots"}
            for s in args
            if s != source
        ):
            for row in rows:
                series = {**row.inputs, **row.auxiliaries}
                for i in range(1, len(row.time) - 1):
                    if series[source][i] != series[source][i - 1] and all(
                        series[s][i] == series[s][i - 1]
                        for s in set(args) - {source}
                        if interpolation.get(s) == "previous_complete_knots"
                    ):
                        changes += 1
        jumps.append({"source": source, "isolated_interior_jumps": changes})
    for z in hidden:
        if set(args) != {target, z} or types.get(z) != "differential":
            continue
        upstream = set(graph["sources"][z])
        drives = upstream & forcings
        if len(drives) == 1 and upstream == {z, *drives}:
            drive = next(iter(drives))
            levels = set()
            for row in rows:
                vals = {**row.inputs, **row.auxiliaries}[drive]
                for i in range(len(vals) - 1):
                    interpolation_mode = interpolation.get(drive)
                    if interpolation_mode == "previous_complete_knots" or (
                        interpolation_mode == "linear_complete_knots"
                        and vals[i] == vals[i + 1]
                    ):
                        levels.add(vals[i])
            cascades.append(
                {"hidden": z, "input": drive, "constant_input_levels": sorted(levels)}
            )
    return jumps, cascades


def matched_starts(
    graph: dict,
    train_rows,
    *,
    time_symbol: str = "t",
    interpolation: Mapping[str, str] | None = None,
    right_rate_bounds: Mapping[str, Mapping[str, Interval]] | None = None,
    algebraic_error: Mapping[str, float] | None = None,
    exact_conditioning: bool = True,
    numeric_slack: float = 1e-10,
) -> dict:
    """Screen every observed target equation at matched starts of training events.

    Quantitative affine exclusions require exact conditioning and independent
    response intervals (right rates for ODEs or a stated algebraic value error).
    First sample increments alone never supply such intervals.
    """
    if not isfinite(numeric_slack) or numeric_slack < 0:
        raise ValueError("Expected finite nonnegative numerical slack")
    if any(not isfinite(e) or e < 0 for e in (algebraic_error or {}).values()):
        raise ValueError("Expected finite nonnegative algebraic measurement error")
    rows = _rows(train_rows, "train")
    tokens = [_initial_tokens(graph, row, time_symbol) for row in rows]
    measured = set(rows[0].targets) | set(rows[0].auxiliaries) | set(rows[0].inputs)
    equations = []
    for target, args in sorted(graph["sources"].items()):
        if target not in measured:
            continue
        if len(set(args)) != len(args):
            raise ValueError(f"Duplicate argument for {target}")
        algebraic = graph["variable_types"][target] == "algebraic"
        hidden = [s for s in args if s in graph["variable_types"] and s not in measured]
        response = []
        for row in rows:
            values = {**row.inputs, **row.auxiliaries, **row.targets}[target]
            dt = row.time[1] - row.time[0]
            estimate = values[0] if algebraic else (values[1] - values[0]) / dt
            interval = None
            if algebraic and target in (algebraic_error or {}):
                eps = algebraic_error[target]
                interval = (
                    (nextafter(estimate - eps, -inf), nextafter(estimate + eps, inf))
                    if eps
                    else (estimate, estimate)
                )
            if not algebraic:
                interval = (
                    (right_rate_bounds or {}).get(target, {}).get(row.trajectory_id)
                )
            if interval is not None and not (
                len(interval) == 2
                and all(map(isfinite, interval))
                and interval[0] <= interval[1]
            ):
                raise ValueError("Invalid measurement or right-rate interval")
            response.append(
                {
                    "id": row.trajectory_id,
                    "estimate": estimate,
                    "dt": dt,
                    "interval": list(interval) if interval is not None else None,
                }
            )
        checks = [
            _contrast(
                s,
                [a for a in args if a != s],
                tokens,
                rows,
                response,
                exact_conditioning,
                numeric_slack,
            )
            for s in args
            if any(k[s][0] == "number" for k in tokens)
        ]
        jumps, cascades = _motifs(
            graph, target, hidden, rows, interpolation or {}, time_symbol
        )
        equals = sum(
            rows[i].time[0] == rows[j].time[0]
            and all(tokens[i][s] == tokens[j][s] for s in hidden)
            for i, j in combinations(range(len(rows)), 2)
        )
        equations.append(
            {
                "target": target,
                "type": graph["variable_types"][target],
                "hidden_sources": hidden,
                "all_pairs": comb(len(rows), 2),
                "equal_hidden_initial_pairs": equals,
                "response": "observed_initial_value"
                if algebraic
                else "first_forward_secant",
                "sources": checks,
                "jump_candidates": jumps,
                "filter_candidates": cascades,
            }
        )
    return {
        "equations": equations,
        "exact_conditioning": exact_conditioning,
        "numeric_slack": numeric_slack,
        "conditions": [
            "Recorded conditioning arguments are taken as exact.",
            "Measured algebraic arguments must reproduce their initial readings.",
            "Forward secants average a changing RHS over the first interval.",
            "Affine exclusions hold only on the matched slice.",
            "A cascade graph alone does not establish a linear filter.",
            "Jump inference requires additive terms and bounded one-sided rates.",
        ],
    }
