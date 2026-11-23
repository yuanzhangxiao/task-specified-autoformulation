"""Train-only, fixed-graph function families for the Phase C construction pilot."""

from __future__ import annotations

from itertools import combinations
from math import sqrt

import numpy as np

from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
    validate_evidence,
)

# A family modifies the affine model in exactly one place.
Family = tuple[str, tuple[int, ...]]
LINEAR: Family = ("affine", ())


def _series(row: CompleteTrajectory, time_name: str) -> dict[str, np.ndarray]:
    time = np.asarray(row.time)
    return {
        time_name: time,
        **{k: np.full(time.shape, v) for k, v in row.covariates.items()},
        **{
            k: np.asarray(v)
            for channels in (row.targets, row.inputs, row.auxiliaries)
            for k, v in channels.items()
        },
    }


def _scale(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(blocks)
    center = np.mean(values, axis=0)
    sd = np.std(values, axis=0)
    return center, np.where(sd > 1e-9, sd, np.maximum(np.abs(center), 1.0))


def _features(x: np.ndarray, family: Family) -> np.ndarray:
    kind, cols = family
    x = x.copy()
    if kind == "saturating":
        x[:, cols[0]] = np.tanh(x[:, cols[0]])
    if kind == "interaction":
        x = np.column_stack((x, x[:, cols[0]] * x[:, cols[1]]))
    return np.column_stack((np.ones(len(x)), x))


def _design(
    blocks: list[np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
    family: Family,
    differential: bool,
    previous: np.ndarray,
) -> np.ndarray:
    rows = []
    for block in blocks:
        x = (block - center) / scale
        if differential:
            # Step-held forcings retain their left value throughout this interval.
            right = x[1:].copy()
            right[:, previous] = x[:-1, previous]
            rows.append((_features(x[:-1], family) + _features(right, family)) / 2)
        else:
            rows.append(_features(x, family))
    return np.concatenate(rows)


def _coefficients(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weak ridge stabilizes collinear training features; intercept stays free."""
    penalty = np.eye(design.shape[1]) * sqrt(1e-6 * len(y))
    penalty[0, 0] = 0
    return np.linalg.lstsq(
        np.vstack((design, penalty)), np.r_[y, np.zeros(len(penalty))], rcond=None
    )[0]


def _name(family: Family, sources: list[str]) -> str:
    kind, cols = family
    return " ".join([kind, *(sources[i] for i in cols)])


def _menu(varying: list[int]) -> list[Family]:
    return [
        LINEAR,
        *(("saturating", (i,)) for i in varying),
        *(("interaction", pair) for pair in combinations(varying, 2)),
    ]


def _rank(blocks, changes, families, differential, previous):
    """Two folds of whole training trajectories. Learn scaling on the fitting fold."""
    squared = np.zeros(len(families))
    for fold in (0, 1):
        train = [i for i in range(len(changes)) if i % 2 != fold]
        check = [i for i in range(len(changes)) if i % 2 == fold]
        train_x, check_x = [blocks[i] for i in train], [blocks[i] for i in check]
        y, truth = (
            np.concatenate([changes[i] for i in train]),
            np.concatenate([changes[i] for i in check]),
        )
        center, scale = _scale(train_x)
        for index, family in enumerate(families):
            design = _design(train_x, center, scale, family, differential, previous)
            held = _design(check_x, center, scale, family, differential, previous)
            estimate = held @ _coefficients(design, y)
            squared[index] += np.sum((estimate - truth) ** 2)
    return np.sqrt(squared / sum(map(len, changes)))


def _comparison(blocks, changes, times, ids, sources, differential, previous):
    """Find two close observed argument tuples and describe their output or rate gap."""
    if not sources:
        return None
    values = []
    for b in blocks:
        if differential:
            x = (b[:-1] + b[1:]) / 2
            x[:, previous] = b[:-1, previous]
        else:
            x = b
        values.append(x)
    x = np.concatenate(values)
    y = np.concatenate(changes)
    width = np.maximum(np.ptp(x, axis=0) * 0.01, 1e-12)
    bins = np.floor((x - x.min(axis=0)) / width).astype(int)
    extrema = {}
    for i, entry in enumerate(map(tuple, bins)):
        lo, hi = extrema.get(entry, (i, i))
        extrema[entry] = i if y[i] < y[lo] else lo, i if y[i] > y[hi] else hi
    low, high = max(extrema.values(), key=lambda ij: y[ij[1]] - y[ij[0]])
    if y[low] == y[high]:
        return None
    boundaries = np.cumsum([0, *map(len, changes)])
    samples = []
    for i in (low, high):
        block = int(np.searchsorted(boundaries, i, side="right") - 1)
        k = int(i - boundaries[block])
        samples.append(
            {
                "trajectory": ids[block],
                "time": list(map(float, times[block][k : k + 1 + int(differential)])),
                "arguments": dict(zip(sources, map(float, x[i]), strict=True)),
                "observed": float(y[i]),
            }
        )
    return {
        "meaning": "Comparable arguments within 1% of each training range. "
        "Differential arguments are interval averages; this is advisory.",
        "samples": samples,
    }


def _equation(node, sources, differential, events, time_name, interpolation):
    known = events[0][1]
    missing = [s for s in sources if s not in known]
    previous = np.array(
        [interpolation.get(s) == "previous_complete_knots" for s in sources], bool
    )
    usable = [
        (key, row) for key, row in events if len(row[time_name]) > int(differential)
    ]
    center, scale = np.zeros(len(sources)), np.ones(len(sources))
    for i, name in enumerate(sources):
        if name in known:
            center[i], scale[i] = _scale([r[name] for _, r in events])
    evidence = {
        "comparison": "interval_average_rate" if differential else "algebraic_value",
        "unobserved": list(
            dict.fromkeys([*([node] if node not in known else []), *missing])
        ),
        "source_ranges": {
            s: [
                float(min(r[s].min() for _, r in events)),
                float(max(r[s].max() for _, r in events)),
            ]
            for s in sources
            if s in known
        },
    }
    varying = [
        i
        for i, src in enumerate(sources)
        if src in missing or np.ptp(evidence["source_ranges"][src]) > 1e-9
    ]
    families = _menu(varying)
    if evidence["unobserved"] or not usable:
        evidence["status"] = "whole_system_rollout_required"
        duration = np.median([np.ptp(row[time_name]) for _, row in events])
        rate = 1 / max(duration, 1.0) if differential else 1.0
        # Nonzero free couplings keep an unobserved pathway active at the initial
        # vector. A chain initialized with every gain zero has a zero Jacobian.
        guesses = np.full(1 + len(sources), rate * 0.1)
        guesses[0] = 0.0
        if differential and node in sources:
            guesses[1 + sources.index(node)] = -rate * scale[sources.index(node)]
        if usable and node in known:
            # Estimate initial weights for the observed part of the balance only.
            # Hidden contributions remain unknown, so this is never a family score.
            observed = [i for i, src in enumerate(sources) if src in known]
            x = [
                np.column_stack([r[sources[i]] for i in observed])
                if observed
                else np.empty((len(r[node]), 0))
                for _, r in usable
            ]
            y = np.concatenate(
                [
                    np.diff(r[node]) / np.diff(r[time_name])
                    if differential
                    else r[node]
                    for _, r in usable
                ]
            )
            design = _design(
                x,
                center[observed],
                scale[observed],
                LINEAR,
                differential,
                previous[observed],
            )
            guesses[[0, *(1 + i for i in observed)]] = _coefficients(design, y)
        options = [(f, guesses, None) for f in families[:2]]
        return options, center, scale, evidence

    x = [
        np.column_stack([r[s] for s in sources])
        if sources
        else np.empty((len(r[node]), 0))
        for _, r in usable
    ]
    t = [r[time_name] for _, r in usable]
    changes = [
        np.diff(r[node]) / np.diff(r[time_name]) if differential else r[node]
        for _, r in usable
    ]
    y = np.concatenate(changes)
    center, scale = _scale(x)
    scores = (
        _rank(x, changes, families, differential, previous)
        if len(changes) >= 2
        else None
    )
    order = list(range(len(families)))
    if scores is not None:
        order.sort(key=lambda i: (scores[i], i))
        # Preserve a simple family when differences are within two percent.
        if scores[0] <= scores[order[0]] * 1.02 + 1e-10:
            order.remove(0)
            order.insert(0, 0)
    options = []
    for i in order[:2]:
        family = families[i]
        design = _design(x, center, scale, family, differential, previous)
        options.append(
            (
                family,
                _coefficients(design, y),
                None if scores is None else float(scores[i]),
            )
        )
    evidence.update(
        status="train_trajectory_cross_validation"
        if scores is not None
        else "one_training_event",
        trajectories=len(changes),
        samples=len(y),
        family_rmse={_name(families[i], sources): float(scores[i]) for i in order[:3]}
        if scores is not None
        else {},
        comparable_arguments=_comparison(
            x, changes, t, [i for i, _ in usable], sources, differential, previous
        ),
    )
    return options, center, scale, evidence


def _normalized(name: str, center: float, scale: float) -> str:
    if center == 0 and scale == 1:
        return name
    # Always include a leading sign-safe pair of parentheses around the center.
    return f"(({name}-({center:.12g}))/{scale:.12g})"


def _expression(sources, family, center, scale, coeffs, parameter):
    parts = [
        _normalized(n, c, s) for n, c, s in zip(sources, center, scale, strict=True)
    ]
    kind, columns = family
    if kind == "saturating":
        parts[columns[0]] = f"tanh({parts[columns[0]]})"
    if kind == "interaction":
        parts.append(f"({parts[columns[0]]}*{parts[columns[1]]})")
    return "+".join(
        [
            parameter(coeffs[0]),
            *(
                f"{parameter(k)}*{part}"
                for k, part in zip(coeffs[1:], parts, strict=True)
            ),
        ]
    )


def _model(graph, all_options, choices):
    equations, initials, params = {}, {}, {}
    reserved = set(graph["sources"]).union(*graph["sources"].values())
    for rule in graph["initializers"].values():
        if "channel" in rule:
            reserved.add(rule["channel"])
        reserved.update(rule.get("public_initials", []))
        reserved.update("state0_" + s for s in rule.get("state_initials", []))
    prefix = "afcoef"
    while any(name.startswith(prefix) for name in reserved):
        prefix += "x"

    def parameter(value):
        name = f"{prefix}{len(params) + 1}"
        params[name] = float(value)
        return name

    for node, (options, center, scale) in all_options.items():
        family, coefs, _ = options[choices[node]]
        equations[node] = _expression(
            graph["sources"][node], family, center, scale, coefs, parameter
        )
    for state, rule in sorted(graph["initializers"].items()):
        if rule["mode"] == "observed_initial":
            continue
        if rule["mode"] not in ("map", "value") or (
            rule["mode"] == "value" and rule.get("scope") != "shared_training"
        ):
            raise ValueError(f"Unsupported initializer in the time-zero pilot: {rule}")
        args = [
            *rule.get("public_initials", []),
            *("state0_" + v for v in rule.get("state_initials", [])),
        ]
        initials[state] = "+".join(
            [parameter(0), *(f"{parameter(0)}*{a}" for a in args)]
        )
    return validate_formula(
        graph, {"equations": equations, "initializers": initials, "parameters": params}
    )


def _basin_outlet(graph, evidence):
    """Isolate outlet flow under the registered two-basin public balance."""
    rhs = graph["sources"]
    required = {
        "h_down": {
            "inflow_down",
            "area_down",
            "h_up",
            "crest_up",
            "h_down",
            "crest_down",
        },
        "h_up": {"inflow_up", "area_up", "h_up", "crest_up"},
    }
    missing = {
        node: sorted(names - set(rhs.get(node, []))) for node, names in required.items()
    }
    groups = []
    for row in sorted(evidence.trajectories, key=lambda item: item.trajectory_id):
        c = row.covariates
        time = np.asarray(row.time)
        dt = np.diff(time)
        up = np.asarray(row.inputs["inflow_up"])
        down = np.asarray(row.inputs["inflow_down"])
        if not len(dt) or min(c["area_up"], c["area_down"]) <= 0 or np.any(up < 0):
            continue
        integrated = np.cumsum((up[:-1] + up[1:]) * dt / 2)
        # Even if all the inlet is retained, the upstream water cannot overflow.
        room = (c["crest_up"] - c["initial_up"]) * c["area_up"]
        isolated = integrated <= room + 1e-12 * max(1.0, abs(room))
        if not np.any(isolated):
            continue
        h = np.asarray(row.targets["h_down"])
        head = np.maximum(h - c["crest_down"], 0)
        outlet = (down[:-1] + down[1:]) / 2 - c["area_down"] * np.diff(h) / dt
        groups.append(
            (
                row.trajectory_id,
                np.array([head[:-1], head[1:]])[:, isolated],
                outlet[isolated],
            )
        )
    result = {
        "status": "insufficient_isolated_training_events",
        "basis": "Two-basin public volume balance; upstream cannot overflow before "
        "integrated runoff fills its surveyed free storage.",
        "missing_direct_sources_for_depth_balances": missing,
        "isolated_intervals": {key: len(outlet) for key, _, outlet in groups},
        "scope": "Outlet discharge on transfer-free training intervals only.",
    }
    if len(groups) < 2:
        return result
    groups.sort(key=lambda x: (-len(x[2]), x[0]))
    train_id, x, y = groups[0]
    scores = []
    for p in (1.0, 1.5, 2.0):
        feature = np.mean(x**p, axis=0)
        if np.dot(feature, feature) == 0:
            continue
        k = max(0.0, float(np.dot(feature, y) / np.dot(feature, feature)))
        errors = np.concatenate(
            [k * np.mean(head**p, axis=0) - outlet for _, head, outlet in groups[1:]]
        )
        scores.append(
            {
                "power": p,
                "coefficient": k,
                "flow_rmse_m3_per_min": float(np.sqrt(np.mean(errors**2))),
            }
        )
    if scores:
        result.update(
            status="ranked_outlet_laws",
            fitting_trajectory=train_id,
            check_trajectories=[key for key, _, _ in groups[1:]],
            scores=sorted(scores, key=lambda item: item["flow_rmse_m3_per_min"]),
        )
    return result


def build_shortlist(graph: dict, frozen: dict) -> dict:
    """Return two fixed-graph hypotheses and compact evidence from training alone."""
    source = frozen["evidence"]
    if source.get("split") != "train":
        raise ValueError("Function family screening requires training evidence")
    evidence = TrainingCompatibilityEvidence.model_validate(source)
    validate_evidence(evidence)
    time = evidence.context.time_symbol
    events = [
        (r.trajectory_id, _series(r, time))
        for r in sorted(evidence.trajectories, key=lambda row: row.trajectory_id)
    ]
    options, diagnostics = {}, {}
    for node, deps in sorted(graph["sources"].items()):
        ranked, center, scale, detail = _equation(
            node,
            deps,
            graph["variable_types"][node] == "differential",
            events,
            time,
            evidence.contract.interpolation,
        )
        options[node] = ranked, center, scale
        diagnostics[node] = detail

    first = dict.fromkeys(options, 0)
    menus = [first]

    # Change one equation with the closest competitive family. Relative error
    # costs are comparable across glucose, insulin, water depth, etc.
    def relative_cost(node):
        ((_, _, a), (_, _, b)), _, _ = options[node]
        return (a is None, abs(a - b) / max(a, b, 1e-10) if a is not None else 0, node)

    eligible = [node for node, (opts, _, _) in options.items() if len(opts) > 1]
    if eligible:
        alternate = min(eligible, key=relative_cost)
        menus.append({**first, alternate: 1})
    result = {
        "candidates": [_model(graph, options, choices) for choices in menus],
        "candidate_families": [
            {
                node: _name(opts[choices[node]][0], graph["sources"][node])
                for node, (opts, _, _) in options.items()
            }
            for choices in menus
        ],
        "evidence": {
            "split": "train",
            "training_evidence_sha256": evidence.sha256,
            "ranking": "Rank affine, single-source tanh saturation and pairwise "
            "interaction alternatives by two-fold whole-trajectory RMSE. "
            "Direct differential scores compare observed interval rates "
            "to integrated RHS estimates. Equations with unobserved "
            "arguments need whole-system rollout.",
            "equations": diagnostics,
        },
    }
    if frozen["case_id"] == "basin_seed0_downstream":
        if any(
            evidence.contract.interpolation.get(s) != "linear_complete_knots"
            for s in ("inflow_up", "inflow_down")
        ):
            raise ValueError(
                "The registered basin diagnostic requires linear runoff knots"
            )
        result["evidence"]["isolated_outlet"] = _basin_outlet(graph, evidence)
    return result
