"""Nominate two complete analytic models on a fixed graph from training data alone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from autoformalism.research import trajectory_function_families as base
from autoformalism.research.trajectory_function_fitting import _rows, validate_formula
from autoformalism.search.topology_compatibility import CompleteTrajectory

WINDOWS = (1, 4, 16)
Family = tuple[str, tuple[int, ...]]
AFFINE: Family = ("affine", ())


@dataclass
class Choice:
    family: Family
    coefficients: np.ndarray
    rmse: float | None
    expression: str | None


def _shapes(sources: int) -> list[Family]:
    """Every shape receives the same unrestricted affine background."""
    return [
        AFFINE,
        *((kind, (i,)) for i in range(sources) for kind in ("hinge", "tanh", "square")),
        *(("product", pair) for pair in combinations(range(sources), 2)),
    ]


def _features(z: np.ndarray, menu: list[Family], median: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(z)), *z.T]
    for kind, indexes in menu[1:]:
        i = indexes[0]
        if kind == "hinge":
            columns.append(np.maximum(z[:, i] - median[i], 0))
        elif kind == "tanh":
            columns.append(np.tanh(z[:, i]))
        elif kind == "square":
            columns.append(z[:, i] ** 2)
        else:
            columns.append(z[:, i] * z[:, indexes[1]])
    return np.column_stack(columns)


def _intervals(t, y, z, menu, median, previous):
    """Average each RHS basis over 1-, 4-, and 16-step windows using actual times."""
    left = _features(z[:-1], menu, median)
    right = z[1:].copy()
    right[:, previous] = z[:-1, previous]
    end = _features(right, menu, median)
    volume = np.cumsum(np.diff(t)[:, None] * (left + end) * 0.5, axis=0)
    integral = np.vstack([np.zeros(volume.shape[1]), volume])
    blocks = []
    for width in WINDOWS:
        if len(t) <= width:
            continue
        dt = t[width:] - t[:-width]
        x = (integral[width:] - integral[:-width]) / dt[:, None]
        blocks.append((x, (y[width:] - y[:-width]) / dt))
    return blocks


def _design(events, training, sources, node, menu, differential, previous, time_symbol):
    raw = [np.column_stack([r[s] for s in sources]) for r in events]
    center, scale = base._scale([raw[i] for i in training])
    med = (
        np.median(np.concatenate([raw[i] for i in training]), axis=0) - center
    ) / scale
    designs = []
    for r, x in zip(events, raw, strict=True):
        z = (x - center) / scale
        if differential:
            blocks = _intervals(r[time_symbol], r[node], z, menu, med, previous)
        else:
            blocks = [(_features(z, menu, med), r[node])]
        # Equal weight for each event and (within event) each available width.
        count = len(blocks)
        design = np.vstack([a / np.sqrt(count * len(b)) for a, b in blocks])
        response = np.concatenate([b / np.sqrt(count * len(b)) for _, b in blocks])
        designs.append((design, response))
    return designs, center, scale, med


def _fits(designs, training, nsrc, menu):
    """Fit all bases with event-normalized weights and a weak slope ridge."""
    x = np.vstack([designs[i][0] for i in training])
    y = np.concatenate([designs[i][1] for i in training])
    # Column sums of squares reflect the number of events after event weighting.
    # The shared helper's ridge scales with the number of rows.
    adjustment = np.sqrt(len(y) / len(training))
    x, y = x * adjustment, y * adjustment
    common = list(range(1 + nsrc))
    fits = []
    for j in range(len(menu)):
        cols = common + ([nsrc + j] if j else [])
        fits.append((cols, base._coefficients(x[:, cols], y)))
    return fits


def _extra(family: Family, names: list[str], center, scale, median) -> str | None:
    kind, ix = family
    if kind == "affine":
        return None
    args = [base._normalized(names[i], center[i], scale[i]) for i in ix]
    if kind == "hinge":
        return f"max({args[0]}-({median[ix[0]]:.12g}),0)"
    if kind == "square":
        return f"({args[0]}**2)"
    if kind == "tanh":
        return f"tanh({args[0]})"
    return f"({args[0]}*{args[1]})"


def _node(events, deps, node, differential, time_symbol, interpolation):
    observed = [s for s in deps if s in events[0]]
    missing = [s for s in deps if s not in observed]
    hidden_target = node not in events[0]
    names = [*([node] if hidden_target else []), *missing]
    detail = {
        "unobserved": list(dict.fromkeys(names)),
        "observed_sources": observed,
        "response": "interval_average_rate" if differential else "observed_value",
    }
    center, scale = np.zeros(len(deps)), np.ones(len(deps))
    duration = max(float(np.median([np.ptp(ev[time_symbol]) for ev in events])), 1.0)
    gain = 0.1 / duration if differential else 0.1
    warm = np.full(1 + len(deps), gain)
    warm[0] = 0
    if hidden_target or not observed:
        for i, s in enumerate(deps):
            if s in observed:
                center[i], scale[i] = base._scale([ev[s] for ev in events])
        if differential and node in deps:
            warm[1 + deps.index(node)] = -scale[deps.index(node)] / duration
        if not hidden_target:
            y = [
                np.diff(ev[node]) / np.diff(ev[time_symbol])
                if differential
                else ev[node]
                for ev in events
            ]
            warm[0] = np.mean([np.mean(v) for v in y])
        detail["status"] = "rollout_required" if names else "constant_only"
        options = [Choice(AFFINE, warm, None, None)]
        if deps:
            f: Family = ("tanh", (0,))
            options.append(
                Choice(
                    f, np.r_[warm, gain], None, _extra(f, deps, center, scale, center)
                )
            )
        return options, center, scale, detail

    menu = _shapes(len(observed))
    previous = np.array(
        [interpolation.get(s) == "previous_complete_knots" for s in observed]
    )
    all_rows = list(range(len(events)))

    def design(indices):
        return _design(
            events, indices, observed, node, menu, differential, previous, time_symbol
        )

    errors = np.zeros(len(menu))
    have_folds = len(events) >= 2
    if have_folds:
        for fold in (0, 1):
            fit = [i for i in all_rows if i % 2 != fold]
            check = [i for i in all_rows if i % 2 == fold]
            designs, _, _, _ = design(fit)
            for j, (cols, coef) in enumerate(_fits(designs, fit, len(observed), menu)):
                for i in check:
                    x, y = designs[i]
                    errors[j] += np.sum((x[:, cols] @ coef - y) ** 2)
    designs, c, d, med = design(all_rows)
    fits = _fits(designs, all_rows, len(observed), menu)
    for j, (_, coef) in enumerate(fits):
        if not have_folds:
            for x, y in designs:
                errors[j] += np.sum((x[:, fits[j][0]] @ coef - y) ** 2)
    rmse = np.sqrt(errors / len(events))
    # Skip constant-source variants so the second slot can express a different shape.
    varying = np.ptp(
        np.vstack([np.column_stack([r[s] for s in observed]) for r in events]), axis=0
    )
    ids = [j for j, (_, ix) in enumerate(menu) if all(varying[i] > 0 for i in ix)]
    order = sorted(ids, key=lambda j: (rmse[j], j))
    if rmse[0] <= 1.02 * rmse[order[0]] + 1e-10:
        order.remove(0)
        order.insert(0, 0)
    mapping = [deps.index(s) for s in observed]
    center[mapping], scale[mapping] = c, d
    labels = [base._name(menu[j], observed) for j in order]
    status = "observed_projection" if missing else "all_arguments_observed"
    detail.update(
        status=status,
        metric="whole_event_cv_rmse" if have_folds else "training_rmse_only",
        shortlist=[
            {"shape": k, "rmse": float(rmse[j])}
            for k, j in zip(labels[:3], order[:3], strict=True)
        ],
        compared=len(order),
    )
    options = []
    for j in order[:2]:
        shape, ix = menu[j]
        beta = fits[j][1]
        coefficients = warm.copy()
        coefficients[[0, *(1 + i for i in mapping)]] = beta[: 1 + len(observed)]
        if j:
            coefficients = np.r_[coefficients, beta[-1]]
        options.append(
            Choice(
                (shape, tuple(mapping[i] for i in ix)),
                coefficients,
                float(rmse[j]) if have_folds else None,
                _extra(menu[j], observed, c, d, med),
            )
        )
    return options, center, scale, detail


def _bundle(graph, nodes, selection):
    """Reuse the graph's parameter allocation and initialization rules."""
    baseline, reserved = {}, set()
    for node, (options, c, s) in nodes.items():
        chosen = options[selection[node]]
        n = len(graph["sources"][node])
        baseline[node] = [(base.LINEAR, chosen.coefficients[: n + 1], None)], c, s
        reserved.update(graph["sources"][node])
    candidate = base._model(graph, baseline, dict.fromkeys(nodes, 0))
    reserved.update(nodes)
    reserved.update(candidate["parameters"])
    for rule in graph["initializers"].values():
        reserved.update(rule.get("public_initials", ()))
        reserved.add(rule.get("channel", ""))
        reserved.update("state0_" + n for n in rule.get("state_initials", ()))
    for i, (node, (options, _, _)) in enumerate(nodes.items()):
        choice = options[selection[node]]
        if choice.expression is None:
            continue
        symbol = f"shape{i + 1}"
        while symbol in reserved:
            symbol += "x"
        reserved.add(symbol)
        candidate["parameters"][symbol] = float(choice.coefficients[-1])
        candidate["equations"][node] += f"+{symbol}*{choice.expression}"
    return validate_formula(graph, candidate)


def build_shape_shortlist(
    graph: dict,
    training_rows: Sequence[dict | CompleteTrajectory],
    *,
    time_symbol: str = "t",
    interpolation: Mapping[str, str] | None = None,
) -> dict:
    """Return whole-model hypotheses and their training evidence."""
    rows = sorted(_rows(training_rows, "train"), key=lambda row: row.trajectory_id)
    events = [base._series(row, time_symbol) for row in rows]
    if any(
        time_symbol in getattr(rows[0], k)
        for k in ("inputs", "targets", "auxiliaries", "covariates")
    ):
        raise ValueError("Time symbol shadows a public channel")
    nodes, evidence = {}, {}
    for node, deps in graph["sources"].items():
        if not set(deps) <= (set(graph["variable_types"]) | set(events[0])):
            raise ValueError(f"RHS of {node} includes an undeclared source")
        options, c, s, detail = _node(
            events,
            deps,
            node,
            graph["variable_types"][node] == "differential",
            time_symbol,
            interpolation or {},
        )
        nodes[node] = options, c, s
        evidence[node] = detail
    selections = [dict.fromkeys(nodes, 0)]
    eligible = [n for n, (opts, _, _) in nodes.items() if len(opts) > 1]
    if eligible:

        def gap(n):
            a, b = nodes[n][0][:2]
            if a.rmse is None:
                return (1, 0.0)
            return (0, abs(a.rmse - b.rmse) / max(a.rmse, b.rmse, 1e-10))

        alternate = min(eligible, key=gap)
        selections.append({**selections[0], alternate: 1})
    return {
        "candidates": [_bundle(graph, nodes, pick) for pick in selections],
        "candidate_families": [
            {
                n: base._name(nodes[n][0][pick[n]].family, graph["sources"][n])
                for n in nodes
            }
            for pick in selections
        ],
        "evidence": {
            "split": "train",
            "events": [r.trajectory_id for r in rows],
            "window_steps": list(WINDOWS),
            "ranking": "Score values or window-average rates on whole event folds. "
            "Partial-source scores concern only the measured projection. "
            "Unmeasured response laws require joint open-loop fitting.",
            "equations": evidence,
        },
    }
