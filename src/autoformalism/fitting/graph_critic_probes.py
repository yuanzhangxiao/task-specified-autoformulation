"""Training-only graph probes for a fixed, fitted continuous-time candidate."""

from __future__ import annotations

import ast
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from autoformalism.data import Trajectory
from autoformalism.expressions import CompiledModel, RuntimeExpressionError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.interaction_response import _bands
from autoformalism.fitting.public_fitting import unpack_split
from autoformalism.fitting.simulation import forcing_segment_indices, trajectory_forcing
from autoformalism.schemas.public_fitting import PublicSplit

RULES = {
    "protocol": "graph-critic-probes-1",
    "selection": "max absolute error in endpoint change / pooled target training SD",
    "windows": ["early", "middle", "late"],
    "partition": "elapsed-time thirds; minimum two samples per window",
    "tie_break": "ascending trajectory ID, band index, target name",
    "sd_floor": 1e-12,
    "constant_targets": "exclude from ranking at or below SD floor",
    "minimum_scaled_change_error": 0.05,
    "driver": "identity-observed target state upstream of winner, or winner itself",
    "other_outputs": "reachable targets excluding direct aliases of the driver state",
    "paths": "shortest syntactic path; lexical tie break",
    "substitution": "linear interpolation of training readings at every sample knot",
    "initials": "each solver's own free state at window onset, replacing driver only",
    "solvers": ["Radau", "DOP853"],
    "max_rhs_calls_per_solver_trace": 40000,
    "balance": "signed RHS integrals, checked on both solvers and decimated knots",
    "balance_relative_atol": 1e-3,
    "balance_scaled_atol": 1e-3,
    "rtol": 1e-9,
    "atol": 1e-11,
    "solver_scaled_atol": 1e-6,
    "saved_nmse_rtol": 1e-6,
    "saved_nmse_atol": 1e-9,
}
_NUMERICAL_ERRORS = (ArithmeticError, ValueError, RuntimeError, RuntimeExpressionError)


def _graph(model: CompiledModel) -> dict[str, set[str]]:
    """Trace symbol dependencies in the validated RHS, processes and readouts."""
    v = model.validated
    nodes = {s: f"state:{s}" for s in model.state_names}
    nodes.update({p: f"process:{p}" for p in v.process_order})
    edges = {v: set() for v in nodes.values()}
    for prefix, expressions in (
        ("state", v.equation_expressions),
        ("process", v.process_expressions),
        ("target", v.observation_expressions),
    ):
        for name, expr in expressions.items():
            for symbol in expr.symbols & nodes.keys():
                edges[nodes[symbol]].add(f"{prefix}:{name}")
    return edges


def _paths(graph: dict[str, set[str]], state: str) -> dict[str, list[str]]:
    """Keep one acyclic shortest path to each structurally reachable readout."""
    first = f"state:{state}"
    queue = deque([[first]])
    seen = {first}
    targets = {}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node.startswith("target:"):
            targets[node.removeprefix("target:")] = path
        for other in sorted(graph.get(node, set()) - seen):
            seen.add(other)
            queue.append([*path, other])
    return targets


@dataclass
class _Trace:
    values: np.ndarray
    states: np.ndarray


def _trace(
    model: CompiledModel,
    row: Trajectory,
    p: Mapping[str, float],
    initial: np.ndarray,
    method: str,
    first: int,
    last: int,
    driver: tuple[int, str] | None = None,
    *,
    cost: dict[str, int],
) -> _Trace:
    """Integrate across forcing corners and every substituted observation knot."""
    time = row.time
    forcing = trajectory_forcing(model, row)
    full = np.empty((last - first + 1, len(initial)))
    full[0] = initial
    active = np.ones(len(initial), dtype=bool)
    if driver is None:
        knots = forcing_segment_indices(model, row)
        knots = np.r_[first, knots[(first < knots) & (knots < last)], last]
    else:
        index, channel = driver
        active[index] = False
        full[:, index] = row.targets[channel][first : last + 1]
        knots = np.arange(first, last + 1)

    calls = 0
    key = "clamped_rhs" if driver else "baseline_rhs"
    cost["solver_traces_started"] += 1

    def rhs(t, z):
        nonlocal calls
        if calls >= RULES["max_rhs_calls_per_solver_trace"]:
            raise ValueError(
                f"{method} exceeded {calls} RHS calls on {row.trajectory_id}"
            )
        calls += 1
        cost[key] += 1
        if driver is None:
            return model.rhs(t, z, p, forcing)
        x = np.empty(len(initial))
        x[active] = z
        x[index] = np.interp(t, time, row.targets[channel])
        return model.rhs(t, x, p, forcing)[active]

    for lo, hi in pairwise(knots):
        if not np.any(active):
            break
        sol = solve_ivp(
            rhs,
            (time[lo], time[hi]),
            full[lo - first, active],
            t_eval=time[lo + 1 : hi + 1],
            method=method,
            atol=RULES["atol"],
            rtol=RULES["rtol"],
        )
        if not sol.success or sol.y.shape != (sum(active), hi - lo):
            raise ValueError(f"{method} integration failed")
        full[lo + 1 - first : hi + 1 - first, active] = sol.y.T

    channels = model.validated.context.targets
    outputs = []
    for t, x in zip(time[first : last + 1], full, strict=True):
        model.validate_state_constraints(x)
        observed = model.observe(t, x, p, forcing)
        outputs.append([observed[c] for c in channels])
    return _Trace(np.asarray(outputs), full)


def _agreement(traces: Sequence[_Trace], scales: np.ndarray) -> float:
    diff = float(np.max(np.abs(traces[0].values - traces[1].values) / scales))
    if not np.isfinite(diff) or diff > RULES["solver_scaled_atol"]:
        raise ValueError("independent solvers disagree on measured outputs")
    return diff


def _winner(rows, channels, observed, free, scales):
    """Use the same time partition on all channels; comparisons need two samples."""
    candidates = []
    for i, row in enumerate(rows):
        bands = _bands(row.time)
        for band, name in enumerate(RULES["windows"]):
            indices = np.flatnonzero(bands == band)
            if len(indices) < 2:
                continue
            a, b = int(indices[0]), int(indices[-1])
            truth = observed[i][b] - observed[i][a]
            estimate = free[i][0].values[b] - free[i][0].values[a]
            for j, target in enumerate(channels):
                if scales[j] <= RULES["sd_floor"]:
                    continue
                error = float(abs(estimate[j] - truth[j]) / scales[j])
                key = (-error, row.trajectory_id, band, target)
                window = {
                    "trajectory_id": row.trajectory_id,
                    "target": target,
                    "band": name,
                    "start": float(row.time[a]),
                    "end": float(row.time[b]),
                    "observed_change": float(truth[j]),
                    "predicted_change": float(estimate[j]),
                    "scaled_error": error,
                }
                candidates.append((key, window, i, a, b))
    winner = min(candidates, key=lambda r: r[0]) if candidates else None
    return len(candidates), winner


def _outcome(actual, original, replaced):
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values - actual))))

    free_rmse, new_rmse = rmse(original), rmse(replaced)
    return {
        "observed_terminal": float(actual[-1]),
        "free_terminal": float(original[-1]),
        "substituted_terminal": float(replaced[-1]),
        "free_rmse": free_rmse,
        "substituted_rmse": new_rmse,
        "terminal_error_reduction": float(
            abs(original[-1] - actual[-1]) - abs(replaced[-1] - actual[-1])
        ),
        "rmse_reduction": free_rmse - new_rmse,
    }


def _identity_readouts(model: CompiledModel) -> dict[str, str]:
    """Identify exact target-to-state observations, including pure process aliases."""
    validated = model.validated
    aliases = {state: state for state in model.state_names}
    for name in validated.process_order:
        expr = validated.process_expressions[name].tree.body
        if isinstance(expr, ast.Name) and expr.id in aliases:
            aliases[name] = aliases[expr.id]
    result = {}
    for target, expr in validated.observation_expressions.items():
        body = expr.tree.body
        if isinstance(body, ast.Name) and body.id in aliases:
            result[target] = aliases[body.id]
    return result


def _signed_terms(node: ast.expr, sign: int = 1):
    """Flatten only additive outer operators of a validated equation."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        yield from _signed_terms(
            node.operand, -sign if isinstance(node.op, ast.USub) else sign
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        yield from _signed_terms(node.left, sign)
        yield from _signed_terms(
            node.right, -sign if isinstance(node.op, ast.Sub) else sign
        )
    else:
        yield sign, node


def _balance(model, row, p, free, window, a, b, scale):
    """Check rate integrals against fitted change, coarser knots and both solvers."""
    state = _identity_readouts(model).get(window["target"])
    result = {"status": "withheld", "reason": None, "state": state, "terms": []}
    if state is None or b - a < 2:
        return result | {"reason": "balance requires a direct state and three samples"}
    expr = model.validated.equation_expressions[state]
    pieces = [
        (sign, RestrictedParser().parse(ast.unparse(node), location="balance"))
        for sign, node in _signed_terms(expr.tree.body)
    ]
    times = row.time[a : b + 1]
    decimated = np.unique(np.r_[np.arange(0, len(times), 2), len(times) - 1])
    forcing = trajectory_forcing(model, row)
    totals = []
    scale_tol = RULES["balance_scaled_atol"] * scale
    rtol = RULES["balance_relative_atol"]
    try:
        for trace in free:
            rates = np.array(
                [
                    [
                        sign * model.evaluate_expression(term, t, x, p, forcing)
                        for sign, term in pieces
                    ]
                    for t, x in zip(times, trace.states[a : b + 1], strict=True)
                ]
            )
            integrals = trapezoid(rates, times, axis=0)
            coarse = trapezoid(rates[decimated], times[decimated], axis=0)
            k = model.state_names.index(state)
            change = float(trace.states[b, k] - trace.states[a, k])
            if (
                not np.all(np.isfinite(integrals))
                or not np.allclose(coarse, integrals, atol=scale_tol, rtol=rtol)
                or not np.isclose(sum(integrals), change, atol=scale_tol, rtol=rtol)
            ):
                return result | {
                    "reason": "sampled quadrature lacks closure or resolution"
                }
            totals.append(integrals)
    except _NUMERICAL_ERRORS as exc:
        return result | {"reason": str(exc)}
    if not np.allclose(*totals, atol=scale_tol, rtol=rtol):
        return result | {"reason": "term integrals disagree between solvers"}
    total = float(sum(totals[0]))
    return result | {
        "status": "ready",
        "fitted_change": window["predicted_change"],
        "integral_sum": total,
        "closure_error": total - window["predicted_change"],
        "terms": [
            {"expression": term.source, "sign": sign, "signed_integral": float(v)}
            for (sign, term), v in zip(pieces, totals[0], strict=True)
        ],
    }


def _substitutions(model, row, p, free, observed, window, a, b, scales, cost):
    """Probe parents of the chosen target and its influence when directly measured."""
    graph = _graph(model)
    channels = model.validated.context.targets
    identities = _identity_readouts(model)
    probes = []
    for driver, state in sorted(identities.items()):
        paths = _paths(graph, state)
        if window["target"] not in paths:
            continue
        paths = {t: path for t, path in paths.items() if identities.get(t) != state}
        if not paths:
            continue
        record = {
            "driver": driver,
            "state": state,
            "paths": paths,
            "status": "withheld",
            "reason": None,
            "outcomes": {},
        }
        probes.append(record)
        try:
            replay = [
                _trace(
                    model,
                    row,
                    p,
                    trace.states[a],
                    method,
                    a,
                    b,
                    (model.state_names.index(state), driver),
                    cost=cost,
                )
                for method, trace in zip(RULES["solvers"], free, strict=True)
            ]
            record["max_scaled_solver_difference"] = _agreement(replay, scales)
        except _NUMERICAL_ERRORS as exc:
            record["reason"] = str(exc)
            continue
        record["status"] = "ready"
        for target in paths:
            j = channels.index(target)
            record["outcomes"][target] = _outcome(
                observed[a : b + 1, j],
                free[0].values[a : b + 1, j],
                replay[0].values[:, j],
            )
    return probes


def probe_endpoint(
    model: CompiledModel,
    training: PublicSplit,
    parameters: Mapping[str, float],
    initial_states: Mapping[str, Sequence[float]],
    *,
    expected_nmse: float | None = None,
) -> dict:
    """Run the frozen discovery and substitution rules on one saved endpoint."""
    if training.name != "train":
        raise ValueError("graph critic requires the public training split")
    ctx = model.validated.context
    if set(model.validated.forcing_symbols) & set(ctx.targets):
        raise ValueError("baseline free rollout cannot use measured target forcing")
    if set(parameters) != set(model.parameter_names):
        raise ValueError("shared parameter identity differs from compiled model")
    rows = unpack_split(training).trajectories
    channels = ctx.targets
    if set(initial_states) != {r.trajectory_id for r in rows} or any(
        set(r.targets) != set(channels) for r in rows
    ):
        raise ValueError("trajectory or target coverage differs")
    observed = [np.column_stack([r.targets[c] for c in channels]) for r in rows]
    scales = np.maximum(np.std(np.concatenate(observed), axis=0), RULES["sd_floor"])
    result = {
        "status": "withheld",
        "reason": None,
        "training_nmse": None,
        "window": None,
        "probes": [],
        "compared_windows": 0,
        "balance": {
            "status": "withheld",
            "reason": "no verified discrepancy",
            "terms": [],
        },
        "compute": {"solver_traces_started": 0, "baseline_rhs": 0, "clamped_rhs": 0},
        "trajectories": len(rows),
        "constant_targets": [
            ch
            for ch, sd in zip(channels, scales, strict=True)
            if sd <= RULES["sd_floor"]
        ],
    }
    free = []
    differences = []
    try:
        for row in rows:
            start = np.asarray(initial_states[row.trajectory_id], dtype=float)
            if start.shape != (len(model.state_names),) or not np.isfinite(start).all():
                raise ValueError("invalid fitted initial state")
            traces = [
                _trace(
                    model,
                    row,
                    parameters,
                    start,
                    method,
                    0,
                    len(row.time) - 1,
                    cost=result["compute"],
                )
                for method in RULES["solvers"]
            ]
            differences.append(_agreement(traces, scales))
            free.append(traces)
    except _NUMERICAL_ERRORS as exc:
        result["reason"] = str(exc)
        return result
    result["max_scaled_solver_difference"] = max(differences)
    residuals = np.concatenate(
        [
            (f[0].values - actual) / scales
            for f, actual in zip(free, observed, strict=True)
        ]
    )
    score = float(np.mean(np.square(residuals)))
    if not np.isfinite(score) or (
        expected_nmse is not None
        and not np.isclose(
            score,
            expected_nmse,
            rtol=RULES["saved_nmse_rtol"],
            atol=RULES["saved_nmse_atol"],
        )
    ):
        result["reason"] = "training score differs from saved endpoint"
        return result
    result["training_nmse"] = score
    result["compared_windows"], choice = _winner(rows, channels, observed, free, scales)
    if choice is None:
        result["reason"] = "no rankable windows"
        return result
    _, window, i, a, b = choice
    result["window"] = window
    if window["scaled_error"] < RULES["minimum_scaled_change_error"]:
        result["reason"] = "window change errors below presentation threshold"
        return result
    probes = _substitutions(
        model,
        rows[i],
        parameters,
        free[i],
        observed[i],
        window,
        a,
        b,
        scales,
        result["compute"],
    )
    result["balance"] = _balance(
        model,
        rows[i],
        parameters,
        free[i],
        window,
        a,
        b,
        scales[channels.index(window["target"])],
    )
    result["probes"] = probes
    if result["balance"]["status"] == "ready" or any(
        p["status"] == "ready" for p in probes
    ):
        result["status"] = "ready"
    else:
        result["reason"] = "no verified balance or cross-target substitution"
    return result
