"""Train-only bounded function screening and independent validation rollouts."""

from __future__ import annotations

import ast
import platform
from graphlib import TopologicalSorter
from importlib.metadata import version
from time import monotonic

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY, RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.search.requirement_feedback import _simplify
from autoformalism.search.topology_compatibility import CompleteTrajectory
from autoformalism.staged_topology import content_hash


class Formula(StrictSchema):
    """One complete graph realization; shared parameters have one start each."""

    equations: dict[Identifier, str] = Field(max_length=64)
    initializers: dict[Identifier, str] = Field(max_length=64)
    parameters: dict[Identifier, FiniteFloat] = Field(max_length=128)


def _initial_sources(rule: dict) -> set[str]:
    return set(rule.get("public_initials", ())) | {
        f"state0_{n}" for n in rule.get("state_initials", ())
    }


def validate_formula(graph: dict, formula: dict) -> dict:
    """Parse every expression and bind its symbols to the frozen dependencies."""
    value = Formula.model_validate(formula)
    kinds, rules = graph["variable_types"], graph["initializers"]
    states = {n for n in kinds if kinds[n] == "differential"}
    if set(rules) != states or set(value.equations) != set(kinds):
        raise ValueError(
            "Every generated variable needs its frozen equation and initial type"
        )
    free = {n for n in states if rules[n]["mode"] != "observed_initial"}
    if set(value.initializers) != free:
        raise ValueError(f"Provide initializers only for {sorted(free)}")
    param = set(value.parameters)
    reserved = set(kinds) | {"t"} | set(APPROVED_FUNCTION_ARITY)
    for names in graph["sources"].values():
        reserved.update(names)
    for rule in rules.values():
        reserved.update(_initial_sources(rule))
        if "channel" in rule:
            reserved.add(rule["channel"])
    if param & reserved:
        raise ValueError(
            f"Parameter names collide with graph symbols: {sorted(param & reserved)}"
        )
    used = set()
    parser = RestrictedParser()
    for key, expressions, deps in (
        ("equation", value.equations, graph["sources"]),
        ("initial", value.initializers, {n: _initial_sources(rules[n]) for n in free}),
    ):
        for name, text in expressions.items():
            try:
                parsed = parser.parse(text, location=f"{key}.{name}")
            except ModelValidationError as error:
                raise ValueError(str(error)) from error
            sources = set(deps[name])
            if parsed.symbols - param != sources:
                raise ValueError(
                    f"{key} {name} needs {sorted(sources)}; "
                    f"got {sorted(parsed.symbols - param)}"
                )
            live = {
                node.id
                for node in ast.walk(_simplify(parsed.tree))
                if isinstance(node, ast.Name)
            }
            if inactive := sources - live:
                raise ValueError(f"{key} {name} cancels sources: {sorted(inactive)}")
            used.update(live & param)
    if remaining := param - used:
        raise ValueError(f"Unused parameters: {sorted(remaining)}")
    return value.model_dump(mode="json")


class _Rollout:
    """Compile one restricted graph and reuse the flow for each trajectory length."""

    def __init__(self, graph, formula, targets, substeps):
        import casadi as ca

        self.ca = ca
        formula = validate_formula(graph, formula)
        self.parameters = sorted(formula["parameters"])
        self.states = sorted(graph["initializers"])
        self.targets = tuple(targets)
        self.rules = graph["initializers"]
        if set(targets) - graph["variable_types"].keys():
            raise ValueError("The graph does not generate all public targets")
        sources = set().union(*map(set, graph["sources"].values()))
        self.forcings = sorted(sources - graph["variable_types"].keys())
        self.readings = sorted(
            {
                r["channel"]
                for r in self.rules.values()
                if r["mode"] == "observed_initial"
            }
            | {n for r in self.rules.values() for n in r.get("public_initials", ())}
        )
        parser = RestrictedParser()

        def expression(text, env):
            return _translate(ca, parser.parse(text, location="function"), env)

        p = ca.SX.sym("parameters", len(self.parameters))
        parameters = dict(zip(self.parameters, ca.vertsplit(p), strict=True))
        z = ca.SX.sym("readings", len(self.readings))
        initials = parameters | dict(zip(self.readings, ca.vertsplit(z), strict=True))
        order = TopologicalSorter(
            {n: r.get("state_initials", ()) for n, r in self.rules.items()}
        ).static_order()
        for n in order:
            rule = self.rules[n]
            initials[f"state0_{n}"] = (
                initials[rule["channel"]]
                if rule["mode"] == "observed_initial"
                else expression(formula["initializers"][n], initials)
            )
        self.initialize = ca.Function(
            "initialize",
            [p, z],
            [ca.vertcat(*(initials[f"state0_{n}"] for n in self.states))],
        )
        x = ca.SX.sym("state", len(self.states))
        u = ca.SX.sym("forcing", len(self.forcings))
        env = parameters | dict(zip(self.states, ca.vertsplit(x), strict=True))
        env.update(zip(self.forcings, ca.vertsplit(u), strict=True))
        algebraic = set(graph["variable_types"]) - set(self.states)
        for n in TopologicalSorter(
            {n: sorted(set(graph["sources"][n]) & algebraic) for n in sorted(algebraic)}
        ).static_order():
            env[n] = expression(formula["equations"][n], env)
        rhs = ca.Function(
            "rhs",
            [x, p, u],
            [
                ca.vertcat(
                    *(expression(formula["equations"][n], env) for n in self.states)
                )
            ],
        )
        self.observe = ca.Function(
            "observe", [x, p, u], [ca.vertcat(*(env[n] for n in targets))]
        )
        right = ca.SX.sym("right", len(self.forcings))
        dt = ca.SX.sym("dt")
        y, step = x, dt / substeps
        for i in range(substeps):
            left = u + (right - u) * (i / substeps)
            mid = u + (right - u) * ((i + 0.5) / substeps)
            end = u + (right - u) * ((i + 1) / substeps)
            k1 = rhs(y, p, left)
            k2 = rhs(y + step * k1 / 2, p, mid)
            k3 = rhs(y + step * k2 / 2, p, mid)
            k4 = rhs(y + step * k3, p, end)
            y = y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.step = ca.Function("step", [x, p, u, right, dt], [y])
        self.flows = {}

    def _flow(self, size):
        if size not in self.flows:
            ca = self.ca
            z = ca.MX.sym("z", len(self.readings))
            u = ca.MX.sym("u", len(self.forcings), size)
            d = ca.MX.sym("d", 1, size - 1)
            p = ca.MX.sym("p", len(self.parameters))
            x0 = self.initialize(p, z)
            xs = ca.horzcat(
                x0,
                self.step.mapaccum(size - 1)(
                    x0, ca.repmat(p, 1, size - 1), u[:, :-1], u[:, 1:], d
                ),
            )
            ys = self.observe.map(size)(xs, ca.repmat(p, 1, size), u)
            self.flows[size] = ca.Function("rollout", [p, z, u, d], [ys, xs])
        return self.flows[size]

    def _initial_args(self, row):
        public = (
            {"t": row.time[0]}
            | {k: v[0] for k, v in (row.inputs | row.auxiliaries).items()}
            | row.covariates
        )
        initial = public | {k: v[0] for k, v in row.targets.items()}
        if missing := set(self.forcings) - public.keys():
            raise ValueError(f"Unavailable ongoing forcings: {sorted(missing)}")
        if missing := set(self.readings) - initial.keys():
            raise ValueError(f"Unavailable public initial readings: {sorted(missing)}")
        return [initial[n] for n in self.readings], [public[n] for n in self.forcings]

    def record(self, row):
        t = np.asarray(row.time)
        public = (
            {"t": t}
            | row.inputs
            | row.auxiliaries
            | {k: np.full(len(t), v) for k, v in row.covariates.items()}
        )
        # The only target observations available to an equation are generated states.
        # Measured target histories enter this routine solely to obtain t0 readings.
        readings, _ = self._initial_args(row)
        args = (
            readings,
            np.array([public[n] for n in self.forcings]).reshape(
                len(self.forcings), len(t)
            ),
            np.diff(t).reshape(1, -1),
        )
        return self._flow(len(t)), args

    def observe_initial(self, row, parameters):
        """Evaluate the prescribed initial states and readouts without a time step."""
        readings, forcings = self._initial_args(row)
        states = self.initialize(parameters, readings)
        values = np.asarray(self.observe(states, parameters, forcings)).ravel()
        if not np.isfinite(states).all() or not np.isfinite(values).all():
            raise ArithmeticError("Nonfinite initial state or observed readout")
        return dict(zip(self.targets, values.tolist(), strict=True))

    def run(self, records, params):
        result = []
        for flow, args in records:
            y, x = (np.asarray(value) for value in flow(params, *args))
            if not (np.isfinite(x).all() and np.isfinite(y).all()) or any(
                np.max(np.abs(v)) > 1e18 for v in (x, y)
            ):
                raise ArithmeticError("Unresolved or nonfinite free rollout")
            result.append((y, x))
        return result


def _rows(raw, split, *, min_samples=2):
    rows = [CompleteTrajectory.model_validate(row) for row in raw]
    if not rows:
        raise ValueError(f"Empty {split} split")
    ids = [r.trajectory_id for r in rows]
    if len(set(ids)) != len(ids) or not all(i.startswith(split + "_") for i in ids):
        raise ValueError(f"Expected unique {split} trajectory IDs")
    for row in rows:
        if len(row.time) < min_samples or any(
            set(getattr(row, k)) != set(getattr(rows[0], k))
            for k in ("targets", "auxiliaries", "inputs", "covariates")
        ):
            raise ValueError(
                f"Invalid public trajectory layout for {row.trajectory_id}"
            )
        groups = [
            set(getattr(row, k))
            for k in ("targets", "auxiliaries", "inputs", "covariates")
        ]
        if sum(map(len, groups)) != len(set.union(*groups)):
            raise ValueError("Public trajectory roles overlap")
    return rows


def _targets(rows, names):
    return np.concatenate(
        [np.array([row.targets[n] for n in names]) for row in rows], axis=1
    )


def initial_target_events(
    graph: dict, formula: dict, rows: list[dict], targets: list[str]
) -> list[dict]:
    """Evaluate a fixed vector on observed algebraic targets at training time zero."""
    training = _rows(rows, "train", min_samples=1)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(
            graph["variable_types"].get(name) != "algebraic"
            or name not in training[0].targets
            for name in targets
        )
    ):
        raise ValueError("Expected distinct observed algebraic targets")
    if any(row.time[0] != 0 for row in training):
        raise ValueError("Training trajectory starts after or before time zero")
    model = _Rollout(graph, formula, targets, 1)
    parameters = [formula["parameters"][name] for name in model.parameters]
    events = []
    for row in training:
        values = model.observe_initial(row, parameters)
        events.append(
            {
                "event": row.trajectory_id,
                "time": [row.time[0]],
                "observed": {n: [row.targets[n][0]] for n in targets},
                "predicted": {n: [v] for n, v in values.items()},
            }
        )
    return events


def _residual(result, targets, scales):
    predictions = np.concatenate([y for y, _ in result], axis=1)
    return (predictions - targets) / scales[:, None]


def _metrics(result, targets, scales, names):
    errors = np.mean(_residual(result, targets, scales) ** 2, axis=1)
    return {
        "pooled_nmse": float(np.mean(errors)),
        "negative_samples": {
            n: sum(int(np.sum(y[k] < -1e-10)) for y, _ in result)
            for k, n in enumerate(names)
        },
        "per_target_nmse": dict(zip(names, errors.tolist(), strict=True)),
        "minimum_prediction": {
            n: min(float(y[k].min()) for y, _ in result) for k, n in enumerate(names)
        },
    }


def fit_formula(
    graph, formula, train, *, seed=0, max_evaluations=100, wall_seconds=60.0, starts=1
):
    """Fit only public training trajectories, charging every residual evaluation."""
    if (
        max_evaluations < starts
        or starts < 1
        or not np.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError(
            "Require a positive budget with at least one evaluation per start"
        )
    started = monotonic()
    rows = _rows(train, "train")
    names = sorted(rows[0].targets)
    model = _Rollout(graph, formula, names, 2)
    records = [model.record(row) for row in rows]
    observed = _targets(rows, names)
    # Rescale before computing variance so a finite large public measurement
    # cannot create an infinite denominator and spuriously perfect fit.
    magnitudes = np.maximum(np.max(abs(observed), axis=1), 1e-8)
    scales = np.maximum((observed / magnitudes[:, None]).std(axis=1) * magnitudes, 1e-8)
    rng = np.random.default_rng(seed)
    guess = np.array([formula["parameters"][n] for n in model.parameters], dtype=float)
    best, loss, calls = None, float("inf"), 0
    failed_rollouts = 0
    diagnostics = []
    deadline = started + wall_seconds
    for i in range(starts):
        stop = (i + 1) * max_evaluations // starts

        def residual(p, stop=stop):
            nonlocal best, loss, calls, failed_rollouts
            if monotonic() >= deadline or calls >= stop:
                raise TimeoutError("wall or evaluation budget reached")
            calls += 1
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    errors = _residual(model.run(records, p), observed, scales).ravel()
                    value = float(np.mean(errors**2))
            except ArithmeticError:
                failed_rollouts += 1
                return np.full(observed.size, 1.0e10)
            if value < loss:
                best, loss = p.copy(), value
            return errors

        try:
            if len(guess):
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    parameter_scales = np.maximum(abs(guess), 0.01)
                    initial = (
                        guess
                        if i == 0
                        else (
                            guess
                            + rng.normal(size=guess.size) * 0.25 * parameter_scales
                        )
                    )
                    fit = least_squares(residual, initial, x_scale=parameter_scales)
                reason = str(fit.message)
            else:
                residual(guess)
                reason = "fixed model evaluated"
        except TimeoutError as error:
            reason = str(error)
        except (ArithmeticError, np.linalg.LinAlgError) as error:
            reason = f"Optimizer numerical failure: {error}"
        diagnostics.append(reason)
        if monotonic() >= deadline:
            break
    base = {
        "graph_sha256": content_hash(graph),
        "formula_sha256": content_hash(formula),
        "evaluations": calls,
        "failed_rollouts": failed_rollouts,
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "casadi")},
        },
        "starts_attempted": len(diagnostics),
        "stop_reasons": diagnostics,
        "seconds": monotonic() - started,
        "scales": dict(zip(names, scales.tolist(), strict=True)),
        "budget": {
            "seed": seed,
            "max_evaluations": max_evaluations,
            "starts": starts,
            "wall_seconds": wall_seconds,
        },
        "substeps": 2,
    }
    if best is None:
        return base | {
            "status": "fit_failed",
            "reason": "No finite training rollout was obtained within this budget",
            "training_nmse": None,
            "parameters": None,
        }
    params = dict(zip(model.parameters, best.tolist(), strict=True))
    report = _check_grid(graph, formula, rows, params, names, scales)
    return base | {
        **report,
        "parameters": params,
        "effective_sources": _effective_sources(graph, formula, params),
        "seconds": monotonic() - started,
        "training_nmse": report["pooled_nmse"],
        "status": "fitted" if report["status"] == "scored" else report["status"],
    }


def _effective_sources(graph: dict, formula: dict, parameters: dict) -> dict:
    """Trace simplified per-equation dependencies at the frozen parameter vector."""
    import casadi as ca

    parser = RestrictedParser()
    active = {}
    for name, sources in graph["sources"].items():
        symbols = ca.SX.sym("source", len(sources))
        env = parameters | dict(zip(sources, ca.vertsplit(symbols), strict=True))
        value = _translate(
            ca, parser.parse(formula["equations"][name], location=name), env
        )
        remaining = {v.name() for v in ca.symvar(ca.simplify(ca.SX(value)))}
        active[name] = [
            s
            for s, v in zip(sources, ca.vertsplit(symbols), strict=True)
            if v.name() in remaining
        ]
    return active


def _check_grid(graph, formula, rows, parameters, names, scales, *, curves=False):
    """Verify predictions over the full development split on a finer time grid."""
    predictions = []
    for substeps in (2, 4):
        model = _Rollout(graph, formula, names, substeps)
        params = [parameters[n] for n in model.parameters]
        try:
            values = model.run([model.record(r) for r in rows], params)
        except ArithmeticError as error:
            return {
                "status": "rollout_failed",
                "pooled_nmse": None,
                "reason": str(error),
            }
        predictions.append(np.concatenate([y for y, _ in values], axis=1))
    difference = float(np.max(abs(predictions[0] - predictions[1]) / scales[:, None]))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            metrics = _metrics(values, _targets(rows, names), scales, names)
    except ArithmeticError as error:
        return {"status": "score_failed", "pooled_nmse": None, "reason": str(error)}
    if {"h_up", "h_down"} <= set(model.states) and all(
        {"area_up", "area_down"} <= r.covariates.keys()
        and {"inflow_up", "inflow_down"} <= r.inputs.keys()
        for r in rows
    ):
        # With nonnegative external discharge, total storage cannot gain more
        # water than the two measured inflows supply. This test uses predicted
        # hidden depth, never a hidden trajectory from the benchmark.
        excess = []
        for (_, x), row in zip(values, rows, strict=True):
            storage = sum(
                row.covariates["area_" + n] * x[model.states.index("h_" + n)]
                for n in ("up", "down")
            )
            inflow = np.asarray(row.inputs["inflow_up"]) + row.inputs["inflow_down"]
            excess.extend(
                np.diff(storage) - np.diff(row.time) * (inflow[1:] + inflow[:-1]) / 2
            )
        metrics["maximum_storage_gain_above_inflow_m3"] = max(0.0, float(max(excess)))
    plot_data = {}
    if curves:
        plot_data["curves"] = [
            {
                "trajectory_id": r.trajectory_id,
                "time": list(r.time),
                "observed": {n: list(r.targets[n]) for n in names},
                "predicted": {n: y[i].tolist() for i, n in enumerate(names)},
            }
            for r, (y, _) in zip(rows, values, strict=True)
        ]
    if difference > 0.01:
        return {
            "status": "resolution_unverified",
            "pooled_nmse": None,
            "maximum_scaled_grid_difference": difference,
            "diagnostic_fine_grid": metrics,
            **plot_data,
        }
    return {
        "status": "scored",
        "maximum_scaled_grid_difference": difference,
        **metrics,
        **plot_data,
    }


def score_formula(graph, formula, fit_result, validation):
    """Replay frozen parameters on validation with a finer grid check."""
    if (fit_result["graph_sha256"], fit_result["formula_sha256"]) != (
        content_hash(graph),
        content_hash(formula),
    ):
        raise ValueError("Fit identity differs from graph or formula")
    if fit_result["status"] != "fitted":
        return {"status": "fit_unavailable", "pooled_nmse": None}
    rows = _rows(validation, "val")
    names = sorted(rows[0].targets)
    if set(names) != set(fit_result["scales"]):
        raise ValueError("Validation target channels differ from training")
    scales = np.array([fit_result["scales"][n] for n in names])
    return _check_grid(
        graph, formula, rows, fit_result["parameters"], names, scales, curves=True
    )
