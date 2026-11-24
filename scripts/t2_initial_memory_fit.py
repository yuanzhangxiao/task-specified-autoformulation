"""Fold-local constrained fits for persistent or fading initial glucose effects."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.signal import lfilter

from autoformalism.research.trajectory_function_fitting import validate_formula
from scripts import compare_positive_insulin_rates as production
from scripts import t2_initial_memory_folds as folds
from scripts.fit_t2_disposal_constrained import DISPOSAL_BUFFER, GAIN_FLOOR
from scripts.fit_t2_initial_disposal_order import order_pairs

ACTIVE = (
    "afcoef9",
    "afcoef10",
    "afcoef11",
    "afcoef12",
    "afcoef16",
    "init_tissue_gain",
    "disp_tissue",
    "meal_balance_offset",
    "meal_balance_egp",
    "meal_balance_gain",
    "meal_balance_tau",
)
TARGETS = ("Gp", "I", "U")
SUBSTEPS = 4


def _check_fold(rows, fold):
    ids = sorted(r["trajectory_id"] for r in rows)
    expected = {f"train_{i:03}" for i in range(16)}
    if (
        len(ids) != 15
        or len(set(ids)) != 15
        or expected - set(ids) not in ({"train_011"}, {"train_012"})
        or sorted(fold["fit_ids"]) != ids
    ):
        raise ValueError("Expected the same 15 events used for the fold insulin fit")
    if any(r["time"][0] != 0 for r in rows):
        raise ValueError("Expected time-zero event initials")
    if any(np.any(np.asarray(r["auxiliaries"]["Uii"]) != 1) for r in rows):
        raise ValueError("Constant-Uii reduction does not apply to this fold")
    return ids


def seed_model(rows, fold, arm, start, persistent_model=None):
    """Build the declared graph and fresh or fold-local nested coordinates."""
    ids = _check_fold(rows, fold)
    if arm not in {"persistent", "fading"} or start not in {"short", "long", "nested"}:
        raise ValueError("Unknown tissue-memory arm or start")
    if (start == "nested") != (persistent_model is not None):
        raise ValueError("The nested start requires its fold's fitted persistent model")
    centers, scales = fold["centers"], fold["scales"]
    insulin, insulin_parameters = folds.insulin_formula(fold)
    ins = fold["insulin"]
    ref = centers["Gt_ref"]
    t_end = max(r["time"][-1] for r in rows)
    gap = (np.array([r["auxiliaries"]["Gt"][0] for r in rows]) - ref) / 100
    u0 = np.array([r["targets"]["U"][0] for r in rows])
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(len(gap)), gap]), u0, rcond=None
    )[0]
    egp = np.mean([r["auxiliaries"]["EGP"][0] for r in rows])
    duration = t_end * (0.1 if start == "short" else 0.5)
    p = {
        **insulin_parameters,
        "afcoef9": 0.0,
        "afcoef10": 1e-5,
        "afcoef11": float(intercept),
        "afcoef12": 0.0,
        "afcoef13": 1.0,
        "afcoef14": 0.0,
        "afcoef15": 0.0,
        "afcoef16": 0.0,
        "init_tissue_gain": float(slope),
        "disp_tissue": 0.0,
        "meal_balance_offset": float(intercept / 2),
        "meal_balance_egp": float(intercept / (2 * egp)),
        "meal_balance_tau": duration,
        "meal_balance_gain": 1.0 if start == "short" else 2.0,
    }
    if persistent_model is not None:
        if (
            persistent_model["fit_training_ids"] != ids
            or persistent_model["memory_arm"] != "persistent"
            or persistent_model["scalar_model"] != ins["scalar_model"]
            or persistent_model["knots"] != ins["knots"]
            or persistent_model["scales"] != scales
            or persistent_model["centers"] != centers
        ):
            raise ValueError(
                "Persistent continuation belongs to a different fold or model"
            )
        p.update(persistent_model["formula"]["parameters"])
    types = dict.fromkeys(("D_meal", "Gp", "I", "I_act"), "differential")
    types["U"] = "algebraic"
    sources = {
        "D_meal": ["D_meal", "meal_rate_g_per_min"],
        "Gp": ["D_meal", "EGP", "U"],
        "I": ins["sources"],
        "I_act": ["I"],
        "U": ["Gp", "Gt", "I_act", "Uii"],
    }
    rules = {
        "D_meal": {"mode": "known"},
        "Gp": {"mode": "observed_initial", "channel": "Gp"},
        "I": {"mode": "observed_initial", "channel": "I"},
        "I_act": {"mode": "map", "public_initials": ["Gt"], "state_initials": ["I"]},
    }
    offset = f"init_tissue_gain*((Gt-({ref!r}))/100)"
    initials = {"D_meal": "0", "I_act": f"afcoef15+afcoef16*state0_I+{offset}"}
    equations = {
        "D_meal": "(meal_rate_g_per_min-D_meal)/meal_balance_tau",
        "Gp": "meal_balance_offset+meal_balance_egp*EGP-U+meal_balance_gain*D_meal",
        "I": insulin,
        "I_act": f"afcoef9+afcoef10*((I-({centers['I']!r}))/{scales['I']!r})",
        "U": f"afcoef11+afcoef12*((Gp-({centers['Gp']!r}))/{scales['Gp']!r})"
        "+afcoef13*I_act+afcoef14*(Uii-1)+disp_tissue*((Gt-100)/100)",
    }
    if "graph" in fold:
        expected = fold["graph"]
        base_sources = expected["sources"] | {"I": ins["sources"]}
        if (
            expected["variable_types"] != types
            or expected["initializers"] != rules
            or base_sources != sources
        ):
            raise ValueError(
                "This fitter requires the authenticated delayed-disposal graph"
            )
    if arm == "fading":
        p["memory_rate"] = 1 / (100 * t_end) if start == "nested" else 1 / duration
        types["M_init"] = "differential"
        sources["M_init"] = ["M_init"]
        sources["I_act"] = ["I", "M_init"]
        rules["M_init"] = {
            "mode": "map",
            "public_initials": ["Gt"],
            "state_initials": [],
        }
        initials["M_init"] = offset
        equations["M_init"] = "-memory_rate*M_init"
        equations["I_act"] += "-memory_rate*M_init"
    graph = {"variable_types": types, "sources": sources, "initializers": rules}
    formula = validate_formula(
        graph, {"equations": equations, "initializers": initials, "parameters": p}
    )
    return {
        "graph": graph,
        "formula": formula,
        "knots": ins["knots"],
        "scalar_model": ins["scalar_model"],
        "centers": centers,
        "scales": scales,
        "memory_arm": arm,
        "fit_training_ids": ids,
    }


def _scalar_insulin(row, knots, scalar, grid):
    """Integrate insulin and its time integral exactly segment-by-segment."""
    t, base, hats, i0 = production.grid(row, knots)
    initial_production = hats @ production.checked_weights(knots, scalar)
    tt = np.union1d(base, grid)
    prod = np.interp(tt, base, initial_production)
    tau = scalar["tau"]
    dt = np.diff(tt)
    z = dt / tau
    mass = -np.expm1(-z) * tau
    right = dt * np.where(
        z < 0.0001, 0.5 - z / 6 + z * z / 24, (1 + np.expm1(-z) / z) / z
    )
    increments = (mass - right) * prod[:-1] + right * prod[1:]
    insulin = np.empty_like(tt)
    insulin[0] = i0
    for j, (a, b) in enumerate(zip(np.exp(-z), increments, strict=True)):
        insulin[j + 1] = a * insulin[j] + b
    integral = tau * (cumulative_trapezoid(prod, tt, initial=0) - (insulin - i0))
    idx = np.searchsorted(tt, grid)
    if t[0] != grid[0] or t[-1] != grid[-1]:
        raise ValueError("Insulin and fitting grids differ in time coverage")
    return insulin[idx], integral[idx]


def _linear_filter(decay, increments, initial):
    """Solve x[j+1] = decay*x[j] + increments[j] for all events."""
    first = np.asarray(initial)[:, None]
    states, _ = lfilter([1.0], [1.0, -decay], increments, axis=1, zi=decay * first)
    return np.concatenate((first, states), axis=1)


class _Trajectories:
    """Evaluate the triangular graph on a fine grid using scalar linear propagation."""

    def __init__(self, model, rows):
        t = np.asarray(rows[0]["time"], dtype=float)
        if any(not np.array_equal(r["time"], t) for r in rows) or not np.allclose(
            np.diff(t), t[1] - t[0], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                "This specialization requires a common regular sample grid"
            )
        self.dt = (t[1] - t[0]) / SUBSTEPS
        self.t = np.linspace(t[0], t[-1], (len(t) - 1) * 2 * SUBSTEPS + 1)

        def history(group, key):
            return np.array([np.interp(self.t, t, r[group][key]) for r in rows])

        self.gas = history("inputs", "meal_rate_g_per_min")
        self.egp = history("auxiliaries", "EGP")
        gt = history("auxiliaries", "Gt")
        self.gt = (gt - 100) / 100
        self.initial_tissue = (gt[:, :1] - model["centers"]["Gt_ref"]) / 100
        ins, integral = (
            np.stack(a)
            for a in zip(
                *[
                    _scalar_insulin(r, model["knots"], model["scalar_model"], self.t)
                    for r in rows
                ],
                strict=True,
            )
        )
        self.i0 = ins[:, :1]
        self.signal = (integral - model["centers"]["I"] * self.t) / model["scales"]["I"]
        self.g0 = np.array([r["targets"]["Gp"][0] for r in rows])
        self.gbar = model["centers"]["Gp"]
        self.scales = np.array([model["scales"][k] for k in TARGETS])
        self.observed = np.array([[r["targets"][k] for k in TARGETS] for r in rows])
        self.ierr = (ins[:, :: 2 * SUBSTEPS] - self.observed[:, 1]) / self.scales[1]
        self.active = ACTIVE + (
            ("memory_rate",) if model["memory_arm"] == "fading" else ()
        )
        self.obs_u0 = self.observed[:, 2, 0]
        self.high, self.low = order_pairs(self.obs_u0)
        self.required = 0.5 * (self.obs_u0[self.high] - self.obs_u0[self.low])
        # Integrate the quadratic interpolant of the glucose forcing against its
        # scalar exponential kernel by fixed Gauss-Legendre weights.
        s, w = np.polynomial.legendre.leggauss(4)
        self.gauss = (1 + s) / 2
        s = self.gauss
        self.weights = (w / 2)[:, None] * np.array(
            [2 * (s - 0.5) * (s - 1), 4 * s * (1 - s), 2 * s * (s - 0.5)]
        ).T

    def predict(self, v):
        p = dict(zip(self.active, v, strict=True))
        z = self.dt / (2 * p["meal_balance_tau"])
        total = -np.expm1(-z)
        right = z * (0.5 - z / 6 + z * z / 24) if abs(z) < 1e-4 else 1 - total / z
        release = _linear_filter(
            np.exp(-z),
            (total - right) * self.gas[:, :-1] + right * self.gas[:, 1:],
            np.zeros(len(self.gas)),
        )
        k = p.get("memory_rate", 0)
        action = (
            p["afcoef16"] * self.i0
            + p["afcoef9"] * self.t
            + p["afcoef10"] * self.signal
            + p["init_tissue_gain"] * self.initial_tissue * np.exp(-k * self.t)
        )
        g_rate = p["afcoef12"] / self.scales[0]
        other_u = (
            p["afcoef11"] - g_rate * self.gbar + action + p["disp_tissue"] * self.gt
        )
        forcing = (
            p["meal_balance_offset"]
            + p["meal_balance_egp"] * self.egp
            + p["meal_balance_gain"] * release
            - other_u
        )
        if -np.real(g_rate) * self.t[-1] > 600:
            raise ArithmeticError("Unstable glucose trial")
        weights = self.dt * np.exp(-g_rate * self.dt * (1 - self.gauss)) @ self.weights
        increments = (
            weights[0] * forcing[:, :-1:2]
            + weights[1] * forcing[:, 1::2]
            + weights[2] * forcing[:, 2::2]
        )
        glucose = _linear_filter(np.exp(-g_rate * self.dt), increments, self.g0)
        disposal = other_u[:, ::2] + g_rate * glucose
        if (
            not np.all(np.isfinite(glucose))
            or not np.all(np.isfinite(disposal))
            or np.max(np.abs(glucose)) > 1e16
            or np.max(np.abs(disposal)) > 1e16
        ):
            raise ArithmeticError("Unresolved glucose/disposal trial")
        return glucose, disposal

    def residual(self, glucose, disposal):
        return np.stack(
            [
                (glucose[:, ::SUBSTEPS] - self.observed[:, 0]) / self.scales[0],
                self.ierr,
                (disposal[:, ::SUBSTEPS] - self.observed[:, 2]) / self.scales[2],
            ],
            axis=1,
        )


def fit(rows, fold, arm, start, *, budget=200, persistent_model=None):
    """Keep the lowest feasible pooled training loss within a fixed SLSQP budget."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("Expected positive evaluation budget")
    model = seed_model(rows, fold, arm, start, persistent_model)
    data = _Trajectories(model, rows)
    p = model["formula"]["parameters"]
    names = data.active
    p0 = np.array([p[n] for n in names])
    initial = data.predict(p0)
    source = float(np.mean(data.residual(*initial) ** 2))
    gap = data.required
    min_rate = 1 / (100 * data.t[-1])
    lower = {
        "afcoef10": GAIN_FLOOR,
        "meal_balance_egp": GAIN_FLOOR,
        "meal_balance_gain": GAIN_FLOOR,
        "meal_balance_tau": data.dt,
        "memory_rate": min_rate,
    }
    upper = {"memory_rate": 1 / (SUBSTEPS * data.dt)}
    lo = np.array([lower.get(n, -np.inf) for n in names])
    hi = np.array([upper.get(n, np.inf) for n in names])
    base_jac = _trajectory_jacobian(data, p0)
    g_j, u_j = base_jac
    rms = np.sqrt(
        (
            np.mean((g_j[:, ::SUBSTEPS] / data.scales[0]) ** 2, axis=(0, 1))
            + np.mean((u_j[:, ::SUBSTEPS] / data.scales[2]) ** 2, axis=(0, 1))
        )
        / 3
    )
    scale = np.minimum(1 / np.maximum(rms, 1e-10), 100 * np.maximum(abs(p0), 1))
    calls, jac_calls, invalid = 0, 1, 0
    best, best_score = None, np.inf
    order0 = initial[1][data.high, 0] - initial[1][data.low, 0] - gap
    slopes = (u_j[data.high, 0] - u_j[data.low, 0]) * scale
    unique = np.unique(np.column_stack((slopes, -order0)), axis=0)
    events = np.arange(len(rows))

    @lru_cache(maxsize=2)
    def evaluate(key):
        nonlocal calls, invalid, best, best_score
        if calls >= budget:
            raise StopIteration("Forward evaluation budget reached")
        calls += 1
        v = np.clip(p0 + scale * np.array(key), lo, hi)
        try:
            g, u = data.predict(v)
        except ArithmeticError:
            invalid += 1
            return 1e28, None, np.full(2 * len(rows), -1e14), None
        e = data.residual(g, u)
        score = float(np.mean(e**2))
        ig, iu = g.argmin(axis=1), u.argmin(axis=1)
        margins = np.concatenate(
            (
                g[events, ig] / data.scales[0],
                (u[events, iu] - 1 - DISPOSAL_BUFFER) / data.scales[2],
            )
        )
        order_margin = u[data.high, 0] - u[data.low, 0] - gap
        if (
            min(margins) >= -1e-9
            and np.all(order_margin >= -1e-9)
            and score < best_score
        ):
            best, best_score = v.copy(), score
        return score, e, margins, (ig, iu)

    @lru_cache(maxsize=2)
    def derivatives(key):
        nonlocal jac_calls
        _, resid, _, at = evaluate(key)
        if resid is None:
            return np.zeros(len(names)), np.zeros((2 * len(rows), len(names)))
        if any(key):
            jac_calls += 1
            gg, uu = _trajectory_jacobian(
                data, np.clip(p0 + scale * np.array(key), lo, hi)
            )
        else:
            gg, uu = base_jac
        dg = gg[:, ::SUBSTEPS, :] / data.scales[0]
        du = uu[:, ::SUBSTEPS, :] / data.scales[2]
        grad = (
            (2 / resid.size)
            * (
                np.einsum("ij,ijk->k", resid[:, 0], dg)
                + np.einsum("ij,ijk->k", resid[:, 2], du)
            )
            * scale
        )
        ig, iu = at
        constraint_jac = (
            np.concatenate(
                (gg[events, ig] / data.scales[0], uu[events, iu] / data.scales[2])
            )
            * scale
        )
        return grad, constraint_jac

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: evaluate(tuple(x))[2],
            "jac": lambda x: derivatives(tuple(x))[1],
        }
    ]
    if len(unique):
        constraints.append(LinearConstraint(unique[:, :-1], unique[:, -1], np.inf))
    zero = np.zeros(len(names))
    evaluate(tuple(zero))
    success, msg, nit, endpoint = False, None, None, None
    try:
        opt = minimize(
            lambda x: evaluate(tuple(x))[0],
            zero,
            jac=lambda x: derivatives(tuple(x))[0],
            bounds=Bounds((lo - p0) / scale, (hi - p0) / scale),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": budget, "ftol": 1e-11},
        )
        success, msg, nit = bool(opt.success), str(opt.message), int(opt.nit)
        endpoint = np.clip(p0 + scale * opt.x, lo, hi)
    except StopIteration as e:
        msg = str(e)
    if best is None:
        return {
            "model": None,
            "training_nmse": None,
            "source_nmse": source,
            "solver_success": False,
            "termination": "No feasible rollout: " + str(msg),
        }
    p.update(dict(zip(names, best.tolist(), strict=True)))
    return {
        "model": model,
        "source_nmse": source,
        "training_nmse": best_score,
        "solver_success": success,
        "termination": msg,
        "solver_iterations": nit,
        "forward_evaluations": calls,
        "jacobian_evaluations": jac_calls,
        "invalid_trials": invalid,
        "budget": budget,
        "retained_solver_endpoint": bool(
            endpoint is not None and np.array_equal(best, endpoint)
        ),
        "decay_rate_bounds": [min_rate, float(upper["memory_rate"])]
        if arm == "fading"
        else None,
        "minimum_meal_tau": data.dt,
        "ordering_pairs": len(gap),
        "free_parameter_names": list(names),
        "start_values": dict(zip(names, p0.tolist(), strict=True)),
    }


def _trajectory_jacobian(data, values):
    """Complex-step sensitivities of analytical filters and their smooth gains."""
    gs, us = [], []
    for i in range(len(values)):
        step = values.astype(complex)
        step[i] += 1e-24j
        g, u = data.predict(step)
        gs.append(g.imag / 1e-24)
        us.append(u.imag / 1e-24)
    return np.stack(gs, axis=-1), np.stack(us, axis=-1)
