"""Fit the total-glucose balance with the archived fading action and meal filter."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from autoformalism.research.trajectory_function_fitting import validate_formula
from scripts import t2_initial_memory_fit as plasma

FIXED_SOURCE = ("meal_balance_offset", "meal_balance_egp")
STARTS = ("short", "long", "warm")


def seed_model(rows: list[dict], fold: dict, start: str, warm_model=None) -> dict:
    """Transplant the fold's disposal and insulin laws into a total-mass balance."""
    if start not in STARTS or (start == "warm") != (warm_model is not None):
        raise ValueError("Pass a matching fading model only for the warm start")
    model = plasma.seed_model(
        rows, fold, "fading", "long" if start == "warm" else start
    )
    graph, f = model["graph"], model["formula"]
    if warm_model is not None:
        same = (
            "graph",
            "memory_arm",
            "fit_training_ids",
            "scalar_model",
            "knots",
            "centers",
            "scales",
        )
        if (
            any(model[k] != warm_model.get(k) for k in same)
            or warm_model["formula"]["initializers"] != f["initializers"]
            or warm_model["formula"]["equations"] != f["equations"]
            or set(warm_model["formula"]["parameters"]) != set(f["parameters"])
        ):
            raise ValueError("Warm model differs from this fold's fading graph")
        shared = warm_model["formula"]["parameters"]
        locked = set(f["parameters"]) - {*plasma.ACTIVE, "memory_rate"}
        if any(shared[k] != f["parameters"][k] for k in locked):
            raise ValueError("Frozen initializer, gauge or insulin fit differs")
        f["parameters"].update(shared)

    for k in FIXED_SOURCE:
        del f["parameters"][k]
    graph["variable_types"].update(Gp="algebraic", S="differential")
    del graph["initializers"]["Gp"]
    graph["initializers"]["S"] = {
        "mode": "map",
        "public_initials": ["Gp", "Gt"],
        "state_initials": [],
    }
    f["initializers"]["S"] = "Gp+Gt"
    graph["sources"].update(Gp=["S", "Gt"], S=["EGP", "E", "U", "D_meal"])
    f["equations"].update(Gp="S-Gt", S="EGP-E-U+meal_balance_gain*D_meal")
    return model | {"formula": validate_formula(graph, f), "balance_arm": "total"}


class _TotalTrajectories:
    """Propagate total mass S with the original insulin solution and filter."""

    def __init__(self, model: dict, rows: list[dict]):
        base = plasma._Trajectories(model, rows)
        self.observed = base.observed
        self.required = base.required
        self.high, self.low = base.high, base.low
        self.t, self.dt, self.scales = base.t, base.dt, base.scales
        self.active = tuple(name for name in base.active if name not in FIXED_SOURCE)
        self.tissue = 100 + 100 * base.gt
        base.g0 = base.g0 + self.tissue[:, 0]
        base.egp = base.egp - np.array(
            [np.interp(base.t, r["time"], r["auxiliaries"]["E"]) for r in rows]
        )
        # Substitute Gp=S-Gt into a*(Gp-Gp_mean)+b*(Gt-100)/100.
        # This gives a*(S-(Gp_mean+100))+(b-100*a)*(Gt-100)/100.
        base.gbar += 100
        self.base = base

    def predict(self, v):
        p = dict(zip(self.active, v, strict=True))
        p.update(meal_balance_offset=0, meal_balance_egp=1)
        p["disp_tissue"] -= 100 * p["afcoef12"] / self.scales[0]
        total, disposal = self.base.predict([p[k] for k in self.base.active])
        return total - self.tissue[:, ::2], disposal

    def residual(self, glucose, disposal):
        return self.base.residual(glucose, disposal)


def fit(rows: list[dict], fold: dict, start: str, *, budget=200, warm_model=None):
    """Select a feasible fit within the original SLSQP evaluation budget."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("Expected a positive evaluation budget")
    m = seed_model(rows, fold, start, warm_model)
    data = _TotalTrajectories(m, rows)
    names = data.active
    p = m["formula"]["parameters"]
    p0 = np.array([p[k] for k in names])
    step = plasma.SUBSTEPS
    min_rate = 1 / (100 * data.t[-1])
    lo_map = {
        "memory_rate": min_rate,
        "afcoef10": plasma.GAIN_FLOOR,
        "meal_balance_gain": plasma.GAIN_FLOOR,
        "meal_balance_tau": data.dt,
    }
    lo = np.array([lo_map.get(k, -np.inf) for k in names])
    hi = np.array(
        [1 / (step * data.dt) if k == "memory_rate" else np.inf for k in names]
    )
    if np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("Initial vector violates the archived fit bounds")
    gg0, uu0 = plasma._trajectory_jacobian(data, p0)
    rms = np.sqrt(
        (
            np.mean((gg0[:, ::step] / data.scales[0]) ** 2, axis=(0, 1))
            + np.mean((uu0[:, ::step] / data.scales[2]) ** 2, axis=(0, 1))
        )
        / 3
    )
    scale = np.minimum(1 / np.maximum(rms, 1e-10), 100 * np.maximum(abs(p0), 1))
    zero = tuple(np.zeros_like(p0))
    events = np.arange(len(rows))
    best, minimum = None, np.inf
    calls, invalid, derivatives = 0, 0, 1

    def vector(key):
        return np.clip(p0 + scale * np.array(key), lo, hi)

    @lru_cache(maxsize=2)
    def evaluate(key):
        nonlocal calls, invalid, best, minimum
        if calls >= budget:
            raise StopIteration("Forward evaluation budget reached")
        calls += 1
        v = vector(key)
        try:
            g, u = data.predict(v)
        except ArithmeticError:
            invalid += 1
            return 1e28, np.full(2 * len(rows), -1e14), None, None
        ig, iu = np.argmin(g, axis=1), np.argmin(u, axis=1)
        domain = np.r_[
            g[events, ig] / data.scales[0],
            (u[events, iu] - 1 - plasma.DISPOSAL_BUFFER) / data.scales[2],
        ]
        order = u[data.high, 0] - u[data.low, 0] - data.required
        e = data.residual(g, u)
        loss = np.mean(e**2).item()
        if min(domain) >= -1e-9 and np.all(order >= -1e-9) and loss < minimum:
            best, minimum = v.copy(), loss
        return loss, domain, e, (ig, iu)

    @lru_cache(maxsize=2)
    def jacobian(key):
        nonlocal derivatives
        _, _, errors, where = evaluate(key)
        if errors is None:
            return np.zeros(len(names)), np.zeros((2 * len(rows), len(names)))
        if key == zero:
            gg, uu = gg0, uu0
        else:
            derivatives += 1
            gg, uu = plasma._trajectory_jacobian(data, vector(key))
        dg, du = gg[:, ::step] / data.scales[0], uu[:, ::step] / data.scales[2]
        grad = (2 * scale / errors.size) * (
            np.einsum("ij,ijk->k", errors[:, 0], dg)
            + np.einsum("ij,ijk->k", errors[:, 2], du)
        )
        ig, iu = where
        bounds_jac = (
            np.vstack(
                (gg[events, ig] / data.scales[0], uu[events, iu] / data.scales[2])
            )
            * scale
        )
        return grad, bounds_jac

    source, _, err0, _ = evaluate(zero)
    if err0 is None:
        raise ArithmeticError("Initial total-mass trajectory is unresolved")
    u0 = data.observed[:, 2, 0] + data.scales[2] * err0[:, 2, 0]
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: evaluate(tuple(x))[1],
            "jac": lambda x: jacobian(tuple(x))[1],
        }
    ]
    if len(data.high):
        diff = (uu0[data.high, 0] - uu0[data.low, 0]) * scale
        threshold = data.required - (u0[data.high] - u0[data.low])
        unique = np.unique(np.column_stack((diff, threshold)), axis=0)
        constraints.append(LinearConstraint(unique[:, :-1], unique[:, -1], np.inf))
    success, nit, endpoint = False, None, None
    try:
        opt = minimize(
            lambda x: evaluate(tuple(x))[0],
            np.array(zero),
            jac=lambda x: jacobian(tuple(x))[0],
            bounds=Bounds((lo - p0) / scale, (hi - p0) / scale),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": budget, "ftol": 1e-11},
        )
        success, msg, nit = bool(opt.success), str(opt.message), int(opt.nit)
        endpoint = vector(opt.x)
    except StopIteration as exc:
        msg = str(exc)
    if best is not None:
        p.update(dict(zip(names, best.tolist(), strict=True)))
    return {
        "model": m if best is not None else None,
        "source_nmse": source,
        "training_nmse": minimum if best is not None else None,
        "solver_success": success,
        "termination": msg,
        "solver_iterations": nit,
        "forward_evaluations": calls,
        "jacobian_evaluations": derivatives,
        "invalid_trials": invalid,
        "budget": budget,
        "retained_solver_endpoint": endpoint is not None
        and np.array_equal(endpoint, best),
        "free_parameter_names": list(names),
        "start_values": dict(zip(names, p0.tolist(), strict=True)),
        "decay_rate_bounds": [min_rate, 1 / (step * data.dt)],
        "minimum_meal_tau": data.dt,
        "ordering_pairs": len(data.required),
    }
