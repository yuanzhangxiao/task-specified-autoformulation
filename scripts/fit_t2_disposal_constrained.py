"""Train-only fixed-insulin constrained fits on the two slot-015 disposal graphs."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.staged_topology import content_hash
from scripts.fit_positive_insulin_handoff import optimizer_start
from scripts.positive_insulin_refit_kernel import FitProblem

# The scale/offset reduction below applies to these complete graph and initializer
# identities. A change in any equation requires another reduction check.
GRAPHS = {
    "45a7bdaf04750ce5b35eb4cb2bdfa0bff66616e35914473c1641c419b6bdbf86",
    "62bcc1666fcd77c0d587e9e4defde0f65b1e7367b3deb8f21ae6dea40b840521",
}
DISPOSAL_BUFFER = 1e-4
GAIN_FLOOR = 1e-10
DOMAIN_TOL = 1e-9


def _midpoints(row: dict) -> dict:
    """Interpolate public signals for constraints, retaining observed knots at 2i."""
    t = np.asarray(row["time"])
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Increasing observation times required")
    grid = np.empty(2 * len(t) - 1)
    grid[::2], grid[1::2] = t, (t[1:] + t[:-1]) / 2
    return (
        row
        | {"time": grid.tolist()}
        | {
            key: {k: np.interp(grid, t, a).tolist() for k, a in row[key].items()}
            for key in ("targets", "inputs", "auxiliaries")
        }
    )


class _Training:
    def __init__(self, model: dict, rows: list[dict]):
        f = model["formula"]
        if (
            content_hash([model["graph"], f["equations"], f["initializers"]])
            not in GRAPHS
        ):
            raise ValueError("Unqualified graph for the slot-015 action reduction")
        expanded = [_midpoints(r) for r in rows]
        self.engine = FitProblem(model, expanded, substeps=2)
        self.names = self.engine.names
        self.reference = self.engine._vector(f["parameters"])
        self.scales = self.engine.scales
        if self.engine.targets != ("Gp", "I", "U"):
            raise ValueError("Unexpected target order")
        self.observed = np.concatenate(
            [
                np.column_stack([r["targets"][k] for k in self.engine.targets])
                for r in expanded
            ]
        )
        self.uii = np.concatenate([r["auxiliaries"]["Uii"] for r in expanded])
        if min(self.uii) < 0 or not np.isfinite(self.uii).all():
            raise ValueError("Expected nonnegative supplied independent disposal")
        lengths = np.array([len(r["time"]) for r in expanded])
        self.edges = np.cumsum([0, *lengths])
        self.samples = np.concatenate(
            [
                np.arange(i, j, 2)
                for i, j in zip(self.edges[:-1], self.edges[1:], strict=True)
            ]
        )
        # The affine latent redefinition A* = s A + b preserves every forecast.
        # Hold a positive output gain and the initializer intercept at source values.
        fixed = {"afcoef13", "afcoef15"} | self.engine.insulin_parameters
        if np.all(self.uii == 1):
            fixed.add("afcoef14")  # Its only regressor Uii - 1 vanishes.
        self.active = [i for i, n in enumerate(self.names) if n not in fixed]
        self.lo = np.array(
            [
                2 * SAFE_DIVISION_EPSILON
                if n == "meal_balance_tau"
                else GAIN_FLOOR
                if n in {"afcoef10", "meal_balance_egp", "meal_balance_gain"}
                else -np.inf
                for n in self.names
            ]
        )
        if (self.reference < self.lo).any() or f["parameters"]["afcoef13"] <= 0:
            raise ValueError("Source violates fixed-insulin/domain coefficient bounds")

    def margins(self, error: np.ndarray):
        predictions = self.observed + error * self.scales
        # Nonnegative I follows from the fixed source law and observed initial.
        # Nonnegative U follows from nonnegative Uii and the dependent margin.
        quantities = np.column_stack(
            [
                predictions[:, 0],
                predictions[:, 2] - self.uii - DISPOSAL_BUFFER,
            ]
        )
        ix = np.vstack(
            [
                a + np.argmin(quantities[a:b], axis=0)
                for a, b in zip(self.edges[:-1], self.edges[1:], strict=True)
            ]
        )
        raw = quantities[ix, np.arange(2)]
        constraint = (raw / self.scales[[0, 2]]).ravel()
        return constraint, ix


class _Limit(Exception):
    """The common rollout budget has been spent."""


def fit(
    model: dict, training_rows: list[dict], start: str, *, budget: int = 80
) -> dict:
    """Minimize pooled training NMSE subject to all event-domain inequalities."""
    if start not in {"warm", "displaced"} or budget < 1:
        raise ValueError("Expected named start and positive rollout budget")
    data = _Training(model, training_rows)
    ix = data.active
    base = data.reference
    jac0 = data.engine.jacobian(base).reshape(-1, 3, len(base))
    rms = np.sqrt(np.mean(jac0[data.samples][:, :, ix] ** 2, axis=(0, 1)))
    scale = np.minimum(1 / np.maximum(rms, 1e-10), 100 * np.maximum(abs(base[ix]), 1))
    displaced = optimizer_start(
        model, model["formula"]["parameters"], "fixed_insulin", start
    )
    origin = np.array([displaced[data.names[i]] for i in ix])
    initial = (origin - base[ix]) / scale
    lower = (data.lo[ix] - base[ix]) / scale
    last_x, last_r, last_c, last_rows, last_j = [None] * 5
    calls, derivs, invalid = 0, 1, 0
    best, best_score, best_margins = None, np.inf, None

    def expand(x):
        p = base.copy()
        p[ix] = base[ix] + scale * x
        # Rounding at an active lower bound must preserve a strict positive gain.
        return np.maximum(p, data.lo)

    def evaluate(x):
        nonlocal last_x, last_r, last_c, last_rows, last_j
        nonlocal best, best_score, best_margins, calls, invalid
        if last_x is not None and np.array_equal(x, last_x):
            return
        if calls >= budget:
            raise _Limit
        calls += 1
        last_x = x.copy()
        last_j = None
        p = expand(x)
        try:
            err = data.engine.residual(p).reshape(-1, 3)
            last_c, last_rows = data.margins(err)
            last_r = err[data.samples]
        except ArithmeticError:
            invalid += 1
            last_r = np.full((len(data.samples), 3), 1e15)
            last_rows = np.zeros((len(training_rows), 2), dtype=int)
            last_c = np.full(len(training_rows) * 2, -1e15)
            return
        value = float(np.mean(last_r**2))
        if min(last_c) >= -DOMAIN_TOL and value < best_score:
            best, best_score, best_margins = p, value, last_c.copy()

    def derivatives(x):
        nonlocal last_j, derivs
        evaluate(x)
        if last_j is None:
            if np.all(x == 0):
                jac = jac0
            else:
                jac = data.engine.jacobian(expand(x)).reshape(-1, 3, len(base))
                derivs += 1
            last_j = jac[:, :, ix] * scale
        return last_j

    def objective(x):
        evaluate(x)
        return float(np.mean(last_r**2))

    def gradient(x):
        j = derivatives(x)[data.samples]
        return 2 * np.einsum("ti,tij->j", last_r, j) / last_r.size

    def constraints(x):
        evaluate(x)
        return last_c

    def constraint_jac(x):
        j = derivatives(x)
        return j[last_rows, [0, 2], :].reshape(-1, len(ix))

    zero = np.zeros(len(ix))
    source_score = objective(zero)
    if best is None:
        raise ValueError("Source fails the optimization grid interior margin")
    fraction = 1.0 if start == "displaced" else 0.0
    report = {"solver_success": False, "termination": "evaluation budget reached"}
    try:
        # The same deterministic feasibility restoration is offered to both arms.
        while fraction > 1 / 1024 and min(constraints(fraction * initial)) < 0:
            fraction /= 2
        x0 = fraction * initial
        if min(constraints(x0)) < 0:
            fraction, x0 = 0.0, zero
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=[(v, None) for v in lower],
            constraints=[{"type": "ineq", "fun": constraints, "jac": constraint_jac}],
            options={"maxiter": budget, "ftol": 1e-11},
        )
        terminal = expand(result.x)
        report = {
            "retained_solver_endpoint": bool(np.array_equal(best, terminal)),
            "terminal_nmse": float(result.fun),
            "solver_success": bool(result.success),
            "termination": result.message,
            "iterations": int(result.nit),
            "solver_status": int(result.status),
        }
        if np.array_equal(last_x, result.x) and last_j is not None:
            multipliers = np.asarray(result.multipliers)
            lag_grad = gradient(result.x) - constraint_jac(result.x).T @ multipliers
            at_lower = result.x <= lower + 1e-9
            lag_grad[at_lower] = np.minimum(0.0, lag_grad[at_lower])
            report["terminal_scaled_projected_lagrangian_gradient_inf"] = float(
                np.linalg.norm(lag_grad, ord=np.inf)
            )
            report["terminal_complementarity_inf"] = float(
                np.linalg.norm(multipliers * last_c, ord=np.inf)
            )
    except _Limit:
        pass
    except ArithmeticError as exc:
        report = {"solver_success": False, "termination": str(exc)}
    params = dict(zip(data.names, best.tolist(), strict=True))
    return {
        "parameters": params,
        "training_nmse": best_score,
        "diagnostics": report
        | {
            "start": start,
            "displacement_fraction": fraction,
            "source_nmse": source_score,
            "retained_minimum_Gp": float(min(best_margins[::2]) * data.scales[0]),
            "retained_minimum_U_minus_Uii": float(
                min(best_margins[1::2]) * data.scales[2] + DISPOSAL_BUFFER
            ),
            "budget": budget,
            "objective_evaluations": calls,
            "jacobian_evaluations": derivs,
            "invalid_rollouts": invalid,
            "active_parameters": [data.names[i] for i in ix],
            "fixed_parameters": [n for i, n in enumerate(data.names) if i not in ix],
            "coordinate_scales": dict(
                zip((data.names[i] for i in ix), scale.tolist(), strict=True)
            ),
            "coefficient_lower_bounds": {
                n: float(v)
                for n, v in zip(data.names, data.lo, strict=True)
                if np.isfinite(v)
            },
            "disposal_interior_margin": DISPOSAL_BUFFER,
            "constraint_grid": "observation times and interval midpoints",
        },
    }
