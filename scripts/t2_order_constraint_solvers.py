"""Compare the original SLSQP fit with enforcement of initial order by coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import Bounds, minimize

from scripts import fit_t2_initial_disposal_order as reference

TOL = reference.base.DOMAIN_TOL


class InitialOrder:
    """Compile the declared time-zero map with all inactive parameters substituted."""

    def __init__(self, data, fraction):
        self.fraction = fraction
        self.active = data.active
        self.reference = data.reference
        engine = data.engine.engine
        ca = engine.ca
        x = ca.SX.sym("active_delta", len(self.active))
        pars = ca.SX(self.reference)
        pars[self.active] += x
        self.arguments = [(a[0], a[1][:, 0]) for _, a, _, _ in data.engine.records]
        initial_u = [
            engine.observe(engine.initialize(pars, z), pars, u)[
                engine.targets.index("U")
            ]
            for z, u in self.arguments
        ]
        prediction = ca.vertcat(*initial_u)
        if not ca.is_linear(prediction, x):
            raise ValueError("Declared initial disposal is not affine in free gains")
        origin = np.zeros(len(self.active))
        a, b = ca.Function(
            "initial_affine", [x], [ca.jacobian(prediction, x), prediction]
        )(origin)
        self.slopes = np.asarray(a)
        self.intercept = np.asarray(b).ravel()
        observed = data.observed[data.edges[:-1], 2]
        self.high, self.low = reference.order_pairs(observed)
        self.required = (0.0 if fraction is None else fraction) * (
            observed[self.high] - observed[self.low]
        )
        self.matrix = self.slopes[self.high] - self.slopes[self.low]
        self.bound = self.required - (
            self.intercept[self.high] - self.intercept[self.low]
        )
        self.engine = engine

    def slack(self, vector):
        return (
            self.matrix @ (vector[self.active] - self.reference[self.active])
            - self.bound
        )

    def check(self, vectors) -> dict:
        largest = 0.0
        for vector in vectors:
            actual = np.array(
                [
                    float(
                        self.engine.observe(
                            self.engine.initialize(vector, z), vector, u
                        )[self.engine.targets.index("U")]
                    )
                    for z, u in self.arguments
                ]
            )
            approx = (
                self.slopes @ (vector[self.active] - self.reference[self.active])
                + self.intercept
            )
            largest = max(largest, float(np.max(abs(actual - approx))))
            np.testing.assert_allclose(actual, approx, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                actual[self.high] - actual[self.low] - self.required,
                self.slack(vector),
                rtol=1e-12,
                atol=1e-12,
            )
        return {
            "vectors": len(vectors),
            "events": len(self.arguments),
            "pairs": len(self.high),
            "max_absolute_U_error": largest,
        }


@dataclass
class Wedge:
    """An invertible affine chart that makes independent order facets box bounds."""

    columns: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray
    offset: np.ndarray
    lower: np.ndarray
    facets: np.ndarray
    rhs: np.ndarray
    certificates: dict

    def coordinates(self, x):
        y = x.copy()
        y[self.columns] = self.transform @ x[self.columns] - self.offset
        return y

    def physical(self, y):
        x = y.copy()
        x[self.columns] = self.inverse @ (y[self.columns] + self.offset)
        return x

    def gradients(self, jac):
        result = jac.copy()
        result[..., self.columns] = jac[..., self.columns] @ self.inverse
        return result

    def project(self, raw):
        """Nearest point in the original scaled Euclidean metric (small convex QP)."""
        if np.all(self.facets @ raw[self.columns] >= self.rhs):
            return raw.copy()
        x0 = raw[self.columns]
        # Enumerating active sets is exact for the one- or two-facet T2 wedge.
        best, distance = None, np.inf
        for count in range(1, len(self.rhs) + 1):
            for active in combinations(range(len(self.rhs)), count):
                ix = list(active)
                c = self.facets[ix]
                multipliers = np.linalg.solve(c @ c.T, self.rhs[ix] - c @ x0)
                proj = x0 + c.T @ multipliers
                if (
                    min(multipliers) < -1e-10
                    or min(self.facets @ proj - self.rhs) < -1e-10
                ):
                    continue
                d = np.linalg.norm(proj - x0)
                if d < distance:
                    best, distance = proj, d
        if best is None:
            raise ArithmeticError("No certified nearest feasible projection")
        x = raw.copy()
        x[self.columns] = best
        return x


def wedge(matrix, rhs, lower) -> Wedge:
    """Certify a bound-preserving chart for the registered one-/two-facet problems."""
    columns = np.flatnonzero(np.any(matrix != 0, axis=0))
    if not len(columns) or np.any(np.isfinite(lower[columns])):
        raise ValueError("Order involves bounded coefficients or has no direction")
    rows = matrix[:, columns]
    norms = np.linalg.norm(rows, axis=1)
    zero = norms == 0
    if np.any(rhs[zero] > 0):
        raise ValueError("Initial constraints infeasible at all coefficients")
    a, b = rows[~zero] / norms[~zero, None], rhs[~zero] / norms[~zero]
    rank = np.linalg.matrix_rank(a, tol=1e-12)
    if rank not in (1, 2):
        raise ValueError("Expected a rank-one or rank-two disposal-order wedge")
    # Constraints may be repeated by many event pairs. Keep the toughest parallel row.
    directions, limits = [], []
    for row, bound in zip(a, b, strict=True):
        j = next(
            (i for i, v in enumerate(directions) if np.max(abs(v - row)) < 1e-13), None
        )
        if j is None:
            directions.append(row)
            limits.append(bound)
        else:
            limits[j] = max(limits[j], bound)
    for ix in combinations(range(len(directions)), rank):
        c, d = np.array(directions)[list(ix)], np.array(limits)[list(ix)]
        if np.linalg.matrix_rank(c, tol=1e-12) < rank:
            continue
        # Each original normalized row must be implied by the independent lower bounds.
        weights = np.linalg.lstsq(c.T, a.T, rcond=None)[0].T
        error = float(np.max(abs(weights @ c - a)))
        if error > 1e-12 or weights.min() < -1e-12 or np.min(weights @ d - b) < -1e-11:
            continue
        transform = np.vstack((c, null_space(c).T))
        inv = np.linalg.inv(transform)
        offset = np.r_[d, np.zeros(len(columns) - rank)]
        low = lower.copy()
        low[columns[:rank]] = 0.0
        return Wedge(
            columns,
            transform,
            inv,
            offset,
            low,
            c,
            d,
            {
                "rank": int(rank),
                "facets": len(c),
                "unique_directions": len(directions),
                "max_reconstruction_error": error,
                "min_weight": float(weights.min()),
                "min_implied_rhs_margin": float(np.min(weights @ d - b)),
                "transform_condition": float(np.linalg.cond(transform)),
            },
        )
    raise ValueError(
        "Could not certify independent bound facets for all observed pairs"
    )


class Counter:
    """Charge actual batched engine calls, including an infeasible source scaling."""

    def __init__(self, data, order, guard):
        self.forward = self.jacobians = self.bad_forward = self.bad_jacobians = 0
        self.residual_seconds = self.jacobian_seconds = 0.0
        self.initial_residual_end = None
        self.setup_jacobian_infeasible = False
        n = len(order.arguments)
        self.event_count = n
        engine = data.engine
        self.charging = None
        self.event_calls = {"residual": 0, "jacobian": 0}
        self.unresolved_residual_batches = 0

        class MeteredRecords(list):
            def __iter__(records):
                for item in super().__iter__():
                    if self.charging is not None:
                        self.event_calls[self.charging] += 1
                    yield item

        engine.records = MeteredRecords(engine.records)
        for name in ("residual", "jacobian"):
            setattr(
                engine, name, self.wrapper(getattr(engine, name), name, order, guard)
            )

    def wrapper(self, original, name, order, guard):
        def measured(vector):
            bad = order.fraction is not None and np.any(order.slack(vector) < -TOL)
            is_jac = name == "jacobian"
            setup_jac = is_jac and self.jacobians == 0
            if guard and bad and not setup_jac:
                raise ValueError(
                    "Order-infeasible vector sent to expensive solver evaluation"
                )
            if is_jac:
                self.jacobians += 1
                self.bad_jacobians += int(bad)
                if setup_jac:
                    self.setup_jacobian_infeasible = bool(bad)
            else:
                self.forward += 1
                self.bad_forward += int(bad)
            t = perf_counter()
            self.charging = name
            try:
                return original(vector)
            except ArithmeticError:
                if not is_jac:
                    self.unresolved_residual_batches += 1
                raise
            finally:
                self.charging = None
                if is_jac:
                    self.jacobian_seconds += perf_counter() - t
                else:
                    self.residual_seconds += perf_counter() - t
                    if self.forward == 1:
                        self.initial_residual_end = perf_counter()

        return measured

    def costs(self, begin, solve, projection):
        end = perf_counter()
        return {
            "full_residual_batches": self.forward,
            "full_jacobian_batches": self.jacobians,
            "infeasible_residual_batches": self.bad_forward,
            "infeasible_jacobian_batches": self.bad_jacobians,
            "setup_jacobians": 1,
            "infeasible_setup_jacobians": int(self.setup_jacobian_infeasible),
            "full_residual_trajectories": self.event_calls["residual"],
            "unresolved_residual_batches": self.unresolved_residual_batches,
            "full_jacobian_trajectories": self.event_calls["jacobian"],
            "elapsed_seconds": end - begin,
            "solver_seconds": end - solve,
            "setup_seconds": solve - begin - projection,
            "projection_seconds": projection,
            "residual_engine_seconds": self.residual_seconds,
            "jacobian_engine_seconds": self.jacobian_seconds,
        }


def _feasible(model, data, order, counter, raw, budget):
    ix, p0 = data.active, data.reference
    j0 = data.engine.jacobian(p0).reshape(-1, 3, len(p0))[:, :, ix]
    rms = np.sqrt(np.mean(j0[data.samples] ** 2, axis=(0, 1)))
    scale = np.minimum(1 / np.maximum(rms, 1e-10), 100 * np.maximum(abs(p0[ix]), 1))
    lower = (data.lo[ix] - p0[ix]) / scale
    t = perf_counter()
    chart = wedge(order.matrix * scale, order.bound, lower)
    xraw = np.maximum((raw[ix] - p0[ix]) / scale, lower)
    projected = chart.project(xraw)
    y0 = np.maximum(chart.coordinates(projected), chart.lower)
    projection_seconds = perf_counter() - t

    def vector(y):
        v = p0.copy()
        v[ix] = np.maximum(data.lo[ix], p0[ix] + scale * chart.physical(y))
        return v

    initial = vector(y0)
    projection = chart.certificates | {
        "distance_in_reference_scaled_coordinates": float(
            np.linalg.norm(projected - xraw)
        ),
        "min_raw_slack": float(order.slack(raw).min()),
        "min_projected_slack": float(order.slack(initial).min()),
        "max_roundtrip_error": float(np.max(abs(chart.physical(y0) - projected))),
        "projected_parameters": dict(zip(data.names, initial.tolist(), strict=True)),
        "reparameterized_names": [data.names[ix[i]] for i in chart.columns],
    }
    best, best_score, source = None, np.inf, None

    @lru_cache(maxsize=2)
    def evaluate(key):
        nonlocal best, best_score, source
        if counter.forward >= budget:
            raise StopIteration("Forward evaluation budget reached")
        v = vector(np.asarray(key))
        try:
            e = data.engine.residual(v).reshape(-1, 3)
        except ArithmeticError:
            # A rejected nonfinite trial lets SLSQP's line search shorten its step.
            return 1e28, np.full(2 * counter.event_count, -1e14), None, None
        c, ids = data.margins(e)
        score = float(np.mean(e[data.samples] ** 2))
        if source is None:
            source = score
        if min(c) >= -TOL and min(order.slack(v)) >= -TOL and score < best_score:
            best, best_score = v, score
        return score, c, ids, e[data.samples]

    @lru_cache(maxsize=2)
    def jac(key):
        v = vector(np.asarray(key))
        j = (
            j0
            if np.array_equal(v, p0)
            else (data.engine.jacobian(v).reshape(-1, 3, len(p0))[:, :, ix])
        )
        return chart.gradients(j * scale)

    def gradient(y):
        k = tuple(y)
        e = evaluate(k)[3]
        if e is None:
            raise ArithmeticError("Solver requested derivatives at an unresolved trial")
        return 2 * np.einsum("ti,tij->j", e, jac(k)[data.samples]) / e.size

    def domain_jac(y):
        k = tuple(y)
        ids = evaluate(k)[2]
        if ids is None:
            raise ArithmeticError("Solver requested derivatives at an unresolved trial")
        return jac(k)[ids, [0, 2], :].reshape(-1, len(ix))

    solve = perf_counter()
    try:
        res = minimize(
            lambda y: evaluate(tuple(y))[0],
            y0,
            jac=gradient,
            bounds=Bounds(chart.lower, np.inf),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda y: evaluate(tuple(y))[1],
                    "jac": domain_jac,
                }
            ],
            method="SLSQP",
            options={"maxiter": budget, "ftol": 1e-11},
        )
        success, message = bool(res.success), str(res.message)
    except (StopIteration, ArithmeticError) as exc:
        success, message = False, str(exc)
    return (
        {
            "parameters": None
            if best is None
            else dict(zip(data.names, best.tolist(), strict=True)),
            "solver_success": success,
            "termination": message,
            "training_nmse": best_score if best is not None else None,
            "projected_start_nmse": source,
            "forward_evaluations": counter.forward,
            "jacobian_evaluations": counter.jacobians,
            "projection": projection,
        },
        solve,
        projection_seconds,
    )


def fit(
    model: dict,
    rows: list[dict],
    arm: str,
    start: str,
    *,
    budget: int = 80,
    fraction: float | None = 0.5,
) -> dict:
    """Run a freshly compiled, costed comparison using only the fitting events."""
    if (
        arm not in ("reference", "feasible")
        or start not in reference.coupled.orion.STARTS
        or budget < 1
    ):
        raise ValueError("Unsupported solver arm, start or forward budget")
    if fraction not in reference.FRACTIONS.values() or (
        arm == "feasible" and fraction is None
    ):
        raise ValueError("Unsupported ordering fraction")
    begin = perf_counter()
    data = reference.base._Training(model, rows)
    order = InitialOrder(data, fraction)
    raw_dict = reference.coupled.orion.optimizer_start(
        model, model["formula"]["parameters"], "fixed_insulin", start
    )
    raw = data.reference.copy()
    raw[data.active] = np.maximum(data.engine._vector(raw_dict), data.lo)[data.active]
    shifted = data.reference.copy()
    shifted[data.active] += np.linspace(-0.37, 0.53, len(data.active))
    shifted = np.maximum(shifted, data.lo)
    check = order.check(
        [data.reference, raw, shifted, 0.5 * (shifted + data.reference)]
    )
    counter = Counter(data, order, guard=arm == "feasible")
    if arm == "reference":
        label = next(k for k, v in reference.FRACTIONS.items() if v == fraction)
        result = reference.optimize(model, data, label, budget, start)
        solve, projection = counter.initial_residual_end, 0.0
    else:
        result, solve, projection = _feasible(model, data, order, counter, raw, budget)
    costs = counter.costs(begin, solve, projection)
    if (
        result["forward_evaluations"] != counter.forward
        or result["jacobian_evaluations"] != counter.jacobians
    ):
        raise ValueError("Solver and instrumented engine counts differ")
    return result | {
        "cost": costs,
        "matrix_check": check,
        "fraction": fraction,
        "raw_start_parameters": dict(zip(data.names, raw.tolist(), strict=True)),
    }
