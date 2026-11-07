"""Convex conditional fitting with explicit sampled concentration inequalities.

Only the inner problem is convex. A feasible, KKT-qualified inner point says
nothing about convergence of the outer matrix-coefficient search or positivity
under an unseen schedule. No predictions or signed coefficients are clipped.
"""

from __future__ import annotations

from time import monotonic

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from autoformalism.fitting import prefix_meal_joint_profile as joint
from autoformalism.fitting import prefix_multioutput_profile as projection
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting.public_fitting import content_sha256

PROTOCOL = "sampled-concentration-convex-profile-1"
PRIMAL_TOLERANCE = 1e-9
KKT_TOLERANCE = 2e-7


def constrained_least_squares(a, b, lower, upper, c, d, deadline):
    """Minimize ||a x+b||² subject to bounds and c x+d >= 0.

    A finite cutting-plane loop supplies violated rows to a small SLSQP QP.
    Acceptance checks *all* original rows, bound feasibility, nonnegative dual
    multipliers, complementarity and stationarity, independently of success.
    Column normalization and QR compression preserve the least-squares problem.
    """
    started = monotonic()
    roots._check(deadline)
    a, b, c, d = (
        roots._finite(v, n)
        for v, n in ((a, "design"), (b, "offset"), (c, "constraints"), (d, "domain"))
    )
    lower, upper = np.asarray(lower), np.asarray(upper)
    n = a.shape[1]
    if (
        a.ndim != 2
        or b.shape != (len(a),)
        or c.ndim != 2
        or c.shape[1] != n
        or d.shape != (len(c),)
        or lower.shape != (n,)
        or upper.shape != (n,)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower >= upper)
    ):
        raise ValueError("invalid free-coordinate convex profile shapes or bounds")
    units = np.linalg.norm(a, axis=0)
    units = np.where(units == 0, 1.0, units)
    h, g = a / units, c / units
    lo, hi = lower * units, upper * units
    # Work with O(n²) objective evaluations, never O(samples*n) inside SLSQP.
    q, r = np.linalg.qr(h, mode="reduced")
    projected = q.T @ b
    singular = np.linalg.svd(h, compute_uv=False)
    initial = lsq_linear(
        r, -projected, bounds=(lo, hi), method="bvls", tol=1e-13, max_iter=200
    )
    x = initial.x
    if not initial.success or not np.isfinite(x).all():
        raise projection.ProfileUnavailable("convex profile start failed", {})
    # Scaling constraint rows improves conditioning without changing their signs.
    row_units = np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    rows, rhs = g / row_units[:, None], -d / row_units
    selected: set[int] = set()
    iterations, raw_success = 0, bool(initial.success)
    stationarity, complementarity = np.inf, np.inf

    def objective(v):
        roots._check(deadline)
        residual = r @ v + projected
        return 0.5 * float(residual @ residual)

    def gradient(v):
        roots._check(deadline)
        return r.T @ (r @ v + projected)

    for cuts in range(33):
        roots._check(deadline)
        slack = g @ x + d
        bad = np.flatnonzero(slack < -PRIMAL_TOLERANCE)
        if not len(bad):
            # Bounds use the same >= convention as concentration constraints.
            eye = np.eye(n)
            gl = np.flatnonzero(np.isfinite(lo))
            gu = np.flatnonzero(np.isfinite(hi))
            chosen = np.asarray(sorted(selected), dtype=int)
            all_rows = np.concatenate((rows[chosen], eye[gl], -eye[gu]))
            all_slack = np.concatenate(
                (rows[chosen] @ x - rhs[chosen], x[gl] - lo[gl], hi[gu] - x[gu])
            )
            active = all_slack <= 2e-7 * max(1.0, float(np.linalg.norm(x, np.inf)))
            grad = gradient(x)
            if active.any():
                dual = lsq_linear(
                    all_rows[active].T,
                    grad,
                    bounds=(0, np.inf),
                    method="bvls",
                    tol=1e-13,
                    max_iter=1000,
                )
                stationarity = float(np.max(np.abs(all_rows[active].T @ dual.x - grad)))
                complementarity = float(np.max(np.abs(dual.x * all_slack[active])))
            else:
                stationarity, complementarity = float(np.max(abs(grad))), 0.0
            tolerance = KKT_TOLERANCE * max(1.0, float(np.linalg.norm(h @ x + b)))
            bounds_ok = bool(
                np.all(x >= lo - PRIMAL_TOLERANCE)
                and np.all(x <= hi + PRIMAL_TOLERANCE)
            )
            if bounds_ok and max(stationarity, complementarity) <= tolerance:
                fitted = x / units
                residual = a @ fitted + b
                audit = {
                    "protocol": PROTOCOL,
                    "status": "conditional_optimum",
                    "normalized_kkt_residual": stationarity,
                    "complementarity_residual": complementarity,
                    "kkt_tolerance": tolerance,
                    "bounds_satisfied": bounds_ok,
                    "minimum_scaled_concentration": float(np.min(slack)),
                    "primal_tolerance": PRIMAL_TOLERANCE,
                    "constraints": len(c),
                    "working_constraints": len(selected),
                    "cutting_plane_solves": cuts,
                    "iterations": iterations,
                    "optimizer_raw_success": raw_success,
                    "design_rank": int(np.sum(singular > 1e-10 * singular[0])),
                    "design_columns": n,
                    "column_units": units.tolist(),
                    "column_normalized_singular_values": singular.tolist(),
                    "training_nmse": float(np.mean(residual**2)),
                    "inner_elapsed_seconds": monotonic() - started,
                    "continuous_positivity_certified": False,
                }
                roots._check(deadline)
                return fitted, residual, audit
            # Polishing the current convex QP may resolve a near-active face.
            selected.update(np.flatnonzero(slack <= 1e-6).tolist())
        else:
            selected.update(bad[np.argsort(slack[bad])[:32]].tolist())
        if cuts == 32:
            break
        chosen = np.asarray(sorted(selected), dtype=int)
        constraints = (
            []
            if not len(chosen)
            else [LinearConstraint(rows[chosen], rhs[chosen], np.inf)]
        )
        solved = minimize(
            objective,
            x,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12},
        )
        iterations += int(solved.nit)
        raw_success = bool(solved.success)
        if not np.isfinite(solved.x).all():
            break
        x = solved.x
    raise projection.ProfileUnavailable(
        "concentration QP feasibility or KKT unresolved",
        {
            "status": "constrained_optimum_unavailable",
            "iterations": iterations,
            "normalized_kkt_residual": float(stationarity)
            if np.isfinite(stationarity)
            else None,
            "minimum_scaled_concentration": float(np.min(g @ x + d)),
            "optimizer_raw_success": raw_success,
        },
    )


class InsulinConcentrationProfile(joint.JointMealProfile):
    """One explicit training-domain control; I is an absolute concentration."""

    def __init__(self, oracle, *, failure_diagnostics=False):
        super().__init__(
            oracle, additional_forcing="c_I", failure_diagnostics=failure_diagnostics
        )
        if "I" not in self.original.system.channels:
            raise ValueError("insulin concentration control requires target I")
        record = {k: v for k, v in self.audit.items() if k != "sha256"}
        record.update(
            protocol=PROTOCOL,
            inner_solver="convex_QP_with_all_row_feasibility_and_KKT",
            domain={
                "channel": "I",
                "lower": 0.0,
                "subdivisions": 2,
                "basis": "public target is absolute insulin concentration",
                "scope": "training horizons at observations and midpoints",
            },
            continuous_positivity_certified=False,
            signed_feedback_domain_unchanged=True,
        )
        self.audit = record | {"sha256": content_sha256(record)}

    def design(self, parameters, deadline):
        return super().design(
            parameters, deadline, subdivisions=2, constrained_channel="I"
        )

    def solve_block(self, block, deadline):
        a, b, lo, hi = (block[n] for n in ("design", "offset", "lower", "upper"))
        c, d = block["constraint_design"], block["constraint_offset"]
        fixed = lo == hi
        fitted = lo.copy()
        fitted[~fixed], residual, audit = constrained_least_squares(
            a[:, ~fixed],
            b + a[:, fixed] @ lo[fixed],
            lo[~fixed],
            hi[~fixed],
            c[:, ~fixed],
            d + c[:, fixed] @ lo[fixed],
            deadline,
        )
        return fitted, residual, audit
