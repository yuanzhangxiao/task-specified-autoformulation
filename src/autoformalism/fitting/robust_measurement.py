"""Fixed robust objectives on M34 readings and certified convex affine profiles.

Huber is an estimation objective, not a probability of measurement faults. Its
scale is the declared known sigma, and its threshold is frozen before fitting.
"""

from __future__ import annotations

from functools import partial
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import lsq_linear, minimize

from autoformalism.fitting import measurement_rollout, profiled_output
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from autoformalism.schemas.sensor_drift import DriftProblem


class Loss(BaseModel):
    """Immutable, predeclared loss on standardized present training readings."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal["gaussian", "huber"] = "huber"
    delta: float = Field(default=1.345, gt=0, allow_inf_nan=False)


def _residual(residual: np.ndarray) -> np.ndarray:
    """Reject malformed or nonfinite loss inputs without silently masking rows."""
    result = np.asarray(residual, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("loss residual must be a finite vector")
    return result


def loss_values(residual: np.ndarray, loss: Loss) -> np.ndarray:
    """Return each native objective contribution, without Gaussian constants."""
    r = _residual(residual)
    magnitude = np.abs(r)
    if loss.kind == "gaussian":
        values = 0.5 * r**2
    else:
        central = np.minimum(magnitude, loss.delta)
        values = 0.5 * central**2 + loss.delta * (magnitude - central)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite loss contribution")
    return values


def score(residual: np.ndarray, loss: Loss) -> np.ndarray:
    """Return rho prime, including the bounded Huber outlier score."""
    r = _residual(residual)
    return r.copy() if loss.kind == "gaussian" else np.clip(r, -loss.delta, loss.delta)


def transformed_residual(residual: np.ndarray, loss: Loss) -> tuple:
    """Return q=sign(r)*sqrt(2 rho(r)) and its exact first derivative.

    The derivative is one at zero and throughout the quadratic region. It is
    not the square root of the IRLS working weight outside that region.
    """
    r = _residual(residual)
    q, derivative = r.copy(), np.ones_like(r)
    if loss.kind == "huber":
        outside = np.abs(r) > loss.delta
        q[outside] = np.sign(r[outside]) * np.sqrt(
            2 * loss.delta * (np.abs(r[outside]) - 0.5 * loss.delta)
        )
        derivative[outside] = loss.delta / np.abs(q[outside])
    if not np.isfinite(q).all():
        raise ValueError("nonfinite transformed residual")
    return q, derivative


class ProfileUnavailable(ValueError):
    """A rejected inner solve or derivative, retaining its measured work audit."""

    def __init__(self, reason: str, audit: dict):
        super().__init__(reason)
        self.audit = audit | {"status": "unavailable", "reason": reason}


def _arrays(design, offset, d_design, d_offset, lower, upper) -> tuple:
    """Validate the complete affine representation and its fixed physical bounds."""
    a, b, da, db, lo, hi = (
        np.asarray(value, dtype=float)
        for value in (design, offset, d_design, d_offset, lower, upper)
    )
    if (
        a.ndim != 2
        or min(a.shape, default=0) == 0
        or b.shape != (len(a),)
        or da.ndim != 3
        or da.shape[:2] != a.shape
        or db.shape != (len(a), da.shape[2])
        or lo.shape != (a.shape[1],)
        or hi.shape != lo.shape
        or np.any(lo >= hi)
        or any(not np.isfinite(v).all() for v in (a, b, da, db))
        or np.isnan(lo).any()
        or np.isnan(hi).any()
    ):
        raise ValueError("invalid bounded robust projection arrays")
    return a, b, da, db, lo, hi


def _active(z, gradient, lower, upper) -> np.ndarray:
    """Identify bound multipliers with the correct KKT signs in scaled units."""
    tolerance = 1e-9 * np.maximum(1.0, np.abs(z))
    active = np.zeros(len(z), dtype=int)
    active[(z - lower <= tolerance) & (gradient >= 0)] = -1
    active[(upper - z <= tolerance) & (gradient <= 0)] = 1
    return active


def _kkt(gradient, active) -> float:
    """Maximum violation of bound-constrained first-order convex optimality."""
    violation = gradient.copy()
    violation[active == -1] = np.minimum(gradient[active == -1], 0)
    violation[active == 1] = np.maximum(gradient[active == 1], 0)
    return float(np.max(np.abs(violation), initial=0))


def project(
    design,
    offset,
    d_design,
    d_offset,
    lower,
    upper,
    *,
    loss: Loss,
    deadline: float = np.inf,
    max_iterations: int = 250,
) -> dict:
    """Solve the convex bounded Huber problem, then differentiate its KKT system.

    The solve uses the full objective and analytic score, followed by bounded
    Newton polishing when the current inlier Hessian is nonsingular. Acceptance
    requires independently checked KKT residuals; no single IRLS step is labeled
    an exact profile. Derivatives use the inlier Hessian and the *full* score in
    the design-derivative term. Unstable derivatives request joint fallback.
    """
    begun = monotonic()
    audit = {
        "loss": loss.model_dump(),
        "solver": "bounded-convex-Huber-LBFGSB-with-Newton-polish",
        "iterations": 0,
        "inner_evaluations": 0,
        "approximate_profile": False,
        "convex_optimum_certified": False,
    }

    def time_check():
        if monotonic() >= deadline:
            error = TimeoutError("robust inner projection allowance exhausted")
            error.audit = audit | {
                "status": "unavailable",
                "reason": str(error),
                "inner_seconds": monotonic() - begun,
            }
            raise error

    def fail(reason):
        raise ProfileUnavailable(reason, audit | {"inner_seconds": monotonic() - begun})

    time_check()
    a, b, da, db, lo, hi = _arrays(design, offset, d_design, d_offset, lower, upper)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("inner iteration limit must be a positive integer")
    if max_iterations <= 0:
        fail("robust inner iteration allowance exhausted")
    if loss.kind == "gaussian":
        result = profiled_output.project(a, b, da, db, lo, hi)
        time_check()
        result["audit"] |= {
            "loss": loss.model_dump(),
            "inner_seconds": monotonic() - begun,
            "approximate_profile": False,
        }
        return result
    units = np.linalg.norm(a, axis=0)
    if np.any(units == 0) or not np.isfinite(units).all():
        fail("rank-deficient robust output design")
    h, low, high = a / units, lo * units, hi * units
    singular = np.linalg.svd(h, compute_uv=False)
    full_rank = len(singular) == a.shape[1] and singular[-1] > 1e-12 * singular[0]
    audit["column_normalized_singular_values"] = singular.tolist()
    audit["design_full_rank"] = bool(full_rank)
    if not full_rank and da.shape[2]:
        fail("rank-deficient robust output design")

    def objective(z):
        time_check()
        audit["inner_evaluations"] += 1
        r = h @ z + b
        return float(loss_values(r, loss).sum()), h.T @ score(r, loss)

    def iteration(_):
        time_check()
        audit["iterations"] += 1

    # Least squares supplies only a deterministic starting point, never the
    # claimed Huber optimum. All subsequent decisions use the Huber objective.
    seed = lsq_linear(h, -b, bounds=(low, high), method="bvls", tol=1e-12)
    audit["seed_iterations"] = int(seed.nit)
    time_check()
    solved = minimize(
        objective,
        np.clip(seed.x, low, high),
        jac=True,
        bounds=list(zip(low, high, strict=True)),
        method="L-BFGS-B",
        callback=iteration,
        options={"maxiter": max_iterations, "gtol": 1e-11, "ftol": 1e-15},
    )
    time_check()
    audit |= {
        "solver_success": bool(solved.success),
        "solver_status": int(solved.status),
        "solver_message": str(solved.message),
    }
    z = np.clip(np.asarray(solved.x), low, high)
    if not np.isfinite(z).all():
        fail("nonfinite robust inner solution")
    # Resolve near-stationary roundoff and smooth-region steps using the actual
    # Huber Hessian. Line search still checks the full piecewise objective.
    for _ in range(min(20, max_iterations - audit["iterations"])):
        value, gradient = objective(z)
        active = _active(z, gradient, low, high)
        if _kkt(gradient, active) <= 1e-10:
            break
        free, inlier = active == 0, np.abs(h @ z + b) < loss.delta
        hf = h[inlier][:, free]
        _, curvature, vectors = np.linalg.svd(hf, full_matrices=False)
        if not free.any() or len(curvature) < int(free.sum()):
            break
        if curvature[-1] <= 1e-10 * curvature[0]:
            break
        direction = np.zeros(len(z))
        direction[free] = -vectors.T @ ((vectors @ gradient[free]) / curvature**2)
        accepted = False
        for backtrack in range(30):
            trial = np.clip(z + 0.5**backtrack * direction, low, high)
            trial_value, _ = objective(trial)
            numerical_slack = 8 * np.finfo(float).eps * (1 + abs(value))
            if trial_value <= value + 1e-4 * gradient @ (trial - z) + numerical_slack:
                z, accepted = trial, True
                break
        audit["iterations"] += 1
        if not accepted:
            break
    value, gradient = objective(z)
    active = _active(z, gradient, low, high)
    # Snap active coordinates to their actual fixed bounds before certifying.
    z[active == -1], z[active == 1] = low[active == -1], high[active == 1]
    gains = np.clip(z / units, lo, hi)
    z = gains * units
    residual = a @ gains + b
    value = float(loss_values(residual, loss).sum())
    gradient = h.T @ score(residual, loss)
    active = _active(z, gradient, low, high)
    psi, free = score(residual, loss), active == 0
    kkt_tolerance = 1e-8 * max(1.0, float(np.max(np.abs(h).T @ np.abs(psi))))
    optimality = _kkt(gradient, active)
    audit |= {
        "active_mask": active.tolist(),
        "objective_value": value,
        "gradient_scaled": gradient.tolist(),
        "optimality": optimality,
        "kkt_tolerance": kkt_tolerance,
        "inner_seconds": monotonic() - begun,
    }
    if not np.isfinite(value) or optimality > kkt_tolerance:
        fail("robust inner KKT check failed")
    audit["convex_optimum_certified"] = True
    inlier = np.abs(residual) < loss.delta
    hf = h[inlier][:, free]
    u, s, vt = np.linalg.svd(hf, full_matrices=False)
    rank = int(np.sum(s > 1e-10 * s[0])) if len(s) else 0
    threshold_distance = float(np.min(np.abs(np.abs(residual) - loss.delta)))
    guard = 1e-7 * max(1.0, loss.delta)
    weak_bounds = bool(np.any((active != 0) & (np.abs(gradient) <= guard)))
    near_free_bounds = bool(
        np.any(
            free & (np.minimum(z - low, high - z) <= guard * np.maximum(1, np.abs(z)))
        )
    )
    reasons = []
    if rank < int(free.sum()):
        reasons.append("rank-deficient inlier Hessian; inner uniqueness unavailable")
    if threshold_distance <= guard:
        reasons.append("Huber threshold proximity")
    if weak_bounds or near_free_bounds:
        reasons.append("weak or changing active bound")
    audit |= {
        "inlier_count": int(inlier.sum()),
        "outlier_region_count": int((~inlier).sum()),
        "inlier_hessian_rank": rank,
        "inlier_hessian_eigenvalues_scaled": (s**2).tolist(),
        "free_inner_coordinates": int(free.sum()),
        "threshold_distance": threshold_distance,
        "derivative_guards": reasons,
        "local_uniqueness_certified": rank == int(free.sum()) and not weak_bounds,
    }
    jacobian = np.einsum("nkp,k->np", da, gains) + db
    if reasons and da.shape[2]:
        fail("robust profile derivative unavailable: " + "; ".join(reasons))
    if free.any() and da.shape[2]:
        term = np.einsum("nkp,n->kp", da[:, free] / units[free][None, :, None], psi)
        dz = -vt.T @ (
            (u.T @ jacobian[inlier]) / s[:, None] + (vt @ term) / s[:, None] ** 2
        )
        jacobian += h[:, free] @ dz
    if not np.isfinite(jacobian).all():
        fail("nonfinite robust profiled derivative")
    time_check()
    audit |= {
        "status": "certified",
        "derivative": "fixed-active-set exact Huber residual derivative"
        if da.shape[2]
        else "not required: zero outer coordinates",
        "inner_seconds": monotonic() - begun,
    }
    return {"gains": gains, "residual": residual, "jacobian": jacobian, "audit": audit}


class RobustRollouts:
    """Transform optimizer residuals only; delegate raw reporting to the base."""

    def __init__(self, base: measurement_rollout.MeasurementRollouts, loss: Loss):
        self.base, self.loss = base, loss

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        """Evaluate q and its chain-rule Jacobian under the requested integrator."""
        residual, derivative = self.base.evaluate(
            vector, deadline, jacobian=jacobian, method=method
        )
        q, chain = transformed_residual(residual, self.loss)
        return q, None if derivative is None else chain[:, None] * derivative


class RobustProfile:
    """Transform a certified profile and retain every success/failure audit."""

    def __init__(self, base, loss: Loss):
        self.base, self.loss = base, loss
        self.inner_audits: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate(self, vector, deadline):
        """Return transformed profiled residuals without hiding rejected solves."""
        begun = monotonic()
        audit = {}
        try:
            fitted, residual, derivative, audit = self.base.evaluate(vector, deadline)
            q, chain = transformed_residual(residual, self.loss)
        except (ValueError, TimeoutError, np.linalg.LinAlgError) as error:
            self.inner_audits.append(
                audit
                | getattr(error, "audit", {})
                | {
                    "status": "unavailable",
                    "reason": str(error),
                    "profile_seconds": monotonic() - begun,
                }
            )
            raise
        audit = audit | {"profile_seconds": monotonic() - begun}
        self.inner_audits.append(audit)
        return fitted, q, chain[:, None] * derivative, audit


def build(problem: MeasurementProblem | DriftProblem, loss: Loss) -> tuple:
    """Reuse M34 certificates and attach the requested fixed optimization loss."""
    if isinstance(problem, DriftProblem):
        from autoformalism.fitting.sensor_drift import build as build_drift

        return build_drift(problem, loss)
    oracle, profile, route = measurement_rollout.build(problem)
    if profile is not None:
        if loss.kind == "huber":
            profile.projector = partial(project, loss=loss)
        profile = RobustProfile(profile, loss)
    route = route | {
        "loss": loss.model_dump(),
        "inner_objective": "convex bounded " + loss.kind,
        "confidence_probability": None,
    }
    return RobustRollouts(oracle, loss), profile, route
