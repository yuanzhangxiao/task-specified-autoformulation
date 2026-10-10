"""Bound-safe recovery searches and distinct fit/uncertainty verification gates."""

from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting.affine_propagation import (
    numerically_verified,
    verify_outputs,
)
from autoformalism.fitting.public_fitting import content_sha256


class DomainViolation(ValueError):
    """A domain violation with the actual vector and reconstruction evidence."""

    def __init__(self, record):
        self.record = record
        super().__init__("material parameter-domain violation")


def restore(anchor, units, q, lower, upper, names, events):
    """Project only bounded floating-point reconstruction error, never inputs."""
    step = units * q
    vector = anchor + step
    if not np.isfinite(vector).all():
        raise ValueError("nonfinite optimizer coordinate reconstruction")
    bad = (vector < lower) | (vector > upper)
    if np.any(bad):
        edge = np.where(vector < lower, lower, upper)
        tolerance = (
            8
            * np.finfo(float).eps
            * np.maximum(1, np.abs(anchor) + np.abs(step) + np.abs(edge))
        )
        excess = np.maximum(lower - vector, vector - upper)
        record = {
            "parameters": dict(zip(names, vector.tolist(), strict=True)),
            "violations": [
                {
                    "parameter": names[i],
                    "value": float(vector[i]),
                    "lower": float(lower[i]),
                    "upper": float(upper[i]),
                    "excess": float(excess[i]),
                    "roundoff_allowance": float(tolerance[i]),
                }
                for i in np.flatnonzero(bad)
            ],
        }
        material = np.any(excess[bad] > tolerance[bad])
        record["action"] = "rejected" if material else "roundoff_projected"
        events.append(record)
        if material:
            raise DomainViolation(record)
        vector = np.clip(vector, lower, upper)
    return vector


def usable(point: dict | None) -> bool:
    """Finite output agreement for fitting, distinct from tight profile evidence."""
    return bool(
        point
        and np.isfinite(
            [
                point["training_nmse"],
                point["alternate_training_nmse"],
                point["solver_loss_discrepancy"],
                point["solver_prediction_discrepancy"],
            ]
        ).all()
        and point["solver_prediction_discrepancy"] <= 1e-7
        and point["solver_loss_discrepancy"]
        <= max(
            1e-12, 1e-8 * max(point["training_nmse"], point["alternate_training_nmse"])
        )
    )


def improves(candidate: dict | None, incumbent: dict | None) -> bool:
    """Require an improvement under both solvers; retain ties and unusable checks."""
    if not usable(candidate):
        return False
    return not usable(incumbent) or max(
        candidate["training_nmse"], candidate["alternate_training_nmse"]
    ) < min(incumbent["training_nmse"], incumbent["alternate_training_nmse"])


def check_point(oracle, parameters, deadline, checkpoint, *, tight=False):
    """Output-only independent verification; tighter settings never change fitting."""
    original = oracle.settings
    if tight:
        oracle.settings = original.model_copy(
            update={"relative_tolerance": 1e-12, "absolute_tolerance": 1e-14}
        )
    try:
        point = verify_outputs(oracle, parameters, deadline, checkpoint)
    finally:
        oracle.settings = original
    return point | {
        "tight_verification": tight,
        "usable_for_retention": usable(point),
        "strict_uncertainty_verified": numerically_verified(point, 1e-12),
    }


def search(
    oracle,
    start,
    deadline,
    checkpoint,
    *,
    calls,
    target,
    profile=None,
    acceptable=None,
    telemetry=False,
    optimizer_scaling="jacobian",
    record_steps=False,
):
    """Joint or projected TRF; charge started attempts, distinguish completed calls."""
    if optimizer_scaling not in {"jacobian", "fixed_coordinates"}:
        raise ValueError("unknown optimizer scaling")
    anchor = oracle.vector(start)
    free = list(range(len(anchor))) if profile is None else list(profile.outer)
    units = oracle.units[free]
    count, best, cached, domain_events, projection = 0, None, None, [], None
    begun, trace, steps = monotonic(), [], []
    accepted = None

    class Reached(Exception):
        pass

    def save():
        checkpoint(
            {"calls": count, "best": best, "domain_events": domain_events}
            | (
                {"completed_calls": len(trace), "evaluations": trace}
                if telemetry
                else {}
            )
            | ({"accepted_steps": steps} if record_steps else {})
        )

    def evaluate(q):
        nonlocal count, best, cached, projection, accepted
        if cached is not None and np.array_equal(q, cached[0]):
            return cached[1:]
        if count >= calls or monotonic() >= deadline:
            raise TimeoutError("search allowance exhausted")
        vector = anchor.copy()
        vector[free] = restore(
            anchor[free],
            units,
            q,
            oracle.lower[free],
            oracle.upper[free],
            [oracle.names[i] for i in free],
            domain_events,
        )
        count += 1
        save()
        if profile is None:
            r, j = oracle.evaluate(vector, deadline)
        else:
            vector, r, j, projection = profile.evaluate(vector, deadline)
        value = float(np.mean(r * r))
        if not np.isfinite(value) or not np.isfinite(j).all():
            raise ValueError("nonfinite fitted residual or derivative")
        if best is None or value < best["training_nmse"]:
            best = {"parameters": oracle.parameters(vector), "training_nmse": value}
        if telemetry:
            # Gradient of unamplified mean squared residual in scaled outer
            # coordinates. Trial evaluations are not accepted optimizer iterates.
            gradient = 2 * (j * units).T @ r / len(r)
            tolerance = 1e-8 * np.maximum(1, np.abs(vector[free]))
            lower_active = vector[free] - oracle.lower[free] <= tolerance
            upper_active = oracle.upper[free] - vector[free] <= tolerance
            projected = gradient.copy()
            projected[
                (lower_active & (gradient > 0)) | (upper_active & (gradient < 0))
            ] = 0
            trace.append(
                {
                    "attempt": count,
                    "elapsed_seconds": monotonic() - begun,
                    "training_nmse": value,
                    "best_training_nmse": best["training_nmse"],
                    "projected_gradient_inf": float(np.max(np.abs(projected))),
                    "active_outer_bounds": {
                        oracle.names[i]: "lower" if lower_active[k] else "upper"
                        for k, i in enumerate(free)
                        if lower_active[k] or upper_active[k]
                    },
                    "parameters_sha256": content_sha256(oracle.parameters(vector)),
                }
            )
        save()
        cached = (q.copy(), r, j * units)
        if record_steps and accepted is None:
            accepted = cached
        if value <= target and (acceptable is None or acceptable(r)):
            raise Reached
        return cached[1:]

    def callback(intermediate_result):
        """Accepted displacement and local linear prediction; no extra evaluations."""
        nonlocal accepted
        q = intermediate_result.x
        if np.array_equal(q, accepted[0]):
            return
        if cached is None or not np.array_equal(q, cached[0]):
            raise ValueError("accepted optimizer point missing from evaluation cache")
        step = q - accepted[0]
        previous = float(np.mean(accepted[1] ** 2))
        actual = previous - float(np.mean(cached[1] ** 2))
        predicted = previous - float(np.mean((accepted[1] + accepted[2] @ step) ** 2))
        steps.append(
            {
                "evaluation_count": count,
                "elapsed_seconds": monotonic() - begun,
                "training_nmse": float(np.mean(cached[1] ** 2)),
                "coordinate_step_norm": float(np.linalg.norm(step)),
                "actual_reduction": actual,
                "linearized_residual_reduction": predicted,
                "reduction_ratio": actual / predicted if predicted > 0 else None,
            }
        )
        accepted = cached
        save()

    try:
        solved = least_squares(
            lambda q: evaluate(q)[0] * 1e6,
            np.zeros(len(free)),
            jac=lambda q: evaluate(q)[1] * 1e6,
            bounds=(
                (oracle.lower[free] - anchor[free]) / units,
                (oracle.upper[free] - anchor[free]) / units,
            ),
            x_scale="jac" if optimizer_scaling == "jacobian" else 1.0,
            ftol=None,
            xtol=1e-12,
            gtol=1e-10,
            max_nfev=calls,
            **({"callback": callback} if record_steps else {}),
        )
        stop = "step_or_gradient_converged" if solved.success else "budget_limited"
        detail = {"message": solved.message, "optimality": float(solved.optimality)}
    except Reached:
        stop, detail = "numerical_target_reached", {}
    except (ValueError, TimeoutError, RuntimeError, ArithmeticError) as error:
        stop = (
            "budget_limited" if isinstance(error, TimeoutError) else "numerical_failure"
        )
        detail = {"message": str(error)[-1000:]}
    save()
    return {
        "best": best,
        "calls": count,
        "stop_reason": stop,
        **detail,
        "domain_events": domain_events,
        "projection": projection,
        **({"completed_calls": len(trace), "evaluations": trace} if telemetry else {}),
        **(
            {"accepted_steps": steps, "optimizer_scaling": optimizer_scaling}
            if record_steps
            else {}
        ),
    }


def diverse_starts(oracle, problem, count: int, seed: int) -> list[dict]:
    """Training/domain-only stratified starts, independent of methods and truth."""
    if count == 0:
        return []
    rng = np.random.default_rng(seed)
    original = oracle.vector(problem.start)
    points = np.zeros((count, len(original)))
    for j in range(len(original)):
        lo, hi = oracle.lower[j], oracle.upper[j]
        if np.isnan([lo, hi]).any() or lo >= hi:
            raise ValueError("portfolio requires ordered declared parameter bounds")
        quantiles = (rng.permutation(count) + rng.uniform(0.1, 0.9, count)) / count
        if lo > 0 and np.isfinite(hi):
            points[:, j] = np.exp(np.log(lo) + quantiles * (np.log(hi) - np.log(lo)))
        else:
            left, right = (
                max(lo, original[j] - 2 * oracle.units[j]),
                min(hi, original[j] + 2 * oracle.units[j]),
            )
            if not np.isfinite([left, right]).all() or left >= right:
                raise ValueError("portfolio requires a finite sampling window")
            points[:, j] = left + quantiles * (right - left)
    result = [oracle.parameters(p) for p in points]
    for item in result:
        oracle.vector(item)
    return result
