"""Bound-safe recovery searches and distinct fit/uncertainty verification gates."""

from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting import poor_fit as stationary
from autoformalism.fitting.affine_propagation import (
    numerically_verified,
    verify_outputs,
)
from autoformalism.fitting.optimizer_scaling import RelativeScaling, relative_scales
from autoformalism.fitting.public_fitting import content_sha256


class DomainViolation(ValueError):
    """A domain violation with the actual vector and reconstruction evidence."""

    def __init__(self, record):
        self.record = record
        super().__init__("material parameter-domain violation")


class NumericalInvalidTrial(ValueError):
    """A numerical overflow/nonfinite trial, not a domain or programming error."""


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
    record_full_steps=False,
    scaling_policy: RelativeScaling | None = None,
    normalized_objective=False,
    objective_multiplier: float | None = None,
    poor_fit: stationary.Policy | None = None,
    reject_invalid_trials: bool = False,
):
    """Joint or projected TRF; optionally shrink after unusable numerical trials.

    Only numerical nonfiniteness is rejectable, and only after a finite starting
    evaluation. TRF receives an infinite residual and no Jacobian for that trial;
    deadlines, domain violations and other errors still terminate the attempt.
    """
    if optimizer_scaling not in {"jacobian", "fixed_coordinates", "guarded_relative"}:
        raise ValueError("unknown optimizer scaling")
    if record_full_steps and not record_steps:
        raise ValueError("full accepted vectors require record_steps")
    multiplier = (
        objective_multiplier
        if objective_multiplier is not None
        else (1.0 if normalized_objective else 1e6)
    )
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("objective multiplier must be finite and positive")
    anchor = oracle.vector(start)
    free = list(range(len(anchor))) if profile is None else list(profile.outer)
    if poor_fit is not None and profile is None:
        free = list(getattr(oracle, "active_indices", free))
    units = oracle.units[free]
    count, best, cached, domain_events, projection = 0, None, None, [], None
    begun, trace, steps = monotonic(), [], []
    accepted = None
    cached_full = accepted_full = None
    scaling = {}
    history, stationarity = [], None
    started = failed = 0
    rejected, invalid_cached = [], None

    class Reached(Exception):
        pass

    class ExactOutput(Exception):
        pass

    class Stationary(Exception):
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
            | full_history_receipt()
            | ({"solver_scaling": scaling} if scaling else {})
            | rejection_receipt()
        )

    def full_history_receipt():
        """Bind retained full-vector steps to their chart and coordinate units."""
        return (
            {
                "accepted_history_protocol": "full-coordinate-accepted-steps-1",
                "history_coordinate_names": list(oracle.names),
                "history_coordinate_units": oracle.units.tolist(),
                "last_accepted_parameters": None
                if accepted_full is None
                else oracle.parameters(accepted_full),
            }
            if record_full_steps
            else {}
        )

    def rejection_receipt():
        """Keep failed starts visible without serializing an infinite penalty."""
        return (
            {
                "calls_started": started,
                "failed_calls": failed,
                "rejected_calls": len(rejected),
                "rejected_trials": rejected,
                "reject_invalid_trials": True,
            }
            if reject_invalid_trials
            else {}
        )

    def evaluate(q):
        nonlocal count, best, cached, projection, accepted, stationarity
        nonlocal started, failed, invalid_cached
        nonlocal cached_full, accepted_full
        if cached is not None and np.array_equal(q, cached[0]):
            return cached[1:]
        if invalid_cached is not None and np.array_equal(q, invalid_cached[0]):
            return invalid_cached[1:]
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
        started += 1
        try:
            if profile is None:
                r, j = oracle.evaluate(vector, deadline)
                j = j[:, free]
            else:
                vector, r, j, projection = profile.evaluate(vector, deadline)
            if reject_invalid_trials and (
                r.ndim != 1
                or not len(r)
                or j.shape != (len(r), len(free))
                or (cached is not None and r.shape != cached[1].shape)
            ):
                raise ValueError("invalid fitted residual or derivative shape")
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(np.mean(r * r))
                scaled_jacobian = j * units
            if not np.isfinite(value) or not np.isfinite(scaled_jacobian).all():
                raise NumericalInvalidTrial("nonfinite fitted residual or derivative")
        except BaseException as error:
            failed += 1
            if not (
                reject_invalid_trials
                and isinstance(error, NumericalInvalidTrial)
                and cached is not None
            ):
                raise
            if monotonic() >= deadline:
                raise TimeoutError("search allowance exhausted") from error
            # A failed trial cannot enter best/history, supply curvature, or
            # overwrite the finite evaluation used by accepted-step callbacks.
            rejected.append(
                {
                    "attempt": count,
                    "parameters_sha256": content_sha256(oracle.parameters(vector)),
                    "error": f"{type(error).__name__}: {error}"[-1000:],
                }
            )
            invalid_cached = (q.copy(), np.full(cached[1].shape, np.inf), None)
            save()
            return invalid_cached[1:]
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
                    "projected_gradient_inf": float(
                        np.max(np.abs(projected), initial=0)
                    ),
                    "active_outer_bounds": {
                        oracle.names[i]: "lower" if lower_active[k] else "upper"
                        for k, i in enumerate(free)
                        if lower_active[k] or upper_active[k]
                    },
                    "parameters_sha256": content_sha256(oracle.parameters(vector)),
                }
            )
        save()
        cached = (q.copy(), r, scaled_jacobian)
        if record_full_steps:
            cached_full = vector.copy()
        if record_steps and accepted is None:
            accepted = cached
            if record_full_steps:
                accepted_full = cached_full.copy()
        if value <= target and (acceptable is None or acceptable(r)):
            raise Reached
        if poor_fit is not None and free:
            history.append(
                stationary.sample(
                    q,
                    vector[free],
                    r,
                    j * units,
                    oracle.lower[free],
                    oracle.upper[free],
                )
            )
            stationarity = stationary.witness(history, poor_fit, target)
            if stationarity:
                stationarity["parameters_sha256"] = content_sha256(
                    oracle.parameters(vector)
                )
                raise Stationary
        return cached[1:]

    def evaluate_jacobian(q):
        """Never provide fabricated curvature for a rejected residual trial."""
        derivative = evaluate(q)[1]
        if derivative is None:
            raise NumericalInvalidTrial("Jacobian requested for rejected trial")
        return derivative * factor

    def callback(intermediate_result):
        """Accepted displacement and local linear prediction; no extra evaluations."""
        nonlocal accepted, accepted_full
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
        if record_full_steps:
            steps[-1].update(
                outer_coordinate_step_norm=steps[-1]["coordinate_step_norm"],
                coordinate_step_norm=float(
                    np.linalg.norm((cached_full - accepted_full) / oracle.units)
                ),
                previous_reduced_vector=accepted_full.tolist(),
                reduced_vector=cached_full.tolist(),
                parameters_sha256=content_sha256(oracle.parameters(cached_full)),
            )
            accepted_full = cached_full.copy()
        accepted = cached
        save()

    try:
        if not free and poor_fit is not None:
            evaluate(np.zeros(0))
            raise ExactOutput
        x_scale = "jac" if optimizer_scaling == "jacobian" else 1.0
        factor = multiplier
        if normalized_objective or optimizer_scaling == "guarded_relative":
            residual, jacobian = evaluate(np.zeros(len(free)))
            if normalized_objective:
                factor /= np.sqrt(len(residual))
            if optimizer_scaling == "guarded_relative":
                scaling = relative_scales(jacobian, scaling_policy or RelativeScaling())
                x_scale = np.asarray(scaling["scales"])
            scaling |= {
                "method": optimizer_scaling,
                "coordinate_units": units.tolist(),
                "mean_objective": normalized_objective,
                "objective_multiplier": multiplier,
                "absolute_gradient_stop_enabled": not normalized_objective,
            }
            save()
        solved = least_squares(
            lambda q: evaluate(q)[0] * factor,
            np.zeros(len(free)),
            jac=evaluate_jacobian,
            bounds=(
                (oracle.lower[free] - anchor[free]) / units,
                (oracle.upper[free] - anchor[free]) / units,
            ),
            x_scale=x_scale,
            ftol=None,
            xtol=1e-12,
            gtol=None if normalized_objective else 1e-10,
            max_nfev=calls,
            **({"callback": callback} if record_steps else {}),
        )
        stop = "step_or_gradient_converged" if solved.success else "budget_limited"
        detail = {"message": solved.message, "optimality": float(solved.optimality)}
    except Reached:
        stop, detail = "numerical_target_reached", {}
    except ExactOutput:
        stop, detail = (
            "output_problem_solved",
            {
                "solution_scope": "bounded_linear_profile"
                if profile
                else "fixed_output",
                "independent_verification_required": True,
            },
        )
    except Stationary:
        stop, detail = "stationary_poor_fit", {"stationarity": stationarity}
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
        **rejection_receipt(),
        **full_history_receipt(),
        **({"solver_scaling": scaling} if scaling else {}),
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
