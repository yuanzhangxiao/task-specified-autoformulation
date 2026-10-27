"""Opt-in TRF residual trials with Jacobians only at requested optimizer points."""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting import poor_fit as stationary
from autoformalism.fitting.optimizer_scaling import RelativeScaling, relative_scales
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial, restore


@dataclass
class _Point:
    """One finite residual checkpoint; absent derivatives are never fabricated."""

    q: np.ndarray
    vector: np.ndarray
    residual: np.ndarray
    value: float
    attempt: int
    derivative: np.ndarray | None = None
    trace: dict | None = None
    witnessed: bool = False


class _Reached(Exception):
    pass


class _Stationary(Exception):
    pass


def search(
    oracle,
    start: dict,
    deadline: float,
    checkpoint: Callable[[dict], None],
    *,
    calls: int,
    target: float,
    profile=None,
    acceptable=None,
    telemetry: bool = False,
    optimizer_scaling: str = "jacobian",
    record_steps: bool = False,
    record_full_steps: bool = False,
    scaling_policy: RelativeScaling | None = None,
    normalized_objective: bool = False,
    objective_multiplier: float | None = None,
    poor_fit: stationary.Policy | None = None,
    reject_invalid_trials: bool = False,
    residual_only_trials: bool = True,
) -> dict:
    """Search one nonrenewable residual-call/wall allowance with separate J costs.

    ``calls`` retains the eager path's residual-trial meaning. Derivative calls
    have separate attempted/started/completed/failed counters and share exactly
    the same deadline. SciPy TRF requests a Jacobian at the initial and accepted
    points, so rejected finite trials never acquire stationarity evidence.
    ``residual_only_trials=False`` is the matched eager control: it computes
    derivatives for trials but uses the same accepted-point stopping protocol.
    The derivative evaluator must return the exact same deterministic/cached
    baseline residual as the residual-only evaluator. Mixing state-only and
    augmented solves with different residuals needs a separate error contract.
    """
    if profile is not None:
        raise ValueError("lazy Jacobian search does not support projected profiles")
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
    free = list(getattr(oracle, "active_indices", range(len(anchor))))
    if len(set(free)) != len(free) or any(
        isinstance(i, bool)
        or not isinstance(i, (int, np.integer))
        or i < 0
        or i >= len(anchor)
        for i in free
    ):
        raise ValueError("invalid active coordinate indices")
    held = [i for i in range(len(anchor)) if i not in free]
    units = oracle.units[free]
    eager_derivatives = not residual_only_trials and bool(free)
    scope = "active_block" if held else "full_coordinates"
    begun = monotonic()
    count = started = completed = failed = 0
    jacobian_attempts = jacobian_calls = completed_jacobians = failed_jacobians = 0
    best = accepted = None
    shape = None
    cache: dict[tuple, _Point | None] = {}
    trace, steps, history, domain_events, rejected = [], [], [], [], []
    scaling, stationarity = {}, None
    factor = multiplier

    def receipt() -> dict:
        """Expose distinct residual and derivative work, including failed work."""
        return {
            "calls": count,
            "calls_started": started,
            "residual_calls": started,
            "completed_residual_calls": completed,
            "failed_calls": failed,
            "jacobian_attempts": jacobian_attempts,
            "jacobian_calls": jacobian_calls,
            "completed_jacobian_calls": completed_jacobians,
            "failed_jacobian_calls": failed_jacobians,
            "oracle_calls": started + (jacobian_calls if residual_only_trials else 0),
            "call_allowance_scope": "residual_trials",
            "lazy_jacobian": residual_only_trials,
            "best": best,
            "domain_events": domain_events,
            "stationarity_scope": scope,
            "active_coordinate_names": [oracle.names[i] for i in free],
            "held_coordinate_names": [oracle.names[i] for i in held],
            "full_endpoint_audit_required": bool(held),
            **({"solver_scaling": scaling} if scaling else {}),
            **(
                {"completed_calls": completed, "evaluations": trace}
                if telemetry
                else {}
            ),
            **({"accepted_steps": steps} if record_steps else {}),
            **(
                {
                    "accepted_history_protocol": "full-coordinate-accepted-steps-1",
                    "history_coordinate_names": list(oracle.names),
                    "history_coordinate_units": oracle.units.tolist(),
                    "last_accepted_parameters": None
                    if accepted is None
                    else oracle.parameters(accepted.vector),
                }
                if record_full_steps
                else {}
            ),
            **(
                {
                    "reject_invalid_trials": True,
                    "rejected_calls": len(rejected),
                    "rejected_trials": rejected,
                }
                if reject_invalid_trials
                else {}
            ),
        }

    def save() -> None:
        checkpoint(receipt())

    def check_deadline() -> None:
        if monotonic() >= deadline:
            raise TimeoutError("search allowance exhausted")

    def residual_array(raw) -> np.ndarray:
        """Validate residual structure before treating nonfiniteness as a trial."""
        residual = np.asarray(raw, dtype=float)
        if (
            residual.ndim != 1
            or not len(residual)
            or (shape is not None and residual.shape != shape)
        ):
            raise ValueError("invalid fitted residual shape")
        return residual

    def residual(q: np.ndarray) -> np.ndarray:
        """Evaluate and checkpoint a residual without requesting derivatives."""
        nonlocal count, started, completed, failed, best, shape
        nonlocal jacobian_attempts, jacobian_calls, failed_jacobians
        check_deadline()
        key = tuple(q)
        if key in cache:
            point = cache[key]
            return np.full(shape, np.inf) if point is None else point.residual
        if count >= calls:
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
        if eager_derivatives:
            jacobian_attempts += 1
        save()
        check_deadline()
        started += 1
        if eager_derivatives:
            jacobian_calls += 1
        try:
            raw, raw_j = oracle.evaluate(vector, deadline, jacobian=eager_derivatives)
            r = residual_array(raw)
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(np.mean(r * r))
                finite_objective = np.isfinite(r * factor).all()
            if not np.isfinite(value) or not finite_objective:
                raise NumericalInvalidTrial("nonfinite fitted residual")
        except BaseException as error:
            failed += 1
            if eager_derivatives:
                failed_jacobians += 1
            if not (
                reject_invalid_trials
                and shape is not None
                and isinstance(error, NumericalInvalidTrial)
            ):
                raise
            check_deadline()
            rejected.append(
                {
                    "attempt": count,
                    "parameters_sha256": content_sha256(oracle.parameters(vector)),
                    "error": f"{type(error).__name__}: {error}"[-1000:],
                }
            )
            cache[key] = None
            save()
            return np.full(shape, np.inf)
        completed += 1
        shape = r.shape
        point = _Point(q.copy(), vector.copy(), r.copy(), value, count)
        cache[key] = point
        if best is None or value < best["training_nmse"]:
            best = {"parameters": oracle.parameters(vector), "training_nmse": value}
        if telemetry:
            tolerance = 1e-8 * np.maximum(1, np.abs(vector[free]))
            lower_active = vector[free] - oracle.lower[free] <= tolerance
            upper_active = oracle.upper[free] - vector[free] <= tolerance
            point.trace = {
                "attempt": count,
                "elapsed_seconds": monotonic() - begun,
                "training_nmse": value,
                "best_training_nmse": best["training_nmse"],
                "projected_gradient_inf": None,
                "gradient_status": "not_requested",
                "gradient_scope": scope,
                "active_outer_bounds": {
                    oracle.names[i]: "lower" if lower_active[k] else "upper"
                    for k, i in enumerate(free)
                    if lower_active[k] or upper_active[k]
                },
                "parameters_sha256": content_sha256(oracle.parameters(vector)),
            }
            trace.append(point.trace)
        # Save useful output progress before any derivative operation can fail.
        save()
        if eager_derivatives:
            install_derivative(point, point.residual, raw_j)
        check_deadline()
        if value <= target and (acceptable is None or acceptable(r)):
            raise _Reached
        return point.residual

    def accept(point: _Point) -> None:
        """Record a TRF-accepted displacement before its derivative can fail."""
        nonlocal accepted
        if accepted is point:
            return
        if accepted is not None and record_steps:
            if accepted.derivative is None:
                raise ValueError("accepted predecessor has no Jacobian")
            step = point.q - accepted.q
            actual = accepted.value - point.value
            predicted = accepted.value - float(
                np.mean((accepted.residual + accepted.derivative @ step) ** 2)
            )
            item = {
                "evaluation_count": point.attempt,
                "elapsed_seconds": monotonic() - begun,
                "training_nmse": point.value,
                "coordinate_step_norm": float(np.linalg.norm(step)),
                "actual_reduction": actual,
                "linearized_residual_reduction": predicted,
                "reduction_ratio": actual / predicted if predicted > 0 else None,
            }
            if record_full_steps:
                item.update(
                    outer_coordinate_step_norm=item["coordinate_step_norm"],
                    coordinate_step_norm=float(
                        np.linalg.norm((point.vector - accepted.vector) / oracle.units)
                    ),
                    previous_reduced_vector=accepted.vector.tolist(),
                    reduced_vector=point.vector.tolist(),
                    parameters_sha256=content_sha256(oracle.parameters(point.vector)),
                )
            steps.append(item)
        accepted = point
        save()

    def install_derivative(point: _Point, raw_r, raw_j) -> None:
        """Validate actual derivative work independently of optimizer acceptance."""
        nonlocal completed_jacobians, failed_jacobians
        try:
            derivative_r = residual_array(raw_r)
            j = np.asarray(raw_j, dtype=float)
            if j.shape != (len(point.residual), len(anchor)):
                raise ValueError("invalid fitted derivative shape")
            # Held slots may deliberately contain NaN sentinels. They cannot
            # affect finite checks, scaling, gradients or stationarity.
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = j[:, free] * units
                finite_j = np.isfinite(scaled * factor).all()
            if not np.isfinite(derivative_r).all() or not finite_j:
                raise NumericalInvalidTrial("nonfinite fitted residual or derivative")
            if not np.array_equal(derivative_r, point.residual):
                raise ValueError(
                    "derivative evaluation changed the cached residual; "
                    "lazy search requires an identical deterministic baseline"
                )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sample = stationary.sample(
                    point.q,
                    point.vector[free],
                    point.residual,
                    scaled,
                    oracle.lower[free],
                    oracle.upper[free],
                )
            if not np.isfinite([sample["gradient"], sample["correlation"]]).all():
                raise NumericalInvalidTrial("nonfinite fitted gradient or correlation")
        except BaseException:
            failed_jacobians += 1
            if point.trace is not None:
                point.trace["gradient_status"] = "failed"
            raise
        completed_jacobians += 1
        point.derivative = scaled.copy()
        if point.trace is not None:
            point.trace.update(
                projected_gradient_inf=sample["gradient"],
                gradient_status="available",
            )
        save()
        check_deadline()

    def jacobian(q: np.ndarray) -> np.ndarray:
        """Supply real active columns only; a residual cache is never curvature."""
        nonlocal jacobian_attempts, jacobian_calls, failed_jacobians, stationarity
        residual(q)
        point = cache[tuple(q)]
        if point is None:
            raise NumericalInvalidTrial("Jacobian requested for rejected trial")
        # TRF asks jac(x_new) only after it accepts x_new, before its callback.
        accept(point)
        if point.derivative is None:
            jacobian_attempts += 1
            if point.trace is not None:
                point.trace["gradient_status"] = "requested"
            save()
            check_deadline()
            jacobian_calls += 1
            try:
                raw_r, raw_j = oracle.evaluate(
                    point.vector.copy(), deadline, jacobian=True
                )
            except BaseException:
                failed_jacobians += 1
                if point.trace is not None:
                    point.trace["gradient_status"] = "failed"
                raise
            install_derivative(point, raw_r, raw_j)
        if poor_fit is not None and free and not point.witnessed:
            sample = stationary.sample(
                q,
                point.vector[free],
                point.residual,
                point.derivative,
                oracle.lower[free],
                oracle.upper[free],
            )
            point.witnessed = True
            history.append(sample)
            stationarity = stationary.witness(history, poor_fit, target)
            if stationarity:
                stationarity.update(
                    parameters_sha256=content_sha256(oracle.parameters(point.vector)),
                    scope=scope,
                    full_model_stationarity_certified=False,
                )
                raise _Stationary
        return point.derivative * factor

    def callback(intermediate_result) -> None:
        """Validate accepted-point history without doing another rollout."""
        check_deadline()
        point = cache.get(tuple(intermediate_result.x))
        if point is None or point.derivative is None:
            raise ValueError("accepted optimizer point missing from derivative cache")
        accept(point)

    try:
        initial = np.zeros(len(free))
        if not free:
            residual(initial)
            accept(cache[tuple(initial)])
            stop, detail = (
                "output_problem_solved",
                {
                    "solution_scope": "fixed_active_block" if held else "fixed_output",
                    "independent_verification_required": True,
                },
            )
        else:
            x_scale = "jac" if optimizer_scaling == "jacobian" else 1.0
            if normalized_objective or optimizer_scaling == "guarded_relative":
                r = residual(initial)
                if normalized_objective:
                    factor /= np.sqrt(len(r))
                if optimizer_scaling == "guarded_relative":
                    scaled_j = jacobian(initial) / factor
                    scaling = relative_scales(
                        scaled_j, scaling_policy or RelativeScaling()
                    )
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
                lambda q: residual(q) * factor,
                initial,
                jac=jacobian,
                bounds=(
                    (oracle.lower[free] - anchor[free]) / units,
                    (oracle.upper[free] - anchor[free]) / units,
                ),
                x_scale=x_scale,
                ftol=None,
                xtol=1e-12,
                gtol=None if normalized_objective else 1e-10,
                max_nfev=calls,
                callback=callback,
            )
            stop = "step_or_gradient_converged" if solved.success else "budget_limited"
            detail = {"message": solved.message, "optimality": float(solved.optimality)}
    except _Reached:
        stop, detail = "numerical_target_reached", {}
    except _Stationary:
        stop = "active_block_stationary_poor_fit" if held else "stationary_poor_fit"
        detail = {"stationarity": stationarity}
    except (ValueError, TimeoutError, RuntimeError, ArithmeticError) as error:
        stop = (
            "budget_limited" if isinstance(error, TimeoutError) else "numerical_failure"
        )
        detail = {"message": str(error)[-1000:]}
    save()
    return receipt() | {
        "stop_reason": stop,
        "projection": None,
        **detail,
        **({"optimizer_scaling": optimizer_scaling} if record_steps else {}),
    }
