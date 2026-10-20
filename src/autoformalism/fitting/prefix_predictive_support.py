"""Conditional prediction support for the declared analytical two-root controls.

Only a retained observed prefix and target-free future timestamp grids enter
this diagnostic. The nonnegative-root support is enclosed by an unconstrained
least-squares ellipsoid. Its forecast bound is conservative, conditional on
fitted shared theta and the exact initial observation, and is not a confidence
interval. An unbounded relaxation need not mean the nonnegative support itself
is unbounded. This helper neither selects roots nor writes checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from time import monotonic, process_time

import numpy as np

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multiroot_controls as controls
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem


def _positive_scalar(value: float, name: str) -> float:
    """Validate a finite positive allowance without accepting booleans."""
    parsed = controls._noise_sigma(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _horizons(
    horizon_times: dict[str, list[float]], cutoff: float
) -> dict[str, list[float]]:
    """Copy strictly future, increasing grids, accepting no observation payload."""
    if not isinstance(horizon_times, dict) or not horizon_times:
        raise ValueError("horizon_times must be a nonempty named timestamp mapping")
    result = {}
    for name, times in horizon_times.items():
        if not isinstance(name, str) or not name or not isinstance(times, list):
            raise ValueError("each named horizon must be a list of timestamps")
        if not times or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            for value in times
        ):
            raise ValueError("horizon timestamps must be finite nonempty scalars")
        copied = [float(value) for value in times]
        if copied[0] <= cutoff or any(
            right <= left for left, right in pairwise(copied)
        ):
            raise ValueError("horizons must increase strictly after the prefix cutoff")
        result[name] = copied
    return result


def _deadline(deadline: float) -> None:
    """Cooperatively stop before starting further analytical work."""
    if monotonic() >= deadline:
        raise TimeoutError("predictive-support allowance exhausted")


def _finite(*values: object) -> None:
    """Reject nonfinite intermediate values before a result can be qualified."""
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("nonfinite predictive-support calculation")


def _horizon_bound(
    times: list[float],
    initial_time: float,
    theta: float,
    rates: tuple[float, float],
    scale: float,
    right: np.ndarray,
    singular: np.ndarray,
    rank: int,
    center: np.ndarray,
    retained: np.ndarray,
    radius: float,
    design: np.ndarray,
) -> dict:
    """Bound normalized future RMS, certifying structural null cancellation.

    The leakage tolerance is 64*eps*max(B.shape)*||B||_2. A small leakage
    measurement alone NEVER establishes cancellation: only the declared equal
    rates and identical prefix/future columns certify the exact null direction.
    Other numerically unresolved directions return an unbounded relaxation.
    The noise-support term also contains the bounded-to-unconstrained minimum
    gap; that extra width comes from the conservative domain relaxation.
    """
    elapsed = np.asarray(times) - initial_time
    future = (
        np.column_stack([controls.response(elapsed, rate, theta) for rate in rates])
        / scale
    )
    _finite(future)
    future_norm = float(np.linalg.norm(future, ord=2))
    null = right[rank:].T
    leak = float(np.linalg.norm(future @ null, ord=2)) if null.size else 0.0
    tolerance = 64 * np.finfo(float).eps * max(future.shape) * future_norm
    _finite(future_norm, leak, tolerance)
    structural = bool(
        rank == 1
        and rates[0] == rates[1]
        and np.array_equal(design[:, 0], design[:, 1])
        and np.array_equal(future[:, 0], future[:, 1])
    )
    cancellation = bool(rank == 2 or (structural and leak <= tolerance))
    result = {
        "status": "finite" if cancellation else "unbounded_relaxation",
        "sample_count": len(times),
        "time": times,
        "normalized_rms_upper_bound": None,
        "retained_shift_contribution": None,
        "noise_support_contribution": None,
        "forecast_amplification_operator_norm": None,
        "nullspace_dimension": 2 - rank,
        "nullspace_leak_operator_norm": leak,
        "nullspace_leak_tolerance": tolerance,
        "nullspace_leak_below_tolerance": bool(leak <= tolerance),
        "nullspace_cancellation_certified": cancellation,
        "nullspace_cancellation_basis": (
            "full_column_rank"
            if rank == 2
            else "declared_equal_rates_and_identical_columns"
            if cancellation
            else "not_certified"
        ),
        "unbounded_domain_relaxation_only": not cancellation,
    }
    if not cancellation:
        return result
    inverse = right[:rank].T / singular[:rank]
    amplification = float(np.linalg.norm(future @ inverse, ord=2))
    shift = float(np.linalg.norm(future @ (center - retained))) / np.sqrt(len(times))
    noise = radius * amplification / np.sqrt(len(times))
    bound = shift + noise
    _finite(leak, tolerance, amplification, shift, noise, bound)
    result.update(
        normalized_rms_upper_bound=float(bound),
        retained_shift_contribution=float(shift),
        noise_support_contribution=float(noise),
        forecast_amplification_operator_norm=amplification,
    )
    return result


def predictive_support(
    model: PrefixFitProblem,
    fit: Mapping,
    prefix: kernel.PrefixView,
    inference: Mapping,
    numerical: kernel.NumericalPolicy,
    known_noise_sigma: float,
    horizon_times: dict[str, list[float]],
    *,
    objective_rise_units: float = 4.0,
    maximum_seconds: float = 5.0,
) -> dict:
    """Enclose prefix-supported forecast differences from the retained roots.

    Support is ``xi >= 0, ||A xi-d||^2 <= S_bounded_min + q*(sigma/scale)^2``.
    Write ``c=A^+d`` and ``rho=sqrt(max(0, limit-S_ls_min))``. For each future
    design B the bound is ``(||B(c-xihat)|| + rho*||B V_r/s_r||_2)/sqrt(n)``.
    It is finite only when B cancels every prefix-null direction. Nonnegativity
    is relaxed for this enclosure, never for the conditional reference check.

    The selected endpoint must pass the existing conditional optimum tolerance.
    Sigma and q are declared inputs, not estimated from future observations.
    Timing includes validation/setup; deadlines are cooperative and completed
    intermediate evidence is retained if a later calculation fails or expires.
    The caller owns durable attempt reservations, checkpointing, and resume.
    """
    started, cpu = monotonic(), process_time()
    if (
        not isinstance(model, PrefixFitProblem)
        or not isinstance(prefix, kernel.PrefixView)
        or not isinstance(numerical, kernel.NumericalPolicy)
        or not isinstance(fit, Mapping)
        or not isinstance(inference, Mapping)
    ):
        raise TypeError("typed problem, prefix, numerical policy and mappings required")
    sigma = controls._noise_sigma(known_noise_sigma)
    units = _positive_scalar(objective_rise_units, "objective rise units")
    seconds = _positive_scalar(maximum_seconds, "maximum seconds")
    grids = _horizons(horizon_times, prefix.cutoff)
    controls._inference_identity(model, fit, prefix, inference)
    rates = controls._control_rates(model)
    deadline = started + seconds
    result = {
        "protocol": "prefix-predictive-support-1",
        "available": False,
        "status": "inference_unavailable",
        "problem_sha256": content_sha256(model),
        "fit_sha256": content_sha256(fit),
        "prefix_sha256": content_sha256(prefix),
        "inference_sha256": content_sha256(inference),
        "numerical_policy_sha256": content_sha256(numerical),
        "horizon_times_sha256": content_sha256(grids),
        "budget_seconds": seconds,
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "conditional_reference_calls_started": 0,
        "conditional_reference_calls_completed": 0,
        "horizon_evaluations_started": 0,
        "horizon_evaluations_completed": 0,
        "integration_calls": 0,
        "known_noise_sigma": sigma,
        "objective_rise_units": units,
        "normalized_sse_rise": None,
        "support_normalized_sse_limit": None,
        "bounded_minimum_normalized_sse": None,
        "unconstrained_minimum_normalized_sse": None,
        "domain_relaxation_normalized_sse_gap": None,
        "retained_normalized_sse": None,
        "ellipsoid_radius": None,
        "root_names": list(controls.ROOT_NAMES),
        "root_rank": None,
        "root_ambiguous": None,
        "root_uniqueness_claimed": False,
        "conditional_reference": None,
        "horizons": {
            name: {
                "status": "unavailable",
                "sample_count": len(times),
                "time": times,
                "normalized_rms_upper_bound": None,
            }
            for name, times in grids.items()
        },
        "all_horizons_finite": False,
        "prefix_cutoff": prefix.cutoff,
        "prefix_sample_count": len(prefix.time),
        "noise_observation_count": len(prefix.time) - 1,
        "output_normalization": "frozen_training_channel_scale",
        "conditional_on_frozen_shared_coefficients": True,
        "conditional_on_exact_initial_observation": True,
        "shared_parameter_uncertainty_included": False,
        "initial_observation_uncertainty_included": False,
        "domain_relaxation": "nonnegative_roots_enclosed_by_unconstrained_ellipsoid",
        "confidence_interval": False,
        "calibrated_probability": False,
        "future_observations_used": False,
        "truth_used": False,
        "roots_replaced": False,
    }
    try:
        if not inference.get("fit_success"):
            return result
        _deadline(deadline)
        result["diagnostic_calls_started"] = 1
        result["conditional_reference_calls_started"] = 1
        reference = controls.conditional_linear_optimum(
            model, fit, prefix, inference, numerical
        )
        result["conditional_reference"] = reference
        result["conditional_reference_calls_completed"] = reference[
            "diagnostic_calls_completed"
        ]
        _deadline(deadline)
        if not reference["available"]:
            result["status"] = (
                "budget_exhausted"
                if reference["status"] == "budget_exhausted"
                else "conditional_reference_unavailable"
            )
            return result
        if not reference["objective_optimal"]:
            result["status"] = "retained_objective_not_optimal"
            return result
        oracle, design, target, _ = controls._design(
            model, fit, prefix, inference, numerical
        )
        _deadline(deadline)
        if np.any(oracle.lower != 0) or np.any(oracle.upper != np.inf):
            raise ValueError("predictive support requires declared nonnegative roots")
        scale = float(oracle.scales["y"])
        retained = oracle.vector(inference["roots"])
        _finite(design, target, retained, scale)
        singular, right, rank, tolerance = controls._spectrum(design)
        center = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = design @ center - target
        minimum = float(residual @ residual)
        bounded = float(reference["reference_normalized_mse"] * len(target))
        rise = units * (sigma / scale) ** 2
        limit = bounded + rise
        radius = float(np.sqrt(max(0.0, limit - minimum)))
        _finite(singular, right, center, minimum, bounded, rise, limit, radius)
        result.update(
            training_scale=scale,
            retained_roots=oracle.parameters(retained),
            unconstrained_center_roots=oracle.parameters(center),
            root_rank=rank,
            root_ambiguous=rank < 2,
            singular_values=singular.tolist(),
            right_singular_vectors=right.tolist(),
            rank_tolerance=tolerance,
            normalized_sse_rise=float(rise),
            support_normalized_sse_limit=float(limit),
            bounded_minimum_normalized_sse=bounded,
            unconstrained_minimum_normalized_sse=minimum,
            domain_relaxation_normalized_sse_gap=float(max(0.0, bounded - minimum)),
            retained_normalized_sse=float(
                reference["retained_normalized_mse"] * len(target)
            ),
            ellipsoid_radius=radius,
        )
        _deadline(deadline)
        for name, times in grids.items():
            _deadline(deadline)
            result["horizon_evaluations_started"] += 1
            horizon = _horizon_bound(
                times,
                prefix.time[0],
                oracle.frozen["theta"],
                rates,
                scale,
                right,
                singular,
                rank,
                center,
                retained,
                radius,
                design,
            )
            _deadline(deadline)
            result["horizons"][name] = horizon
            result["horizon_evaluations_completed"] += 1
        result.update(
            available=True,
            status="complete",
            diagnostic_calls_completed=1,
            all_horizons_finite=all(
                value["status"] == "finite" for value in result["horizons"].values()
            ),
        )
    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        TimeoutError,
        np.linalg.LinAlgError,
    ) as error:
        result.update(
            status="budget_exhausted"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            error=str(error),
        )
    finally:
        result["diagnostic_wall_seconds"] = monotonic() - started
        result["diagnostic_cpu_seconds"] = process_time() - cpu
    return result
