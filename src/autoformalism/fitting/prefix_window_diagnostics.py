"""Common forecast horizons and conditional information from a sealed prefix.

Scoring is evaluator-only. The information helper receives no held-out suffix,
performs one bounded sensitivity evaluation, and never estimates or selects a
parameter. Its noise-unit scale is a local sensitivity description, not a
confidence interval or a statement about globally identifiable initial states.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import monotonic

import numpy as np

from autoformalism.fitting.prefix_initialization import (
    Checkpoint,
    NumericalPolicy,
    PrefixOracle,
    PrefixView,
    _rank,
)
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicTrajectory


def common_horizon_score(
    predictions: Mapping,
    heldout: PublicTrajectory,
    score_start: float,
    scales: Mapping[str, float],
) -> dict:
    """Score strictly after one common start without changing the forecast receipt."""
    if (
        predictions["trajectory_id"] != heldout.trajectory_id
        or tuple(predictions["time"]) != heldout.time
    ):
        raise ValueError("forecast evaluation trajectory/time differs")
    cutoff = predictions["cutoff"]
    if (
        not np.isfinite(score_start)
        or not np.isfinite(cutoff)
        or score_start < cutoff
        or score_start >= heldout.time[-1]
    ):
        raise ValueError(
            "common scoring start must follow cutoff and precede last time"
        )
    if set(scales) != set(heldout.targets) or any(
        not np.isfinite(value) or value <= 0 for value in scales.values()
    ):
        raise ValueError("invalid frozen training scales")
    mask = np.asarray(heldout.time) > score_start
    result = {
        "protocol": "prefix-common-horizon-score-1",
        "available": False,
        "normalized_mse": None,
        "per_target_normalized_mse": {},
        "sample_count": 0,
        "horizon_sample_count": int(mask.sum()),
        "forecast_cutoff": cutoff,
        "score_start": float(score_start),
        "scoring_rule": "strictly_after_common_score_start",
        "forecast_sha256": content_sha256(predictions),
        "inference_sha256": predictions.get("inference_sha256"),
        "schedule_sha256": predictions.get("schedule_sha256"),
    }
    if not predictions.get("forecast_success"):
        return result
    outputs = predictions["predictions"]
    if set(outputs) != set(heldout.targets):
        raise ValueError("forecast evaluation channels differ")
    scores = {}
    for name, target in heldout.targets.items():
        values = np.asarray(outputs[name], float)
        if values.shape != (len(target),) or not np.isfinite(values).all():
            raise ValueError("invalid forecast values")
        scores[name] = float(
            np.mean(((values[mask] - np.asarray(target)[mask]) / scales[name]) ** 2)
        )
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("nonfinite forecast score")
    return result | {
        "available": True,
        "normalized_mse": float(np.mean(list(scores.values()))),
        "per_target_normalized_mse": scores,
        "sample_count": int(mask.sum()),
    }


def _inference_identity(problem, fit_result, prefix, inference) -> None:
    """Reject a mismatched retained point before spending diagnostic work."""
    expected = {
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
    }
    if any(inference.get(name) != value for name, value in expected.items()):
        raise ValueError("prefix information inference identity differs")
    if inference.get("shared_parameters") != fit_result.get(
        "shared_parameters"
    ) or inference.get("training_scales") != fit_result.get("training_scales"):
        raise ValueError("prefix information frozen shared values/scales differ")


def _noise_scale(
    derivative: np.ndarray, channels: tuple[str, ...], scales: Mapping, sigma: float
) -> tuple[float | None, str]:
    """Describe a one-root, one-output response in one declared noise unit."""
    if derivative.shape[1] != 1:
        return None, "not_one_root"
    if len(channels) != 1:
        return None, "not_one_output"
    norm = float(np.linalg.norm(derivative[:, 0]))
    if norm == 0:
        return None, "uninformative"
    if sigma == 0:
        return None, "noiseless"
    value = (sigma / scales[channels[0]]) / norm
    if not np.isfinite(value):
        return None, "numerically_unavailable"
    return float(value), "available"


def prefix_information(
    problem: PrefixFitProblem,
    fit_result: Mapping,
    prefix: PrefixView,
    inference: Mapping,
    numerical: NumericalPolicy,
    known_noise_sigma: float,
    checkpoint: Checkpoint | None = None,
) -> dict:
    """Evaluate native root sensitivities using only the retained observed prefix.

    ``numerical.verification_seconds`` bounds this separate diagnostic operation.
    The started call is published before integration. The caller owns durable
    replay and retains an interrupted call's charge; this function has no retry.
    """
    if not isinstance(prefix, PrefixView) or not isinstance(numerical, NumericalPolicy):
        raise TypeError("typed prefix and numerical policy required")
    if (
        isinstance(known_noise_sigma, bool)
        or not isinstance(known_noise_sigma, (int, float, np.integer, np.floating))
        or not np.isfinite(known_noise_sigma)
        or known_noise_sigma < 0
    ):
        raise ValueError("known noise sigma must be a finite nonnegative scalar")
    _inference_identity(problem, fit_result, prefix, inference)
    result = {
        "protocol": "prefix-local-information-1",
        "available": False,
        "status": "inference_unavailable",
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "inference_sha256": content_sha256(inference),
        "numerical_policy_sha256": content_sha256(numerical),
        "budget_seconds": numerical.verification_seconds,
        "prefix_cutoff": prefix.cutoff,
        "prefix_last_observation_time": prefix.time[-1],
        "known_noise_sigma": float(known_noise_sigma),
        "root_names": [],
        "root_values": None,
        "singular_values": None,
        "physical_output_singular_values": None,
        "rank": None,
        "coordinate_system": "native_physical_roots",
        "output_normalization": "frozen_training_channel_scales",
        "rank_observation_scope": "complete_prefix_including_exact_initial_boundary",
        "noise_scale_observation_scope": "prefix_samples_strictly_after_initial_time",
        "noise_scale_residual_count": None,
        "one_noise_unit_root_scale": None,
        "scale_status": "unavailable",
        "conditional_on_frozen_shared_coefficients": True,
        "confidence_interval": False,
        "scope": "local prefix sensitivity; no uniqueness or confidence claim",
    }
    save = checkpoint or (lambda value: None)
    if not inference.get("fit_success"):
        save(deepcopy(result))
        return result
    started = monotonic()
    deadline = started + numerical.verification_seconds
    try:
        oracle = PrefixOracle(
            problem, fit_result, prefix, numerical, root_guesses=inference["roots"]
        )
        vector = oracle.vector(inference["roots"])
        result.update(
            root_names=list(oracle.names), root_values=oracle.parameters(vector)
        )
        if monotonic() >= deadline:
            raise TimeoutError("prefix diagnostic allowance exhausted during setup")
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result.update(
            status="budget_exhausted"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            error=str(error),
            diagnostic_wall_seconds=monotonic() - started,
        )
        save(deepcopy(result))
        return result
    result["diagnostic_calls_started"] = 1
    save(deepcopy(result))
    try:
        residual, derivative = oracle.evaluate(vector, deadline)
        result["diagnostic_calls_completed"] = 1
        if (
            derivative.shape != (len(residual), len(oracle.names))
            or not np.isfinite(derivative).all()
        ):
            raise ValueError("prefix root sensitivity is nonfinite or has wrong shape")
        rank = _rank(derivative, np.ones(len(oracle.names)))
        row_scales = np.tile(
            [oracle.scales[name] for name in oracle.system.channels], len(prefix.time)
        )
        physical = np.linalg.svd(derivative * row_scales[:, None], compute_uv=False)
        noise_derivative = derivative[len(oracle.system.channels) :]
        scale, scale_status = _noise_scale(
            noise_derivative,
            oracle.system.channels,
            oracle.scales,
            float(known_noise_sigma),
        )
        if monotonic() >= deadline:
            raise TimeoutError("prefix diagnostic allowance exhausted")
        result.update(
            available=True,
            status="complete",
            singular_values=rank["singular_values"],
            physical_output_singular_values=physical.tolist(),
            rank=rank["rank"],
            rank_tolerance=rank["tolerance"],
            full_column_rank=rank["full_column_rank"],
            residual_count=len(residual),
            observation_times=len(prefix.time),
            noise_scale_residual_count=len(noise_derivative),
            one_noise_unit_root_scale=scale,
            scale_status=scale_status,
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
    result["diagnostic_wall_seconds"] = monotonic() - started
    save(deepcopy(result))
    return result
