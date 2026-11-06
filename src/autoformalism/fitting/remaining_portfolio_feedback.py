"""Descriptive training mismatches at one frozen, independently verified fit.

These records locate observed errors. They do not diagnose a scientific cause,
certify optimization adequacy, or prescribe a change to the equations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "remaining-portfolio-training-mismatches-1"
MEAL_INPUT = "meal_rate_g_per_min"


def _windows(inputs: Mapping, count: int) -> dict[str, np.ndarray]:
    """Partition sampled times by the supplied ingestion rate and its history."""
    if MEAL_INPUT not in inputs:
        return {"unclassified": np.ones(count, dtype=bool)}
    rate = np.asarray(inputs[MEAL_INPUT], dtype=float)
    if rate.shape != (count,) or not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError("nonnegative supplied ingestion rate required")
    positive = rate > 0
    if not positive.any():
        return {"no_meal_schedule": np.ones(count, dtype=bool)}
    occurred = np.maximum.accumulate(positive)
    return {
        "pre_meal": ~occurred,
        "ingestion": positive,
        "post_ingestion": occurred & ~positive,
    }


def _metrics(error: np.ndarray, scale: float) -> dict:
    return {
        "samples": len(error),
        "normalized_mse": float(np.mean((error / scale) ** 2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_prediction_minus_observation": float(np.mean(error)),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def summarize_residuals(
    training: PublicSplit,
    channels: Sequence[str],
    scales: Mapping[str, float],
    residual: np.ndarray,
) -> dict:
    """Decode the oracle's row/time/channel ordering without another model fit."""
    if training.name != "train":
        raise ValueError("feedback accepts public training only")
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("distinct output channels required")
    if set(scales) != set(channels) or any(
        not np.isfinite(value) or value <= 0 for value in scales.values()
    ):
        raise ValueError("positive frozen channel scales required")
    if any(set(row.targets) != set(channels) for row in training.rows):
        raise ValueError("output channels differ from the supplied target contract")
    residual = np.asarray(residual, dtype=float)
    count = sum(len(row.time) for row in training.rows) * len(channels)
    if residual.shape != (count,) or not np.isfinite(residual).all():
        raise ValueError("finite residual with exact row/time/channel shape required")
    columns = residual.reshape(-1, len(channels))
    rows, offset = [], 0
    for row in training.rows:
        length = len(row.time)
        block = columns[offset : offset + length]
        offset += length
        time = np.asarray(row.time, dtype=float)
        windows = _windows(row.external_inputs, length)
        targets = {}
        for index, channel in enumerate(channels):
            observed = np.asarray(row.targets[channel], dtype=float)
            error = block[:, index] * scales[channel]
            predicted = observed + error
            if not np.isfinite(predicted).all():
                raise ValueError("nonfinite reconstructed prediction")
            actual_peak, fitted_peak = (
                int(np.argmax(observed)),
                int(np.argmax(predicted)),
            )
            targets[channel] = {
                **_metrics(error, scales[channel]),
                "initial_error": float(error[0]),
                "final_error": float(error[-1]),
                "sampled_peak": {
                    "observation_time": float(time[actual_peak]),
                    "prediction_time": float(time[fitted_peak]),
                    "time_difference": float(time[fitted_peak] - time[actual_peak]),
                    "observation_value": float(observed[actual_peak]),
                    "prediction_value": float(predicted[fitted_peak]),
                    "observation_plateau_samples": int(
                        np.sum(observed == observed[actual_peak])
                    ),
                    "prediction_plateau_samples": int(
                        np.sum(predicted == predicted[fitted_peak])
                    ),
                    "observation_at_boundary": actual_peak in (0, length - 1),
                    "prediction_at_boundary": fitted_peak in (0, length - 1),
                    "rule": (
                        "first maximum among supplied sample times; "
                        "no peak interpolation"
                    ),
                },
                "windows": {
                    name: _metrics(error[mask], scales[channel])
                    for name, mask in windows.items()
                    if mask.any()
                },
                "prediction": predicted.tolist(),
            }
        rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "time": list(row.time),
                "targets": targets,
            }
        )
    return {
        "training_nmse": float(np.mean(residual**2)),
        "per_target_normalized_mse": dict(
            zip(channels, np.mean(columns**2, axis=0).tolist(), strict=True)
        ),
        "training_scales": dict(scales),
        "rows": rows,
        "metric_weighting": (
            "equal supplied samples and channels; global training channel scales"
        ),
        "window_rule": (
            "positive meal rate, before first positive rate, or zero rate after "
            "a positive rate; all-zero meal schedules separate"
        ),
        "window_interpretation": (
            "sampled input phases; post_ingestion may include intervals between meals"
        ),
        "initialization": "full-training fitted roots; not prefix-only forecasting",
        "global_minimum_certified": False,
        "parameter_confidence_certified": False,
        "scientific_cause_identified": False,
        "automatic_revision": False,
        "test_data_opened": False,
    }


def describe_endpoint(
    original,
    endpoint: dict,
    seconds: float,
    checkpoint: Callable[[dict], None],
) -> dict:
    """Replay original equations once and describe only the frozen verified fit."""
    if not endpoint.get("original_equations_verified"):
        raise ValueError("independently verified endpoint required")
    if isinstance(seconds, bool) or not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("bounded positive feedback allowance required")
    budget = execution._budget(seconds, seconds + 60)
    result = {
        "protocol": PROTOCOL,
        "passed": False,
        "endpoint_sha256": content_sha256(endpoint),
        "parameters_sha256": content_sha256(endpoint["parameters"]),
        "training_sha256": content_sha256(original.training),
        "calls_started": 0,
        "calls_completed": 0,
    }
    interrupted = None
    try:
        point = original.vector(endpoint["parameters"])

        def evaluate(deadline):
            result["calls_started"] += 1
            residual, _ = original.evaluate(
                point, deadline, jacobian=False, method="Radau"
            )
            result["calls_completed"] += 1
            return residual

        residual = budget.numerical(evaluate)
        budget.check()
        evidence = summarize_residuals(
            original.training, original.system.channels, original.scales, residual
        )
        reference = float(endpoint["training_nmse"])
        tolerance = max(1e-12, 1e-8 * max(abs(reference), evidence["training_nmse"]))
        if (
            not np.isfinite(reference)
            or abs(evidence["training_nmse"] - reference) > tolerance
        ):
            raise ValueError("feedback replay disagrees with retained endpoint loss")
        result.update(evidence, passed=True, replay_loss_tolerance=tolerance)
    except Exception as error:
        continuation._invalidate(result, error, "feedback")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "feedback")
    result["calls_failed"] = result["calls_started"] - result["calls_completed"]
    return continuation._finish(result, budget, checkpoint, "feedback", interrupted)
