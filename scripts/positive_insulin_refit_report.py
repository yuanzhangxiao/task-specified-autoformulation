"""Compare training-frozen positive-insulin fits within each restoring parent."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scripts import positive_insulin_refit_inputs as inputs
from scripts.transfer_insulin_order_law import paired_results

STAGES = ("start", "fixed_insulin", "joint")
TAGS = inputs.TAGS
TOLERANCE = 1.0


def _insulin_metrics(event: dict) -> dict:
    """Measure timing, height and worst error on the recorded time grid."""
    time = np.asarray(event["time"], dtype=float)
    observed = np.asarray(event["observed"]["I"], dtype=float)
    predicted = np.asarray(event["predicted"]["I"], dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or observed.shape != time.shape
        or predicted.shape != time.shape
        or not np.isfinite([time, predicted, observed]).all()
        or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Invalid time or insulin series in scored trajectory")
    obs_peak, pred_peak = int(observed.argmax()), int(predicted.argmax())
    errors = abs(observed - predicted)
    worst = int(errors.argmax())
    return {
        "observed_peak_time_min": float(time[obs_peak]),
        "predicted_peak_time_min": float(time[pred_peak]),
        "peak_time_error_min": float(time[pred_peak] - time[obs_peak]),
        "observed_peak_pmol_L": float(observed[obs_peak]),
        "predicted_peak_pmol_L": float(predicted[pred_peak]),
        "peak_height_error_pmol_L": float(predicted[pred_peak] - observed[obs_peak]),
        "observed_peak_at_boundary": obs_peak in (0, len(time) - 1),
        "predicted_peak_at_boundary": pred_peak in (0, len(time) - 1),
        "minimum_predicted_pmol_L": float(predicted.min()),
        "max_abs_error_pmol_L": float(errors[worst]),
        "worst_error_time_min": float(time[worst]),
        "samples_above_1_pmol_L": int(np.count_nonzero(errors > TOLERANCE)),
        "within_1_pmol_L": bool(errors[worst] <= TOLERANCE),
    }


def _score_stage(
    replay: Callable, model: dict, selection: dict, rows: list[dict]
) -> dict:
    fitted = inputs.with_parameters(model, selection["parameters"])
    result = replay(fitted, rows)
    events = [dict(e) for e in result["events"]]
    if sorted(e["event"] for e in events) != sorted(r["trajectory_id"] for r in rows):
        raise ValueError("Replay did not account for every supplied trajectory")
    stats = {}
    for e in events:
        if e["status"] == "scored":
            e["insulin_behavior"] = _insulin_metrics(e)
            stats[e["event"]] = e["insulin_behavior"]
    complete = len(stats) == len(events)
    passes = sum(m["within_1_pmol_L"] for m in stats.values())
    uniform_pass = passes == len(events) if complete or passes < len(stats) else None
    worst = max(stats, key=lambda k: stats[k]["max_abs_error_pmol_L"], default=None)
    summary = result["summary"] | {
        "per_target_nmse": (
            result["summary"]["available_per_target_nmse"] if complete else None
        ),
        "median_abs_peak_time_error_min": (
            float(np.median([abs(m["peak_time_error_min"]) for m in stats.values()]))
            if complete
            else None
        ),
        "minimum_predicted_I_pmol_L": (
            min(m["minimum_predicted_pmol_L"] for m in stats.values())
            if complete
            else None
        ),
        "worst_I_event": worst if complete else None,
        "max_abs_I_error_pmol_L": (
            stats[worst]["max_abs_error_pmol_L"] if complete else None
        ),
        "events_within_1_pmol_L": passes,
        "all_events_within_1_pmol_L": uniform_pass,
        "events_with_insulin_peak": len(stats),
    }
    return {
        "events": events,
        "summary": summary,
        "diagnostics": selection["diagnostics"],
    }


def _paired(parent: dict, child: dict) -> dict:
    """Count matched events and only difference whole-cohort scores when complete."""
    comparison = paired_results(parent["events"], child["events"])
    a, b = parent["summary"], child["summary"]
    differences = {}
    if a["pooled_nmse"] is not None and b["pooled_nmse"] is not None:
        for name in (
            "pooled_nmse",
            "max_abs_I_error_pmol_L",
            "median_abs_peak_time_error_min",
        ):
            differences[name] = b[name] - a[name]
        differences["per_target_nmse"] = {
            k: b["per_target_nmse"][k] - a["per_target_nmse"][k] for k in inputs.TARGETS
        }
    return comparison | {"child_minus_parent": differences or None}


def report(
    seal: dict, models: dict, rows: list[dict], replay: Callable, split: str
) -> dict:
    """Replay all three frozen stages and summarize matched slot/law differences."""
    if set(models) != set(TAGS) or set(seal["models"]) != set(TAGS):
        raise ValueError("Expected all four frozen models")
    if split not in ("train", "val"):
        raise ValueError("Only training and development validation are supported")
    expected = {f"{split}_{i:03d}" for i in range(16 if split == "train" else 4)}
    if len(rows) != len(expected) or {r["trajectory_id"] for r in rows} != expected:
        raise ValueError("Expected the entire registered split")
    for tag in TAGS:
        stages = seal["models"][tag]
        source_parameters = models[tag]["formula"]["parameters"]
        if (
            set(stages) != set(STAGES)
            or stages["start"]["parameters"] != source_parameters
        ):
            raise ValueError(f"Missing fitting stage or changed source start for {tag}")
        for stage in stages.values():
            parameters = stage["parameters"]
            if parameters.keys() != source_parameters.keys() or not all(
                np.isfinite(v) for v in parameters.values()
            ):
                raise ValueError(f"Incomplete or nonfinite parameter vector for {tag}")
    for slot in ("002", "015"):
        if (
            models[f"{slot}-restoring"]["scales"]
            != models[f"{slot}-positive_EGP"]["scales"]
        ):
            raise ValueError(f"Paired normalization scales differ for slot {slot}")
    scores = {
        tag: {
            stage: _score_stage(replay, models[tag], seal["models"][tag][stage], rows)
            for stage in STAGES
        }
        for tag in TAGS
    }
    paired = {
        slot: {
            stage: _paired(
                scores[f"{slot}-restoring"][stage],
                scores[f"{slot}-positive_EGP"][stage],
            )
            for stage in STAGES
        }
        for slot in ("002", "015")
    }
    return {
        "fit_identity": seal["identity"],
        "split": split,
        "validation_previously_used_for_development": split == "val",
        "models": scores,
        "paired": paired,
        "insulin_criteria": {
            "peak": "Earliest sampled event maximum, including both endpoints.",
            "minimum": "Lowest predicted insulin on the observation grid.",
            "pointwise": "Every recorded absolute insulin error at or below 1 pmol/L.",
        },
    }
