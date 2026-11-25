"""Nominate complete nonnegative, threshold-activated laws from interval means.

Callers supply the signed source displacement from a known activation threshold
and interval-mean rates from independent training balances. Interpolation of
the source between observed endpoints is linear. Each event carries equal
weight in both fitting and leave-one-training-event-out scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

EXPRESSIONS = {
    "square_root": "gain * sqrt(max(x, 0))",
    "linear": "gain * max(x, 0)",
    "power": "amplitude * (max(x, 0) / source_scale) ** exponent",
    "saturating": "maximum * max(x, 0) / (half_saturation + max(x, 0))",
}
POWER_BOUNDS = (0.25, 4.0)
HALF_SATURATION_RATIO_BOUNDS = (0.001, 1000.0)


def _finite(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Each source, interval mean and prediction must be finite.")
    return array


def interval_mean_law(candidate: dict, left, right) -> float | np.ndarray:
    """Mean complete rate along a linear source segment, including activation."""
    a, b = np.broadcast_arrays(_finite(left), _finite(right))
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    wet_low = np.maximum(low, 0)
    wet_high = np.maximum(high, 0)
    wet_width = wet_high - wet_low
    # The segment can enter or leave the active domain, or stay at a point.
    fraction = np.divide(
        wet_width, high - low, out=np.array(high > 0, dtype=float), where=high != low
    )
    family = candidate["family"]
    params = candidate["parameters"]
    if family in {"square_root", "linear", "power"}:
        if family == "power":
            exponent = params["exponent"]
            end_rate = params["amplitude"] * (wet_high / params["source_scale"]) ** (
                exponent
            )
        else:
            exponent = 0.5 if family == "square_root" else 1.0
            end_rate = params["gain"] * wet_high**exponent
        ratio = np.divide(
            wet_width, wet_high, out=np.zeros_like(wet_high), where=wet_high > 0
        )
        # Integral of x**p from wet_low to wet_high, divided by wet_width.
        # At a point the factor is one; from zero it is exactly 1/(p+1).
        factor = np.ones_like(ratio)
        factor[ratio == 1] = 1.0 / (exponent + 1.0)
        partial = (ratio > 0) & (ratio < 1)
        t = ratio[partial]
        factor[partial] = -np.expm1((exponent + 1) * np.log1p(-t)) / (
            (exponent + 1) * t
        )
        mean = fraction * end_rate * factor
    elif family == "saturating":
        half = params["half_saturation"]
        base = half + wet_low
        t = wet_width / base
        # Integrating x/(half+x) produces 1-log1p(t)/t. The expansion
        # avoids cancellation for nearby observations, including a flat head.
        correction = np.empty_like(t)
        small = t < 1e-4
        s = t[small]
        correction[small] = s * (0.5 - s / 3 + s**2 / 4)
        correction[~small] = 1 - np.log1p(t[~small]) / t[~small]
        mean = (
            fraction * params["maximum"] * (wet_low / base + half / base * correction)
        )
    else:
        raise ValueError(f"Unknown law family {family!r}.")
    result = _finite(mean)
    return float(result) if result.ndim == 0 else result


def predict_law(candidate: dict, x) -> float | np.ndarray:
    """Evaluate the complete instantaneous rate, for use in simulation."""
    wet = np.maximum(_finite(x), 0)
    params = candidate["parameters"]
    family = candidate["family"]
    if family == "square_root":
        rate = params["gain"] * np.sqrt(wet)
    elif family == "linear":
        rate = params["gain"] * wet
    elif family == "power":
        rate = (
            params["amplitude"] * (wet / params["source_scale"]) ** params["exponent"]
        )
    elif family == "saturating":
        rate = params["maximum"] * (wet / (params["half_saturation"] + wet))
    else:
        raise ValueError(f"Unknown law family {family!r}.")
    result = _finite(rate)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class _Event:
    name: str
    left: np.ndarray
    right: np.ndarray
    rate: np.ndarray
    error_bound: np.ndarray


def _event(row: dict) -> _Event:
    name = row["trajectory_id"]
    if not isinstance(name, str) or not name.startswith("train_"):
        raise ValueError("Law nomination accepts identified training events only.")
    arrays = [_finite(row[key]) for key in ("left", "right", "mean_rate")]
    if (
        arrays[0].ndim != 1
        or not arrays[0].size
        or any(a.shape != arrays[0].shape for a in arrays)
    ):
        raise ValueError(
            "Supply equally sized nonempty one-dimensional interval arrays."
        )
    supplied = np.asarray(row.get("absolute_error_bound"), dtype=object)
    if supplied.ndim and supplied.shape != arrays[0].shape:
        raise ValueError("Supply one error bound or a bound per interval.")
    bounds = np.full(arrays[0].shape, np.nan)
    for i, value in enumerate(np.broadcast_to(supplied, bounds.shape)):
        if value is not None:
            bound = float(value)
            if not np.isfinite(bound) or bound < 0:
                raise ValueError(
                    "An absolute error bound must be finite and nonnegative."
                )
            bounds[i] = bound
    return _Event(name, *arrays, bounds)


def _shape_check(events: Sequence[_Event]) -> dict:
    """Find a reversed pair of ordered source intervals in O(n log n) time."""
    report = {
        "status": "no_witness",
        "assumptions": [
            "One shared nondecreasing function of the signed source alone.",
            "Linear source interpolation keeps values within known endpoint ranges.",
            "Supplied bounds cover total absolute error in each interval-mean rate.",
        ],
    }
    if not events:
        return report
    starts = np.concatenate([np.minimum(ev.left, ev.right) for ev in events])
    ends = np.concatenate([np.maximum(ev.left, ev.right) for ev in events])
    rate = np.concatenate([ev.rate for ev in events])
    error = np.concatenate([ev.error_bound for ev in events])
    known = np.isfinite(error)
    delta = np.where(known, error, 0.0)
    # Round outwards before comparing. Unknown bounds count as zero solely
    # for advisory comparisons; the certified search excludes those intervals.
    guard = 8 * np.finfo(float).eps * np.maximum(np.abs(rate), delta)
    lower_rate = _finite(rate - delta - guard)
    upper_rate = _finite(rate + delta + guard)

    def strongest(lower_rows, higher_rows) -> tuple[float, int, int]:
        by_end = sorted(lower_rows, key=ends.__getitem__)
        pos = 0
        highest = None
        best = (0.0, -1, -1)
        for j in sorted(higher_rows, key=starts.__getitem__):
            while pos < len(by_end) and ends[by_end[pos]] <= starts[j]:
                i = by_end[pos]
                if highest is None or lower_rate[i] > lower_rate[highest]:
                    highest = i
                pos += 1
            if highest is not None:
                gap = lower_rate[highest] - upper_rate[j]
                if gap > best[0]:
                    best = (float(gap), int(highest), int(j))
        return best

    bounded = np.flatnonzero(known)
    unknown = np.flatnonzero(~known)
    gap, i, j = strongest(bounded, bounded)
    if gap > 0:
        status = "contradicted"
    else:
        # An advisory pair must have at least one missing bound. A measured
        # reversal fully covered by two supplied bounds is compatible.
        gap, i, j = max(
            strongest(unknown, np.arange(len(rate))), strongest(bounded, unknown)
        )
        if gap <= 0:
            return report
        status = "advisory"
    references = [(ev.name, k) for ev in events for k in range(ev.rate.size)]

    def interval(k):
        name, index = references[k]
        return {
            "trajectory_id": name,
            "interval_index": index,
            "source_range": [float(starts[k]), float(ends[k])],
            "mean_rate": float(rate[k]),
            "absolute_error_bound": float(error[k]) if known[k] else None,
        }

    measured_gap = float(rate[i] - rate[j])
    return {
        **report,
        "status": status,
        "lower_source_interval": interval(i),
        "higher_source_interval": interval(j),
        "observed_rate_gap": measured_gap,
        "gap_after_supplied_bounds": float(measured_gap - delta[i] - delta[j]),
        "gap_after_roundoff_guard": gap,
    }


def _excited(events: Sequence[_Event]) -> bool:
    """Require distinct active intervals to distinguish amplitude from shape."""
    if not events:
        return False
    left = np.concatenate([ev.left for ev in events])
    right = np.concatenate([ev.right for ev in events])
    rates = np.concatenate([ev.rate for ev in events])
    # Two intervals differing only in time below the threshold have the same
    # active range and cannot distinguish exponents with a free amplitude.
    ends = np.sort(np.maximum(0, np.column_stack((left, right))), axis=1)
    wet = ends[:, 1] > 0
    if np.sum(wet) < 2 or not np.any(rates[wet] > 0):
        return False
    levels = ends[wet] / np.max(ends[wet])
    return bool(np.max(np.ptp(levels, axis=0)) > 1e-8)


def _fit(family: str, events: Sequence[_Event]) -> dict:
    """Profile the positive amplitude analytically for each shape parameter."""
    left = np.concatenate([ev.left for ev in events])
    right = np.concatenate([ev.right for ev in events])
    y = np.concatenate([ev.rate for ev in events])
    weights = np.concatenate([np.full(ev.rate.size, 1 / ev.rate.size) for ev in events])
    scale = float(max(np.max(left), np.max(right)))

    def profile(shape: float) -> tuple[float, dict]:
        if family == "power":
            params = {"source_scale": scale, "exponent": shape}
            amplitude = "amplitude"
        elif family == "saturating":
            params = {"half_saturation": float(scale * np.exp(shape))}
            amplitude = "maximum"
        else:
            params = {}
            amplitude = "gain"
        params[amplitude] = 1.0
        law = {
            "family": family,
            "expression": EXPRESSIONS[family],
            "parameters": params,
        }
        basis = interval_mean_law(law, left, right)
        denom = np.dot(weights, basis**2)
        params[amplitude] = float(max(0, np.dot(weights * y, basis) / denom))
        mse = np.dot(weights, (params[amplitude] * basis - y) ** 2) / len(events)
        return float(mse), law

    if family in {"linear", "square_root"}:
        mse, law = profile(0)
    else:
        bounds = (
            POWER_BOUNDS if family == "power" else np.log(HALF_SATURATION_RATIO_BOUNDS)
        )
        result = minimize_scalar(
            lambda shape: profile(shape)[0],
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-10},
        )
        # Bounded minimization does not evaluate either endpoint.
        mse, law = min(
            (profile(float(shape)) for shape in (*bounds, result.x)),
            key=lambda pair: pair[0],
        )
    return {**law, "train_rmse": float(np.sqrt(mse))}


def fit_law_menu(events: Sequence[dict]) -> dict:
    """Fit a bounded family menu and rank it on left-out whole training events.

    Rows carry trajectory_id, left, right and mean_rate. An optional scalar or
    per-interval absolute_error_bound enables a certified monotonicity check;
    omitted bounds or individual nulls permit advisory evidence only. The same
    target RMS scales every candidate score after fitting. A single informative
    event can be evaluated externally, but has no event-validation score.
    Scores normalized by the common scale are rounded to eight decimals for
    ties. Within a bin fixed shapes rank first, then the lower raw score.
    """
    data = [_event(event) for event in events]
    if len({event.name for event in data}) != len(data):
        raise ValueError("Each training trajectory must appear once.")
    shape_check = _shape_check(data)
    if shape_check["status"] == "contradicted":
        return {
            "status": "menu_contradicted",
            "reason": "Ordered source intervals have incompatible mean rates.",
            "shape_check": shape_check,
            "candidates": [],
        }
    if not _excited(data):
        return {
            "status": "insufficient_excitation",
            "reason": "Need positive observed rates and two distinct active intervals.",
            "shape_check": shape_check,
            "candidates": [],
        }
    target_scale = float(np.sqrt(np.mean([np.mean(ev.rate**2) for ev in data])))
    candidates = [_fit(family, data) for family in EXPRESSIONS]
    for candidate in candidates:
        folds = []
        for index, event in enumerate(data):
            others = data[:index] + data[index + 1 :]
            fold = {"held_out": event.name, "train_events": [ev.name for ev in others]}
            if _excited(others):
                law = _fit(candidate["family"], others)
                residual = interval_mean_law(law, event.left, event.right) - event.rate
                fold.update(
                    parameters=law["parameters"],
                    rmse=float(np.sqrt(np.mean(residual**2))),
                )
            else:
                fold.update(rmse=None, reason="Insufficient training excitation.")
            folds.append(fold)
        cv = None
        if all(f["rmse"] is not None for f in folds):
            cv = float(np.sqrt(np.mean([f["rmse"] ** 2 for f in folds])))
        candidate.update(
            cv_rmse=cv,
            cv_normalized_rmse=cv / target_scale if cv is not None else None,
            folds=folds,
        )
    ranked = candidates[0]["cv_rmse"] is not None
    score = "cv_rmse" if ranked else "train_rmse"
    # Disregard optimizer roundoff in numerical ties; prefer fixed shapes.
    candidates.sort(
        key=lambda c: (
            round(c[score] / target_scale, 8),
            c["family"] in {"power", "saturating"},
            c[score],
        )
    )
    return {
        "status": "ranked" if ranked else "fit_only",
        "target_scale_rms": target_scale,
        "ranking_score": score,
        "shape_check": shape_check,
        "power_exponent_bounds": list(POWER_BOUNDS),
        "half_saturation_bounds_relative_to_max_displacement": list(
            HALF_SATURATION_RATIO_BOUNDS
        ),
        "candidates": candidates,
    }
