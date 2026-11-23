"""Measure open-loop training error over increasing physical-time horizons."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

HORIZONS = (0.125, 0.25, 0.5, 1.0)


def _vector(x, n):
    values = np.asarray(x, dtype=float)
    if values.shape != (n,):
        raise ValueError("Invalid number of predictions for the requested prefix")
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("Nonfinite forecast on prefix")
    return values


def event_horizons(
    event: str,
    time: Sequence[float],
    observed: Sequence[float],
    training_sd: float,
    prefix: Callable[[int], tuple[Sequence[float], Sequence[float] | None]],
    *,
    verified_full_curve: bool = False,
) -> list[dict]:
    """Audit one target from its allowed initial condition, with no state resets.

    prefix(n) returns the fine and coarse forecasts through the first n samples.
    A caller using a previously grid-verified complete curve may omit the coarse
    vector. Failures refer only to the requested prefix, and abstain from scoring.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(observed, dtype=float)
    if not event.startswith("train_"):
        raise ValueError("Only training events may enter the diagnostic")
    if (
        t.ndim != 1
        or len(t) < 2
        or y.shape != t.shape
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(y))
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("Expected aligned finite observations and increasing time")
    if not np.isfinite(training_sd) or training_sd <= 0:
        raise ValueError("A finite positive training SD is required")
    out = []
    for h in HORIZONS:
        end = t[-1] if h == 1.0 else t[0] + h * (t[-1] - t[0])
        n = int(np.searchsorted(t, end, side="right"))
        row = {"fraction": h, "last_time": float(t[n - 1]), "samples": n}
        out.append(row)
        if n < 2:
            row["status"] = "insufficient_observations"
            continue
        try:
            fine, coarse = prefix(n)
            fine = _vector(fine, n)
            if coarse is None:
                if not verified_full_curve:
                    raise ValueError(
                        "Missing coarse forecast or full-curve verification"
                    )
            else:
                with np.errstate(over="raise", invalid="raise"):
                    delta = float(
                        np.max(np.abs(_vector(coarse, n) - fine)) / training_sd
                    )
                row["maximum_scaled_grid_difference"] = delta
                if delta > 0.01:
                    row["status"] = "resolution_unverified"
                    continue
            with np.errstate(over="raise", invalid="raise"):
                e = (fine - y[:n]) / training_sd
                # Exact integral of a linearly interpolated sample residual squared.
                energy = (e[:-1] ** 2 + e[:-1] * e[1:] + e[1:] ** 2) / 3
                row["nrmse"] = float(
                    np.sqrt(np.average(energy, weights=np.diff(t[:n])))
                )
            row["status"] = "scored"
        except ArithmeticError as err:
            row.update(status="rollout_failed", reason=str(err))
    return out


def summarize(events: dict[str, list[dict]], threshold: float = 1.0) -> dict:
    """Keep numerical failures explicit and weight events equally within a target."""
    if not events or any(not k.startswith("train_") for k in events):
        raise ValueError("Expected at least one training event")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive finite threshold required")
    if any([r["fraction"] for r in rs] != list(HORIZONS) for rs in events.values()):
        raise ValueError("Inconsistent recorded horizons")
    horizons = []
    first = {}
    for k, h in enumerate(HORIZONS):
        values, flags, statuses = [], [], Counter()
        for event, rs in events.items():
            r = rs[k]
            statuses[r["status"]] += 1
            if r["status"] == "scored":
                values.append(r["nrmse"])
            flag = flag_reason(r, threshold)
            if flag:
                flags.append(event)
                first.setdefault(event, {"fraction": h, "reason": flag})
        horizons.append(
            {
                "fraction": h,
                "status_counts": dict(statuses),
                "median_scored_nrmse": float(np.median(values)) if values else None,
                "worst_scored_nrmse": float(max(values)) if values else None,
                "flagged": len(flags),
            }
        )
    return {
        "events": len(events),
        "warning_threshold": threshold,
        "horizons": horizons,
        "first_warning": first,
    }


def flag_reason(row: dict, threshold: float) -> str | None:
    """A magnitude warning or a separate warning about numerical uncertainty."""
    if row["status"] in {"rollout_failed", "resolution_unverified"}:
        return row["status"]
    if row["status"] == "scored" and row["nrmse"] >= threshold:
        return "large_residual"
    return None
