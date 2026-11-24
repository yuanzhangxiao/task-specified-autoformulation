"""Compare observed and model-generated initial values across training events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from math import comb, isfinite
from numbers import Integral, Real

from autoformalism.research.ordered_driver_certificate import _lower


def _scalar(value: Real) -> Fraction:
    """Read a finite initial measurement, prediction, time or tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("Expected a finite numeric scalar")
    return Fraction(int(value) if isinstance(value, Integral) else float(value))


def _initial_values(row: dict, target: str) -> tuple[str, Fraction, Fraction]:
    """Read only the first sample of an aligned, identified training event."""
    if not isinstance(row, Mapping):
        raise ValueError("Expected a training event record")
    event = row.get("event", row.get("trajectory_id"))
    if (
        not isinstance(event, str)
        or not event.startswith("train_")
        or not event[6:]
        or row.get("trajectory_id", event) != event
        or row.get("split", "train") != "train"
    ):
        raise ValueError("Expected an identified training event")
    try:
        time = row["time"]
        observed = row["observed"][target]
        predicted = row["predicted"][target]
        arrays = (time, observed, predicted)
        if any(isinstance(a, (str, bytes, Mapping)) for a in arrays):
            raise ValueError("Expected aligned sample arrays")
        if not len(time) or len(time) != len(observed) or len(time) != len(predicted):
            raise ValueError("Expected aligned nonempty sample arrays")
        if _scalar(time[0]) != 0:
            raise ValueError("Expected a time-zero sample")
        return event, _scalar(observed[0]), _scalar(predicted[0])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Missing initial data for {event} and {target}") from exc


def check_initial_order(
    events: Sequence[dict], target: str, *, tolerance: float
) -> dict:
    """Check a fixed vector's algebraic predictions at time zero.

    The caller supplies a generated algebraic target and the same vector for all
    events. An observed separation greater than twice the absolute per-sample
    tolerance fixes a strict order for any predictions meeting that tolerance.
    A predicted tie or reversal violates that necessary condition. Level errors
    are reported separately, including on pairs without a required order.
    """
    tol = _scalar(tolerance)
    if tol < 0 or not isinstance(target, str) or not target.strip() or not events:
        raise ValueError("Expected a nonnegative tolerance, target and training events")
    points = sorted(_initial_values(row, target) for row in events)
    if len({event for event, _, _ in points}) != len(points):
        raise ValueError("Duplicate training event")
    errors = [(event, abs(p - y)) for event, y, p in points]
    worst_event, worst_error = max(errors, key=lambda item: item[1])
    comparable = reversed_pairs = predicted_ties = observed_ties = 0
    worst_pair = None
    worst_bound = Fraction(0)
    for a, b in combinations(points, 2):
        low, high = sorted((a, b), key=lambda point: point[1])
        gap = high[1] - low[1]
        if gap <= 2 * tol:
            observed_ties += int(gap == 0)
            continue
        comparable += 1
        predicted_gap = high[2] - low[2]
        if predicted_gap > 0:
            continue
        reversed_pairs += int(predicted_gap < 0)
        predicted_ties += int(predicted_gap == 0)
        bound = (gap - predicted_gap) / 2
        if bound > worst_bound:
            worst_bound = bound
            worst_pair = {
                "low_event": low[0],
                "high_event": high[0],
                "observed_low": float(low[1]),
                "observed_high": float(high[1]),
                "predicted_low": float(low[2]),
                "predicted_high": float(high[2]),
                "max_abs_error_lower_bound": _lower(bound),
            }
    pair_count = comb(len(points), 2)
    return {
        "scope": "fixed_parameter_vector_at_time_zero",
        "target": target,
        "tolerance": float(tol),
        "status": "ordering_conflict" if worst_pair else "no_ordering_conflict",
        "event_count": len(points),
        "pair_count": pair_count,
        "comparable_pairs": comparable,
        "unresolved_observed_pairs": pair_count - comparable,
        "observed_tie_pairs": observed_ties,
        "reversed_pairs": reversed_pairs,
        "tied_prediction_pairs": predicted_ties,
        "worst_order_witness": worst_pair,
        "worst_error_event": worst_event,
        "max_abs_error": float(worst_error),
        "events_exceeding_tolerance": sum(error > tol for _, error in errors),
    }
