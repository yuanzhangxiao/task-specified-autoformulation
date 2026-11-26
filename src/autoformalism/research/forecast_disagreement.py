"""Choose fitted functions using prediction disagreement on selection events."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from statistics import mean

import numpy as np

FAMILIES = ("affine", "square")
POLICIES = ("fixed_affine", "all_events", "high_disagreement", "random_same_size")
# Ties at numerical precision use the simpler conventional reference.
TIE_ATOL = 1e-12
TIE_RTOL = 1e-10


def prediction_gap(
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    scales: Mapping[str, float],
) -> float:
    """Mean squared post-initial difference across outputs; uses predictions alone."""
    if set(predictions) != set(FAMILIES) or not scales:
        raise ValueError("Need both families and a shared set of targets")
    if any(set(series) != set(scales) for series in predictions.values()):
        raise ValueError("Forecast target sets differ")
    errors = []
    for key, scale in scales.items():
        left, right = (np.asarray(predictions[f][key], dtype=float) for f in FAMILIES)
        if (
            not math.isfinite(scale)
            or scale <= 0
            or left.ndim != 1
            or left.shape != right.shape
            or left.size < 2
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            raise ValueError("Invalid forecast pair or target scale")
        errors.append(np.mean(np.square((left[1:] - right[1:]) / scale)))
    value = float(np.mean(errors))
    if not math.isfinite(value):
        raise ValueError("Nonfinite forecast difference")
    return value


def largest_gaps(gaps: Mapping[str, float]) -> list[str]:
    """Return the largest half of the gaps, breaking exact ties by event ID."""
    if len(gaps) < 2 or any(not math.isfinite(x) or x < 0 for x in gaps.values()):
        raise ValueError("Need at least two valid forecast gaps")
    return sorted(gaps, key=lambda event: (-gaps[event], event))[: (len(gaps) + 1) // 2]


def _best(losses: Mapping[str, Mapping[str, float | None]], ids: Iterable[str]) -> str:
    events = tuple(ids)
    left, right = (mean(losses[i][f] for i in events) for f in FAMILIES)
    if right < left and not math.isclose(
        left, right, rel_tol=TIE_RTOL, abs_tol=TIE_ATOL
    ):
        return "square"
    return "affine"


def choose(
    losses: Mapping[str, Mapping[str, float | None]], gaps: Mapping[str, float]
) -> dict:
    """Freeze policies on the selection events; None indicates a failed forecast."""
    if len(losses) < 2 or any(set(row) != set(FAMILIES) for row in losses.values()):
        raise ValueError("Incomplete selection design")
    if any(
        v is not None and (not math.isfinite(v) or v < 0)
        for row in losses.values()
        for v in row.values()
    ):
        raise ValueError("Invalid selection losses")
    paired_ids = {key for key, row in losses.items() if None not in row.values()}
    if set(gaps) != paired_ids:
        raise ValueError("Expected a gap for every paired event")
    eligible = [f for f in FAMILIES if all(r[f] is not None for r in losses.values())]
    selected = []
    votes = {"fixed_affine": {"affine": 1}}
    ids = sorted(losses)
    if len(eligible) == 2:
        selected = largest_gaps(gaps)
        random_votes = Counter(
            _best(losses, s) for s in combinations(ids, len(selected))
        )
        votes.update(
            all_events={_best(losses, ids): 1},
            high_disagreement={_best(losses, selected): 1},
            random_same_size=dict(random_votes),
        )
    else:
        for policy in POLICIES[1:]:
            votes[policy] = {eligible[0]: 1} if eligible else {}
    return {
        "eligible": eligible,
        "high_disagreement_events": selected,
        "selection_gaps": dict(gaps),
        "selection_losses": dict(losses),
        "votes": votes,
    }


def score_votes(
    votes: Mapping[str, int], held_losses: Mapping[str, Mapping[str, float | None]]
) -> dict:
    """Score the frozen choice on every held event; preserve unavailable weight."""
    if not held_losses or any(
        set(row) != set(FAMILIES) for row in held_losses.values()
    ):
        raise ValueError("Incomplete held design")
    if not set(votes) <= set(FAMILIES) or any(
        type(n) is not int or n < 1 for n in votes.values()
    ):
        raise ValueError("Invalid selected family counts")
    if any(
        v is not None and (not math.isfinite(v) or v < 0)
        for row in held_losses.values()
        for v in row.values()
    ):
        raise ValueError("Invalid held losses")
    total = sum(votes.values())
    available = {
        f for f in votes if all(row[f] is not None for row in held_losses.values())
    }
    coverage = sum(votes[f] for f in available) / total if total else 0.0
    event_losses = {}
    if coverage == 1.0:
        for event_id, row in held_losses.items():
            event_losses[event_id] = sum(row[f] * n / total for f, n in votes.items())
    return {
        "full_forecast_probability": coverage,
        "mean_event_nmse": mean(event_losses.values()) if event_losses else None,
        "event_nmse": event_losses,
    }
