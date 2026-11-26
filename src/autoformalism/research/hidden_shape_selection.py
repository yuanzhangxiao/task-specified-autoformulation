"""Compare competing shape hypotheses on whole omitted events."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import permutations

import numpy as np


def compare_event_errors(
    errors: Mapping[str, Mapping[str, float]], margin: float
) -> dict:
    """Require paired, material forecast improvements across independent events.

    All entries are RMSE in the same output units. The margin is an experiment
    setting. This is a predictive decision rule, not a structural identity test.
    """
    if len(errors) < 2 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Need competing models and a finite nonnegative margin")
    first = next(iter(errors.values()))
    events = sorted(first)
    if any(set(row) != set(events) for row in errors.values()):
        raise ValueError("Compare each hypothesis on the same events")
    if len(events) < 2:
        return {"status": "unresolved", "reason": "too few events", "winner": None}
    arrays = {
        k: np.asarray([r[e] for e in events], dtype=float) for k, r in errors.items()
    }
    if any(not np.isfinite(v).all() or min(v) < 0 for v in arrays.values()):
        return {
            "status": "unresolved",
            "reason": "incomplete forecasts",
            "winner": None,
        }
    comparisons = []
    dominates = dict.fromkeys(arrays, 0)
    for left, right in permutations(sorted(arrays), 2):
        gain = arrays[right] - arrays[left]
        wins = int(np.count_nonzero(gain > margin))
        losses = int(np.count_nonzero(gain < -margin))
        if wins > len(events) / 2 and losses == 0:
            dominates[left] += 1
        comparisons.append(
            {
                "candidate": left,
                "against": right,
                "wins": wins,
                "losses": losses,
                "median_rmse_gain": float(np.median(gain)),
            }
        )
    winner = next((k for k, n in dominates.items() if n == len(arrays) - 1), None)
    return {
        "status": "preferred_on_omitted_events" if winner else "unresolved",
        "winner": winner,
        "events": events,
        "rmse_margin": margin,
        "comparisons": comparisons,
    }


def equivalent_power_readouts(
    power: int, other_power: int, decay: Fraction, other_decay: Fraction
) -> bool:
    """Prove the change of coordinate w=z**(power/other_power) in a fixed family.

    Both models have y'=z**power and z'=-decay*z, with a positive hidden
    state, the same observed initial value and corresponding hidden initials.
    A specified common unit hidden initial satisfies the correspondence.
    """
    if not all(isinstance(n, int) and n > 0 for n in (power, other_power)):
        raise ValueError("Readout powers must be positive integers")
    if not all(isinstance(v, Fraction) and v > 0 for v in (decay, other_decay)):
        raise ValueError("Use exact positive rational decay rates for this proof")
    return power * decay == other_power * other_decay
