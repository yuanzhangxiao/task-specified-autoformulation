"""Select a qualified ODE using nonnegative whole-event forecast losses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite


@dataclass(frozen=True)
class FittedCandidate:
    """Training error sets evaluation order. Parameter count breaks score ties."""

    name: str
    parameter_count: int
    training_nmse: float


def select_on_all_events(
    candidates: Sequence[FittedCandidate],
    events: Sequence[str],
    loss: Callable[[str, str], float | None],
    *,
    early_stop: bool = True,
) -> dict:
    """Return the minimum mean loss among candidates qualified on every event.

    A failed qualification returns None. Each new event contributes a
    nonnegative amount to the final sum, even when earlier errors are small.
    Strict pruning leaves an exact tie available to the complexity tie-break.
    """
    if not events or len(set(events)) != len(events):
        raise ValueError("Supply distinct, nonempty selection events")
    if len({c.name for c in candidates}) != len(candidates) or any(
        c.parameter_count < 0 or not isfinite(c.training_nmse) or c.training_nmse < 0
        for c in candidates
    ):
        raise ValueError("Invalid fitted candidate inventory")
    best: tuple[Fraction, int, str] | None = None
    outcomes = []
    calls = 0
    for candidate in sorted(candidates, key=lambda c: (c.training_nmse, c.name)):
        partial = Fraction()
        visited = []
        status = "complete"
        for event in sorted(events):
            value = loss(candidate.name, event)
            visited.append(event)
            calls += 1
            if value is None:
                status = "failed_qualification"
                break
            if not isfinite(value) or value < 0:
                raise ValueError("Expected a finite nonnegative event loss")
            # Exact summation of the recorded binary floats protects near ties.
            partial += Fraction(float(value))
            if early_stop and best and partial > best[0] and len(visited) < len(events):
                status = "above_incumbent"
                break
        if status == "complete":
            key = (partial, candidate.parameter_count, candidate.name)
            best = key if best is None else min(key, best)
        outcomes.append(
            {
                "candidate": candidate.name,
                "events_evaluated": visited,
                "status": status,
                "sum_of_observed_losses": float(partial),
            }
        )
    return {
        "winner": best[2] if best else None,
        "selection_nmse": float(best[0] / len(events)) if best else None,
        "event_evaluations": calls,
        "full_grid_size": len(events) * len(candidates),
        "trace": outcomes,
    }
