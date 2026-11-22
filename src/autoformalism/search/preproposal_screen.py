"""Exact, train-only information-set screening for an observed differential target.

An option identifies a permitted source of distinguishing information. The
proposer still has to realize its path and initializer in a joint dynamical model.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from itertools import combinations
from math import comb, inf, isfinite, nextafter
from typing import Literal

from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.search import topology_compatibility as compatibility
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
)
from autoformalism.search.topology_repair_generator import _disjoint
from autoformalism.staged_topology import content_hash

Verdict = Literal["screened", "contradicted", "unresolved"]
Difference = Literal["different", "equal", "unknown"]


class ScreenWitness(StrictSchema):
    """Earliest disjoint observations for a pair and their causal comparisons."""

    id: str
    trajectories: tuple[str, str]
    time: FiniteFloat
    observations: tuple[FiniteFloat, FiniteFloat]
    different: tuple[str, ...]
    unknown: tuple[str, ...]
    pair_error_floor: FiniteFloat


class ScreenCandidate(StrictSchema):
    """A unit-cost inclusion-minimal support with a private witness per option."""

    id: str
    options: tuple[str, ...]
    private_witnesses: dict[str, str]


class ScreenResult(StrictSchema):
    """Portable witness table; absent entries in a witnessed comparison mean equal."""

    protocol: Literal["preproposal-differential-screen-1"] = (
        "preproposal-differential-screen-1"
    )
    evidence_sha256: str
    target: Identifier
    minimum_target_gap: FiniteFloat = 0.0
    status: Verdict
    required_options: tuple[str, ...]
    optional_options: tuple[str, ...]
    pair_count: int
    remaining_witness_count: int
    witnesses: tuple[ScreenWitness, ...]
    noncomparable_pairs: tuple[tuple[str, str], ...]
    unavailable: tuple[str, ...]
    candidates: tuple[ScreenCandidate, ...]
    subsets_examined: int
    subset_counts: dict[Verdict, int]


class SubsetAssessment(StrictSchema):
    """Pair counts and a worst-observation error floor for this support."""

    status: Verdict
    distinguished_pairs: int
    conflicting_pairs: int
    unresolved_pairs: int
    max_pair_error_floor: FiniteFloat


def _compare_readings(a: float, b: float, bound: float | None) -> Difference:
    if bound is None:
        return "unknown"
    if _disjoint(a, b, bound):
        return "different"
    return "equal" if bound == 0 else "unknown"


def _forcing_value(row: CompleteTrajectory, name: str, t: float, mode: str) -> Fraction:
    """Exactly interpolate the measured center of one bounded forcing interval."""
    values = row.inputs[name] if name in row.inputs else row.auxiliaries[name]
    index = bisect_right(row.time, t) - 1
    start = Fraction(values[index])
    if row.time[index] == t or mode == "previous_complete_knots":
        return start
    weight = (Fraction(t) - Fraction(row.time[index])) / (
        Fraction(row.time[index + 1]) - Fraction(row.time[index])
    )
    return start + weight * (Fraction(values[index + 1]) - start)


def _history(
    left: CompleteTrajectory,
    right: CompleteTrajectory,
    name: str,
    end: float,
    evidence: TrainingCompatibilityEvidence,
) -> Difference:
    """Compare forcing on the positive-duration part of an initial-value problem."""
    if end == left.time[0]:
        return "equal"
    mode = evidence.contract.interpolation.get(name, "unknown")
    bound = evidence.contract.absolute_error_bounds.get(name)
    if mode == "unknown" or bound is None:
        return "unknown"
    if mode == "previous_complete_knots":
        end = max(t for row in (left, right) for t in row.time if t < end)
    if bound == 0:
        equal, missing = compatibility._forcing_equal(
            left, right, {name}, evidence, end
        )
        if missing:
            return "unknown"
        return "equal" if equal else "different"

    # A disjoint interval at a union knot certifies different linear functions
    # on a neighboring open interval, or different held values after that knot.
    knots = {t for row in (left, right) for t in row.time if t <= end}
    gap = 2 * Fraction(bound)
    if any(
        abs(_forcing_value(left, name, t, mode) - _forcing_value(right, name, t, mode))
        > gap
        for t in knots
    ):
        return "different"
    return "unknown"


def _witness(
    left: CompleteTrajectory,
    right: CompleteTrajectory,
    target: str,
    time: float,
    observations: tuple[float, float],
    options: tuple[str, ...],
    evidence: TrainingCompatibilityEvidence,
    number: int,
) -> ScreenWitness:
    different, unknown = [], []
    for option in options:
        kind, channel = option.split(":", 1)
        if kind == "history":
            status = _history(left, right, channel, time, evidence)
        elif time == left.time[0] and channel != target:
            # Another state's initializer cannot change a measured target at t0.
            status = "equal"
        else:
            status = _compare_readings(
                compatibility._value(left, channel, 0, evidence.context.time_symbol),
                compatibility._value(right, channel, 0, evidence.context.time_symbol),
                evidence.contract.absolute_error_bounds.get(channel),
            )
        if status == "different":
            different.append(option)
        elif status == "unknown":
            unknown.append(option)

    a, b = map(Fraction, observations)
    exact_floor = abs(a - b) / 2 - Fraction(
        evidence.contract.absolute_error_bounds[target]
    )
    floor = float(exact_floor)
    if Fraction(floor) > exact_floor:
        floor = nextafter(floor, -inf)
    return ScreenWitness(
        id=f"w{number}",
        trajectories=(left.trajectory_id, right.trajectory_id),
        time=time,
        observations=observations,
        different=tuple(different),
        unknown=tuple(unknown),
        pair_error_floor=floor,
    )


def _witnesses(
    evidence: TrainingCompatibilityEvidence,
    target: str,
    options: tuple[str, ...],
    minimum_target_gap: float,
) -> tuple[tuple[ScreenWitness, ...], tuple[tuple[str, str], ...]]:
    rows, noncomparable = [], []
    trajectories = sorted(evidence.trajectories, key=lambda r: r.trajectory_id)
    bound = evidence.contract.absolute_error_bounds[target]
    threshold = 2 * Fraction(bound) + Fraction(minimum_target_gap)
    for left, right in combinations(trajectories, 2):
        if left.time[0] != right.time[0]:
            noncomparable.append((left.trajectory_id, right.trajectory_id))
            continue
        right_samples = dict(zip(right.time, right.targets[target], strict=True))
        common = [
            (t, a, right_samples[t])
            for t, a in zip(left.time, left.targets[target], strict=True)
            if t in right_samples
        ]
        if len(common) == 1 and (len(left.time) > 1 or len(right.time) > 1):
            noncomparable.append((left.trajectory_id, right.trajectory_id))
            continue
        # Each option uses an initial reading or a complete forcing prefix.
        # A distinction, once attained, cannot be lost by extending that prefix.
        for t, a, b in common:
            if a != b and (not threshold or abs(Fraction(a) - Fraction(b)) > threshold):
                rows.append(
                    _witness(
                        left, right, target, t, (a, b), options, evidence, len(rows)
                    )
                )
                break
    return tuple(rows), tuple(noncomparable)


def _verdict(conflicts: int, unresolved: int) -> Verdict:
    if conflicts:
        return "contradicted"
    return "unresolved" if unresolved else "screened"


def assess_options(result: ScreenResult, options: tuple[str, ...]) -> SubsetAssessment:
    """Assess any optional selection on the same table, retaining the fixed roots."""
    if len(options) != len(set(options)) or set(options) - set(result.optional_options):
        raise ValueError("Select unique options from this screen's optional catalogue")
    selected = set(options) | set(result.required_options)
    distinguished = conflicts = 0
    unknown = (
        result.pair_count if result.unavailable else len(result.noncomparable_pairs)
    )
    floor = 0.0
    for witness in result.witnesses:
        if not selected.isdisjoint(witness.different):
            distinguished += 1
        elif not selected.isdisjoint(witness.unknown):
            unknown += 1
        else:
            conflicts += 1
            floor = max(floor, witness.pair_error_floor)
    return SubsetAssessment(
        status=_verdict(conflicts, unknown or len(result.unavailable)),
        distinguished_pairs=distinguished,
        conflicting_pairs=conflicts,
        unresolved_pairs=unknown,
        max_pair_error_floor=floor,
    )


def _enumerate(
    target: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    rows: tuple[ScreenWitness, ...],
    incomplete: bool,
) -> tuple[list[ScreenCandidate], dict[Verdict, int], int]:
    """Enumerate all masks. A selected option must uniquely cover at least one row."""
    all_bits = (1 << len(rows)) - 1
    differs = dict.fromkeys((*required, *optional), 0)
    unknown = differs.copy()
    for i, row in enumerate(rows):
        for option in row.different:
            differs[option] |= 1 << i
        for option in row.unknown:
            unknown[option] |= 1 << i
    base = uncertain = 0
    for option in required:
        base |= differs[option]
        uncertain |= unknown[option]
    counts: dict[Verdict, int] = {"screened": 0, "contradicted": 0, "unresolved": 0}
    candidates = []
    for size in range(len(optional) + 1):
        for chosen in combinations(optional, size):
            covered, unresolved, private = base, uncertain, 0
            for option in chosen:
                distinguishes = differs[option]
                # Keep witnesses unique to a single optional source. A witness
                # already covered by a fixed root cannot justify an addition.
                private = (private & ~distinguishes) | (distinguishes & ~covered)
                covered |= distinguishes
                unresolved |= unknown[option]
            conflicts = all_bits & ~(covered | unresolved)
            verdict = _verdict(conflicts, (all_bits & ~covered) or incomplete)
            counts[verdict] += 1
            if verdict != "screened" or not all(
                private & differs[name] for name in chosen
            ):
                continue
            certificates = {}
            for option in chosen:
                bits = private & differs[option]
                certificates[option] = rows[(bits & -bits).bit_length() - 1].id
            digest = content_hash(
                {"target": target, "fixed": required, "chosen": chosen}
            )
            candidates.append(
                ScreenCandidate(
                    id=f"candidate-{digest[:16]}",
                    options=chosen,
                    private_witnesses=certificates,
                )
            )
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("Candidate identifier collision")
    return candidates, counts, (all_bits & ~base).bit_count()


def screen_target(
    evidence: TrainingCompatibilityEvidence,
    target: str,
    *,
    histories: tuple[str, ...],
    initials: tuple[str, ...] = (),
    required_histories: tuple[str, ...] = (),
    required_initials: tuple[str, ...] = (),
    max_subsets: int = 65536,
    minimum_target_gap: float = 0.0,
) -> ScreenResult:
    """Screen a declared differential target with its observed time-zero value.

    The caller supplies the forecast-permitted catalogue after variable
    declaration, excluding generated auxiliaries from supplied histories. An
    initial option may transmit a measured time-zero reading over a causal path
    after t0. Only the target itself can set its own measured initial value.
    A witness's two target observation intervals must be separated by more
    than minimum_target_gap in target units. This sensitivity margin leaves
    the certified uncertainty of sources and target initials unchanged.
    """
    compatibility.validate_evidence(evidence)
    if not isfinite(minimum_target_gap) or minimum_target_gap < 0:
        raise ValueError("Minimum target gap must be finite and nonnegative")
    context, contract = evidence.context, evidence.contract
    if target not in context.targets:
        raise ValueError("Screen target must be a measured public target")
    permitted_histories = set(context.external_inputs + context.auxiliaries)
    permitted_initials = permitted_histories | set(
        context.targets + context.fixed_covariates
    )
    for names, allowed in (
        (histories, permitted_histories),
        (initials, permitted_initials),
        (required_histories, set(histories)),
        (required_initials, set(initials)),
    ):
        if len(set(names)) != len(names) or set(names) - allowed:
            raise ValueError(
                "Invalid or duplicate source in the permitted information catalogue"
            )
    fixed = tuple(
        sorted(
            {f"initial:{target}"}
            | {f"history:{v}" for v in required_histories}
            | {f"initial:{v}" for v in required_initials}
        )
    )
    optional = tuple(
        sorted(
            {*(f"history:{v}" for v in histories), *(f"initial:{v}" for v in initials)}
            - set(fixed)
        )
    )
    subset_count = 1 << len(optional)
    if subset_count > max_subsets:
        raise ValueError(
            f"Exact screening requires {subset_count} subsets, budget {max_subsets}"
        )

    unavailable = compatibility._common_missing(evidence)
    if len(evidence.trajectories) < 2:
        unavailable.append("At least two training trajectories are required")
    if not contract.reset_impulse_free or not contract.uniqueness_assumed:
        unavailable.append(
            "A reset-free interval and unique ODE solutions are required"
        )
    if contract.absolute_error_bounds.get(target) is None:
        unavailable.append("Target observation error bound is unknown")
    if (
        target not in contract.initial_rules
        or contract.initial_rules[target].mode != "observed_initial"
    ):
        unavailable.append(
            "The target must be initialized from its observed time-zero reading"
        )

    pairs = comb(len(evidence.trajectories), 2)
    if unavailable:
        rows, noncomparable = (), ()
        candidates, counts, remaining = (
            [],
            {"screened": 0, "contradicted": 0, "unresolved": 0},
            0,
        )
        status = "unresolved"
    else:
        rows, noncomparable = _witnesses(
            evidence, target, (*fixed, *optional), minimum_target_gap
        )
        candidates, counts, remaining = _enumerate(
            target, fixed, optional, rows, bool(noncomparable)
        )
        status = (
            "screened"
            if candidates
            else ("unresolved" if counts["unresolved"] else "contradicted")
        )
    return ScreenResult(
        evidence_sha256=evidence.sha256,
        target=target,
        minimum_target_gap=minimum_target_gap,
        status=status,
        required_options=fixed,
        optional_options=optional,
        pair_count=pairs,
        remaining_witness_count=remaining,
        witnesses=rows,
        noncomparable_pairs=noncomparable,
        unavailable=tuple(unavailable),
        candidates=tuple(candidates),
        subsets_examined=sum(counts.values()),
        subset_counts=counts,
    )


def assess_realized_roots(
    evidence: TrainingCompatibilityEvidence,
    target: str,
    *,
    histories: tuple[str, ...],
    initials: tuple[str, ...],
    minimum_target_gap: float = 0.0,
) -> tuple[ScreenResult, SubsetAssessment]:
    """Check exactly the realized causal roots of one differential target.

    The caller must trace every reachable initializer and forcing in the
    proposed graph. Unknown or trajectory-specific free hidden initials cannot
    be represented by measured channels and must be marked unresolved upstream.
    """
    screen = screen_target(
        evidence,
        target,
        histories=histories,
        initials=initials,
        required_histories=histories,
        required_initials=initials,
        max_subsets=1,
        minimum_target_gap=minimum_target_gap,
    )
    return screen, assess_options(screen, ())
