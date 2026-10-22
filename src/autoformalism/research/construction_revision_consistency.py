"""Exact, candidate-specific training equality witnesses without simulation.

This opt-in boundary uses complete executable dependencies and the actual scoped
initializer graph. It proves only necessary conditions for simultaneous target
reproduction. Missing conditioning contracts and relevant trajectory roots defer
the affected check; they are never interpreted as equal hidden initial values.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from itertools import combinations
from math import nextafter
from typing import Any

from autoformalism.expressions import compile_candidate
from autoformalism.fitting.initialization_graph import (
    InitialGraphMap,
    InitialGraphValue,
    apply_initialization_graph,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    EvidenceContract,
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-revision-training-consistency-1"
TARGET_COMPARISON_POLICY = {
    "protocol": "target-discrepancy-numerical-guard-1",
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 1e-7,
    "formula": "atol + rtol * max(abs(left), abs(right))",
    "application": "remaining gap after declared observation error intervals",
    "provenance": "Frozen numerical convention from FitConfig defaults; not an "
    "authenticated noise bound or a guarantee of solver/export error.",
    "suppressed_discrepancies": "numerically unresolved; defer, never certify equal",
    "conditioning_comparison": "exact; this guard applies to targets only",
}
_ASSUMPTIONS = [
    "Parameters and initializer-map coefficients are shared across trajectories.",
    "Unique deterministic, open-loop, reset-free solutions are a declared "
    "candidate-class assumption, not a theorem established by this check.",
    "Conditioning values match exactly; complete forcing knots and authenticated "
    "interpolation define the compared functions. No approximate match is used.",
    "Conflicts concern simultaneous reproduction within declared absolute error "
    "bounds, not the usefulness or attainable quality of an approximate fit.",
    "A separate frozen numerical guard defers small target discrepancies. It "
    "does not modify observation uncertainty or conditioning equality.",
]


def _closure(
    symbols: set[str],
    definitions: dict[str, set[str]],
    states: set[str],
    *,
    initial_only: bool,
) -> set[str]:
    """Resolve complete algebraic laws and, optionally, dynamic ancestors."""
    pending, seen = set(symbols), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in definitions and not (initial_only and name in states):
            pending.update(definitions[name] - seen)
    return seen


def _initial_sources(
    state: str,
    problem: PrefixFitProblem,
    observed: dict[str, str],
) -> tuple[set[str], set[str]]:
    """Trace actual initializer ancestry, retaining only relevant scoped roots."""
    if state in observed:
        return {observed[state]}, set()
    rule = problem.graph.rules[state].initial
    if isinstance(rule, InitialGraphValue):
        return set(), {state} if rule.scope == "trajectory" else set()
    if not isinstance(rule, InitialGraphMap):
        return set(), set()
    public, roots = set(rule.public_initials.values()), set()
    for ancestor in rule.state_initials.values():
        channels, upstream_roots = _initial_sources(ancestor, problem, observed)
        public.update(channels)
        roots.update(upstream_roots)
    return public, roots


def _value(row: CompleteTrajectory, channel: str, index: int, time: str) -> float:
    """Read a declared public value; missing channels never become zero."""
    if channel == time:
        return row.time[index]
    if channel in row.covariates:
        return row.covariates[channel]
    for group in (row.targets, row.auxiliaries, row.inputs):
        if channel in group:
            return group[channel][index]
    raise ValueError(f"missing public conditioning channel: {channel}")


def _initial_key(
    state: str,
    row: CompleteTrajectory,
    problem: PrefixFitProblem,
    observed: dict[str, str],
) -> tuple:
    """Identical keys suffice for equality without evaluating an untrusted map."""
    time = problem.context.time_symbol
    if state in observed:
        return ("observed", state, _value(row, observed[state], 0, time))
    rule = problem.graph.rules[state].initial
    if isinstance(rule, InitialGraphValue):
        # Trajectory roots have already caused this target's check to defer.
        return (rule.scope, state)
    if not isinstance(rule, InitialGraphMap):
        return ("known", state, rule.value)
    return (
        "map",
        state,
        tuple(
            (alias, _value(row, channel, 0, time))
            for alias, channel in sorted(rule.public_initials.items())
        ),
        tuple(
            (alias, _initial_key(ancestor, row, problem, observed))
            for alias, ancestor in sorted(rule.state_initials.items())
        ),
    )


def _sample(
    row: CompleteTrajectory,
    channel: str,
    time: float,
    mode: str,
) -> Fraction:
    """Interpolate exact represented numbers, never extrapolating or rounding."""
    if not row.time[0] <= time <= row.time[-1]:
        raise ValueError("forcing comparison outside recorded support")
    values = row.inputs.get(channel, row.auxiliaries.get(channel))
    if values is None:
        raise ValueError(f"unavailable forcing: {channel}")
    index = bisect_right(row.time, time) - 1
    value = Fraction(values[index])
    if mode == "linear_complete_knots" and time != row.time[index]:
        fraction = (Fraction(time) - Fraction(row.time[index])) / (
            Fraction(row.time[index + 1]) - Fraction(row.time[index])
        )
        value += fraction * (Fraction(values[index + 1]) - value)
    return value


def _forcing_equal(
    left: CompleteTrajectory,
    right: CompleteTrajectory,
    channels: set[str],
    contract: EvidenceContract,
    start: float,
    end: float,
) -> bool:
    """Compare one interval; callers retain all preceding interval comparisons."""
    knots = sorted(
        {start, end}
        | {t for row in (left, right) for t in row.time if start <= t <= end}
    )
    for name in sorted(channels):
        if name in left.covariates:
            if left.covariates[name] != right.covariates[name]:
                return False
        elif any(
            _sample(left, name, t, contract.interpolation[name])
            != _sample(right, name, t, contract.interpolation[name])
            for t in knots
        ):
            return False
    return True


def _lower_float(value: Fraction) -> float | None:
    """Render nonnegative exact bounds conservatively; overflow remains unknown."""
    try:
        rendered = float(value)
    except OverflowError:
        return None
    return nextafter(rendered, 0.0) if Fraction(rendered) > value else rendered


def _gap(witness: dict[str, Any]) -> Fraction:
    """Compare witness strengths without overflow or floating-point rounding."""
    a, b = witness["observations"]
    return abs(Fraction(a) - Fraction(b))


def _witness(
    left: CompleteTrajectory,
    i: int,
    right: CompleteTrajectory,
    j: int,
    target: str,
    bound: float,
) -> dict[str, Any] | None:
    """Record disjoint observation intervals and an exact arithmetic error floor."""
    a, b = left.targets[target][i], right.targets[target][j]
    gap = abs(Fraction(a) - Fraction(b))
    excess = gap - 2 * Fraction(bound)
    if excess <= 0:
        return None
    guard = Fraction(TARGET_COMPARISON_POLICY["absolute_tolerance"]) + Fraction(
        TARGET_COMPARISON_POLICY["relative_tolerance"]
    ) * max(abs(Fraction(a)), abs(Fraction(b)))
    return {
        "trajectories": [left.trajectory_id, right.trajectory_id],
        "sample_indices": [i, j],
        "times": [left.time[i], right.time[j]],
        "observations": [a, b],
        "absolute_error_bounds": [bound, bound],
        "error_intervals_disjoint": True,
        "observation_gap": _lower_float(gap),
        "target_numerical_guard": _lower_float(guard),
        "exceeds_target_numerical_guard": excess > guard,
        "pair_squared_error_lower_bound": _lower_float(gap * gap / 2),
        "maximum_absolute_error_lower_bound": _lower_float(gap / 2),
        "pair_squared_excess_error_lower_bound": _lower_float(excess * excess / 2),
    }


def _check(
    target: str,
    output_symbols: set[str],
    definitions: dict[str, set[str]],
    problem: PrefixFitProblem,
    evidence: TrainingCompatibilityEvidence,
    observed: dict[str, str],
    *,
    initial_only: bool,
) -> dict[str, Any]:
    """Apply one conservative equality theorem to one complete output closure."""
    states = {s.name for s in problem.base_candidate.states}
    parameters = {p.name for p in problem.base_candidate.parameters}
    scope = _closure(output_symbols, definitions, states, initial_only=initial_only)
    relevant_states = scope & states
    time = problem.context.time_symbol
    supplied = scope - definitions.keys() - parameters - {time}
    initial_channels, roots = set(), set()
    for state in relevant_states:
        channels, ancestors = _initial_sources(state, problem, observed)
        initial_channels.update(channels)
        roots.update(ancestors)
    missing = []
    if roots:
        missing.append("relevant trajectory roots are not established equal")
    for name in sorted((supplied | initial_channels) - {time}):
        if evidence.contract.absolute_error_bounds.get(name) != 0:
            missing.append(f"exact conditioning channel {name} is not established")
    if not initial_only:
        for name in sorted(supplied - set(problem.context.fixed_covariates)):
            if name not in problem.context.forcing_channels:
                missing.append(f"unmodeled target history {name} is unavailable")
            elif evidence.contract.interpolation.get(name, "unknown") == "unknown":
                missing.append(f"complete interpolation for {name} is unavailable")
    bound = evidence.contract.absolute_error_bounds.get(target)
    if bound is None:
        missing.append(f"observation error bound for {target} is unavailable")
    result: dict[str, Any] = {
        "property": "initial_readout_equality"
        if initial_only
        else "closed_subsystem_equality",
        "target": target,
        "status": "deferred",
        "scope": sorted(scope),
        "initialized_states": sorted(relevant_states),
        "initial_public_channels": sorted(initial_channels),
        "forcing_channels": sorted(supplied),
        "trajectory_roots": sorted(roots),
        "unavailable": missing,
        "comparisons": 0,
        "witnesses": [],
        "numerically_unresolved": [],
        "conditioning_distinctions": {},
        "equations": {
            s.state: s.rhs
            for s in problem.base_candidate.state_equations
            if s.state in relevant_states and not initial_only
        },
        "algebraic_definitions": {
            p.name: p.expression
            for p in problem.base_candidate.processes
            if p.name in scope
        },
    }
    if missing:
        return result
    best = unresolved = None
    distinctions: dict[str, int] = {}

    def distinguish(reason: str) -> None:
        distinctions[reason] = distinctions.get(reason, 0) + 1

    for left, right in combinations(evidence.trajectories, 2):
        if not initial_only and left.time[0] != right.time[0]:
            distinguish("different_absolute_time_origins")
            continue
        if any(
            _initial_key(s, left, problem, observed)
            != _initial_key(s, right, problem, observed)
            for s in relevant_states
        ):
            distinguish("different_initial_information")
            continue
        if initial_only:
            if any(
                _value(left, n, 0, time) != _value(right, n, 0, time)
                for n in supplied | ({time} if time in scope else set())
            ):
                distinguish("different_initial_readout_arguments")
                continue
            points = [(0, 0)]
        else:
            right_indices = {t: j for j, t in enumerate(right.time)}
            points = [
                (i, right_indices[t])
                for i, t in enumerate(left.time)
                if i > 0 and t in right_indices
            ]
            if not points:
                distinguish("no_common_absolute_observation_times")
        matched_through = left.time[0]
        for i, j in points:
            if not initial_only and not _forcing_equal(
                left, right, supplied, evidence.contract, matched_through, left.time[i]
            ):
                distinguish("different_complete_forcing_history")
                break
            matched_through = left.time[i]
            result["comparisons"] += 1
            witness = _witness(left, i, right, j, target, bound)
            if witness is not None:
                if witness["exceeds_target_numerical_guard"]:
                    if best is None or _gap(witness) > _gap(best):
                        best = witness
                elif unresolved is None or _gap(witness) > _gap(unresolved):
                    unresolved = witness
    result["conditioning_distinctions"] = distinctions
    if unresolved is not None:
        result["numerically_unresolved"] = [unresolved]
    if best is not None:
        result.update(status="conditional_conflict", witnesses=[best])
    elif unresolved is not None:
        result["unavailable"] = [
            "target discrepancy is within the frozen numerical guard"
        ]
    elif result["comparisons"]:
        result["status"] = "no_conflict_found"
    else:
        result["unavailable"] = ["no pair has established equal complete conditioning"]
    return result


def assess(problem: PrefixFitProblem, training: dict) -> dict[str, Any]:
    """Assess an executable problem against sealed TrainingCompatibilityEvidence.

    ``training`` is the existing evidence schema serialized as JSON. Malformed,
    stale, non-training or context-mismatched evidence raises ``ValueError``;
    valid evidence with unknown scientific metadata produces deferred checks.
    Historical ``contract.initial_rules`` are deliberately not used: the actual
    compiled observations and scoped graph own initialization in this protocol.
    """
    if not isinstance(problem, PrefixFitProblem):
        raise ValueError("typed PrefixFitProblem required")
    problem = PrefixFitProblem.model_validate(problem.model_dump(mode="json"))
    evidence = TrainingCompatibilityEvidence.model_validate(training)
    validate_evidence(evidence)
    if evidence.context != problem.context:
        raise ValueError("training evidence context differs from problem context")
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise ValueError("unlowered open-loop problem required")
    if any(p.scope.value != "global" for p in problem.base_candidate.parameters):
        raise ValueError("dynamics parameters must be globally shared")
    model = compile_candidate(problem.base_candidate, problem.context)
    apply_initialization_graph(model, problem.graph)
    observed = dict(model.direct_state_observation_channels)
    parsed = model.validated
    definitions = {
        name: set(expr.symbols)
        for name, expr in (
            *parsed.equation_expressions.items(),
            *parsed.process_expressions.items(),
        )
    }
    missing = [
        name
        for name in (
            "dependencies_exhaustive",
            "shared_parameters",
            "open_loop",
            "reset_impulse_free",
            "uniqueness_assumed",
        )
        if getattr(evidence.contract, name) is not True
    ]
    if missing:
        checks = [
            {
                "property": "candidate_class_contract",
                "status": "deferred",
                "unavailable": [f"{name} is not established" for name in missing],
            }
        ]
    else:
        checks = [
            _check(
                target,
                set(expr.symbols),
                definitions,
                problem,
                evidence,
                observed,
                initial_only=initial_only,
            )
            for target, expr in sorted(parsed.observation_expressions.items())
            if target in problem.context.targets
            for initial_only in (True, False)
        ]
    conflicts = [c for c in checks if c["status"] == "conditional_conflict"]
    deferred = [c for c in checks if c["status"] == "deferred"]
    return {
        "protocol": PROTOCOL,
        "status": "conflict"
        if conflicts
        else "deferred"
        if deferred
        else "no_conflict_found",
        "child_problem_sha256": content_hash(problem.model_dump(mode="json")),
        "evidence_sha256": evidence.sha256,
        "conflicts": conflicts,
        "deferred": deferred,
        "checks": checks,
        "assumptions": list(_ASSUMPTIONS),
        "target_comparison_policy": dict(TARGET_COMPARISON_POLICY),
        "contract_provenance": evidence.contract.provenance,
        "initialization_source": "compiled_observed_identities_and_actual_scoped_graph",
        "scientific_correctness_established": False,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
