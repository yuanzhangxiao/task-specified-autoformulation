"""Offline, conditional topology/data diagnostics; never an acceptance gate.

The evidence type deliberately differs from the bounded prompt packet. Equality
of a few displayed samples does not establish equality of complete forcing.
Contract assertions must come from the public task/runtime, not proposer prose.
"""

from __future__ import annotations

import ast
from bisect import bisect_right
from fractions import Fraction
from itertools import combinations, pairwise
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.expressions import ValidationContext
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.search.construction_ledger import Draft
from autoformalism.staged_topology import content_hash

PROTOCOL = "offline-topology-compatibility-1"


class InitialRule(StrictSchema):
    """How a state is initialized, without fitting or evaluating an initializer."""

    mode: Literal["observed_initial", "shared_parameter", "common_map", "unknown"]
    arguments: tuple[Identifier, ...] = ()


class EvidenceContract(StrictSchema):
    """Explicit conditioning assertions; unknown metadata never becomes true.

    Interpolation labels assert that every defining knot is present. Error bounds
    are absolute observation/forcing uncertainty, not descriptive tolerances.
    Uniqueness is a hypothesis of the candidate class, not certified by topology.
    """

    provenance: str = Field(min_length=1)
    dependencies_exhaustive: bool | None = None
    shared_parameters: bool | None = None
    open_loop: bool | None = None
    reset_impulse_free: bool | None = None
    uniqueness_assumed: bool | None = None
    interpolation: dict[
        str, Literal["linear_complete_knots", "previous_complete_knots", "unknown"]
    ] = Field(default_factory=dict)
    absolute_error_bounds: dict[str, FiniteFloat | None] = Field(default_factory=dict)
    initial_rules: dict[str, InitialRule] = Field(default_factory=dict)

    @model_validator(mode="after")
    def nonnegative_bounds(self):
        if any(v is not None and v < 0 for v in self.absolute_error_bounds.values()):
            raise ValueError("absolute error bounds must be nonnegative")
        return self


class CompleteTrajectory(StrictSchema):
    """All public training samples; no derivatives or hidden labels are read."""

    trajectory_id: str
    time: tuple[FiniteFloat, ...] = Field(min_length=1)
    targets: dict[str, tuple[FiniteFloat, ...]]
    auxiliaries: dict[str, tuple[FiniteFloat, ...]]
    inputs: dict[str, tuple[FiniteFloat, ...]]
    covariates: dict[str, FiniteFloat]

    @model_validator(mode="after")
    def aligned(self):
        if any(b <= a for a, b in pairwise(self.time)):
            raise ValueError("times must strictly increase")
        if any(
            len(v) != len(self.time)
            for group in (self.targets, self.auxiliaries, self.inputs)
            for v in group.values()
        ):
            raise ValueError("all channels must be aligned to full times")
        return self


class TrainingCompatibilityEvidence(StrictSchema):
    """Self-contained train-only source and conditioning, with a content digest."""

    protocol: Literal["complete-training-compatibility-evidence-1"] = (
        "complete-training-compatibility-evidence-1"
    )
    split: Literal["train"] = "train"
    context: ValidationContext
    contract: EvidenceContract
    trajectories: tuple[CompleteTrajectory, ...] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Witness(StrictSchema):
    """Recorded samples supporting a conditional exact-interpolation conflict."""

    trajectories: tuple[str, ...]
    sample_indices: tuple[int, ...]
    times: tuple[FiniteFloat, ...]
    observations: tuple[FiniteFloat, ...]
    error_intervals_disjoint: bool | None = None
    pair_squared_error_lower_bound: FiniteFloat | None = None


class Diagnostic(StrictSchema):
    """Separate performed comparisons, conditional claims and unavailable evidence."""

    property: Literal["algebraic_map", "closed_subsystem", "scalar_autonomous"]
    target: str
    status: Literal["conditional_conflict", "no_conflict_found", "deferred"]
    scope: tuple[str, ...]
    verified: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    comparisons: int = 0
    witnesses: tuple[Witness, ...] = ()
    question: str | None = None


class CompatibilityReport(StrictSchema):
    """Offline advice, independent of structural eligibility and fitting quality."""

    protocol: Literal["offline-topology-compatibility-1"] = PROTOCOL
    evidence_sha256: str
    topology_sha256: str
    diagnostics: tuple[Diagnostic, ...]
    acceptance_gate: Literal[False] = False
    llm_calls: Literal[0] = 0
    optimizer_calls: Literal[0] = 0
    test_data_opened: Literal[False] = False


def build_evidence(
    train: DatasetSplit, context: ValidationContext, contract: EvidenceContract
) -> TrainingCompatibilityEvidence:
    """Read complete train arrays only, then hash all values and conditioning."""
    if not isinstance(train, DatasetSplit) or train.name is not SplitName.TRAIN:
        raise ValueError("compatibility evidence requires the train split only")
    trajectories = []
    for item in sorted(train.trajectories, key=lambda v: v.trajectory_id):
        trajectories.append(
            CompleteTrajectory(
                trajectory_id=item.trajectory_id,
                time=tuple(item.time),
                targets={k: tuple(v) for k, v in item.targets.items()},
                auxiliaries={k: tuple(v) for k, v in item.auxiliaries.items()},
                inputs={k: tuple(v) for k, v in item.external_inputs.items()},
                covariates=dict(item.fixed_covariates),
            )
        )
    value = TrainingCompatibilityEvidence(
        context=context,
        contract=contract,
        trajectories=tuple(trajectories),
        sha256="0" * 64,
    )
    value = value.model_copy(update={"sha256": content_hash(_unsigned(value))})
    validate_evidence(value)
    return value


def _unsigned(evidence: TrainingCompatibilityEvidence) -> dict:
    return evidence.model_dump(mode="json", exclude={"sha256"})


def validate_evidence(evidence: TrainingCompatibilityEvidence) -> None:
    """Reject stale/malformed copies and missing channels rather than assuming zero."""
    # Revalidation also catches model_copy updates that bypass Pydantic validation.
    TrainingCompatibilityEvidence.model_validate(evidence.model_dump(mode="json"))
    if evidence.sha256 != content_hash(_unsigned(evidence)):
        raise ValueError("compatibility evidence digest differs")
    ids = [v.trajectory_id for v in evidence.trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("training trajectory IDs must be unique")
    context = evidence.context
    channel_names = set(
        context.targets
        + context.auxiliaries
        + context.external_inputs
        + context.fixed_covariates
    )
    if context.time_symbol in channel_names:
        raise ValueError("time symbol and measurement channel must be distinct")
    for item in evidence.trajectories:
        for actual, expected in (
            (item.targets, context.targets),
            (item.auxiliaries, context.auxiliaries),
            (item.inputs, context.external_inputs),
            (item.covariates, context.fixed_covariates),
        ):
            if set(actual) != set(expected):
                raise ValueError(
                    "complete evidence channels differ from public context"
                )


def _graph(draft: Draft, context: ValidationContext) -> tuple[dict, dict, list[str]]:
    """Expand ordinary and shared contributions, retaining conversion dependencies."""
    kinds = {v.name: v.definition for v in draft.variables}
    variables = set(kinds)
    graph = {name: set() for name in {*kinds, *(p.name for p in draft.processes)}}
    errors = []
    defined = set()
    for process in draft.processes:
        if kinds.get(process.name) == "differential":
            errors.append(f"process/state ownership conflict: {process.name}")
        kinds[process.name] = "algebraic"
        graph[process.name].update(process.depends_on)
        defined.add(process.name)
        for use in process.uses:
            if use.target not in variables:
                errors.append(
                    f"process receiver is not a generated variable: {use.target}"
                )
            if use.target not in graph:
                errors.append(f"undeclared receiver: {use.target}")
                continue
            graph[use.target].add(process.name)
            defined.add(use.target)
            if use.conversion is None:
                # A later permitted conversion may use any public fixed covariate.
                graph[use.target].update(context.fixed_covariates)
            else:
                try:
                    graph[use.target].update(
                        node.id
                        for node in ast.walk(ast.parse(use.conversion))
                        if isinstance(node, ast.Name)
                    )
                except (SyntaxError, ValueError, RecursionError):
                    errors.append(f"unparsed conversion for {use.target}")
    for equation in draft.equations:
        if equation.name not in graph:
            errors.append(f"undeclared equation: {equation.name}")
            continue
        if equation.name in {p.name for p in draft.processes} and equation.terms:
            errors.append(f"multiple definitions: {equation.name}")
        defined.add(equation.name)
        graph[equation.name].update(s for term in equation.terms for s in term.sources)
    errors.extend(f"missing topology: {name}" for name in sorted(set(kinds) - defined))
    supplied = set(
        context.external_inputs + context.auxiliaries + context.fixed_covariates
    )
    supplied.add(context.time_symbol)
    errors.extend(
        f"unresolved source: {name}"
        for name in sorted(set().union(*graph.values()) - set(graph) - supplied)
    )

    forbidden = set(context.external_inputs + context.fixed_covariates)
    forbidden.add(context.time_symbol)
    errors.extend(
        f"supplied-only generated quantity: {n}" for n in graph if n in forbidden
    )
    # Iterative elimination avoids recursive traversal of untrusted long graphs.
    remaining = {n for n in graph if kinds[n] == "algebraic"}
    while remaining:
        ready = {n for n in remaining if not graph[n] & remaining}
        if not ready:
            break
        remaining -= ready
    if remaining:
        errors.append("algebraic cycle")
    return graph, kinds, errors


def _closure(target: str, graph: dict) -> set[str]:
    result, pending = set(), [target]
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(graph.get(name, ()))
    return result


def _common_missing(evidence: TrainingCompatibilityEvidence) -> list[str]:
    contract = evidence.contract
    missing = []
    for key in ("dependencies_exhaustive", "shared_parameters", "open_loop"):
        if getattr(contract, key) is not True:
            missing.append(f"{key} is not established")
    if evidence.context.lagged_targets:
        missing.append("history/observation-update conditioning is unsupported")
    return missing


def _value(item: CompleteTrajectory, name: str, index: int, time_symbol: str) -> float:
    if name == time_symbol:
        return item.time[index]
    if name in item.covariates:
        return item.covariates[name]
    for group in (item.inputs, item.auxiliaries, item.targets):
        if name in group:
            return group[name][index]
    raise KeyError(name)


def _pair(
    left: CompleteTrajectory,
    i: int,
    right: CompleteTrajectory,
    j: int,
    target: str,
    evidence: TrainingCompatibilityEvidence,
) -> Witness:
    a, b = left.targets[target][i], right.targets[target][j]
    bound = evidence.contract.absolute_error_bounds.get(target)
    difference = Fraction(a) - Fraction(b)
    exact_lower_bound = difference * difference / 2
    try:
        lower_bound = float(exact_lower_bound)
        if Fraction(lower_bound) > exact_lower_bound:
            lower_bound = float(np.nextafter(lower_bound, 0.0))
    except (OverflowError, ValueError):
        lower_bound = None
    return Witness(
        trajectories=(left.trajectory_id, right.trajectory_id),
        sample_indices=(i, j),
        times=(left.time[i], right.time[j]),
        observations=(a, b),
        error_intervals_disjoint=(
            None if bound is None else abs(difference) > 2 * Fraction(bound)
        ),
        pair_squared_error_lower_bound=lower_bound,
    )


def _base(property_: str, target: str, scope: set[str], unavailable: list[str]) -> dict:
    return {
        "property": property_,
        "target": target,
        "scope": tuple(sorted(scope)),
        "unavailable": tuple(sorted(set(unavailable))),
        "status": "deferred" if unavailable else "no_conflict_found",
    }


def _keep_largest_pair(witnesses: list[Witness], candidate: Witness) -> None:
    """Keep one reproducible strongest contrast, without changing logical outcomes."""

    def gap(item: Witness) -> Fraction:
        return abs(Fraction(item.observations[0]) - Fraction(item.observations[1]))

    if not witnesses:
        witnesses.append(candidate)
    elif gap(candidate) > gap(witnesses[0]):
        witnesses[0] = candidate


def _finish(
    row: dict, witnesses: list[Witness], comparisons: int, question: str
) -> Diagnostic:
    row["comparisons"] = comparisons
    if witnesses:
        row.update(
            status="conditional_conflict", witnesses=tuple(witnesses), question=question
        )
        row["assumptions"] = (
            *row.get("assumptions", ()),
            "Exact reproduction of recorded targets is being tested; "
            "this is not a fit-quality gate.",
        )
    elif comparisons == 0 and not row["unavailable"]:
        row.update(
            status="deferred", unavailable=("no qualifying observed comparison",)
        )
    return Diagnostic(**row)


def _algebraic(target, graph, kinds, evidence, graph_errors) -> Diagnostic:
    scope = _closure(target, graph)
    supplied = scope - set(graph)
    missing = [*_common_missing(evidence), *graph_errors]
    if kinds.get(target) != "algebraic":
        missing.append("target is not an algebraic readout")
    if any(kinds.get(v) == "differential" for v in scope):
        missing.append(
            "generated dynamic arguments cannot be replaced by measured histories"
        )
    for name in supplied - {evidence.context.time_symbol}:
        if evidence.contract.absolute_error_bounds.get(name) != 0:
            missing.append(f"exact supplied argument {name} is not established")
    row = _base("algebraic_map", target, scope, missing)
    if missing:
        return Diagnostic(**row)
    groups, witnesses, comparisons = {}, [], 0
    for item in evidence.trajectories:
        for index in range(len(item.time)):
            key = tuple(
                _value(item, n, index, evidence.context.time_symbol)
                for n in sorted(supplied)
            )
            if key not in groups:
                groups[key] = (item, index), (item, index)
                continue
            low, high = groups[key]
            comparisons += 1
            for previous, i in (low, high):
                if previous.targets[target][i] != item.targets[target][index]:
                    _keep_largest_pair(
                        witnesses, _pair(previous, i, item, index, target, evidence)
                    )
            value = item.targets[target][index]
            if value < low[0].targets[target][low[1]]:
                low = item, index
            if value > high[0].targets[target][high[1]]:
                high = item, index
            groups[key] = low, high
    row["verified"] = (
        "All generated ancestors are algebraic; any performed comparison "
        "matches the complete supplied argument tuple exactly.",
    )
    return _finish(
        row,
        witnesses,
        comparisons,
        "These equal complete arguments give different recorded outputs. "
        "Is this an intended approximation, "
        "or which permitted distinguishing dependence or representation should change?",
    )


def _forcing_equal(left, right, sources, evidence, end) -> tuple[bool, list[str]]:
    """Compare the functions defined by all knots, never just a displayed subset."""
    missing = []
    equal = True
    for name in sorted(sources - {evidence.context.time_symbol}):
        if evidence.contract.absolute_error_bounds.get(name) != 0:
            missing.append(f"exact conditioning channel {name} is not established")
        if name in left.covariates:
            equal &= left.covariates[name] == right.covariates[name]
            continue
        mode = evidence.contract.interpolation.get(name, "unknown")
        if mode == "unknown":
            missing.append(f"complete interpolation for {name} is unavailable")
            continue
        union = sorted(
            {t for item in (left, right) for t in item.time if t <= end} | {end}
        )

        def samples(item, name=name, mode=mode, union=union):
            values = item.inputs.get(name, item.auxiliaries.get(name))
            if values is None:
                raise ValueError(f"forcing channel unavailable: {name}")
            if mode == "linear_complete_knots":
                result = []
                for time in union:
                    i = bisect_right(item.time, time) - 1
                    value = Fraction(values[i])
                    if item.time[i] != time and i + 1 < len(item.time):
                        # Exact arithmetic on represented values is intentional:
                        # rounded near-equal histories do not imply equal solutions.
                        fraction = (Fraction(time) - Fraction(item.time[i])) / (
                            Fraction(item.time[i + 1]) - Fraction(item.time[i])
                        )
                        value += fraction * (Fraction(values[i + 1]) - value)
                    result.append(value)
                return result
            return [values[bisect_right(item.time, time) - 1] for time in union]

        equal &= samples(left) == samples(right)
    return bool(equal), missing


def _initial_key(item, state, evidence) -> tuple[tuple | None, list[str]]:
    contract = evidence.contract
    rule = contract.initial_rules.get(state, InitialRule(mode="unknown"))
    if rule.mode == "shared_parameter":
        return ("shared_parameter", state), []
    if rule.mode == "unknown":
        return None, [f"initialization of {state} is unavailable"]
    arguments = (state,) if rule.mode == "observed_initial" else rule.arguments
    if rule.mode == "common_map" and not arguments:
        return None, [f"initialization arguments of {state} are unavailable"]
    missing = []
    values = []
    for name in arguments:
        if (
            name != evidence.context.time_symbol
            and contract.absolute_error_bounds.get(name) != 0
        ):
            missing.append(f"exact initial argument {name} is not established")
        try:
            values.append(_value(item, name, 0, evidence.context.time_symbol))
        except KeyError:
            missing.append(f"initial argument {name} is unavailable")
    return tuple(values), missing


def _dynamic(target, graph, kinds, evidence, graph_errors) -> Diagnostic:
    scope = _closure(target, graph)
    states = {n for n in scope if kinds.get(n) == "differential"}
    sources = scope - set(graph)
    missing = [*_common_missing(evidence), *graph_errors]
    if not states:
        missing.append("no differential ancestor subsystem")
    if evidence.contract.reset_impulse_free is not True:
        missing.append("reset/impulse-free open-loop interval is not established")
    if evidence.contract.uniqueness_assumed is not True:
        missing.append("unique-solution candidate-class hypothesis is unavailable")
    row = _base("closed_subsystem", target, scope, missing)
    if missing:
        return Diagnostic(**row)
    witnesses, comparisons, unavailable = [], 0, []
    for left, right in combinations(evidence.trajectories, 2):
        if left.time[0] != right.time[0]:
            unavailable.append("different time origins; no identical-start comparison")
            continue
        init_equal = True
        pair_missing = []
        for state in sorted(states):
            a, am = _initial_key(left, state, evidence)
            b, bm = _initial_key(right, state, evidence)
            pair_missing.extend(am + bm)
            init_equal &= a == b
        end = min(left.time[-1], right.time[-1])
        forcing_equal, fm = _forcing_equal(left, right, sources, evidence, end)
        pair_missing.extend(fm)
        unavailable.extend(pair_missing)
        if pair_missing or not init_equal or not forcing_equal:
            continue
        other = {t: i for i, t in enumerate(right.time)}
        for i, time in enumerate(left.time):
            if time not in other:
                continue
            j = other[time]
            comparisons += 1
            if left.targets[target][i] != right.targets[target][j]:
                _keep_largest_pair(
                    witnesses, _pair(left, i, right, j, target, evidence)
                )
    row["verified"] = (
        "When performed, conditioning comparisons include all ancestor initializers, "
        "supplied forcing functions and covariates.",
    )
    row["assumptions"] = (
        "Shared deterministic functions and parameters have unique solutions "
        "on the common existence interval.",
    )
    row["unavailable"] = tuple(sorted(set(unavailable)))
    if unavailable and not comparisons:
        row["status"] = "deferred"
    return _finish(
        row,
        witnesses,
        comparisons,
        "This complete subsystem receives identical permitted initial information "
        "and forcing, but recorded targets differ. Identify a missed permitted "
        "distinction, intended approximation, "
        "or the scientific dependence to revise; no particular edge is prescribed.",
    )


def _scalar(target, graph, kinds, evidence, graph_errors) -> Diagnostic:
    scope = _closure(target, graph)
    states = {n for n in scope if kinds.get(n) == "differential"}
    sources = scope - set(graph)
    missing = [*_common_missing(evidence), *graph_errors]
    if kinds.get(target) != "differential" or states != {target}:
        missing.append("target is not a directly observed closed scalar state")
    if evidence.context.time_symbol in sources:
        missing.append("explicit time dependence is not autonomous")
    for key in ("reset_impulse_free", "uniqueness_assumed"):
        if getattr(evidence.contract, key) is not True:
            missing.append(f"{key} is not established")
    row = _base("scalar_autonomous", target, scope, missing)
    if missing:
        return Diagnostic(**row)
    witnesses, comparisons, unavailable = [], 0, []
    for item in evidence.trajectories:
        _, source_missing = _forcing_equal(item, item, sources, evidence, item.time[-1])
        unavailable.extend(source_missing)
        if source_missing:
            continue
        if any(
            len(set(item.inputs.get(n, item.auxiliaries.get(n, ())))) > 1
            for n in sources
            if n not in item.covariates
        ):
            continue
        values = item.targets[target]
        if len(values) < 3:
            continue
        comparisons += 1
        indices, amplitude = _strongest_reversal(values)
        if indices is not None:
            if witnesses:
                previous = witnesses[0].observations
                previous_amplitude = min(
                    abs(Fraction(previous[1]) - Fraction(previous[0])),
                    abs(Fraction(previous[1]) - Fraction(previous[2])),
                )
                if amplitude <= previous_amplitude:
                    continue
            bound = evidence.contract.absolute_error_bounds.get(target)
            robust = None if bound is None else amplitude > 2 * Fraction(bound)
            witnesses[:] = [
                Witness(
                    trajectories=(item.trajectory_id,),
                    sample_indices=indices,
                    times=tuple(item.time[i] for i in indices),
                    observations=tuple(values[i] for i in indices),
                    error_intervals_disjoint=robust,
                )
            ]
    row["verified"] = (
        "Only one differential ancestor; target is the state itself. "
        "Performed comparisons establish constant external arguments.",
    )
    row["assumptions"] = (
        "The scalar autonomous ODE has unique solutions and no resets or impulses.",
    )
    row["unavailable"] = tuple(sorted(set(unavailable)))
    if unavailable and not comparisons:
        row["status"] = "deferred"
    return _finish(
        row,
        witnesses,
        comparisons,
        "The directly observed scalar state reverses direction under constant "
        "arguments. A unique scalar autonomous ODE cannot reproduce this exactly. "
        "Is this measurement variation/intended approximation, "
        "or which permitted coupling or representation is missing?",
    )


def _strongest_reversal(
    values: tuple[float, ...],
) -> tuple[tuple[int, int, int] | None, Fraction]:
    """Largest supported peak/trough prominence in linear time; retain exact indices."""
    prefix_low, prefix_high, suffix_low, suffix_high = [], [], [], []
    low = high = 0
    for i, value in enumerate(values):
        if value < values[low]:
            low = i
        if value > values[high]:
            high = i
        prefix_low.append(low)
        prefix_high.append(high)
    low = high = len(values) - 1
    for i in range(len(values) - 1, -1, -1):
        if values[i] <= values[low]:
            low = i
        if values[i] >= values[high]:
            high = i
        suffix_low.append(low)
        suffix_high.append(high)
    suffix_low.reverse()
    suffix_high.reverse()
    best, amplitude = None, Fraction(0)
    for middle in range(1, len(values) - 1):
        for left, right, sign in (
            (prefix_low[middle - 1], suffix_low[middle + 1], 1),
            (prefix_high[middle - 1], suffix_high[middle + 1], -1),
        ):
            candidate = min(
                sign * (Fraction(values[middle]) - Fraction(values[left])),
                sign * (Fraction(values[middle]) - Fraction(values[right])),
            )
            if candidate > amplitude:
                best, amplitude = (left, middle, right), candidate
    return best, amplitude


def diagnose(
    draft: Draft, evidence: TrainingCompatibilityEvidence
) -> CompatibilityReport:
    """Evaluate three necessary properties without changing a draft or its validity."""
    validate_evidence(evidence)
    graph, kinds, errors = _graph(draft, evidence.context)
    rows = []
    for target in evidence.context.targets:
        target_errors = errors + (
            [] if target in graph else ["target is not generated"]
        )
        for check in (_algebraic, _dynamic, _scalar):
            rows.append(check(target, graph, kinds, evidence, target_errors))
    return CompatibilityReport(
        evidence_sha256=evidence.sha256,
        topology_sha256=content_hash(draft.model_dump(mode="json")),
        diagnostics=tuple(rows),
    )


def compact(report: CompatibilityReport) -> str:
    """Short text projection; full assumptions and witness provenance stay in JSON."""
    lines = ["Training compatibility (conditional advice; not an acceptance gate):"]
    for row in report.diagnostics:
        lines.append(
            f"{row.target} / {row.property}: {row.status}; "
            f"comparisons={row.comparisons}."
        )
        if row.unavailable:
            lines.append("Unavailable: " + "; ".join(row.unavailable))
        if row.assumptions:
            lines.append("Conditional on: " + " ".join(row.assumptions))
        for witness in row.witnesses:
            lines.append(
                f"Evidence: {','.join(witness.trajectories)}; "
                f"t={witness.times}; observed={witness.observations}."
            )
            lines.append(
                "Difference exceeds recorded uncertainty bounds: "
                + {True: "yes", False: "no", None: "unknown"}[
                    witness.error_intervals_disjoint
                ]
                + "."
            )
        if row.question:
            lines.append(row.question)
    return "\n".join(lines)
