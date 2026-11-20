"""Bounded dependency edits for a witnessed contradiction in a differential output.

The two operators change a source set or route it through one new, commonly
initialized state. They preserve every existing edge, sign and equation owner.
Their output describes a graph, before rate laws, fitting or scientific selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal

from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.search import topology_compatibility as compatibility
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
)

Status = Literal["distinguishes", "still_forced_equal", "unresolved", "invalid"]


class TopologyEdit(StrictSchema):
    """One member of the edit grammar shared by enumeration and the proposer."""

    kind: Literal["add_dependency", "via_new_memory"]
    source: Identifier
    target: Identifier


class PairObservation(StrictSchema):
    """A common sample with disjoint reported target error intervals."""

    trajectories: tuple[str, str]
    time: FiniteFloat
    observations: tuple[FiniteFloat, FiniteFloat]


class EditAssessment(StrictSchema):
    """Check of a graph edit, with one remaining conflict per affected pair."""

    edit: TopologyEdit
    status: Status
    reason: str
    memory_state: str | None = None
    # Each edge is (source, receiver).
    added_dependencies: tuple[tuple[str, str], ...] = ()
    distinctions: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    remaining_conflicts: tuple[PairObservation, ...] = ()
    unresolved_pairs: tuple[tuple[str, str], ...] = ()


def _disjoint(a: float, b: float, bound: float) -> bool:
    return (
        a != b if bound == 0 else abs(Fraction(a) - Fraction(b)) > 2 * Fraction(bound)
    )


@dataclass(frozen=True)
class _Comparison:
    sample: PairObservation
    differing: frozenset[str]
    unknown: frozenset[str]


class _RepairCheck:
    """Validate once and reuse prefix comparisons across alternative edits."""

    def __init__(
        self,
        draft: Draft,
        evidence: TrainingCompatibilityEvidence,
        target: str,
        allowed_sources: tuple[str, ...],
        pair: tuple[str, str],
        time: float,
    ) -> None:
        compatibility.validate_evidence(evidence)
        self.evidence = evidence
        graph, kinds, errors = compatibility._graph(draft, evidence.context)
        errors += compatibility._common_missing(evidence)
        if not (
            evidence.contract.reset_impulse_free
            and evidence.contract.uniqueness_assumed
        ):
            errors.append("a reset-free interval with unique solutions is required")
        if (
            target not in evidence.context.targets
            or kinds.get(target) != "differential"
        ):
            errors.append("a generated differential target is required")
        if errors:
            raise ValueError("; ".join(errors))
        self.target, self.graph, self.kinds = target, graph, kinds
        self.allowed = set(allowed_sources)
        self.base_scope = compatibility._closure(target, graph)
        self.domains = (
            set(graph)
            | set(evidence.context.forcing_channels)
            | {evidence.context.time_symbol}
        )
        if self.allowed - self.domains:
            raise ValueError(
                "allowed sources must belong to the generated graph or public context"
            )
        reserved = self.domains | set(evidence.context.unavailable_observed_channels)
        self.memory = "repair_memory"
        suffix = 2
        while self.memory in reserved:
            self.memory = f"repair_memory_{suffix}"
            suffix += 1
        bound = evidence.contract.absolute_error_bounds.get(target)
        if bound is None:
            raise ValueError(f"no target observation error bound for {target}")
        trajectories = {v.trajectory_id: v for v in evidence.trajectories}
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= trajectories.keys():
            raise ValueError("select two distinct public training trajectories")
        self.rows: list[_Comparison] = []
        self.noncomparable: list[tuple[str, str]] = []
        self.selected: _Comparison | None = None
        # Canonical ordering makes scores independent of channel/trajectory order.
        for a, b in combinations(sorted(trajectories), 2):
            left, right = trajectories[a], trajectories[b]
            ri = {t: i for i, t in enumerate(right.time)}
            if left.time[0] != right.time[0]:
                self.noncomparable.append((a, b))
                continue
            samples = (
                (t, left.targets[target][i], right.targets[target][ri[t]])
                for i, t in enumerate(left.time)
                if t in ri
            )
            first = next(
                ((t, x, y) for t, x, y in samples if _disjoint(x, y, bound)), None
            )
            if first is not None:
                self.rows.append(self._compare(left, right, *first))
            if {a, b} == set(pair) and time in left.time and time in ri:
                x = left.targets[target][left.time.index(time)]
                y = right.targets[target][ri[time]]
                if _disjoint(x, y, bound):
                    self.selected = self._compare(left, right, time, x, y)
        if self.selected is None:
            raise ValueError(
                "selected evidence is not a disjoint, matched training sample"
            )
        if self.status(self.base_scope, self.selected) != "still_forced_equal":
            raise ValueError(
                "the parent does not certify equal predictions for the selected sample"
            )

    def _compare(
        self,
        left: CompleteTrajectory,
        right: CompleteTrajectory,
        time: float,
        x: float,
        y: float,
    ) -> _Comparison:
        """Check initial conditions and forcing within the common causal prefix."""
        differing, unknown = set(), set()
        start = left.time[0]
        for name in sorted(self.domains):
            if name in self.graph:
                if self.kinds[name] != "differential" or (
                    time == start and name != self.target
                ):
                    continue
                lk, lm = compatibility._initial_key(left, name, self.evidence)
                rk, rm = compatibility._initial_key(right, name, self.evidence)
                if lm or rm:
                    unknown.add(name)
                elif lk != rk:
                    differing.add(name)
            elif time > start and name != self.evidence.context.time_symbol:
                end = time
                if (
                    self.evidence.contract.interpolation.get(name)
                    == "previous_complete_knots"
                ):
                    # A held input jumping exactly at time cannot change a
                    # continuous state at that same instant.
                    end = max(
                        t for item in (left, right) for t in item.time if t < time
                    )
                equal, missing = compatibility._forcing_equal(
                    left, right, {name}, self.evidence, end
                )
                if missing:
                    unknown.add(name)
                elif not equal:
                    differing.add(name)
        sample = PairObservation(
            trajectories=(left.trajectory_id, right.trajectory_id),
            time=time,
            observations=(x, y),
        )
        return _Comparison(sample, frozenset(differing), frozenset(unknown))

    @staticmethod
    def status(scope: set[str], row: _Comparison) -> Status:
        if scope & row.differing:
            return "distinguishes"
        return "unresolved" if scope & row.unknown else "still_forced_equal"

    def assess(self, edit: TopologyEdit) -> EditAssessment:
        if edit.target != self.target or edit.source not in self.allowed:
            return EditAssessment(
                edit=edit,
                status="invalid",
                reason="Edit outside allowed sources or target.",
            )
        memory = edit.kind == "via_new_memory"
        if memory and self.evidence.context.forbid_latent_states:
            return EditAssessment(
                edit=edit, status="invalid", reason="Latent states are prohibited."
            )
        if not memory and edit.source in self.graph[self.target]:
            return EditAssessment(
                edit=edit, status="invalid", reason="Dependency already present."
            )
        # Both edits add a path from source to target without changing the
        # source's current initial information or forcing. The extra memory has
        # the same initial value on all trajectories; it introduces no new root.
        scope = self.base_scope | compatibility._closure(edit.source, self.graph)
        status = self.status(scope, self.selected)
        reasons = {
            "distinguishes": "Distinguishing information can reach the target.",
            "unresolved": "Unresolved initial information or forcing.",
            "still_forced_equal": "Equal predictions remain necessary at the witness.",
        }
        findings = sorted(scope & self.selected.differing)
        edges = ((edit.source, self.target),)
        if memory:
            edges = (
                (edit.source, self.memory),
                (self.memory, self.memory),
                (self.memory, self.target),
            )
        return EditAssessment(
            edit=edit,
            status=status,
            reason=reasons[status],
            memory_state=self.memory if memory else None,
            added_dependencies=edges,
            distinctions=tuple(
                f"{n} initial information differs"
                if n in self.graph
                else f"{n} forcing differs"
                for n in findings
            ),
            unavailable=tuple(sorted(scope & self.selected.unknown)),
            remaining_conflicts=tuple(
                row.sample
                for row in self.rows
                if self.status(scope, row) == "still_forced_equal"
            ),
            unresolved_pairs=tuple(self.noncomparable)
            + tuple(
                row.sample.trajectories
                for row in self.rows
                if self.status(scope, row) == "unresolved"
            ),
        )


def generate_repairs(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    target: str,
    *,
    allowed_sources: tuple[str, ...],
    pair: tuple[str, str],
    time: float,
) -> tuple[EditAssessment, ...]:
    """Enumerate each bounded edit, including failures, on the same selected witness.

    remaining_conflicts covers every equal-start pair with a disjoint target
    sample; it records the earliest such sample still forced equal. A difference
    in admitted conditioning can only accumulate as this causal prefix grows.
    """
    check = _RepairCheck(draft, evidence, target, allowed_sources, pair, time)
    return tuple(
        check.assess(TopologyEdit(kind=kind, source=source, target=target))
        for source in sorted(check.allowed)
        for kind in ("add_dependency", "via_new_memory")
    )


def assess_edit(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    edit: TopologyEdit,
    *,
    allowed_sources: tuple[str, ...],
    pair: tuple[str, str],
    time: float,
) -> EditAssessment:
    """Score a single proposal with exactly the enumerator's evidence and operators."""
    return _RepairCheck(
        draft, evidence, edit.target, allowed_sources, pair, time
    ).assess(edit)
