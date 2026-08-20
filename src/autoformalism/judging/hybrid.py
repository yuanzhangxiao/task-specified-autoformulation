"""Calibration-only hybrid scientific-judge protocol.

This module owns only public-prompt requirement extraction, certified facts for
the canonical executable candidate, and deterministic aggregation. It never
loads private benchmark truth, fit metrics, or trajectories.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from enum import Enum

from pydantic import Field

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    CandidateModel,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    ProposerClaim,
    RequirementEnforcement,
    RequirementRegistry,
    RequirementSource,
    ScientificRequirement,
)
from autoformalism.schemas.base import Identifier, StrictSchema, UnitInterval

_CANDIDATE_SUBJECT = "candidate"
_SEMANTIC_CANDIDATE_CRITERIA = (
    AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
    AbsoluteCriterion.SINK_ROLES_CONSISTENT,
    AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
    AbsoluteCriterion.MECHANISM_CLAIMS_NOT_CONFLICTING,
    AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED,
    AbsoluteCriterion.CLAIMED_DELAYS_MEANINGFUL,
    AbsoluteCriterion.CLAIMED_SATURATIONS_APPROPRIATE,
    AbsoluteCriterion.PROPOSER_CLAIMS_SUPPORTED,
)


def extract_public_requirements(prompt: str) -> RequirementRegistry:
    """Extract the frozen task-required bullet list from a public prompt.

    Phase-B prompts explicitly introduce these bullets as task-required
    mechanisms. No requirement is inferred when that public marker is absent.
    """
    marker = "The primary objective is to recover the following task-required"
    lines = prompt.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if marker in line)
    except StopIteration:
        return RequirementRegistry()
    requirements: list[ScientificRequirement] = []
    collecting = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("-"):
            collecting = True
            text = stripped.removeprefix("-").strip().rstrip(".")
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            requirements.append(
                ScientificRequirement(
                    requirement_id=f"requirement_{digest}",
                    text=text,
                    source=RequirementSource.BENCHMARK,
                    enforcement=RequirementEnforcement.HARD,
                )
            )
        elif collecting and not stripped:
            break
    return RequirementRegistry(requirements=tuple(requirements))


def candidate_claims(candidate: CandidateModel) -> tuple[ProposerClaim, ...]:
    """Return component-local proposer claims without granting task authority."""
    claims: list[ProposerClaim] = []
    for component in (*candidate.states, *candidate.processes):
        for mechanism in component.mechanisms:
            digest = hashlib.sha256(
                f"{component.name}\0{mechanism}".encode()
            ).hexdigest()[:12]
            claims.append(
                ProposerClaim(
                    claim_id=f"claim_{digest}",
                    subject_id=component.name,
                    mechanism=mechanism,
                )
            )
    return tuple(claims)


def structural_facts(
    candidate: CandidateModel,
    *,
    task_inputs: tuple[str, ...],
) -> dict[str, object]:
    """Return certified dependency facts for one canonical candidate."""
    parser = RestrictedParser()
    graph: dict[str, set[str]] = {}

    def edge(source: str, destination: str) -> None:
        graph.setdefault(source, set()).add(destination)

    expression_symbols: dict[str, set[str]] = {}
    for process in candidate.processes:
        symbols = set(
            parser.parse(
                process.expression, location=f"process:{process.name}"
            ).symbols
        )
        expression_symbols[process.name] = symbols
        for symbol in symbols:
            edge(symbol, process.name)
    for equation in candidate.state_equations:
        symbols = set(
            parser.parse(
                equation.rhs, location=f"equation:{equation.state}"
            ).symbols
        )
        expression_symbols[equation.state] = symbols
        for symbol in symbols:
            edge(symbol, equation.state)
    target_nodes: dict[str, str] = {}
    for mapping in candidate.observation_mappings:
        node = f"target_{mapping.channel}"
        target_nodes[node] = mapping.channel
        for symbol in parser.parse(
            mapping.expression, location=f"observation:{mapping.channel}"
        ).symbols:
            edge(symbol, node)

    def paths_from(source: str) -> dict[str, list[str]]:
        pending = deque([(source, [source])])
        visited = {source}
        paths: dict[str, list[str]] = {}
        while pending:
            current, path = pending.popleft()
            for destination in sorted(graph.get(current, ())):
                next_path = [*path, destination]
                if destination in target_nodes:
                    paths.setdefault(target_nodes[destination], next_path)
                elif destination not in visited:
                    visited.add(destination)
                    pending.append((destination, next_path))
        return paths

    parameters = {item.name for item in candidate.parameters}
    state_kind = {item.name: item.kind.value for item in candidate.states}
    components: dict[str, object] = {}
    for name, kind in sorted(
        {
            **{item.name: "state" for item in candidate.states},
            **{item.name: "process" for item in candidate.processes},
        }.items()
    ):
        paths = paths_from(name)
        drivers = sorted(
            expression_symbols.get(name, set()) - parameters - {name, "t"}
        )
        components[name] = {
            "component_kind": kind,
            "state_kind": state_kind.get(name),
            "drivers": drivers,
            "reaches_requested_target": bool(paths),
            "target_paths": paths,
        }
    return {
        "task_inputs": {
            name: {
                "reaches_requested_target": bool(paths_from(name)),
                "target_paths": paths_from(name),
            }
            for name in task_inputs
        },
        "components": components,
        "proposer_claims": [
            item.model_dump(mode="json") for item in candidate_claims(candidate)
        ],
    }


def _runtime_candidate_assessments(
    candidate: CandidateModel,
    *,
    task_inputs: tuple[str, ...],
) -> dict[AbsoluteCriterion, CandidateAbsoluteAssessment]:
    facts = structural_facts(candidate, task_inputs=task_inputs)
    components = facts["components"]
    assert isinstance(components, dict)
    inputs = facts["task_inputs"]
    assert isinstance(inputs, dict)
    claims = candidate_claims(candidate)
    claimed_subjects = sorted({item.subject_id for item in claims})
    latent_subjects = sorted(
        item.name for item in candidate.states if item.kind.value == "latent"
    )

    def all_or_na(
        subjects: list[str],
        predicate,
        *,
        noun: str,
    ) -> CandidateAbsoluteAssessment:
        if not subjects:
            return CandidateAbsoluteAssessment(
                verdict=AbsoluteVerdict.NOT_APPLICABLE,
                evidence=f"No {noun} are present in the canonical candidate.",
            )
        failures = [name for name in subjects if not predicate(name)]
        return CandidateAbsoluteAssessment(
            verdict=(AbsoluteVerdict.FAIL if failures else AbsoluteVerdict.PASS),
            evidence=(
                f"Failing {noun}: {', '.join(failures)}."
                if failures
                else f"All {noun} satisfy the certified dependency predicate."
            ),
        )

    return {
        AbsoluteCriterion.TASK_INPUTS_REACH_TARGETS: all_or_na(
            list(inputs),
            lambda name: bool(inputs[name]["reaches_requested_target"]),
            noun="declared task inputs",
        ),
        AbsoluteCriterion.CLAIMED_COMPONENTS_REACH_TARGETS: all_or_na(
            claimed_subjects,
            lambda name: bool(components[name]["reaches_requested_target"]),
            noun="components carrying proposer mechanism claims",
        ),
        AbsoluteCriterion.LATENT_STATES_HAVE_INCOMING_PATHWAYS: all_or_na(
            latent_subjects,
            lambda name: bool(components[name]["drivers"]),
            noun="latent states",
        ),
        AbsoluteCriterion.LATENT_STATES_REACH_TARGETS: all_or_na(
            latent_subjects,
            lambda name: bool(components[name]["reaches_requested_target"]),
            noun="latent states",
        ),
    }


def deterministic_pair_assessments(
    candidate_a: CandidateModel,
    candidate_b: CandidateModel,
    *,
    task_inputs: tuple[str, ...],
) -> tuple[PairedAbsoluteAssessment, ...]:
    """Evaluate exact absolute predicates for both canonical candidates."""
    left = _runtime_candidate_assessments(candidate_a, task_inputs=task_inputs)
    right = _runtime_candidate_assessments(candidate_b, task_inputs=task_inputs)
    return tuple(
        PairedAbsoluteAssessment(
            criterion=criterion,
            subject_id=_CANDIDATE_SUBJECT,
            candidate_a=left[criterion],
            candidate_b=right[criterion],
        )
        for criterion in left
    )


def semantic_absolute_units(
    requirements: RequirementRegistry,
) -> tuple[tuple[AbsoluteCriterion, str], ...]:
    """Return the exact semantic units that the LLM must assess."""
    units: list[tuple[AbsoluteCriterion, str]] = []
    for requirement in requirements.requirements:
        units.extend(
            (
                (
                    AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED,
                    requirement.requirement_id,
                ),
                (
                    AbsoluteCriterion.REQUIRED_MECHANISM_CONNECTED,
                    requirement.requirement_id,
                ),
            )
        )
    units.extend(
        (criterion, _CANDIDATE_SUBJECT)
        for criterion in _SEMANTIC_CANDIDATE_CRITERIA
    )
    return tuple(units)


class GroupKind(str, Enum):
    """Generic group templates that never encode hidden mechanisms."""

    USER_REQUIREMENT = "user_requirement"
    TASK_CONNECTIVITY = "task_connectivity"
    CLAIM_INTEGRITY = "claim_integrity"
    BALANCE_SEMANTICS = "balance_semantics"
    LATENT_VALIDITY = "latent_validity"
    DYNAMIC_CLAIMS = "dynamic_claims"


@dataclass(frozen=True)
class GroupDefinition:
    """One conjunction over absolute assessment keys."""

    group_id: str
    kind: GroupKind
    weight: float
    keys: tuple[tuple[AbsoluteCriterion, str], ...]
    enforcement: RequirementEnforcement = RequirementEnforcement.SOFT


@dataclass(frozen=True)
class HybridScoringConfig:
    """Exploratory weights retained outside production search."""

    task_connectivity_weight: float = 0.5
    claim_integrity_weight: float = 0.25
    balance_weight: float = 0.5
    latent_weight: float = 0.25
    dynamic_claim_weight: float = 0.25
    partial_tiebreak_weight: float = 0.05
    comparative_weight: float = 0.25
    tie_threshold: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.task_connectivity_weight,
            self.claim_integrity_weight,
            self.balance_weight,
            self.latent_weight,
            self.dynamic_claim_weight,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("group weights must be positive")
        if not 0.0 <= self.partial_tiebreak_weight < 1.0:
            raise ValueError("partial tiebreak weight must be in [0, 1)")
        if self.comparative_weight < 0.0:
            raise ValueError("comparative weight must be nonnegative")
        if not 0.0 <= self.tie_threshold <= 1.0:
            raise ValueError("tie threshold must be in [0, 1]")


def build_group_registry(
    requirements: RequirementRegistry,
    config: HybridScoringConfig | None = None,
) -> tuple[GroupDefinition, ...]:
    """Instantiate generic groups from public requirements and candidate schema."""
    config = config or HybridScoringConfig()
    groups = [
        GroupDefinition(
            group_id=f"requirement_{item.requirement_id}",
            kind=GroupKind.USER_REQUIREMENT,
            weight=item.weight,
            enforcement=item.enforcement,
            keys=(
                (
                    AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED,
                    item.requirement_id,
                ),
                (
                    AbsoluteCriterion.REQUIRED_MECHANISM_CONNECTED,
                    item.requirement_id,
                ),
            ),
        )
        for item in requirements.requirements
        if item.enforcement is not RequirementEnforcement.DESCRIPTIVE
    ]
    groups.extend(
        (
            GroupDefinition(
                "task_connectivity",
                GroupKind.TASK_CONNECTIVITY,
                config.task_connectivity_weight,
                ((AbsoluteCriterion.TASK_INPUTS_REACH_TARGETS, _CANDIDATE_SUBJECT),),
            ),
            GroupDefinition(
                "claim_integrity",
                GroupKind.CLAIM_INTEGRITY,
                config.claim_integrity_weight,
                (
                    (AbsoluteCriterion.PROPOSER_CLAIMS_SUPPORTED, _CANDIDATE_SUBJECT),
                    (
                        AbsoluteCriterion.CLAIMED_COMPONENTS_REACH_TARGETS,
                        _CANDIDATE_SUBJECT,
                    ),
                    (
                        AbsoluteCriterion.MECHANISM_CLAIMS_NOT_CONFLICTING,
                        _CANDIDATE_SUBJECT,
                    ),
                ),
            ),
            GroupDefinition(
                "balance_semantics",
                GroupKind.BALANCE_SEMANTICS,
                config.balance_weight,
                (
                    (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, _CANDIDATE_SUBJECT),
                    (AbsoluteCriterion.SINK_ROLES_CONSISTENT, _CANDIDATE_SUBJECT),
                    (
                        AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
                        _CANDIDATE_SUBJECT,
                    ),
                ),
            ),
            GroupDefinition(
                "latent_validity",
                GroupKind.LATENT_VALIDITY,
                config.latent_weight,
                (
                    (
                        AbsoluteCriterion.LATENT_STATES_HAVE_INCOMING_PATHWAYS,
                        _CANDIDATE_SUBJECT,
                    ),
                    (
                        AbsoluteCriterion.LATENT_STATES_REACH_TARGETS,
                        _CANDIDATE_SUBJECT,
                    ),
                    (
                        AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED,
                        _CANDIDATE_SUBJECT,
                    ),
                ),
            ),
            GroupDefinition(
                "dynamic_claims",
                GroupKind.DYNAMIC_CLAIMS,
                config.dynamic_claim_weight,
                (
                    (AbsoluteCriterion.CLAIMED_DELAYS_MEANINGFUL, _CANDIDATE_SUBJECT),
                    (
                        AbsoluteCriterion.CLAIMED_SATURATIONS_APPROPRIATE,
                        _CANDIDATE_SUBJECT,
                    ),
                ),
            ),
        )
    )
    return tuple(groups)


class GroupOutcome(StrictSchema):
    """Deterministic result for one conjunctive requirement group."""

    group_id: Identifier
    kind: str
    weight: float = Field(gt=0.0)
    complete: UnitInterval | None
    partial: UnitInterval | None
    coverage: UnitInterval


class CandidateHybridScore(StrictSchema):
    """Candidate-specific scientific score and uncertainty."""

    hard_requirement_status: bool | None
    conjunctive_score: UnitInterval | None
    partial_score: UnitInterval | None
    shaped_score: UnitInterval | None
    coverage: UnitInterval
    groups: tuple[GroupOutcome, ...]


class HybridPairScore(StrictSchema):
    """Runtime-owned pair score; the LLM never emits these values."""

    candidate_a: CandidateHybridScore
    candidate_b: CandidateHybridScore
    relative_preference_for_a: UnitInterval | None
    decision_value: float | None = Field(ge=-2.0, le=2.0)
    preferred: str


def _candidate_score(
    assessments: dict[tuple[AbsoluteCriterion, str], PairedAbsoluteAssessment],
    groups: tuple[GroupDefinition, ...],
    *,
    side: str,
    partial_weight: float,
) -> CandidateHybridScore:
    outcomes: list[GroupOutcome] = []
    hard_statuses: list[bool | None] = []
    for group in groups:
        verdicts = []
        for key in group.keys:
            item = assessments.get(key)
            if item is None:
                verdicts.append(AbsoluteVerdict.INDETERMINATE)
            else:
                verdicts.append(getattr(item, side).verdict)
        applicable = [
            value for value in verdicts if value is not AbsoluteVerdict.NOT_APPLICABLE
        ]
        known = [
            value
            for value in applicable
            if value is not AbsoluteVerdict.INDETERMINATE
        ]
        if not applicable:
            complete = None
            partial = None
            coverage = 1.0
        else:
            coverage = len(known) / len(applicable)
            partial = (
                sum(value is AbsoluteVerdict.PASS for value in known) / len(known)
                if known
                else None
            )
            if AbsoluteVerdict.FAIL in known:
                complete = 0.0
            elif len(known) == len(applicable):
                complete = 1.0
            else:
                complete = None
        outcomes.append(
            GroupOutcome(
                group_id=group.group_id,
                kind=group.kind.value,
                weight=group.weight,
                complete=complete,
                partial=partial,
                coverage=coverage,
            )
        )
        if group.enforcement is RequirementEnforcement.HARD:
            hard_statuses.append(None if complete is None else bool(complete))

    applicable_outcomes = [item for item in outcomes if item.partial is not None]
    total_weight = sum(item.weight for item in applicable_outcomes)
    determined = [item for item in applicable_outcomes if item.complete is not None]
    determined_weight = sum(item.weight for item in determined)
    conjunctive = (
        sum(item.weight * float(item.complete) for item in determined)
        / determined_weight
        if determined_weight
        else None
    )
    partial = (
        sum(item.weight * float(item.partial) for item in applicable_outcomes)
        / total_weight
        if total_weight
        else None
    )
    shaped = (
        (conjunctive + partial_weight * partial) / (1.0 + partial_weight)
        if conjunctive is not None and partial is not None
        else None
    )
    hard_status: bool | None
    if any(value is False for value in hard_statuses):
        hard_status = False
    elif hard_statuses and all(value is True for value in hard_statuses):
        hard_status = True
    else:
        hard_status = None
    return CandidateHybridScore(
        hard_requirement_status=hard_status,
        conjunctive_score=conjunctive,
        partial_score=partial,
        shaped_score=shaped,
        coverage=(determined_weight / total_weight if total_weight else 0.0),
        groups=tuple(outcomes),
    )


def score_hybrid_pair(
    result: HybridJudgeResult,
    deterministic: tuple[PairedAbsoluteAssessment, ...],
    requirements: RequirementRegistry,
    config: HybridScoringConfig | None = None,
) -> HybridPairScore:
    """Combine absolute conjunctions and a separate comparative residual."""
    config = config or HybridScoringConfig()
    combined = (*deterministic, *result.absolute_assessments)
    index: dict[tuple[AbsoluteCriterion, str], PairedAbsoluteAssessment] = {}
    for item in combined:
        key = (item.criterion, item.subject_id)
        if key in index:
            raise ValueError(f"duplicate assessment key: {key}")
        index[key] = item
    groups = build_group_registry(requirements, config)
    left = _candidate_score(
        index,
        groups,
        side="candidate_a",
        partial_weight=config.partial_tiebreak_weight,
    )
    right = _candidate_score(
        index,
        groups,
        side="candidate_b",
        partial_weight=config.partial_tiebreak_weight,
    )
    relative = result.numeric_relative_preference
    if left.hard_requirement_status is True and right.hard_requirement_status is False:
        return HybridPairScore(
            candidate_a=left,
            candidate_b=right,
            relative_preference_for_a=relative,
            decision_value=1.0,
            preferred="candidate_a",
        )
    if left.hard_requirement_status is False and right.hard_requirement_status is True:
        return HybridPairScore(
            candidate_a=left,
            candidate_b=right,
            relative_preference_for_a=relative,
            decision_value=-1.0,
            preferred="candidate_b",
        )
    if left.shaped_score is None or right.shaped_score is None:
        decision = None
    else:
        comparative_delta = 0.0 if relative is None else 2.0 * relative - 1.0
        decision = (
            left.shaped_score
            - right.shaped_score
            + config.comparative_weight * comparative_delta
        )
    preferred = (
        "indeterminate"
        if decision is None
        else (
            "candidate_a"
            if decision > config.tie_threshold
            else (
                "candidate_b"
                if decision < -config.tie_threshold
                else "tie"
            )
        )
    )
    return HybridPairScore(
        candidate_a=left,
        candidate_b=right,
        relative_preference_for_a=relative,
        decision_value=decision,
        preferred=preferred,
    )
