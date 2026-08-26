"""Calibration-only hybrid scientific-judge protocol.

This module owns only public-prompt requirement extraction, certified facts for
the canonical executable candidate, and deterministic aggregation. It never
loads private benchmark truth, fit metrics, or trajectories.
"""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Literal

from pydantic import Field

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    AtomicJudgeResult,
    CandidateAbsoluteAssessment,
    CandidateModel,
    ExpectedContributionDirection,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    ProposerClaim,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RepeatedContributionRelation,
    RequirementEnforcement,
    RequirementRegistry,
    RequirementSource,
    ScientificRequirement,
)
from autoformalism.schemas.base import Identifier, StrictSchema, UnitInterval

_CANDIDATE_SUBJECT = "candidate"
STRUCTURAL_FACTS_SCHEMA_VERSION = "structural-facts-2"
ATOMIC_EVIDENCE_SCHEMA_VERSION = "atomic-evidence-plan-1"
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
_MODEL_SEMANTIC_CANDIDATE_CRITERIA = (
    AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT,
    AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT,
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


def _signed_additive_terms(
    node: ast.expr, polarity: int = 1
) -> list[tuple[int, ast.expr]]:
    """Flatten top-level addition while preserving deterministic polarity."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [
            *_signed_additive_terms(node.left, polarity),
            *_signed_additive_terms(node.right, polarity),
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return [
            *_signed_additive_terms(node.left, polarity),
            *_signed_additive_terms(node.right, -polarity),
        ]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _signed_additive_terms(node.operand, -polarity)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _signed_additive_terms(node.operand, polarity)
    return [(polarity, node)]


def _algebraic_expression_facts(
    source: str,
    *,
    location: str,
    parser: RestrictedParser,
) -> dict[str, object]:
    """Return syntax-only additive, polarity, and exact-repeat facts."""
    parsed = parser.parse(source, location=location)
    raw_terms = _signed_additive_terms(parsed.tree.body)
    terms = []
    repeat_groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    symbol_occurrences: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"positive_term_ids": [], "negative_term_ids": []}
    )
    for index, (polarity, node) in enumerate(raw_terms):
        term_id = f"term_{index}"
        normalized = ast.unparse(ast.fix_missing_locations(node))
        term_symbols = sorted(
            parser.parse(normalized, location=f"{location}:{term_id}").symbols
        )
        polarity_name = "positive" if polarity > 0 else "negative"
        terms.append(
            {
                "term_id": term_id,
                "polarity": polarity_name,
                "normalized_expression": normalized,
                "symbols": term_symbols,
            }
        )
        fingerprint = ast.dump(node, annotate_fields=True, include_attributes=False)
        repeat_groups[(polarity, fingerprint)].append(term_id)
        occurrence_key = f"{polarity_name}_term_ids"
        for symbol in term_symbols:
            symbol_occurrences[symbol][occurrence_key].append(term_id)
    repeated = []
    terms_by_id = {str(item["term_id"]): item for item in terms}
    for (polarity, _fingerprint), term_ids in repeat_groups.items():
        if len(term_ids) < 2:
            continue
        exemplar = terms_by_id[term_ids[0]]
        repeated.append(
            {
                "polarity": "positive" if polarity > 0 else "negative",
                "normalized_expression": exemplar["normalized_expression"],
                "count": len(term_ids),
                "term_ids": term_ids,
            }
        )
    return {
        "source_expression": source,
        "top_level_additive_terms": terms,
        "signed_symbol_occurrences": {
            symbol: {
                **occurrences,
                "positive_term_count": len(occurrences["positive_term_ids"]),
                "negative_term_count": len(occurrences["negative_term_ids"]),
            }
            for symbol, occurrences in sorted(symbol_occurrences.items())
        },
        "exact_repeated_additive_terms": sorted(
            repeated,
            key=lambda item: (
                str(item["polarity"]), str(item["normalized_expression"])
            ),
        ),
    }


def structural_facts(
    candidate: CandidateModel,
    *,
    task_inputs: tuple[str, ...],
) -> dict[str, object]:
    """Return certified dependency facts for one canonical candidate."""
    parser = RestrictedParser()
    graph: dict[str, set[str]] = {}
    algebraic_expressions: dict[str, object] = {}

    def edge(source: str, destination: str) -> None:
        graph.setdefault(source, set()).add(destination)

    expression_symbols: dict[str, set[str]] = {}
    for process in candidate.processes:
        location = f"process:{process.name}"
        parsed = parser.parse(process.expression, location=location)
        symbols = set(parsed.symbols)
        algebraic_expressions[location] = _algebraic_expression_facts(
            process.expression,
            location=location,
            parser=parser,
        )
        expression_symbols[process.name] = symbols
        for symbol in symbols:
            edge(symbol, process.name)
    for equation in candidate.state_equations:
        location = f"equation:{equation.state}"
        parsed = parser.parse(equation.rhs, location=location)
        symbols = set(parsed.symbols)
        algebraic_expressions[location] = _algebraic_expression_facts(
            equation.rhs,
            location=location,
            parser=parser,
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
        "schema_version": STRUCTURAL_FACTS_SCHEMA_VERSION,
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
        "algebraic_expressions": algebraic_expressions,
    }


@dataclass(frozen=True)
class SignedOccurrence:
    """One additive state-equation term whose outer sign can be withheld."""

    occurrence_id: str
    candidate_side: str
    equation_location: str
    governed_quantity: str
    unsigned_expression: str
    symbols: tuple[str, ...]
    actual_polarity: str

    def prompt_payload(self) -> dict[str, object]:
        """Return the scientific question without exposing actual polarity."""
        return {
            "occurrence_id": self.occurrence_id,
            "candidate_side": self.candidate_side,
            "equation_location": self.equation_location,
            "governed_quantity": self.governed_quantity,
            "unsigned_expression": self.unsigned_expression,
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True)
class ExactRepeatCandidate:
    """One pair of exact same-polarity terms needing scientific interpretation."""

    repeat_pair_id: str
    candidate_side: str
    equation_location: str
    governed_quantity: str
    unsigned_expression: str
    occurrence_ids: tuple[str, str]

    def prompt_payload(self) -> dict[str, object]:
        """Return an atomic repeat question without declaring duplication."""
        return {
            "repeat_pair_id": self.repeat_pair_id,
            "candidate_side": self.candidate_side,
            "equation_location": self.equation_location,
            "governed_quantity": self.governed_quantity,
            "unsigned_expression": self.unsigned_expression,
            "occurrence_ids": list(self.occurrence_ids),
        }


@dataclass(frozen=True)
class AtomicEvidencePlan:
    """Sign-blinded questions plus runtime-private occurrence polarities."""

    occurrences: tuple[SignedOccurrence, ...]
    repeat_candidates: tuple[ExactRepeatCandidate, ...]

    @property
    def occurrence_ids(self) -> set[str]:
        """Return the exact expected occurrence identifiers."""
        return {item.occurrence_id for item in self.occurrences}

    @property
    def repeat_pair_ids(self) -> set[str]:
        """Return the exact expected repeat-pair identifiers."""
        return {item.repeat_pair_id for item in self.repeat_candidates}

    def prompt_payload(self) -> dict[str, object]:
        """Return only sign-blinded facts safe for the first LLM stage."""
        return {
            "schema_version": ATOMIC_EVIDENCE_SCHEMA_VERSION,
            "signed_occurrences": [
                item.prompt_payload() for item in self.occurrences
            ],
            "exact_repeat_candidates": [
                item.prompt_payload() for item in self.repeat_candidates
            ],
        }


def _candidate_atomic_occurrences(
    candidate: CandidateModel,
    *,
    candidate_side: str,
) -> tuple[tuple[SignedOccurrence, ...], tuple[ExactRepeatCandidate, ...]]:
    """Build stable sign-blinded units for one canonical candidate."""
    parser = RestrictedParser()
    occurrences: list[SignedOccurrence] = []
    repeats: list[ExactRepeatCandidate] = []
    expressions = [
        (f"process:{item.name}", item.name, item.expression)
        for item in candidate.processes
    ]
    expressions.extend(
        (f"equation:{item.state}", item.state, item.rhs)
        for item in candidate.state_equations
    )
    for location, governed_quantity, source_expression in expressions:
        facts = _algebraic_expression_facts(
            source_expression,
            location=location,
            parser=parser,
        )
        by_term_id: dict[str, SignedOccurrence] = {}
        for term in facts["top_level_additive_terms"]:
            assert isinstance(term, dict)
            term_id = str(term["term_id"])
            expression = str(term["normalized_expression"])
            digest = hashlib.sha256(
                f"{candidate_side}\0{location}\0{term_id}\0{expression}".encode()
            ).hexdigest()[:16]
            occurrence = SignedOccurrence(
                occurrence_id=f"occurrence_{digest}",
                candidate_side=candidate_side,
                equation_location=location,
                governed_quantity=governed_quantity,
                unsigned_expression=expression,
                symbols=tuple(str(item) for item in term["symbols"]),
                actual_polarity=str(term["polarity"]),
            )
            occurrences.append(occurrence)
            by_term_id[term_id] = occurrence
        for group in facts["exact_repeated_additive_terms"]:
            assert isinstance(group, dict)
            term_ids = tuple(str(item) for item in group["term_ids"])
            for left_id, right_id in combinations(term_ids, 2):
                left = by_term_id[left_id]
                right = by_term_id[right_id]
                digest = hashlib.sha256(
                    f"{left.occurrence_id}\0{right.occurrence_id}".encode()
                ).hexdigest()[:16]
                repeats.append(
                    ExactRepeatCandidate(
                        repeat_pair_id=f"repeat_{digest}",
                        candidate_side=candidate_side,
                        equation_location=location,
                        governed_quantity=governed_quantity,
                        unsigned_expression=str(group["normalized_expression"]),
                        occurrence_ids=(left.occurrence_id, right.occurrence_id),
                    )
                )
    return tuple(occurrences), tuple(repeats)


def build_atomic_evidence_plan(
    candidate_a: CandidateModel,
    candidate_b: CandidateModel,
) -> AtomicEvidencePlan:
    """Build symmetric atomic questions while retaining signs only at runtime."""
    left_occurrences, left_repeats = _candidate_atomic_occurrences(
        candidate_a, candidate_side="candidate_a"
    )
    right_occurrences, right_repeats = _candidate_atomic_occurrences(
        candidate_b, candidate_side="candidate_b"
    )
    return AtomicEvidencePlan(
        occurrences=(*left_occurrences, *right_occurrences),
        repeat_candidates=(*left_repeats, *right_repeats),
    )


def atomic_candidate_context(candidate: CandidateModel) -> dict[str, object]:
    """Return component meaning without exposing state-equation outer signs."""
    return {
        "states": [
            {
                "name": item.name,
                "kind": item.kind.value,
                "description": item.description,
                "mechanisms": list(item.mechanisms),
            }
            for item in candidate.states
        ],
        "processes": [
            {
                "name": item.name,
                "description": item.description,
                "mechanisms": list(item.mechanisms),
            }
            for item in candidate.processes
        ],
        "proposer_claims": [
            item.model_dump(mode="json") for item in candidate_claims(candidate)
        ],
    }


def _direction_assessment(
    *,
    side: str,
    expected_direction: ExpectedContributionDirection,
    result: AtomicJudgeResult,
    plan: AtomicEvidencePlan,
) -> CandidateAbsoluteAssessment:
    """Compare LLM-inferred direction with runtime-private certified polarity."""
    answers = {
        item.occurrence_id: item for item in result.signed_occurrence_assessments
    }
    side_occurrences = [
        item for item in plan.occurrences if item.candidate_side == side
    ]
    relevant = [
        item
        for item in side_occurrences
        if answers[item.occurrence_id].expected_direction is expected_direction
    ]
    unresolved = [
        item
        for item in side_occurrences
        if answers[item.occurrence_id].expected_direction
        in {
            ExpectedContributionDirection.CONTEXT_DEPENDENT,
            ExpectedContributionDirection.INSUFFICIENT_PUBLIC_INFORMATION,
        }
    ]
    expected_polarity = (
        "positive"
        if expected_direction is ExpectedContributionDirection.POSITIVE
        else "negative"
    )
    mismatched = [
        item for item in relevant if item.actual_polarity != expected_polarity
    ]
    if mismatched:
        return CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.FAIL,
            evidence=(
                f"Expected {expected_polarity} contribution but certified the "
                f"opposite outer polarity for: "
                f"{', '.join(item.occurrence_id for item in mismatched)}."
            ),
        )
    relevant_quantities = {item.governed_quantity for item in relevant}
    relevant_unresolved = [
        item for item in unresolved if item.governed_quantity in relevant_quantities
    ]
    if relevant_unresolved:
        return CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.INDETERMINATE,
            evidence=(
                "At least one occurrence in a quantity with a determinate role "
                "lacks an expected direction; unresolved occurrences: "
                f"{', '.join(item.occurrence_id for item in relevant_unresolved)}."
            ),
        )
    if relevant:
        return CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.PASS,
            evidence=(
                f"All {len(relevant)} occurrences inferred as {expected_polarity} "
                "contributions have matching certified outer polarity."
            ),
        )
    if unresolved:
        return CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.INDETERMINATE,
            evidence=(
                "No occurrence received this determinate role; unresolved "
                "occurrences remain: "
                f"{', '.join(item.occurrence_id for item in unresolved)}."
            ),
        )
    return CandidateAbsoluteAssessment(
        verdict=AbsoluteVerdict.NOT_APPLICABLE,
        evidence=f"No occurrence was inferred as a {expected_polarity} contribution.",
    )


def atomic_role_compatibility_assessments(
    result: AtomicJudgeResult,
    plan: AtomicEvidencePlan,
) -> tuple[PairedAbsoluteAssessment, PairedAbsoluteAssessment]:
    """Derive candidate-level source/sink consistency from atomic directions."""
    result.validate_expected_units(
        occurrence_ids=plan.occurrence_ids,
        repeat_pair_ids=plan.repeat_pair_ids,
    )
    assessments = []
    for criterion, direction in (
        (
            AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
            ExpectedContributionDirection.POSITIVE,
        ),
        (
            AbsoluteCriterion.SINK_ROLES_CONSISTENT,
            ExpectedContributionDirection.NEGATIVE,
        ),
    ):
        assessments.append(
            PairedAbsoluteAssessment(
                criterion=criterion,
                subject_id=_CANDIDATE_SUBJECT,
                candidate_a=_direction_assessment(
                    side="candidate_a",
                    expected_direction=direction,
                    result=result,
                    plan=plan,
                ),
                candidate_b=_direction_assessment(
                    side="candidate_b",
                    expected_direction=direction,
                    result=result,
                    plan=plan,
                ),
            )
        )
    return assessments[0], assessments[1]


def atomic_findings_payload(
    result: AtomicJudgeResult,
    plan: AtomicEvidencePlan,
    role_assessments: tuple[PairedAbsoluteAssessment, ...],
) -> dict[str, object]:
    """Expose stage-one findings, now including polarity compatibility, to stage two."""
    repeat_side = {
        item.repeat_pair_id: item.candidate_side
        for item in plan.repeat_candidates
    }
    return {
        "atomic_evidence_plan": plan.prompt_payload(),
        "certified_outer_polarities": {
            item.occurrence_id: item.actual_polarity
            for item in plan.occurrences
        },
        "signed_occurrence_inferences": [
            item.model_dump(mode="json")
            for item in result.signed_occurrence_assessments
        ],
        "runtime_role_compatibility": [
            item.model_dump(mode="json") for item in role_assessments
        ],
        "exact_repeat_interpretations": [
            {
                **item.model_dump(mode="json"),
                "candidate_side": repeat_side[item.repeat_pair_id],
            }
            for item in result.repeated_contribution_assessments
        ],
    }


def merge_atomic_assessments(
    result: HybridJudgeResult,
    atomic_result: AtomicJudgeResult,
    plan: AtomicEvidencePlan,
    role_assessments: tuple[PairedAbsoluteAssessment, ...],
) -> HybridJudgeResult:
    """Replace broad role answers and enforce determinate exact-repeat failures."""
    role_criteria = {
        AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
        AbsoluteCriterion.SINK_ROLES_CONSISTENT,
    }
    assessments = {
        (item.criterion, item.subject_id): item
        for item in result.absolute_assessments
        if item.criterion not in role_criteria
    }
    for item in role_assessments:
        assessments[(item.criterion, item.subject_id)] = item
    duplicate_key = (
        AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
        _CANDIDATE_SUBJECT,
    )
    duplicate = assessments.get(duplicate_key)
    if duplicate is not None:
        repeat_sides = {
            item.repeat_pair_id: item.candidate_side
            for item in plan.repeat_candidates
        }
        same_by_side: dict[str, list[str]] = defaultdict(list)
        for item in atomic_result.repeated_contribution_assessments:
            if (
                item.relation
                is RepeatedContributionRelation.SAME_PHYSICAL_CONTRIBUTION
            ):
                same_by_side[repeat_sides[item.repeat_pair_id]].append(
                    item.repeat_pair_id
                )

        def repeated_override(
            side: str, current: CandidateAbsoluteAssessment
        ) -> CandidateAbsoluteAssessment:
            identifiers = same_by_side.get(side, [])
            if not identifiers:
                return current
            return CandidateAbsoluteAssessment(
                verdict=AbsoluteVerdict.FAIL,
                evidence=(
                    "Atomic repeat assessment identified the same physical "
                    f"contribution in: {', '.join(identifiers)}."
                ),
            )

        assessments[duplicate_key] = PairedAbsoluteAssessment(
            criterion=duplicate.criterion,
            subject_id=duplicate.subject_id,
            candidate_a=repeated_override("candidate_a", duplicate.candidate_a),
            candidate_b=repeated_override("candidate_b", duplicate.candidate_b),
        )
    payload = result.model_dump(mode="json")
    payload["absolute_assessments"] = [
        item.model_dump(mode="json") for item in assessments.values()
    ]
    return HybridJudgeResult.model_validate(payload)


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
    *,
    include_role_consistency: bool = True,
    include_model_semantics: bool = False,
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
        if include_role_consistency
        or criterion
        not in {
            AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
            AbsoluteCriterion.SINK_ROLES_CONSISTENT,
        }
    )
    if include_model_semantics:
        units.extend(
            (criterion, _CANDIDATE_SUBJECT)
            for criterion in _MODEL_SEMANTIC_CANDIDATE_CRITERIA
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
    comparative_indeterminate_policy: Literal[
        "exclude", "neutral_fixed_denominator"
    ] = "exclude"
    include_model_semantics: bool = False

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
        if self.comparative_indeterminate_policy not in {
            "exclude",
            "neutral_fixed_denominator",
        }:
            raise ValueError("unsupported comparative indeterminate policy")


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
    balance_keys = [
        (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, _CANDIDATE_SUBJECT),
        (AbsoluteCriterion.SINK_ROLES_CONSISTENT, _CANDIDATE_SUBJECT),
        (AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED, _CANDIDATE_SUBJECT),
    ]
    dynamic_keys = [
        (AbsoluteCriterion.CLAIMED_DELAYS_MEANINGFUL, _CANDIDATE_SUBJECT),
        (AbsoluteCriterion.CLAIMED_SATURATIONS_APPROPRIATE, _CANDIDATE_SUBJECT),
    ]
    if config.include_model_semantics:
        balance_keys.append(
            (
                AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT,
                _CANDIDATE_SUBJECT,
            )
        )
        dynamic_keys.append(
            (
                AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT,
                _CANDIDATE_SUBJECT,
            )
        )
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
                tuple(balance_keys),
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
                tuple(dynamic_keys),
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
    relative = (
        result.numeric_relative_preference_fixed_denominator
        if config.comparative_indeterminate_policy
        == "neutral_fixed_denominator"
        else result.numeric_relative_preference
    )
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


def reverse_paired_assessments(
    assessments: tuple[PairedAbsoluteAssessment, ...],
) -> tuple[PairedAbsoluteAssessment, ...]:
    """Map B/A-oriented absolute assessments back to A/B identity order."""
    return tuple(
        PairedAbsoluteAssessment(
            criterion=item.criterion,
            subject_id=item.subject_id,
            candidate_a=item.candidate_b,
            candidate_b=item.candidate_a,
        )
        for item in assessments
    )


def reverse_hybrid_result(result: HybridJudgeResult) -> HybridJudgeResult:
    """Map a B/A judge result back to the stable A/B candidate identities."""
    reversed_relative = []
    for item in result.comparative_assessments:
        verdict = item.verdict
        if verdict is RelativeVerdict.CANDIDATE_A:
            verdict = RelativeVerdict.CANDIDATE_B
        elif verdict is RelativeVerdict.CANDIDATE_B:
            verdict = RelativeVerdict.CANDIDATE_A
        reversed_relative.append(
            RelativeAssessment(
                criterion=item.criterion,
                verdict=verdict,
                evidence=item.evidence,
            )
        )
    return HybridJudgeResult(
        absolute_assessments=reverse_paired_assessments(
            result.absolute_assessments
        ),
        comparative_assessments=tuple(reversed_relative),
    )


def question_consensus(
    first: HybridJudgeResult,
    second: HybridJudgeResult,
) -> tuple[HybridJudgeResult, tuple[str, ...], tuple[str, ...]]:
    """Retain only question verdicts shared by two identity-aligned results.

    Evidence is replaced by a runtime-owned consensus statement. The function
    never chooses between disagreeing scientific answers.
    """
    first_absolute = {
        (item.criterion, item.subject_id): item
        for item in first.absolute_assessments
    }
    second_absolute = {
        (item.criterion, item.subject_id): item
        for item in second.absolute_assessments
    }
    if first_absolute.keys() != second_absolute.keys():
        raise ValueError("orientation absolute-unit sets differ")
    absolute = []
    absolute_disagreements = []
    for key in sorted(first_absolute, key=lambda item: (item[0].value, item[1])):
        left = first_absolute[key]
        right = second_absolute[key]
        candidates = []
        for side in ("candidate_a", "candidate_b"):
            left_verdict = getattr(left, side).verdict
            right_verdict = getattr(right, side).verdict
            if left_verdict is right_verdict:
                verdict = left_verdict
                evidence = (
                    "Symmetric orientations agree. First orientation: "
                    f"{getattr(left, side).evidence} Second orientation: "
                    f"{getattr(right, side).evidence}"
                )
            else:
                verdict = AbsoluteVerdict.INDETERMINATE
                evidence = "Orientations disagreed; withheld as indeterminate."
                absolute_disagreements.append(
                    f"{key[0].value}:{key[1]}:{side}"
                )
            candidates.append(
                CandidateAbsoluteAssessment(verdict=verdict, evidence=evidence)
            )
        absolute.append(
            PairedAbsoluteAssessment(
                criterion=key[0],
                subject_id=key[1],
                candidate_a=candidates[0],
                candidate_b=candidates[1],
            )
        )

    first_relative = {
        item.criterion: item for item in first.comparative_assessments
    }
    second_relative = {
        item.criterion: item for item in second.comparative_assessments
    }
    if first_relative.keys() != second_relative.keys() or set(
        first_relative
    ) != set(RelativeCriterion):
        raise ValueError("orientation comparative criteria differ")
    relative = []
    relative_disagreements = []
    for criterion in RelativeCriterion:
        left_verdict = first_relative[criterion].verdict
        right_verdict = second_relative[criterion].verdict
        if left_verdict is right_verdict:
            verdict = left_verdict
            evidence = (
                "Symmetric orientations agree. First orientation: "
                f"{first_relative[criterion].evidence} Second orientation: "
                f"{second_relative[criterion].evidence}"
            )
        else:
            verdict = RelativeVerdict.INDETERMINATE
            evidence = "Orientations disagreed; withheld as indeterminate."
            relative_disagreements.append(criterion.value)
        relative.append(
            RelativeAssessment(
                criterion=criterion,
                verdict=verdict,
                evidence=evidence,
            )
        )
    return (
        HybridJudgeResult(
            absolute_assessments=tuple(absolute),
            comparative_assessments=tuple(relative),
        ),
        tuple(absolute_disagreements),
        tuple(relative_disagreements),
    )


def require_deterministic_orientation_consensus(
    first: tuple[PairedAbsoluteAssessment, ...],
    second: tuple[PairedAbsoluteAssessment, ...],
) -> tuple[PairedAbsoluteAssessment, ...]:
    """Fail closed unless identity-aligned runtime facts exactly agree."""
    first_index = {(item.criterion, item.subject_id): item for item in first}
    second_index = {(item.criterion, item.subject_id): item for item in second}
    if first_index.keys() != second_index.keys():
        raise ValueError("orientation deterministic-unit sets differ")
    for key, left in first_index.items():
        right = second_index[key]
        if (
            left.candidate_a.verdict is not right.candidate_a.verdict
            or left.candidate_b.verdict is not right.candidate_b.verdict
        ):
            raise ValueError(f"deterministic orientation mismatch: {key}")
    return tuple(
        first_index[key]
        for key in sorted(first_index, key=lambda item: (item[0].value, item[1]))
    )
