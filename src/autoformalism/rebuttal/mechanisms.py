"""Judge-independent mechanism coverage and structural graph metrics."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel


class MechanismRequirement(BaseModel):
    """Public task predicate for one required generated mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tag_aliases: tuple[str, ...] = ()
    required_drivers: tuple[str, ...] = ()
    required_targets: tuple[str, ...] = ()
    must_be_generated: bool = True
    requires_dynamic_memory: bool = False
    required_sign: Literal["positive", "negative", "unspecified"] = "unspecified"
    forbid_current_future_target_input: bool = True


class MechanismEvaluationSpec(BaseModel):
    """Public benchmark/tier mechanism-evaluation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    tier: str
    required_mechanisms: tuple[MechanismRequirement, ...] = Field(min_length=1)


class PredicateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    predicate: str
    status: Literal["satisfied", "failed", "ambiguous"]
    evidence: str


class MechanismEvaluation(BaseModel):
    """Independent coverage and structural-validity result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_coverage: float
    structural_validity: float
    covered_mechanisms: tuple[str, ...]
    predicates: tuple[PredicateResult, ...]
    manual_review_required: bool


def evaluate_mechanisms(
    candidate: CandidateModel, spec: MechanismEvaluationSpec
) -> MechanismEvaluation:
    """Evaluate explicit tagged claims against the expression dependency graph."""
    graph = _dependency_graph(candidate)
    reverse = _reverse_graph(graph)
    state_names = {item.name for item in candidate.states}
    tags: dict[str, set[str]] = defaultdict(set)
    for state in candidate.states:
        for tag in state.mechanisms:
            tags[tag].add(state.name)
    for process in candidate.processes:
        for tag in process.mechanisms:
            tags[tag].add(process.name)

    predicates: list[PredicateResult] = []
    covered: list[str] = []
    for requirement in spec.required_mechanisms:
        accepted_tags = (requirement.id, *requirement.tag_aliases)
        claims = set().union(*(tags.get(tag, set()) for tag in accepted_tags))
        target_nodes = {f"target:{name}" for name in requirement.required_targets}
        connected_claims = {
            claim
            for claim in claims
            if not target_nodes or _reaches_any(graph, claim, target_nodes)
        }
        coverage_ok = bool(connected_claims)
        if coverage_ok:
            covered.append(requirement.id)
        predicates.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate="generated_connected_coverage",
                status="satisfied" if coverage_ok else "failed",
                evidence=(
                    f"connected tagged components: {sorted(connected_claims)}"
                    if coverage_ok
                    else (
                        "tagged components without required target path: "
                        f"{sorted(claims)}"
                    )
                ),
            )
        )
        for driver in requirement.required_drivers:
            driven = any(
                driver in _ancestors(reverse, claim) for claim in connected_claims
            )
            predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate=f"driver:{driver}",
                    status="satisfied" if driven else "failed",
                    evidence=f"driver ancestry checked for {sorted(connected_claims)}",
                )
            )
        if requirement.requires_dynamic_memory:
            dynamic = any(claim in state_names for claim in connected_claims)
            predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate="dynamic_memory",
                    status="satisfied" if dynamic else "failed",
                    evidence=(
                        "dynamic tagged components: "
                        f"{sorted(connected_claims & state_names)}"
                    ),
                )
            )
        if requirement.required_sign != "unspecified":
            predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate=f"regulatory_sign:{requirement.required_sign}",
                    status="ambiguous",
                    evidence="nonlinear multi-path sign requires blinded manual review",
                )
            )
    scored = [item for item in predicates if item.status != "ambiguous"]
    validity = (
        sum(item.status == "satisfied" for item in scored) / len(scored)
        if scored
        else 0.0
    )
    return MechanismEvaluation(
        mechanism_coverage=len(covered) / len(spec.required_mechanisms),
        structural_validity=validity,
        covered_mechanisms=tuple(covered),
        predicates=tuple(predicates),
        manual_review_required=any(
            item.status == "ambiguous" for item in predicates
        ),
    )


def _dependency_graph(candidate: CandidateModel) -> dict[str, set[str]]:
    parser = RestrictedParser()
    graph: dict[str, set[str]] = defaultdict(set)
    for process in candidate.processes:
        parsed = parser.parse(process.expression, location=f"process:{process.name}")
        for symbol in parsed.symbols:
            graph[symbol].add(process.name)
    for equation in candidate.state_equations:
        parsed = parser.parse(equation.rhs, location=f"equation:{equation.state}")
        for symbol in parsed.symbols:
            graph[symbol].add(equation.state)
    for mapping in candidate.observation_mappings:
        parsed = parser.parse(
            mapping.expression, location=f"observation:{mapping.channel}"
        )
        for symbol in parsed.symbols:
            graph[symbol].add(f"target:{mapping.channel}")
    return {key: set(values) for key, values in graph.items()}


def _reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, destinations in graph.items():
        for destination in destinations:
            reverse[destination].add(source)
    return {key: set(values) for key, values in reverse.items()}


def _reaches_any(
    graph: dict[str, set[str]], source: str, targets: set[str]
) -> bool:
    pending = deque([source])
    visited = {source}
    while pending:
        current = pending.popleft()
        if current in targets:
            return True
        for item in graph.get(current, set()) - visited:
            visited.add(item)
            pending.append(item)
    return False


def _ancestors(reverse: dict[str, set[str]], node: str) -> set[str]:
    pending = deque([node])
    visited: set[str] = set()
    while pending:
        current = pending.popleft()
        for item in reverse.get(current, set()) - visited:
            visited.add(item)
            pending.append(item)
    return visited
