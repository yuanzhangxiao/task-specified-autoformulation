"""Deterministic evaluation of public target-generation contracts."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema


class RequiredTargetDependency(StrictSchema):
    """One public channel that must contribute to a generated target."""

    dependency_id: Identifier
    acceptable_symbols: tuple[Identifier, ...] = Field(min_length=1)
    public_requirement: NonEmptyText


class PublicTargetRequirement(StrictSchema):
    """Public generation and composition requirements for one target channel."""

    target_channel: Identifier
    public_requirement: NonEmptyText
    required_dependencies: tuple[RequiredTargetDependency, ...] = ()

    @model_validator(mode="after")
    def dependency_ids_are_unique(self) -> PublicTargetRequirement:
        identifiers = [item.dependency_id for item in self.required_dependencies]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("target dependency identifiers must be unique")
        return self


class PublicTargetContract(StrictSchema):
    """Prompt-committed, benchmark-specific instance of the shared target schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["public-target-contract-1"] = "public-target-contract-1"
    source: Literal["public_prompt"] = "public_prompt"
    benchmark_id: Identifier
    tier: str
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: tuple[PublicTargetRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def target_ids_are_unique(self) -> PublicTargetContract:
        identifiers = [item.target_channel for item in self.targets]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("public target channels must be unique")
        return self


class TargetPredicateResult(StrictSchema):
    """One deterministic target-generation predicate and its evidence."""

    target_channel: Identifier
    predicate: str
    status: Literal["satisfied", "failed"]
    evidence: str


class TargetRequirementResult(StrictSchema):
    """Conjunctive result for one public target channel."""

    target_channel: Identifier
    status: Literal["satisfied", "failed"]
    predicates: tuple[TargetPredicateResult, ...] = Field(min_length=1)


class PublicTargetEvaluation(StrictSchema):
    """Deterministic feasibility result kept separate from model quality scores."""

    passed: bool
    mapped_target_fraction: float = Field(ge=0.0, le=1.0)
    required_dependency_fraction: float = Field(ge=0.0, le=1.0)
    target_results: tuple[TargetRequirementResult, ...] = Field(min_length=1)
    predicates: tuple[TargetPredicateResult, ...] = Field(min_length=1)


def evaluate_public_targets(
    candidate: CandidateModel,
    contract: PublicTargetContract,
) -> PublicTargetEvaluation:
    """Certify explicit mappings and public-channel ancestry for every target."""
    graph = _dependency_graph(candidate)
    mappings = {item.channel for item in candidate.observation_mappings}
    modeled_components = {
        *(item.name for item in candidate.states),
        *(item.name for item in candidate.processes),
    }
    target_results: list[TargetRequirementResult] = []
    all_predicates: list[TargetPredicateResult] = []
    mapped = 0
    dependency_total = 0
    dependency_satisfied = 0

    for requirement in contract.targets:
        target_node = f"target:{requirement.target_channel}"
        mapping_ok = requirement.target_channel in mappings
        if mapping_ok:
            mapped += 1
        predicates = [
            TargetPredicateResult(
                target_channel=requirement.target_channel,
                predicate="explicit_observation_mapping",
                status="satisfied" if mapping_ok else "failed",
                evidence=(
                    f"observation mapping exists for {requirement.target_channel}"
                    if mapping_ok
                    else f"no observation mapping for {requirement.target_channel}"
                ),
            )
        ]
        generated_by = tuple(
            sorted(
                component
                for component in modeled_components
                if _reaches(graph, component, target_node)
            )
        )
        predicates.append(
            TargetPredicateResult(
                target_channel=requirement.target_channel,
                predicate="generated_model_path",
                status="satisfied" if generated_by else "failed",
                evidence=(
                    f"modeled components reaching target: {list(generated_by)}"
                    if generated_by
                    else "no modeled state or process reaches the target mapping"
                ),
            )
        )
        for dependency in requirement.required_dependencies:
            dependency_total += 1
            reaching = tuple(
                symbol
                for symbol in dependency.acceptable_symbols
                if _reaches(graph, symbol, target_node)
            )
            dependency_ok = bool(reaching)
            if dependency_ok:
                dependency_satisfied += 1
            predicates.append(
                TargetPredicateResult(
                    target_channel=requirement.target_channel,
                    predicate=f"required_dependency:{dependency.dependency_id}",
                    status="satisfied" if dependency_ok else "failed",
                    evidence=(
                        f"public symbols reaching target: {list(reaching)}"
                        if dependency_ok
                        else (
                            "none of the required public symbols reaches the "
                            f"target: {list(dependency.acceptable_symbols)}"
                        )
                    ),
                )
            )
        result_status: Literal["satisfied", "failed"] = (
            "satisfied"
            if all(item.status == "satisfied" for item in predicates)
            else "failed"
        )
        result = TargetRequirementResult(
            target_channel=requirement.target_channel,
            status=result_status,
            predicates=tuple(predicates),
        )
        target_results.append(result)
        all_predicates.extend(predicates)

    return PublicTargetEvaluation(
        passed=all(item.status == "satisfied" for item in target_results),
        mapped_target_fraction=mapped / len(contract.targets),
        required_dependency_fraction=(
            dependency_satisfied / dependency_total if dependency_total else 1.0
        ),
        target_results=tuple(target_results),
        predicates=tuple(all_predicates),
    )


def _dependency_graph(candidate: CandidateModel) -> dict[str, set[str]]:
    """Return forward symbol dependencies through processes, states, and targets."""
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


def _reaches(graph: dict[str, set[str]], source: str, target: str) -> bool:
    pending = deque([source])
    visited = {source}
    while pending:
        current = pending.popleft()
        if current == target:
            return True
        for destination in graph.get(current, set()) - visited:
            visited.add(destination)
            pending.append(destination)
    return False
