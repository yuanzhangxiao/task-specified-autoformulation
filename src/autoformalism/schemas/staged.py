"""Typed schemas for staged mechanistic-model construction."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema
from autoformalism.schemas.candidate import (
    ConstraintSpec,
    InitialConditionSpec,
    ParameterSpec,
    StateSpec,
)
from autoformalism.schemas.proposal import ProposedInitialValue, ProposedParameter

Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class InteractionTargetKind(str, Enum):
    """Kind of definition assembled from an interaction hyperedge."""

    STATE_DERIVATIVE = "state_derivative"
    ALGEBRAIC_PROCESS = "algebraic_process"


class InteractionPolarity(str, Enum):
    """Outer algebraic operator used to assemble an interaction term.

    This is not by itself a proof that the assigned function is nonnegative or
    monotone in every source. Parameter roles and scientific checks own those
    separate claims.
    """

    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"


class TopologyProcessSpec(StrictSchema):
    """An instantaneous generated quantity without a functional form yet."""

    name: Identifier
    unit: NonEmptyText = "unspecified"
    description: NonEmptyText = "unspecified"
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class TopologyInteraction(StrictSchema):
    """One signed dependency hyperedge in a staged topology."""

    interaction_id: Identifier
    target: Identifier
    target_kind: InteractionTargetKind
    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)
    polarity: InteractionPolarity = InteractionPolarity.ADDITIVE
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)
    description: NonEmptyText = "unspecified"

    @model_validator(mode="after")
    def sources_are_unique(self) -> TopologyInteraction:
        """Reject a redundant source list before functional assignment."""
        if len(self.sources) != len(set(self.sources)):
            raise ValueError(
                f"duplicate interaction source in {self.interaction_id}"
            )
        return self


class TopologyStateMeasurement(StrictSchema):
    """Identity measurement that makes one modeled state directly observed."""

    state: Identifier
    channel: Identifier
    unit: NonEmptyText = "unspecified"


class TopologyTargetMapping(StrictSchema):
    """Direct mapping from a generated node to one prediction target."""

    channel: Identifier
    source: Identifier
    unit: NonEmptyText = "unspecified"


# Import compatibility for the unrun staged-v1 prototype. New artifacts use
# the scientifically narrower ``TopologyTargetMapping`` name.
TopologyObservationMapping = TopologyTargetMapping


class TopologyCandidate(StrictSchema):
    """Immutable graph-stage candidate with no interaction functions."""

    schema_version: Literal["topology-candidate-2"] = "topology-candidate-2"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    states: tuple[StateSpec, ...] = Field(min_length=1, max_length=64)
    processes: tuple[TopologyProcessSpec, ...] = Field(default=(), max_length=256)
    external_symbols: tuple[Identifier, ...] = Field(default=(), max_length=256)
    interactions: tuple[TopologyInteraction, ...] = Field(
        min_length=1,
        max_length=512,
    )
    state_measurements: tuple[TopologyStateMeasurement, ...] = Field(
        default=(),
        max_length=64,
    )
    target_mappings: tuple[TopologyTargetMapping, ...] = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_graph(self) -> TopologyCandidate:
        """Require graph closure and an acyclic algebraic subgraph."""
        state_names = _unique_attribute("state", self.states, "name")
        process_names = _unique_attribute("process", self.processes, "name")
        _unique_attribute("interaction", self.interactions, "interaction_id")
        _unique_attribute("measured state", self.state_measurements, "state")
        _unique_attribute(
            "state measurement channel", self.state_measurements, "channel"
        )
        _unique_attribute("target channel", self.target_mappings, "channel")
        if len(self.external_symbols) != len(set(self.external_symbols)):
            raise ValueError("duplicate external symbol")

        declared = state_names | process_names
        collisions = state_names & process_names
        if collisions:
            raise ValueError(
                f"state/process name collision: {sorted(collisions)}"
            )
        external_collisions = declared & set(self.external_symbols)
        if external_collisions:
            raise ValueError(
                "generated/external symbol collision: "
                f"{sorted(external_collisions)}"
            )

        available_sources = declared | set(self.external_symbols)
        covered_states: set[str] = set()
        covered_processes: set[str] = set()
        process_dependencies: dict[str, set[str]] = {
            name: set() for name in process_names
        }
        for interaction in self.interactions:
            unknown_sources = set(interaction.sources) - available_sources
            if unknown_sources:
                raise ValueError(
                    f"interaction {interaction.interaction_id} has undefined "
                    f"sources: {sorted(unknown_sources)}"
                )
            if interaction.target_kind is InteractionTargetKind.STATE_DERIVATIVE:
                if interaction.target not in state_names:
                    raise ValueError(
                        f"interaction {interaction.interaction_id} targets "
                        f"undeclared state: {interaction.target}"
                    )
                covered_states.add(interaction.target)
            else:
                if interaction.target not in process_names:
                    raise ValueError(
                        f"interaction {interaction.interaction_id} targets "
                        f"undeclared process: {interaction.target}"
                    )
                covered_processes.add(interaction.target)
                process_dependencies[interaction.target].update(
                    set(interaction.sources) & process_names
                )

        missing_states = state_names - covered_states
        if missing_states:
            raise ValueError(
                f"states without derivative interactions: {sorted(missing_states)}"
            )
        missing_processes = process_names - covered_processes
        if missing_processes:
            raise ValueError(
                f"processes without defining interactions: {sorted(missing_processes)}"
            )

        for measurement in self.state_measurements:
            if measurement.state not in state_names:
                raise ValueError(
                    "state measurement references an undeclared state: "
                    f"{measurement.state}"
                )

        generated = state_names | process_names
        for mapping in self.target_mappings:
            if mapping.source not in generated:
                raise ValueError(
                    f"target {mapping.channel} maps from a non-generated "
                    f"symbol: {mapping.source}"
                )

        _require_acyclic_process_graph(process_dependencies)
        return self


class InteractionFunction(StrictSchema):
    """Restricted expression assigned to one topology interaction."""

    interaction_id: Identifier
    expression: NonEmptyText


class FunctionalCandidate(StrictSchema):
    """Functional assignment that references an immutable topology artifact."""

    schema_version: Literal["functional-candidate-1"] = "functional-candidate-1"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    topology_commitment_sha256: Sha256Digest
    interaction_functions: tuple[InteractionFunction, ...] = Field(
        min_length=1,
        max_length=512,
    )
    parameters: tuple[ParameterSpec, ...] = Field(default=(), max_length=256)
    initial_conditions: tuple[InitialConditionSpec, ...] = Field(
        default=(),
        max_length=64,
    )
    constraints: tuple[ConstraintSpec, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def function_bindings_are_unique(self) -> FunctionalCandidate:
        """Require exactly one local declaration for each named binding."""
        _unique_attribute(
            "interaction function", self.interaction_functions, "interaction_id"
        )
        _unique_attribute("parameter", self.parameters, "name")
        _unique_attribute("initial condition", self.initial_conditions, "state")
        return self


class ProposedTopologyState(StrictSchema):
    """One generated state identity in the compact topology contract.

    State observability is deliberately omitted. The runtime derives it from
    validated public measurement and target mappings instead of asking the
    proposer to emit a fallible observed/latent label.
    """

    name: Identifier
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class ProposedTopologyProcess(StrictSchema):
    """One instantaneous generated quantity before a function is assigned."""

    name: Identifier
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class ProposedTopologyCandidate(StrictSchema):
    """Minimal provider-facing graph proposal without runtime-owned metadata."""

    schema_version: Literal["proposed-topology-candidate-2"] = (
        "proposed-topology-candidate-2"
    )
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    states: tuple[ProposedTopologyState, ...] = Field(min_length=1, max_length=64)
    processes: tuple[ProposedTopologyProcess, ...] = Field(
        default=(), max_length=256
    )
    interactions: tuple[TopologyInteraction, ...] = Field(
        min_length=1,
        max_length=512,
    )
    state_measurements: tuple[TopologyStateMeasurement, ...] = Field(
        default=(),
        max_length=64,
    )
    target_mappings: tuple[TopologyTargetMapping, ...] = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def declarations_are_locally_unique(self) -> ProposedTopologyCandidate:
        """Reject ambiguity before context-dependent topology enrichment."""
        states = _unique_attribute("state", self.states, "name")
        processes = _unique_attribute("process", self.processes, "name")
        _unique_attribute("interaction", self.interactions, "interaction_id")
        _unique_attribute("measured state", self.state_measurements, "state")
        _unique_attribute(
            "state measurement channel", self.state_measurements, "channel"
        )
        _unique_attribute("target channel", self.target_mappings, "channel")
        collisions = states & processes
        if collisions:
            raise ValueError(
                f"state/process name collision: {sorted(collisions)}"
            )
        return self


class ProposedFunctionalInitial(StrictSchema):
    """Initial value supplied only for one runtime-derived latent state."""

    state: Identifier
    initial: ProposedInitialValue


class ProposedFunctionalCandidate(StrictSchema):
    """Minimal provider-facing functions for one committed topology."""

    schema_version: Literal["proposed-functional-candidate-1"] = (
        "proposed-functional-candidate-1"
    )
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    interaction_functions: tuple[InteractionFunction, ...] = Field(
        min_length=1,
        max_length=512,
    )
    parameters: tuple[ProposedParameter, ...] = Field(default=(), max_length=256)
    latent_initials: tuple[ProposedFunctionalInitial, ...] = Field(
        default=(),
        max_length=64,
    )

    @model_validator(mode="after")
    def declarations_are_unique(self) -> ProposedFunctionalCandidate:
        """Require one function, parameter, and latent initializer per name."""
        _unique_attribute(
            "interaction function", self.interaction_functions, "interaction_id"
        )
        _unique_attribute("parameter", self.parameters, "name")
        _unique_attribute("latent initial", self.latent_initials, "state")
        return self


def _unique_attribute(
    label: str,
    values: tuple[object, ...],
    attribute: str,
) -> set[str]:
    names = [str(getattr(value, attribute)) for value in values]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate {label}: {duplicates}")
    return set(names)


def _require_acyclic_process_graph(
    dependencies: dict[str, set[str]],
) -> None:
    states: dict[str, int] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        state = states.get(name, 0)
        if state == 2:
            return
        if state == 1:
            start = path.index(name) if name in path else 0
            cycle = (*path[start:], name)
            raise ValueError(
                f"cyclic algebraic process dependency: {' -> '.join(cycle)}"
            )
        states[name] = 1
        for dependency in sorted(dependencies[name]):
            visit(dependency, (*path, name))
        states[name] = 2

    for name in sorted(dependencies):
        visit(name, ())
