"""Minimal provider-facing actions for incremental model construction."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import (
    FiniteFloat,
    Identifier,
    NonEmptyText,
    StrictSchema,
)
from autoformalism.schemas.proposal import ProposedInitialValue, ProposedParameter
from autoformalism.schemas.staged import (
    InteractionPolarity,
    ProposedFunctionalInitial,
    ProposedTopologyProcess,
    ProposedTopologyState,
    Sha256Digest,
    TopologyStateMeasurement,
    TopologyTargetMapping,
)


class ConstructionObjective(str, Enum):
    """Scientific responsibility addressed by one incremental transaction."""

    INITIAL_CONSTRUCTION = "initial_construction"
    TARGET_PATH_REPAIR = "target_path_repair"
    MECHANISM_REPAIR = "mechanism_repair"
    FUNCTION_REPAIR = "function_repair"
    NUMERICAL_REPAIR = "numerical_repair"
    SIMPLIFICATION = "simplification"


class ConstructionIntent(StrictSchema):
    """Compact decision state externalized before proposing model edits.

    Public requirement and target identifiers are anchors into runtime-owned
    contracts.  They are not explanatory prose and do not duplicate model
    structure.
    """

    objective: ConstructionObjective
    requirement_ids: tuple[Identifier, ...] = Field(default=(), max_length=32)
    target_channels: tuple[Identifier, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def require_unique_anchor(self) -> ConstructionIntent:
        """Require a bounded, unambiguous focus for every action transaction."""
        for label, values in (
            ("requirement", self.requirement_ids),
            ("target channel", self.target_channels),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} in construction intent")
        if not self.requirement_ids and not self.target_channels:
            raise ValueError(
                "construction intent requires a requirement or target anchor"
            )
        return self


class ProposedConstructionIntent(StrictSchema):
    """Provider-selected scientific focus for one subsequent action call."""

    schema_version: Literal["proposed-construction-intent-1"] = (
        "proposed-construction-intent-1"
    )
    objective: ConstructionObjective
    requirement_ids: tuple[Identifier, ...] = Field(default=(), max_length=32)
    target_channels: tuple[Identifier, ...] = Field(default=(), max_length=64)

    def as_runtime_intent(self) -> ConstructionIntent:
        """Convert the provider response into the runtime-owned decision state."""
        return ConstructionIntent(
            objective=self.objective,
            requirement_ids=self.requirement_ids,
            target_channels=self.target_channels,
        )


class AddStateAction(StrictSchema):
    """Add one generated dynamic state to the runtime-maintained draft."""

    action: Literal["add_state"] = "add_state"
    name: Identifier


class AddProcessAction(StrictSchema):
    """Add one generated instantaneous process to the topology draft."""

    action: Literal["add_process"] = "add_process"
    name: Identifier


class RemoveGeneratedNodeAction(StrictSchema):
    """Remove one state or process after references are edited explicitly."""

    action: Literal["remove_generated_node"] = "remove_generated_node"
    name: Identifier


class AddInteractionAction(StrictSchema):
    """Add one signed dependency hyperedge without a functional expression."""

    action: Literal["add_interaction"] = "add_interaction"
    interaction_id: Identifier
    target: Identifier
    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)
    polarity: InteractionPolarity = InteractionPolarity.ADDITIVE

    @model_validator(mode="after")
    def sources_are_unique(self) -> AddInteractionAction:
        """Reject repeated dependency declarations in one hyperedge."""
        if len(self.sources) != len(set(self.sources)):
            raise ValueError(f"duplicate source in {self.interaction_id}")
        return self


class RemoveInteractionAction(StrictSchema):
    """Remove one interaction by its stable runtime identifier."""

    action: Literal["remove_interaction"] = "remove_interaction"
    interaction_id: Identifier


class SetStateMeasurementAction(StrictSchema):
    """Bind one state directly to one supplied auxiliary channel."""

    action: Literal["set_state_measurement"] = "set_state_measurement"
    state: Identifier
    channel: Identifier


class RemoveStateMeasurementAction(StrictSchema):
    """Remove the direct auxiliary measurement attached to one state."""

    action: Literal["remove_state_measurement"] = "remove_state_measurement"
    state: Identifier


class SetTargetMappingAction(StrictSchema):
    """Set or replace the generated source of one public target."""

    action: Literal["set_target_mapping"] = "set_target_mapping"
    channel: Identifier
    source: Identifier


class RemoveTargetMappingAction(StrictSchema):
    """Remove one target mapping before replacing or rebuilding it."""

    action: Literal["remove_target_mapping"] = "remove_target_mapping"
    channel: Identifier


TopologyAction = Annotated[
    AddStateAction
    | AddProcessAction
    | RemoveGeneratedNodeAction
    | AddInteractionAction
    | RemoveInteractionAction
    | SetStateMeasurementAction
    | RemoveStateMeasurementAction
    | SetTargetMappingAction
    | RemoveTargetMappingAction,
    Field(discriminator="action"),
]


class ProposedTopologyActionTransaction(StrictSchema):
    """One focused bundle of topology edits; unmentioned structure is retained."""

    schema_version: Literal["proposed-topology-action-transaction-1"] = (
        "proposed-topology-action-transaction-1"
    )
    actions: tuple[TopologyAction, ...] = Field(min_length=1, max_length=64)


class TopologyDraftInteraction(StrictSchema):
    """Runtime-owned interaction state before target-kind enrichment."""

    interaction_id: Identifier
    target: Identifier
    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)
    polarity: InteractionPolarity = InteractionPolarity.ADDITIVE
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class TopologyDraft(StrictSchema):
    """Possibly incomplete topology maintained by the deterministic runtime."""

    schema_version: Literal["topology-draft-1"] = "topology-draft-1"
    states: tuple[ProposedTopologyState, ...] = Field(default=(), max_length=64)
    processes: tuple[ProposedTopologyProcess, ...] = Field(
        default=(), max_length=256
    )
    interactions: tuple[TopologyDraftInteraction, ...] = Field(
        default=(), max_length=512
    )
    state_measurements: tuple[TopologyStateMeasurement, ...] = Field(
        default=(), max_length=64
    )
    target_mappings: tuple[TopologyTargetMapping, ...] = Field(
        default=(), max_length=64
    )

    @model_validator(mode="after")
    def declarations_are_unique(self) -> TopologyDraft:
        """Keep every partial draft locally unambiguous."""
        collections = (
            ("state", self.states, "name"),
            ("process", self.processes, "name"),
            ("interaction", self.interactions, "interaction_id"),
            ("measured state", self.state_measurements, "state"),
            ("measurement channel", self.state_measurements, "channel"),
            ("target channel", self.target_mappings, "channel"),
        )
        for label, values, attribute in collections:
            names = [str(getattr(item, attribute)) for item in values]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} in topology draft")
        states = {item.name for item in self.states}
        processes = {item.name for item in self.processes}
        if states & processes:
            raise ValueError("state/process collision in topology draft")
        return self


class InteractionFunctionDraft(StrictSchema):
    """One localized function assignment and only the parameters it uses."""

    interaction_id: Identifier
    expression: NonEmptyText
    parameters: tuple[ProposedParameter, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def parameters_are_unique(self) -> InteractionFunctionDraft:
        """Reject ambiguous local parameter-role declarations."""
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(
                f"duplicate parameter in interaction {self.interaction_id}"
            )
        return self


class SetInteractionFunctionAction(StrictSchema):
    """Set or replace the function assigned to one committed interaction."""

    action: Literal["set_interaction_function"] = "set_interaction_function"
    interaction_id: Identifier
    expression: NonEmptyText
    parameters: tuple[ProposedParameter, ...] = Field(default=(), max_length=64)


class RemoveInteractionFunctionAction(StrictSchema):
    """Remove one function assignment while retaining its topology edge."""

    action: Literal["remove_interaction_function"] = (
        "remove_interaction_function"
    )
    interaction_id: Identifier


class SetLatentInitialAction(StrictSchema):
    """Set or replace the current initializer for one latent state."""

    action: Literal["set_latent_initial"] = "set_latent_initial"
    state: Identifier
    initial: ProposedInitialValue


class RemoveLatentInitialAction(StrictSchema):
    """Remove a latent initializer before revising it."""

    action: Literal["remove_latent_initial"] = "remove_latent_initial"
    state: Identifier


FunctionalAction = Annotated[
    SetInteractionFunctionAction
    | RemoveInteractionFunctionAction
    | SetLatentInitialAction
    | RemoveLatentInitialAction,
    Field(discriminator="action"),
]


class ProposedFunctionalActionTransaction(StrictSchema):
    """One focused function transaction conditioned on an exact topology."""

    schema_version: Literal["proposed-functional-action-transaction-1"] = (
        "proposed-functional-action-transaction-1"
    )
    actions: tuple[FunctionalAction, ...] = Field(min_length=1, max_length=128)


class FunctionalDraft(StrictSchema):
    """Possibly incomplete localized function assignments for one topology."""

    schema_version: Literal["functional-draft-1"] = "functional-draft-1"
    topology_commitment_sha256: Sha256Digest
    interaction_functions: tuple[InteractionFunctionDraft, ...] = Field(
        default=(), max_length=512
    )
    latent_initials: tuple[ProposedFunctionalInitial, ...] = Field(
        default=(), max_length=64
    )

    @model_validator(mode="after")
    def declarations_are_unique(self) -> FunctionalDraft:
        """Keep localized function and initializer ownership unambiguous."""
        interactions = [
            item.interaction_id for item in self.interaction_functions
        ]
        initials = [item.state for item in self.latent_initials]
        if len(interactions) != len(set(interactions)):
            raise ValueError("duplicate interaction function in functional draft")
        if len(initials) != len(set(initials)):
            raise ValueError("duplicate latent initializer in functional draft")
        return self


class ConstructionDiagnostic(StrictSchema):
    """One deterministic action or compatibility finding."""

    code: Identifier
    location: NonEmptyText
    message: NonEmptyText


class FunctionalCompatibilityReport(StrictSchema):
    """Compatibility of a partial function draft with one fixed topology."""

    schema_version: Literal["functional-compatibility-report-1"] = (
        "functional-compatibility-report-1"
    )
    status: Literal["incomplete", "compatible", "incompatible"]
    missing_interaction_ids: tuple[Identifier, ...] = ()
    missing_latent_initial_states: tuple[Identifier, ...] = ()
    diagnostics: tuple[ConstructionDiagnostic, ...] = ()


class TopologyActionApplication(StrictSchema):
    """Auditable result of deterministically applying a topology transaction."""

    schema_version: Literal["topology-action-application-1"] = (
        "topology-action-application-1"
    )
    before_sha256: Sha256Digest
    after_sha256: Sha256Digest
    intent: ConstructionIntent
    changed: bool
    added_nodes: tuple[Identifier, ...] = ()
    removed_nodes: tuple[Identifier, ...] = ()
    added_interactions: tuple[Identifier, ...] = ()
    removed_interactions: tuple[Identifier, ...] = ()
    draft: TopologyDraft


class FunctionalActionApplication(StrictSchema):
    """Auditable result of deterministically applying a function transaction."""

    schema_version: Literal["functional-action-application-1"] = (
        "functional-action-application-1"
    )
    before_sha256: Sha256Digest
    after_sha256: Sha256Digest
    intent: ConstructionIntent
    changed: bool
    set_interaction_ids: tuple[Identifier, ...] = ()
    removed_interaction_ids: tuple[Identifier, ...] = ()
    set_initial_states: tuple[Identifier, ...] = ()
    removed_initial_states: tuple[Identifier, ...] = ()
    draft: FunctionalDraft


class ConditionalBeamEntry(StrictSchema):
    """Scored compatible topology/function branch used by beam selection."""

    topology_sha256: Sha256Digest
    functional_sha256: Sha256Digest
    score: FiniteFloat


__all__ = [
    "AddInteractionAction",
    "AddProcessAction",
    "AddStateAction",
    "ConditionalBeamEntry",
    "ConstructionDiagnostic",
    "ConstructionIntent",
    "ConstructionObjective",
    "FunctionalActionApplication",
    "FunctionalCompatibilityReport",
    "FunctionalDraft",
    "InteractionFunctionDraft",
    "ProposedConstructionIntent",
    "ProposedFunctionalActionTransaction",
    "ProposedTopologyActionTransaction",
    "RemoveGeneratedNodeAction",
    "RemoveInteractionAction",
    "RemoveInteractionFunctionAction",
    "RemoveLatentInitialAction",
    "RemoveStateMeasurementAction",
    "RemoveTargetMappingAction",
    "SetInteractionFunctionAction",
    "SetLatentInitialAction",
    "SetStateMeasurementAction",
    "SetTargetMappingAction",
    "TopologyActionApplication",
    "TopologyDraft",
    "TopologyDraftInteraction",
]
