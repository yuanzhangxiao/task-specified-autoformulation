"""Deterministic compiler for incremental topology and function actions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from autoformalism.expressions import (
    ModelValidationError,
    RestrictedParser,
    ValidationContext,
)
from autoformalism.schemas.candidate import ParameterRole, StateKind
from autoformalism.schemas.construction import (
    AddInteractionAction,
    AddProcessAction,
    AddStateAction,
    ConditionalBeamEntry,
    ConstructionDiagnostic,
    ConstructionIntent,
    FunctionalActionApplication,
    FunctionalCompatibilityReport,
    FunctionalDraft,
    InteractionFunctionDraft,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    RemoveGeneratedNodeAction,
    RemoveInteractionAction,
    RemoveInteractionFunctionAction,
    RemoveLatentInitialAction,
    RemoveStateMeasurementAction,
    RemoveTargetMappingAction,
    SetInteractionFunctionAction,
    SetLatentInitialAction,
    SetStateMeasurementAction,
    SetTargetMappingAction,
    TopologyActionApplication,
    TopologyConstructionPhase,
    TopologyDraft,
    TopologyDraftInteraction,
)
from autoformalism.schemas.proposal import ProposedParameter
from autoformalism.schemas.staged import (
    InteractionFunction,
    InteractionTargetKind,
    ProposedFunctionalCandidate,
    ProposedFunctionalInitial,
    ProposedTopologyCandidate,
    ProposedTopologyProcess,
    ProposedTopologyState,
    TopologyCandidate,
    TopologyInteraction,
    TopologyStateMeasurement,
    TopologyTargetMapping,
)
from autoformalism.staging import (
    StagedCandidateExpansion,
    enrich_functional_proposal,
    enrich_topology_proposal,
    expand_staged_candidate,
    topology_commitment_sha256,
)

TranspositionStage = Literal["topology", "functional"]


def topology_draft_sha256(draft: TopologyDraft) -> str:
    """Return a history-independent identity for one partial topology."""
    canonical = _canonical_topology_draft(draft)
    return _sha256(canonical.model_dump(mode="json"))


def functional_draft_sha256(draft: FunctionalDraft) -> str:
    """Return a history-independent identity for localized function choices."""
    canonical = _canonical_functional_draft(draft)
    return _sha256(canonical.model_dump(mode="json"))


def normalize_topology_action_transaction(
    transaction: ProposedTopologyActionTransaction,
) -> tuple[ProposedTopologyActionTransaction, dict[str, object]]:
    """Remove only byte-equivalent repeated topology actions.

    This is a lossless representation repair: actions that merely share a
    target or identifier but differ in content or order are never collapsed.
    """
    actions, removed = _deduplicate_exact_actions(transaction.actions)
    normalized = transaction.model_copy(update={"actions": actions})
    return normalized, {
        "schema_version": "topology-action-repair-1",
        "exact_duplicate_action_indices_removed": removed,
    }


def normalize_topology_action_transaction_for_context(
    transaction: ProposedTopologyActionTransaction,
    *,
    parent: TopologyDraft,
    context: ValidationContext,
) -> tuple[ProposedTopologyActionTransaction, dict[str, object]]:
    """Repair an unambiguous target/auxiliary mapping action transposition.

    A target identity mapping and an auxiliary state measurement carry distinct
    action names, but an open model can occasionally transpose those names. The
    runtime can correct the action only when the referenced generated state and
    public channel class make the intended identity binding unambiguous.
    """
    state_names = {item.name for item in parent.states}
    state_names.update(
        action.name
        for action in transaction.actions
        if isinstance(action, AddStateAction)
    )
    target_channels = set(context.targets)
    auxiliary_channels = set(context.auxiliaries)
    rewritten: list[object] = []
    repairs: list[dict[str, object]] = []
    for index, action in enumerate(transaction.actions):
        replacement = action
        if (
            isinstance(action, SetStateMeasurementAction)
            and action.channel in target_channels
            and action.state in state_names
        ):
            replacement = SetTargetMappingAction(
                channel=action.channel,
                source=action.state,
            )
            repairs.append(
                {
                    "action_index": index,
                    "from": "set_state_measurement",
                    "to": "set_target_mapping",
                }
            )
        elif (
            isinstance(action, SetTargetMappingAction)
            and action.channel in auxiliary_channels
            and action.source in state_names
        ):
            replacement = SetStateMeasurementAction(
                state=action.source,
                channel=action.channel,
            )
            repairs.append(
                {
                    "action_index": index,
                    "from": "set_target_mapping",
                    "to": "set_state_measurement",
                }
            )
        rewritten.append(replacement)
    contextual = transaction.model_copy(update={"actions": tuple(rewritten)})
    normalized, exact_repairs = normalize_topology_action_transaction(contextual)
    audit = dict(exact_repairs)
    if repairs:
        audit["channel_action_transpositions"] = repairs
    return normalized, audit


def normalize_functional_action_transaction(
    transaction: ProposedFunctionalActionTransaction,
) -> tuple[ProposedFunctionalActionTransaction, dict[str, object]]:
    """Remove only byte-equivalent repeated functional actions."""
    actions, removed = _deduplicate_exact_actions(transaction.actions)
    normalized = transaction.model_copy(update={"actions": actions})
    return normalized, {
        "schema_version": "functional-action-repair-1",
        "exact_duplicate_action_indices_removed": removed,
    }


def apply_topology_actions(
    draft: TopologyDraft,
    intent: ConstructionIntent,
    transaction: ProposedTopologyActionTransaction,
    context: ValidationContext,
    *,
    allowed_requirement_ids: Iterable[str] | None = None,
    topology_phase: TopologyConstructionPhase = TopologyConstructionPhase.MIXED,
    attach_intent_mechanisms: bool = True,
    maximum_generated_nodes: int = 64,
    maximum_interactions: int = 512,
) -> TopologyActionApplication:
    """Apply one ordered topology transaction and validate its public boundary.

    Unmentioned nodes, interactions, and mappings are preserved.  Deletions do
    not cascade: a transaction must explicitly repair every resulting dangling
    target or reference, which prevents a small edit from silently changing
    unrelated science.
    """
    _validate_intent(
        intent.requirement_ids,
        intent.target_channels,
        context,
        allowed_requirement_ids=allowed_requirement_ids,
    )
    before = _canonical_topology_draft(draft)
    before_sha256 = topology_draft_sha256(before)
    states = {item.name: item for item in before.states}
    processes = {item.name: item for item in before.processes}
    interactions = {item.interaction_id: item for item in before.interactions}
    _validate_topology_phase_actions(
        transaction,
        phase=topology_phase,
        state_names=frozenset(states),
        process_names=frozenset(processes),
        interactions=interactions,
    )
    measurements = {item.state: item for item in before.state_measurements}
    mappings = {item.channel: item for item in before.target_mappings}
    mechanism_ids = (
        tuple(sorted(intent.requirement_ids)) if attach_intent_mechanisms else ()
    )
    added_nodes: list[str] = []
    removed_nodes: list[str] = []
    added_interactions: list[str] = []
    removed_interactions: list[str] = []

    for action in transaction.actions:
        if isinstance(action, AddStateAction):
            if action.name in states or action.name in processes:
                raise ValueError(f"generated node already exists: {action.name}")
            states[action.name] = ProposedTopologyState(
                name=action.name,
                mechanisms=mechanism_ids,
            )
            added_nodes.append(action.name)
        elif isinstance(action, AddProcessAction):
            if action.name in states or action.name in processes:
                raise ValueError(f"generated node already exists: {action.name}")
            processes[action.name] = ProposedTopologyProcess(
                name=action.name,
                mechanisms=mechanism_ids,
            )
            added_nodes.append(action.name)
        elif isinstance(action, RemoveGeneratedNodeAction):
            if action.name in states:
                del states[action.name]
            elif action.name in processes:
                del processes[action.name]
            else:
                raise ValueError(f"cannot remove unknown generated node: {action.name}")
            removed_nodes.append(action.name)
        elif isinstance(action, AddInteractionAction):
            if action.interaction_id in interactions:
                raise ValueError(f"interaction already exists: {action.interaction_id}")
            interactions[action.interaction_id] = TopologyDraftInteraction(
                interaction_id=action.interaction_id,
                target=action.target,
                sources=action.sources,
                polarity=action.polarity,
                mechanisms=mechanism_ids,
            )
            added_interactions.append(action.interaction_id)
        elif isinstance(action, RemoveInteractionAction):
            if action.interaction_id not in interactions:
                raise ValueError(
                    f"cannot remove unknown interaction: {action.interaction_id}"
                )
            del interactions[action.interaction_id]
            removed_interactions.append(action.interaction_id)
        elif isinstance(action, SetStateMeasurementAction):
            measurements[action.state] = TopologyStateMeasurement(
                state=action.state,
                channel=action.channel,
            )
        elif isinstance(action, RemoveStateMeasurementAction):
            if action.state not in measurements:
                raise ValueError(
                    f"cannot remove missing state measurement: {action.state}"
                )
            del measurements[action.state]
        elif isinstance(action, SetTargetMappingAction):
            mappings[action.channel] = TopologyTargetMapping(
                channel=action.channel,
                source=action.source,
            )
        elif isinstance(action, RemoveTargetMappingAction):
            if action.channel not in mappings:
                raise ValueError(
                    f"cannot remove missing target mapping: {action.channel}"
                )
            del mappings[action.channel]
        else:  # pragma: no cover - Pydantic owns the closed action union.
            raise TypeError(f"unsupported topology action: {type(action).__name__}")

    after = _canonical_topology_draft(
        TopologyDraft(
            states=tuple(states.values()),
            processes=tuple(processes.values()),
            interactions=tuple(interactions.values()),
            state_measurements=tuple(measurements.values()),
            target_mappings=tuple(mappings.values()),
        )
    )
    generated_node_count = len(after.states) + len(after.processes)
    if generated_node_count > maximum_generated_nodes:
        raise ValueError(
            "topology draft contains "
            f"{generated_node_count} generated nodes; maximum is "
            f"{maximum_generated_nodes}"
        )
    if len(after.interactions) > maximum_interactions:
        raise ValueError(
            "topology draft contains "
            f"{len(after.interactions)} interactions; maximum is "
            f"{maximum_interactions}"
        )
    _validate_partial_topology(after, context)
    after_sha256 = topology_draft_sha256(after)
    return TopologyActionApplication(
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        intent=intent,
        topology_phase=topology_phase,
        changed=before_sha256 != after_sha256,
        added_nodes=tuple(added_nodes),
        removed_nodes=tuple(removed_nodes),
        added_interactions=tuple(added_interactions),
        removed_interactions=tuple(removed_interactions),
        draft=after,
    )


def _validate_topology_phase_actions(
    transaction: ProposedTopologyActionTransaction,
    *,
    phase: TopologyConstructionPhase,
    state_names: frozenset[str],
    process_names: frozenset[str],
    interactions: dict[str, TopologyDraftInteraction],
) -> None:
    """Reject actions outside the current runtime-owned construction phase."""
    if phase in {
        TopologyConstructionPhase.MIXED,
        TopologyConstructionPhase.CLOSURE_REPAIR,
    }:
        return
    for action in transaction.actions:
        if phase == TopologyConstructionPhase.COMPONENT_SPECIFICATION:
            if not isinstance(
                action,
                (AddStateAction, AddProcessAction, RemoveGeneratedNodeAction),
            ):
                raise ValueError(
                    "component_specification accepts only generated state or "
                    "process actions"
                )
            continue
        if isinstance(action, AddInteractionAction):
            target = action.target
        elif isinstance(action, RemoveInteractionAction):
            existing = interactions.get(action.interaction_id)
            if existing is None:
                raise ValueError(
                    f"cannot remove unknown interaction: {action.interaction_id}"
                )
            target = existing.target
        else:
            target = None
        if phase == TopologyConstructionPhase.DYNAMIC_TOPOLOGY:
            if target is None or target not in state_names:
                raise ValueError(
                    "dynamic_topology accepts only interaction actions whose "
                    "target is a declared generated state"
                )
            continue
        if phase == TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY:
            if isinstance(
                action,
                (
                    SetStateMeasurementAction,
                    RemoveStateMeasurementAction,
                    SetTargetMappingAction,
                    RemoveTargetMappingAction,
                ),
            ):
                continue
            if target is not None and target in process_names:
                continue
            raise ValueError(
                "algebraic_readout_topology accepts process-targeted "
                "interactions, measurements, and target mappings only"
            )


def finalize_topology_draft(
    draft: TopologyDraft,
    context: ValidationContext,
) -> TopologyCandidate:
    """Compile a complete topology draft into the existing immutable artifact."""
    canonical = _canonical_topology_draft(draft)
    _validate_partial_topology(canonical, context)
    states = {item.name for item in canonical.states}
    processes = {item.name for item in canonical.processes}
    digest = topology_draft_sha256(canonical)
    proposal = ProposedTopologyCandidate(
        candidate_id=f"topology_{digest[:16]}",
        change_summary="Runtime-compiled incremental topology.",
        states=canonical.states,
        processes=canonical.processes,
        interactions=tuple(
            TopologyInteraction(
                interaction_id=item.interaction_id,
                target=item.target,
                target_kind=(
                    InteractionTargetKind.STATE_DERIVATIVE
                    if item.target in states
                    else InteractionTargetKind.ALGEBRAIC_PROCESS
                ),
                sources=item.sources,
                polarity=item.polarity,
                mechanisms=item.mechanisms,
            )
            for item in canonical.interactions
        ),
        state_measurements=canonical.state_measurements,
        target_mappings=canonical.target_mappings,
    )
    if not states:
        raise ValueError("complete topology requires at least one state")
    if any(item.target not in states | processes for item in canonical.interactions):
        raise ValueError("complete topology contains an unresolved target")
    return enrich_topology_proposal(proposal, context)


def apply_functional_actions(
    draft: FunctionalDraft,
    intent: ConstructionIntent,
    transaction: ProposedFunctionalActionTransaction,
    topology: TopologyCandidate,
    context: ValidationContext,
    *,
    allowed_requirement_ids: Iterable[str] | None = None,
) -> FunctionalActionApplication:
    """Apply localized function edits while preserving unmentioned assignments."""
    commitment = topology_commitment_sha256(topology)
    if draft.topology_commitment_sha256 != commitment:
        raise ValueError("functional draft belongs to a different topology")
    _validate_intent(
        intent.requirement_ids,
        intent.target_channels,
        context,
        allowed_requirement_ids=allowed_requirement_ids,
    )
    before = _canonical_functional_draft(draft)
    before_sha256 = functional_draft_sha256(before)
    functions = {item.interaction_id: item for item in before.interaction_functions}
    initials = {item.state: item for item in before.latent_initials}
    interaction_ids = {item.interaction_id for item in topology.interactions}
    latent_states = {
        item.name for item in topology.states if item.kind is StateKind.LATENT
    }
    set_interactions: list[str] = []
    removed_interactions: list[str] = []
    set_initials: list[str] = []
    removed_initials: list[str] = []

    for action in transaction.actions:
        if isinstance(action, SetInteractionFunctionAction):
            if action.interaction_id not in interaction_ids:
                raise ValueError(
                    "function action references an unknown topology interaction: "
                    f"{action.interaction_id}"
                )
            functions[action.interaction_id] = InteractionFunctionDraft(
                interaction_id=action.interaction_id,
                expression=action.expression,
                parameters=action.parameters,
            )
            set_interactions.append(action.interaction_id)
        elif isinstance(action, RemoveInteractionFunctionAction):
            if action.interaction_id not in functions:
                raise ValueError(
                    "cannot remove missing interaction function: "
                    f"{action.interaction_id}"
                )
            del functions[action.interaction_id]
            removed_interactions.append(action.interaction_id)
        elif isinstance(action, SetLatentInitialAction):
            if action.state not in latent_states:
                raise ValueError(
                    f"latent initializer references a non-latent state: {action.state}"
                )
            initials[action.state] = ProposedFunctionalInitial(
                state=action.state,
                initial=action.initial,
            )
            set_initials.append(action.state)
        elif isinstance(action, RemoveLatentInitialAction):
            if action.state not in initials:
                raise ValueError(
                    f"cannot remove missing latent initializer: {action.state}"
                )
            del initials[action.state]
            removed_initials.append(action.state)
        else:  # pragma: no cover - Pydantic owns the closed action union.
            raise TypeError(f"unsupported functional action: {type(action).__name__}")

    after = _canonical_functional_draft(
        FunctionalDraft(
            topology_commitment_sha256=commitment,
            interaction_functions=tuple(functions.values()),
            latent_initials=tuple(initials.values()),
        )
    )
    after_sha256 = functional_draft_sha256(after)
    return FunctionalActionApplication(
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        intent=intent,
        changed=before_sha256 != after_sha256,
        set_interaction_ids=tuple(set_interactions),
        removed_interaction_ids=tuple(removed_interactions),
        set_initial_states=tuple(set_initials),
        removed_initial_states=tuple(removed_initials),
        draft=after,
    )


def assess_functional_compatibility(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    *,
    parser: RestrictedParser | None = None,
) -> FunctionalCompatibilityReport:
    """Check whether localized functions faithfully implement their topology."""
    diagnostics: list[ConstructionDiagnostic] = []
    commitment = topology_commitment_sha256(topology)
    if draft.topology_commitment_sha256 != commitment:
        diagnostics.append(
            ConstructionDiagnostic(
                code="TOPOLOGY_COMMITMENT_MISMATCH",
                location="functional_draft",
                message="functional draft belongs to a different topology",
            )
        )
    interactions = {item.interaction_id: item for item in topology.interactions}
    functions = {item.interaction_id: item for item in draft.interaction_functions}
    missing_interactions = tuple(sorted(set(interactions) - set(functions)))
    for interaction_id in sorted(set(functions) - set(interactions)):
        diagnostics.append(
            ConstructionDiagnostic(
                code="UNKNOWN_INTERACTION",
                location=f"interaction:{interaction_id}",
                message="function assignment has no matching topology interaction",
            )
        )

    parameter_roles: dict[str, ParameterRole] = {}
    for function in draft.interaction_functions:
        for parameter in function.parameters:
            previous = parameter_roles.get(parameter.name)
            if previous is not None and previous is not parameter.role:
                diagnostics.append(
                    ConstructionDiagnostic(
                        code="PARAMETER_ROLE_CONFLICT",
                        location=f"parameter:{parameter.name}",
                        message=(
                            "shared parameter has conflicting qualitative roles: "
                            f"{previous.value}, {parameter.role.value}"
                        ),
                    )
                )
            parameter_roles[parameter.name] = parameter.role

    restricted_parser = parser or RestrictedParser()
    for interaction_id in sorted(set(functions) & set(interactions)):
        function = functions[interaction_id]
        interaction = interactions[interaction_id]
        try:
            parsed = restricted_parser.parse(
                function.expression,
                location=f"interaction:{interaction_id}",
            )
        except ModelValidationError as exc:
            diagnostics.extend(
                ConstructionDiagnostic(
                    code=item.code,
                    location=item.location,
                    message=item.message,
                )
                for item in exc.diagnostics
            )
            continue
        local_parameters = {item.name for item in function.parameters}
        used_parameters = set(parsed.symbols) & set(parameter_roles)
        missing_local_declarations = used_parameters - local_parameters
        if missing_local_declarations:
            diagnostics.append(
                ConstructionDiagnostic(
                    code="NONLOCAL_PARAMETER_REFERENCE",
                    location=f"interaction:{interaction_id}",
                    message=(
                        "localized function must redeclare every shared parameter "
                        f"it uses: {sorted(missing_local_declarations)}"
                    ),
                )
            )
        unused_local = local_parameters - set(parsed.symbols)
        if unused_local:
            diagnostics.append(
                ConstructionDiagnostic(
                    code="UNUSED_LOCAL_PARAMETER",
                    location=f"interaction:{interaction_id}",
                    message=f"unused local parameters: {sorted(unused_local)}",
                )
            )
        used_sources = set(parsed.symbols) - set(parameter_roles)
        expected_sources = set(interaction.sources)
        if used_sources != expected_sources:
            diagnostics.append(
                ConstructionDiagnostic(
                    code="TOPOLOGY_SOURCE_MISMATCH",
                    location=f"interaction:{interaction_id}",
                    message=(
                        f"missing_sources={sorted(expected_sources - used_sources)}, "
                        f"extra_sources={sorted(used_sources - expected_sources)}"
                    ),
                )
            )
        signed = sorted(
            name
            for name in used_parameters
            if parameter_roles[name] is ParameterRole.COEFFICIENT
        )
        if signed:
            diagnostics.append(
                ConstructionDiagnostic(
                    code="SIGNED_WEIGHT_WITH_TOPOLOGY_POLARITY",
                    location=f"interaction:{interaction_id}",
                    message=(
                        "topology owns the outer sign; scalar edge weights need "
                        f"nonnegative or positive roles: {signed}"
                    ),
                )
            )

    latent_states = {
        item.name for item in topology.states if item.kind is StateKind.LATENT
    }
    supplied_initials = {item.state for item in draft.latent_initials}
    missing_initials = tuple(sorted(latent_states - supplied_initials))
    for state in sorted(supplied_initials - latent_states):
        diagnostics.append(
            ConstructionDiagnostic(
                code="INITIAL_FOR_NONLATENT_STATE",
                location=f"initial:{state}",
                message="functional draft initializes a non-latent state",
            )
        )

    status: Literal["incomplete", "compatible", "incompatible"]
    if diagnostics:
        status = "incompatible"
    elif missing_interactions or missing_initials:
        status = "incomplete"
    else:
        status = "compatible"
    return FunctionalCompatibilityReport(
        status=status,
        missing_interaction_ids=missing_interactions,
        missing_latent_initial_states=missing_initials,
        diagnostics=tuple(diagnostics),
    )


def finalize_functional_draft(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    context: ValidationContext,
    *,
    parser: RestrictedParser | None = None,
) -> StagedCandidateExpansion:
    """Compile one compatible function draft through the existing safe runtime."""
    report = assess_functional_compatibility(topology, draft, parser=parser)
    if report.status != "compatible":
        details = "; ".join(item.message for item in report.diagnostics)
        if not details:
            details = (
                f"missing_interactions={list(report.missing_interaction_ids)}, "
                f"missing_initials={list(report.missing_latent_initial_states)}"
            )
        raise ValueError(f"functional draft is {report.status}: {details}")
    parameters = _merged_parameters(draft)
    digest = functional_draft_sha256(draft)
    proposal = ProposedFunctionalCandidate(
        candidate_id=f"functional_{digest[:16]}",
        change_summary="Runtime-compiled incremental function assignment.",
        interaction_functions=tuple(
            InteractionFunction(
                interaction_id=item.interaction_id,
                expression=item.expression,
            )
            for item in draft.interaction_functions
        ),
        parameters=parameters,
        latent_initials=draft.latent_initials,
    )
    functional = enrich_functional_proposal(proposal, topology)
    return expand_staged_candidate(
        topology,
        functional,
        context,
        parser=parser,
    )


def select_conditional_beam(
    entries: Iterable[ConditionalBeamEntry],
    *,
    beam_size: int,
    maximum_functions_per_topology: int,
) -> tuple[ConditionalBeamEntry, ...]:
    """Select a score-ordered beam while preserving topology diversity.

    Lower scores are better.  Exact topology/function duplicates collapse to
    their best score.  Each topology contributes its best compatible function
    before remaining slots are filled globally, subject to the beam capacity.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be at least one")
    if maximum_functions_per_topology < 1:
        raise ValueError("maximum_functions_per_topology must be at least one")
    deduplicated: dict[tuple[str, str], ConditionalBeamEntry] = {}
    for entry in entries:
        key = (entry.topology_sha256, entry.functional_sha256)
        previous = deduplicated.get(key)
        if previous is None or _beam_key(entry) < _beam_key(previous):
            deduplicated[key] = entry
    grouped: dict[str, list[ConditionalBeamEntry]] = defaultdict(list)
    for entry in deduplicated.values():
        grouped[entry.topology_sha256].append(entry)
    eligible: dict[str, list[ConditionalBeamEntry]] = {
        topology: sorted(values, key=_beam_key)[:maximum_functions_per_topology]
        for topology, values in grouped.items()
    }
    representatives = sorted(
        (values[0] for values in eligible.values()),
        key=_beam_key,
    )
    selected = representatives[:beam_size]
    selected_keys = {
        (item.topology_sha256, item.functional_sha256) for item in selected
    }
    remaining = sorted(
        (
            item
            for values in eligible.values()
            for item in values
            if (item.topology_sha256, item.functional_sha256) not in selected_keys
        ),
        key=_beam_key,
    )
    selected.extend(remaining[: max(0, beam_size - len(selected))])
    return tuple(selected)


class ConstructionTranspositionTable:
    """Small content-addressed duplicate table with checkpointable snapshots."""

    def __init__(
        self,
        entries: Iterable[tuple[TranspositionStage, str]] = (),
    ) -> None:
        self._entries = set(entries)

    def register(self, stage: TranspositionStage, sha256: str) -> bool:
        """Register a canonical state; return false when already encountered."""
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("transposition key must be a lowercase SHA-256")
        key = (stage, sha256)
        if key in self._entries:
            return False
        self._entries.add(key)
        return True

    def snapshot(self) -> tuple[tuple[TranspositionStage, str], ...]:
        """Return stable data suitable for an ordinary controller checkpoint."""
        return tuple(sorted(self._entries))


def _validate_intent(
    requirement_ids: tuple[str, ...],
    target_channels: tuple[str, ...],
    context: ValidationContext,
    *,
    allowed_requirement_ids: Iterable[str] | None,
) -> None:
    unknown_targets = set(target_channels) - set(context.targets)
    if unknown_targets:
        raise ValueError(
            f"construction intent references unknown targets: {sorted(unknown_targets)}"
        )
    if allowed_requirement_ids is not None:
        unknown_requirements = set(requirement_ids) - set(allowed_requirement_ids)
        if unknown_requirements:
            raise ValueError(
                "construction intent references unknown requirements: "
                f"{sorted(unknown_requirements)}"
            )


def _validate_partial_topology(
    draft: TopologyDraft,
    context: ValidationContext,
) -> None:
    states = {item.name for item in draft.states}
    processes = {item.name for item in draft.processes}
    generated = states | processes
    available = generated | set(context.forcing_channels) | {context.time_symbol}
    for interaction in draft.interactions:
        if interaction.target not in generated:
            raise ValueError(
                f"interaction targets an unknown generated node: {interaction.target}"
            )
        unknown_sources = set(interaction.sources) - available
        if unknown_sources:
            raise ValueError(
                f"interaction {interaction.interaction_id} uses unavailable sources: "
                f"{sorted(unknown_sources)}"
            )
    for measurement in draft.state_measurements:
        if measurement.state not in states:
            raise ValueError(
                f"measurement references an unknown state: {measurement.state}"
            )
        if measurement.channel not in set(context.auxiliaries):
            raise ValueError(
                "measurement references an unavailable auxiliary channel: "
                f"{measurement.channel}"
            )
    for mapping in draft.target_mappings:
        if mapping.channel not in set(context.targets):
            raise ValueError(
                f"mapping references an unknown public target: {mapping.channel}"
            )
        if mapping.source not in generated:
            raise ValueError(
                f"target mapping source is not generated: {mapping.source}"
            )


def _merged_parameters(draft: FunctionalDraft) -> tuple[ProposedParameter, ...]:
    merged: dict[str, ProposedParameter] = {}
    for function in draft.interaction_functions:
        for parameter in function.parameters:
            previous = merged.get(parameter.name)
            if previous is not None and previous.role is not parameter.role:
                raise ValueError(
                    f"parameter role conflict for {parameter.name}: "
                    f"{previous.role.value}, {parameter.role.value}"
                )
            merged[parameter.name] = parameter
    return tuple(merged[name] for name in sorted(merged))


def _canonical_topology_draft(draft: TopologyDraft) -> TopologyDraft:
    return draft.model_copy(
        update={
            "states": tuple(sorted(draft.states, key=lambda item: item.name)),
            "processes": tuple(sorted(draft.processes, key=lambda item: item.name)),
            "interactions": tuple(
                sorted(draft.interactions, key=lambda item: item.interaction_id)
            ),
            "state_measurements": tuple(
                sorted(
                    draft.state_measurements,
                    key=lambda item: (item.state, item.channel),
                )
            ),
            "target_mappings": tuple(
                sorted(draft.target_mappings, key=lambda item: item.channel)
            ),
        }
    )


def _canonical_functional_draft(draft: FunctionalDraft) -> FunctionalDraft:
    functions = tuple(
        item.model_copy(
            update={
                "parameters": tuple(
                    sorted(item.parameters, key=lambda value: value.name)
                )
            }
        )
        for item in sorted(
            draft.interaction_functions,
            key=lambda value: value.interaction_id,
        )
    )
    return draft.model_copy(
        update={
            "interaction_functions": functions,
            "latent_initials": tuple(
                sorted(draft.latent_initials, key=lambda item: item.state)
            ),
        }
    )


def _beam_key(entry: ConditionalBeamEntry) -> tuple[float, str, str]:
    return (entry.score, entry.topology_sha256, entry.functional_sha256)


def _deduplicate_exact_actions(actions: tuple) -> tuple[tuple, tuple[int, ...]]:
    retained: list[object] = []
    removed: list[int] = []
    observed: set[str] = set()
    for index, action in enumerate(actions):
        canonical = json.dumps(
            action.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if canonical in observed:
            removed.append(index)
            continue
        observed.add(canonical)
        retained.append(action)
    return tuple(retained), tuple(removed)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ConstructionTranspositionTable",
    "apply_functional_actions",
    "apply_topology_actions",
    "assess_functional_compatibility",
    "finalize_functional_draft",
    "finalize_topology_draft",
    "functional_draft_sha256",
    "normalize_functional_action_transaction",
    "normalize_topology_action_transaction",
    "select_conditional_beam",
    "topology_draft_sha256",
]
