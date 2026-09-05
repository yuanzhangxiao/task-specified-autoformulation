"""Checkpointed intent and action calls for incremental model construction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.construction import (
    ConstructionTranspositionTable,
    apply_functional_actions,
    apply_topology_actions,
    assess_functional_compatibility,
    finalize_functional_draft,
    functional_draft_sha256,
    topology_draft_sha256,
)
from autoformalism.expressions import ValidationContext
from autoformalism.llm.models import (
    IncrementalConstructionLLMClient,
    LLMCallResult,
)
from autoformalism.schemas import (
    FunctionalActionApplication,
    FunctionalCompatibilityReport,
    FunctionalDraft,
    ProposedConstructionFocus,
    ProposedConstructionIntent,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    TopologyActionApplication,
    TopologyCandidate,
    TopologyConstructionPhase,
    TopologyDraft,
)
from autoformalism.schemas.base import NonEmptyText, StrictSchema
from autoformalism.schemas.staged import Sha256Digest
from autoformalism.search.construction_agenda import (
    ConstructionAgenda,
    select_construction_agenda,
)
from autoformalism.search.feedback_routing import (
    RevisionStage,
    RoutedProposerFeedback,
)
from autoformalism.search.staged_proposer import StageCallReceipt
from autoformalism.staging import (
    StagedCandidateExpansion,
    topology_commitment_sha256,
)

ActionStage = Literal["topology", "functional_form"]


class IncrementalProposerConfig(StrictSchema):
    """Immutable prompts and checkpoint namespace for action construction."""

    schema_version: Literal["incremental-proposer-config-1"] = (
        "incremental-proposer-config-1"
    )
    checkpoint_directory: Path
    run_fingerprint: Sha256Digest
    intent_system_prompt: NonEmptyText
    topology_action_system_prompt: NonEmptyText
    functional_action_system_prompt: NonEmptyText
    decision_policy: Literal["llm_objective_v1", "runtime_priority_v2"] = (
        "llm_objective_v1"
    )
    maximum_topology_actions_per_transaction: int = Field(
        default=64, ge=1, le=64
    )
    maximum_functional_actions_per_transaction: int = Field(
        default=128, ge=1, le=128
    )
    maximum_generated_nodes_per_topology: int = Field(
        default=64, ge=1, le=64
    )
    maximum_interactions_per_topology: int = Field(
        default=512, ge=1, le=512
    )
    cache_only: bool = False


class IncrementalTopologyResult(StrictSchema):
    """Auditable decision, action, and application for one topology edit."""

    schema_version: Literal["incremental-topology-result-1"] = (
        "incremental-topology-result-1"
    )
    intent: ProposedConstructionIntent
    agenda: ConstructionAgenda | None = None
    focus: ProposedConstructionFocus | None = None
    transaction: ProposedTopologyActionTransaction
    application: TopologyActionApplication
    intent_call: StageCallReceipt
    action_call: StageCallReceipt
    transposition_new: bool


class IncrementalFunctionalResult(StrictSchema):
    """Auditable function edit and its exact topology compatibility result."""

    schema_version: Literal["incremental-functional-result-1"] = (
        "incremental-functional-result-1"
    )
    intent: ProposedConstructionIntent
    agenda: ConstructionAgenda | None = None
    focus: ProposedConstructionFocus | None = None
    transaction: ProposedFunctionalActionTransaction
    application: FunctionalActionApplication
    compatibility: FunctionalCompatibilityReport
    expansion: StagedCandidateExpansion | None = None
    intent_call: StageCallReceipt
    action_call: StageCallReceipt
    transposition_new: bool


class IncrementalProposer:
    """Apply small provider-proposed edits to runtime-owned model drafts."""

    def __init__(
        self,
        *,
        client: IncrementalConstructionLLMClient,
        config: IncrementalProposerConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._transpositions = _load_transpositions(
            config.checkpoint_directory / "transpositions.json"
        )

    def revise_topology(
        self,
        *,
        public_problem: str,
        context: ValidationContext,
        allowed_requirement_ids: tuple[str, ...],
        feedback: RoutedProposerFeedback,
        parent: TopologyDraft,
        proposal_slot: int = 0,
        topology_phase: TopologyConstructionPhase = (TopologyConstructionPhase.MIXED),
        attach_intent_mechanisms: bool = True,
        cache_only: bool | None = None,
    ) -> IncrementalTopologyResult:
        """Select a scientific focus and apply one topology transaction."""
        _require_public_problem(public_problem)
        _require_proposal_slot(proposal_slot)
        intent, intent_call, agenda, focus = self._decision_stage(
            public_problem=public_problem,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            feedback=feedback,
            stage="topology",
            parent_topology=parent,
            topology=None,
            parent_functional=None,
            proposal_slot=proposal_slot,
            cache_only=cache_only,
        )
        action_prompt = _topology_action_prompt(
            public_problem=public_problem,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            feedback=feedback,
            parent=parent,
            intent=intent,
            agenda=agenda,
            topology_phase=topology_phase,
            proposal_slot=proposal_slot,
            maximum_action_count=_topology_action_limit(
                self._config.maximum_topology_actions_per_transaction,
                topology_phase,
            ),
        )
        input_hash = self._stage_input_hash(
            "topology_action",
            self._config.topology_action_system_prompt,
            action_prompt,
        )
        checkpoint = self._checkpoint_path("topology-action", input_hash)
        restored = _load_topology_checkpoint(checkpoint)
        if restored is not None:
            return restored.model_copy(
                update={
                    "intent_call": intent_call,
                    "action_call": _checkpoint_receipt(
                        restored.action_call.request_hash
                    ),
                }
            )
        call = self._client.propose_topology_actions(
            system_prompt=self._config.topology_action_system_prompt,
            user_prompt=action_prompt,
            parent=parent,
            intent=intent,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            topology_phase=topology_phase,
            attach_intent_mechanisms=attach_intent_mechanisms,
            cache_only=self._cache_only(cache_only),
        )
        action_limit = _topology_action_limit(
            self._config.maximum_topology_actions_per_transaction,
            topology_phase,
        )
        _require_action_budget(
            count=len(call.parsed.actions),
            maximum=action_limit,
            stage=f"topology/{topology_phase.value}",
        )
        application = apply_topology_actions(
            parent,
            intent.as_runtime_intent(),
            call.parsed,
            context,
            allowed_requirement_ids=allowed_requirement_ids,
            topology_phase=topology_phase,
            attach_intent_mechanisms=attach_intent_mechanisms,
            maximum_generated_nodes=(
                self._config.maximum_generated_nodes_per_topology
            ),
            maximum_interactions=(
                self._config.maximum_interactions_per_topology
            ),
        )
        if not application.changed:
            raise ValueError("topology action transaction makes no change")
        transposition_new = self._register_transposition(
            "topology", application.after_sha256
        )
        result = IncrementalTopologyResult(
            intent=intent,
            agenda=agenda,
            focus=focus,
            transaction=call.parsed,
            application=application,
            intent_call=intent_call,
            action_call=_call_receipt(call),
            transposition_new=transposition_new,
        )
        _write_result_checkpoint(checkpoint, input_hash, result)
        return result

    def revise_functions(
        self,
        *,
        public_problem: str,
        context: ValidationContext,
        allowed_requirement_ids: tuple[str, ...],
        feedback: RoutedProposerFeedback,
        topology: TopologyCandidate,
        parent: FunctionalDraft,
        proposal_slot: int = 0,
        cache_only: bool | None = None,
    ) -> IncrementalFunctionalResult:
        """Select a scientific focus and apply one topology-bound function edit."""
        _require_public_problem(public_problem)
        _require_proposal_slot(proposal_slot)
        commitment = topology_commitment_sha256(topology)
        if parent.topology_commitment_sha256 != commitment:
            raise ValueError("functional parent belongs to a different topology")
        intent, intent_call, agenda, focus = self._decision_stage(
            public_problem=public_problem,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            feedback=feedback,
            stage="functional_form",
            parent_topology=None,
            topology=topology,
            parent_functional=parent,
            proposal_slot=proposal_slot,
            cache_only=cache_only,
        )
        action_prompt = _functional_action_prompt(
            public_problem=public_problem,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            feedback=feedback,
            topology=topology,
            parent=parent,
            intent=intent,
            agenda=agenda,
            proposal_slot=proposal_slot,
            maximum_action_count=(
                self._config.maximum_functional_actions_per_transaction
            ),
        )
        input_hash = self._stage_input_hash(
            "functional_action",
            self._config.functional_action_system_prompt,
            action_prompt,
        )
        checkpoint = self._checkpoint_path("functional-action", input_hash)
        restored = _load_functional_checkpoint(checkpoint)
        if restored is not None:
            return restored.model_copy(
                update={
                    "intent_call": intent_call,
                    "action_call": _checkpoint_receipt(
                        restored.action_call.request_hash
                    ),
                }
            )
        call = self._client.propose_functional_actions(
            system_prompt=self._config.functional_action_system_prompt,
            user_prompt=action_prompt,
            parent=parent,
            intent=intent,
            topology=topology,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            cache_only=self._cache_only(cache_only),
        )
        _require_action_budget(
            count=len(call.parsed.actions),
            maximum=self._config.maximum_functional_actions_per_transaction,
            stage="functional_form",
        )
        application = apply_functional_actions(
            parent,
            intent.as_runtime_intent(),
            call.parsed,
            topology,
            context,
            allowed_requirement_ids=allowed_requirement_ids,
        )
        if not application.changed:
            raise ValueError("functional action transaction makes no change")
        compatibility = assess_functional_compatibility(topology, application.draft)
        if compatibility.status == "incompatible":
            details = "; ".join(
                f"{item.code} at {item.location}: {item.message}"
                for item in compatibility.diagnostics
            )
            raise ValueError(
                f"functional actions are incompatible with the topology: {details}"
            )
        expansion = (
            finalize_functional_draft(topology, application.draft, context)
            if compatibility.status == "compatible"
            else None
        )
        transposition_new = self._register_transposition(
            "functional", application.after_sha256
        )
        result = IncrementalFunctionalResult(
            intent=intent,
            agenda=agenda,
            focus=focus,
            transaction=call.parsed,
            application=application,
            compatibility=compatibility,
            expansion=expansion,
            intent_call=intent_call,
            action_call=_call_receipt(call),
            transposition_new=transposition_new,
        )
        _write_result_checkpoint(checkpoint, input_hash, result)
        return result

    def _decision_stage(
        self,
        *,
        public_problem: str,
        context: ValidationContext,
        allowed_requirement_ids: tuple[str, ...],
        feedback: RoutedProposerFeedback,
        stage: ActionStage,
        parent_topology: TopologyDraft | None,
        topology: TopologyCandidate | None,
        parent_functional: FunctionalDraft | None,
        proposal_slot: int,
        cache_only: bool | None,
    ) -> tuple[
        ProposedConstructionIntent,
        StageCallReceipt,
        ConstructionAgenda | None,
        ProposedConstructionFocus | None,
    ]:
        """Choose a legacy intent or a focus inside a runtime-owned agenda."""
        if self._config.decision_policy == "llm_objective_v1":
            prompt = _intent_prompt(
                public_problem=public_problem,
                context=context,
                allowed_requirement_ids=allowed_requirement_ids,
                feedback=feedback,
                stage=stage,
                parent_topology=parent_topology,
                topology=topology,
                parent_functional=parent_functional,
                proposal_slot=proposal_slot,
            )
            intent, receipt = self._intent_stage(
                prompt,
                context=context,
                allowed_requirement_ids=allowed_requirement_ids,
                cache_only=cache_only,
            )
            return intent, receipt, None, None

        agenda = select_construction_agenda(
            stage=stage,
            feedback=feedback,
            allowed_requirement_ids=allowed_requirement_ids,
            target_channels=context.targets,
        )
        prompt = _focus_prompt(
            public_problem=public_problem,
            context=context,
            agenda=agenda,
            parent_topology=parent_topology,
            topology=topology,
            parent_functional=parent_functional,
            proposal_slot=proposal_slot,
        )
        focus, receipt = self._focus_stage(
            prompt,
            context=context,
            agenda=agenda,
            cache_only=cache_only,
        )
        intent = ProposedConstructionIntent(
            objective=agenda.objective,
            requirement_ids=focus.requirement_ids,
            target_channels=focus.target_channels,
        )
        return intent, receipt, agenda, focus

    def _intent_stage(
        self,
        user_prompt: str,
        *,
        context: ValidationContext,
        allowed_requirement_ids: tuple[str, ...],
        cache_only: bool | None,
    ) -> tuple[ProposedConstructionIntent, StageCallReceipt]:
        input_hash = self._stage_input_hash(
            "construction_intent",
            self._config.intent_system_prompt,
            user_prompt,
        )
        checkpoint = self._checkpoint_path("intent", input_hash)
        restored = _load_intent_checkpoint(checkpoint)
        if restored is not None:
            intent, request_hash = restored
            return intent, _checkpoint_receipt(request_hash)
        call = self._client.propose_construction_intent(
            system_prompt=self._config.intent_system_prompt,
            user_prompt=user_prompt,
            context=context,
            allowed_requirement_ids=allowed_requirement_ids,
            cache_only=self._cache_only(cache_only),
        )
        _write_intent_checkpoint(checkpoint, input_hash, call)
        return call.parsed, _call_receipt(call)

    def _focus_stage(
        self,
        user_prompt: str,
        *,
        context: ValidationContext,
        agenda: ConstructionAgenda,
        cache_only: bool | None,
    ) -> tuple[ProposedConstructionFocus, StageCallReceipt]:
        input_hash = self._stage_input_hash(
            "construction_focus",
            self._config.intent_system_prompt,
            user_prompt,
        )
        checkpoint = self._checkpoint_path("focus", input_hash)
        restored = _load_focus_checkpoint(checkpoint)
        if restored is not None:
            focus, request_hash = restored
            return focus, _checkpoint_receipt(request_hash)
        allowed_indices = tuple(range(len(agenda.feedback_items)))
        call = self._client.propose_construction_focus(
            system_prompt=self._config.intent_system_prompt,
            user_prompt=user_prompt,
            context=context,
            allowed_requirement_ids=agenda.eligible_requirement_ids,
            allowed_target_channels=agenda.eligible_target_channels,
            allowed_feedback_item_indices=allowed_indices,
            cache_only=self._cache_only(cache_only),
        )
        _write_focus_checkpoint(checkpoint, input_hash, call)
        return call.parsed, _call_receipt(call)

    def _register_transposition(self, stage: str, sha256: str) -> bool:
        is_new = self._transpositions.register(stage, sha256)  # type: ignore[arg-type]
        if is_new:
            _write_transpositions(
                self._config.checkpoint_directory / "transpositions.json",
                self._transpositions,
            )
        return is_new

    def _cache_only(self, override: bool | None) -> bool:
        return self._config.cache_only if override is None else override

    def _stage_input_hash(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return _sha256(
            {
                "run_fingerprint": self._config.run_fingerprint,
                "protocol": self._config.schema_version,
                "stage": stage,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    def _checkpoint_path(self, stage: str, input_hash: str) -> Path:
        return self._config.checkpoint_directory / f"{stage}-{input_hash}.json"


def _intent_prompt(
    *,
    public_problem: str,
    context: ValidationContext,
    allowed_requirement_ids: tuple[str, ...],
    feedback: RoutedProposerFeedback,
    stage: ActionStage,
    parent_topology: TopologyDraft | None,
    topology: TopologyCandidate | None,
    parent_functional: FunctionalDraft | None,
    proposal_slot: int,
) -> str:
    payload = {
        "schema_version": "construction-intent-request-1",
        "task": (
            "Choose one bounded scientific objective and the public requirement "
            "or target anchors that the next localized action should address. "
            "Do not propose model structure, equations, identifiers, or numbers."
        ),
        "stage": stage,
        "proposal_slot": proposal_slot,
        "public_problem": public_problem,
        "runtime_contract": _public_contract(
            context, allowed_requirement_ids=allowed_requirement_ids
        ),
        "routed_feedback": feedback.for_stage(
            RevisionStage.TOPOLOGY
            if stage == "topology"
            else RevisionStage.FUNCTIONAL_FORM,
            include_integrated_repairs=True,
        ),
        "parent_topology_draft": (
            None if parent_topology is None else parent_topology.model_dump(mode="json")
        ),
        "immutable_topology": (
            None if topology is None else topology.model_dump(mode="json")
        ),
        "parent_functional_draft": (
            None
            if parent_functional is None
            else parent_functional.model_dump(mode="json")
        ),
    }
    return _render(payload)


def _focus_prompt(
    *,
    public_problem: str,
    context: ValidationContext,
    agenda: ConstructionAgenda,
    parent_topology: TopologyDraft | None,
    topology: TopologyCandidate | None,
    parent_functional: FunctionalDraft | None,
    proposal_slot: int,
) -> str:
    payload = {
        "schema_version": "construction-focus-request-1",
        "task": (
            "The runtime has already selected the highest-priority feedback "
            "category and objective. Select one or more listed feedback items "
            "and one or more eligible public mechanism/target anchors for the "
            "next action. Multiple mechanisms may be selected in one call. "
            "Do not repeat the objective or propose model content."
        ),
        "proposal_slot": proposal_slot,
        "public_problem": public_problem,
        "runtime_contract": _public_contract(
            context,
            allowed_requirement_ids=agenda.eligible_requirement_ids,
        ),
        "runtime_selected_agenda": agenda.model_dump(mode="json"),
        "feedback_item_indices": list(range(len(agenda.feedback_items))),
        "parent_topology_draft": (
            None if parent_topology is None else parent_topology.model_dump(mode="json")
        ),
        "immutable_topology": (
            None if topology is None else topology.model_dump(mode="json")
        ),
        "parent_functional_draft": (
            None
            if parent_functional is None
            else parent_functional.model_dump(mode="json")
        ),
    }
    return _render(payload)


def _topology_action_prompt(
    *,
    public_problem: str,
    context: ValidationContext,
    allowed_requirement_ids: tuple[str, ...],
    feedback: RoutedProposerFeedback,
    parent: TopologyDraft,
    intent: ProposedConstructionIntent,
    agenda: ConstructionAgenda | None,
    topology_phase: TopologyConstructionPhase,
    proposal_slot: int,
    maximum_action_count: int,
) -> str:
    payload: dict[str, object] = {
        "schema_version": (
            "topology-action-request-1"
            if agenda is None
            else "topology-action-request-2"
        ),
        "task": (
            (
                "Propose only the smallest topology action transaction that "
                "implements the selected intent."
                if agenda is None
                else "Propose one coherent phase-scoped topology action "
                "transaction that implements the selected intent. It may "
                "jointly cover multiple selected mechanisms."
            )
            + " Unmentioned runtime-owned structure is preserved. Do not "
            "provide equations, functions, parameters, units, ranges, "
            "scopes, summaries, or parent IDs. The displayed parent is the "
            "complete current draft. A rejected transaction is applied zero "
            "times, so never remove or reference an item absent from that parent."
        ),
        "public_problem": public_problem,
        "proposal_slot": proposal_slot,
        "runtime_contract": _public_contract(
            context, allowed_requirement_ids=allowed_requirement_ids
        ),
        "selected_intent": intent.model_dump(mode="json"),
        "parent_topology_draft": parent.model_dump(mode="json"),
        "parent_topology_draft_sha256": topology_draft_sha256(parent),
        "routed_feedback": (
            feedback.for_stage(
                RevisionStage.TOPOLOGY,
                include_integrated_repairs=True,
            )
            if agenda is None
            else None
        ),
        "edit_rules": {
            "no_implicit_deletion": True,
            "generated_node_types": ["state", "process"],
            "outer_sign_owned_by_topology": True,
            "target_kind_derived_by_runtime": True,
            "observability_derived_from_measurements_and_target_mappings": True,
            "maximum_action_count": maximum_action_count,
            "every_generated_node_requires_a_distinct_scientific_role": True,
            "name_extension_chains_are_not_scientific_roles": True,
            "transaction_is_atomic": True,
            "displayed_parent_is_authoritative": True,
        },
    }
    if agenda is not None:
        payload["runtime_selected_agenda"] = agenda.model_dump(mode="json")
        payload["topology_phase"] = topology_phase.value
        edit_rules = payload["edit_rules"]
        assert isinstance(edit_rules, dict)
        edit_rules["allowed_action_types"] = _allowed_topology_actions(topology_phase)
        payload["phase_contract"] = _topology_phase_contract(
            parent,
            context,
            topology_phase,
        )
    return _render(payload)


def _functional_action_prompt(
    *,
    public_problem: str,
    context: ValidationContext,
    allowed_requirement_ids: tuple[str, ...],
    feedback: RoutedProposerFeedback,
    topology: TopologyCandidate,
    parent: FunctionalDraft,
    intent: ProposedConstructionIntent,
    agenda: ConstructionAgenda | None,
    proposal_slot: int,
    maximum_action_count: int,
) -> str:
    payload: dict[str, object] = {
        "schema_version": (
            "functional-action-request-1"
            if agenda is None
            else "functional-action-request-2"
        ),
        "task": (
            "Propose only localized interaction-function or latent-initial "
            "actions under the immutable topology. "
            + (
                "A coherent transaction may cover multiple selected mechanisms. "
                if agenda is not None
                else ""
            )
            + "Unmentioned assignments are preserved. Each expression is "
            "an RHS value, not an equation. "
            "Declare only parameter names and qualitative roles; omit ranges, "
            "scopes, summaries, parent IDs, and topology hashes."
        ),
        "public_problem": public_problem,
        "proposal_slot": proposal_slot,
        "runtime_contract": _public_contract(
            context, allowed_requirement_ids=allowed_requirement_ids
        ),
        "selected_intent": intent.model_dump(mode="json"),
        "immutable_topology": topology.model_dump(mode="json"),
        "topology_commitment_sha256": topology_commitment_sha256(topology),
        "parent_functional_draft": parent.model_dump(mode="json"),
        "parent_functional_draft_sha256": functional_draft_sha256(parent),
        "routed_feedback": (
            feedback.for_stage(
                RevisionStage.FUNCTIONAL_FORM,
                include_integrated_repairs=True,
            )
            if agenda is None
            else None
        ),
        "compatibility_rules": {
            "interaction_id_must_exist": True,
            "expression_sources_must_match_topology_exactly": True,
            "outer_sign_owned_by_topology": True,
            "signed_scalar_weight_forbidden": True,
            "latent_initials_required_before_fitting": True,
            "maximum_action_count": maximum_action_count,
        },
    }
    if agenda is not None:
        payload["runtime_selected_agenda"] = agenda.model_dump(mode="json")
    return _render(payload)


def _allowed_topology_actions(
    phase: TopologyConstructionPhase,
) -> list[str]:
    if phase == TopologyConstructionPhase.COMPONENT_SPECIFICATION:
        return ["add_state", "add_process", "remove_generated_node"]
    if phase == TopologyConstructionPhase.DYNAMIC_TOPOLOGY:
        return ["add_interaction", "remove_interaction"]
    if phase == TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY:
        return [
            "add_interaction",
            "remove_interaction",
            "set_state_measurement",
            "remove_state_measurement",
            "set_target_mapping",
            "remove_target_mapping",
        ]
    return [
        "add_state",
        "add_process",
        "remove_generated_node",
        "add_interaction",
        "remove_interaction",
        "set_state_measurement",
        "remove_state_measurement",
        "set_target_mapping",
        "remove_target_mapping",
    ]


def _topology_phase_contract(
    parent: TopologyDraft,
    context: ValidationContext,
    phase: TopologyConstructionPhase,
) -> dict[str, object]:
    """Expose exact phase domains so the provider need not infer action legality."""
    state_names = sorted(item.name for item in parent.states)
    process_names = sorted(item.name for item in parent.processes)
    mapped_targets = {item.channel for item in parent.target_mappings}
    contract: dict[str, object] = {
        "declared_state_names": state_names,
        "declared_process_names": process_names,
        "public_target_channels": list(context.targets),
        "public_auxiliary_measurement_channels": list(context.auxiliaries),
        "unmapped_target_channels": sorted(set(context.targets) - mapped_targets),
        "target_channels_are_outputs_not_causal_inputs": True,
        "set_target_mapping_accepts_only_public_target_channels": True,
        "set_state_measurement_accepts_only_public_auxiliary_channels": True,
    }
    if phase == TopologyConstructionPhase.COMPONENT_SPECIFICATION:
        contract.update(
            {
                "responsibility": (
                    "Declare every generated dynamic state and instantaneous "
                    "process needed by the selected mechanisms and target "
                    "readouts. Existing runtime-scaffolded components and "
                    "mappings must be preserved."
                ),
                "public_channel_names_are_not_implicitly_generated_nodes": True,
            }
        )
    elif phase == TopologyConstructionPhase.DYNAMIC_TOPOLOGY:
        contract.update(
            {
                "responsibility": "Add hyperedges for state ODE right-hand sides.",
                "allowed_interaction_targets": state_names,
                "process_targets_forbidden_in_this_phase": True,
            }
        )
    elif phase == TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY:
        contract.update(
            {
                "responsibility": (
                    "Add hyperedges for algebraic processes, auxiliary state "
                    "measurements, and one generated source for every target."
                ),
                "allowed_interaction_targets": process_names,
            }
        )
    else:
        contract["responsibility"] = (
            "Repair only the reported closure defect against the current parent."
        )
    return contract


def _topology_action_limit(
    configured_maximum: int,
    phase: TopologyConstructionPhase,
) -> int:
    """Return the bounded transaction size for one topology responsibility."""
    phase_maximum = {
        TopologyConstructionPhase.COMPONENT_SPECIFICATION: 12,
        TopologyConstructionPhase.DYNAMIC_TOPOLOGY: 16,
        TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY: 16,
        TopologyConstructionPhase.CLOSURE_REPAIR: 8,
    }.get(phase, configured_maximum)
    return min(configured_maximum, phase_maximum)


def _require_action_budget(*, count: int, maximum: int, stage: str) -> None:
    """Reject an oversized response before it can mutate a maintained draft."""
    if count > maximum:
        raise ValueError(
            f"{stage} action transaction contains {count} actions; "
            f"maximum is {maximum}"
        )


def _public_contract(
    context: ValidationContext,
    *,
    allowed_requirement_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "target_channels": list(context.targets),
        "auxiliary_channels": list(context.auxiliaries),
        "forcing_channels": sorted(context.forcing_channels),
        "time_symbol": context.time_symbol,
        "allowed_requirement_ids": list(allowed_requirement_ids),
    }


def _require_public_problem(public_problem: str) -> None:
    if not public_problem.strip():
        raise ValueError("public_problem must not be empty")


def _require_proposal_slot(proposal_slot: int) -> None:
    if proposal_slot < 0:
        raise ValueError("proposal_slot must be nonnegative")


def _write_intent_checkpoint(
    path: Path,
    input_hash: str,
    result: LLMCallResult[ProposedConstructionIntent],
) -> None:
    _write_checkpoint_payload(
        path,
        {
            "schema_version": "incremental-intent-checkpoint-1",
            "input_sha256": input_hash,
            "request_hash": result.request_hash,
            "intent": result.parsed.model_dump(mode="json"),
        },
    )


def _load_intent_checkpoint(
    path: Path,
) -> tuple[ProposedConstructionIntent, str] | None:
    payload = _load_checkpoint_payload(
        path, expected_schema="incremental-intent-checkpoint-1"
    )
    if payload is None:
        return None
    request_hash = _required_sha256(payload, "request_hash", path)
    intent = ProposedConstructionIntent.model_validate(payload.get("intent"))
    return intent, request_hash


def _write_focus_checkpoint(
    path: Path,
    input_hash: str,
    result: LLMCallResult[ProposedConstructionFocus],
) -> None:
    _write_checkpoint_payload(
        path,
        {
            "schema_version": "incremental-focus-checkpoint-1",
            "input_sha256": input_hash,
            "request_hash": result.request_hash,
            "focus": result.parsed.model_dump(mode="json"),
        },
    )


def _load_focus_checkpoint(
    path: Path,
) -> tuple[ProposedConstructionFocus, str] | None:
    payload = _load_checkpoint_payload(
        path, expected_schema="incremental-focus-checkpoint-1"
    )
    if payload is None:
        return None
    request_hash = _required_sha256(payload, "request_hash", path)
    focus = ProposedConstructionFocus.model_validate(payload.get("focus"))
    return focus, request_hash


def _write_result_checkpoint(
    path: Path,
    input_hash: str,
    result: IncrementalTopologyResult | IncrementalFunctionalResult,
) -> None:
    _write_checkpoint_payload(
        path,
        {
            "schema_version": "incremental-action-checkpoint-1",
            "input_sha256": input_hash,
            "result_type": (
                "topology"
                if isinstance(result, IncrementalTopologyResult)
                else "functional"
            ),
            "result": result.model_dump(mode="json"),
        },
    )


def _load_topology_checkpoint(path: Path) -> IncrementalTopologyResult | None:
    payload = _load_checkpoint_payload(
        path, expected_schema="incremental-action-checkpoint-1"
    )
    if payload is None:
        return None
    if payload.get("result_type") != "topology":
        raise ValueError(f"incremental checkpoint type mismatch: {path}")
    return IncrementalTopologyResult.model_validate(payload.get("result"))


def _load_functional_checkpoint(path: Path) -> IncrementalFunctionalResult | None:
    payload = _load_checkpoint_payload(
        path, expected_schema="incremental-action-checkpoint-1"
    )
    if payload is None:
        return None
    if payload.get("result_type") != "functional":
        raise ValueError(f"incremental checkpoint type mismatch: {path}")
    return IncrementalFunctionalResult.model_validate(payload.get("result"))


def _write_checkpoint_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, sort_keys=True, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint_payload(
    path: Path,
    *,
    expected_schema: str,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"incremental checkpoint is not an object: {path}")
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported incremental checkpoint: {path}")
    expected_input = path.stem.rsplit("-", 1)[-1]
    if payload.get("input_sha256") != expected_input:
        raise ValueError(f"incremental checkpoint input hash mismatch: {path}")
    return payload


def _required_sha256(
    payload: dict[str, object],
    key: str,
    path: Path,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"incremental checkpoint {key} is invalid: {path}")
    return value


def _load_transpositions(path: Path) -> ConstructionTranspositionTable:
    if not path.exists():
        return ConstructionTranspositionTable()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "construction-transposition-checkpoint-1"
    ):
        raise ValueError(f"unsupported transposition checkpoint: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"invalid transposition entries: {path}")
    normalized: list[tuple[Literal["topology", "functional"], str]] = []
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or entry[0] not in {"topology", "functional"}
            or not isinstance(entry[1], str)
        ):
            raise ValueError(f"invalid transposition entry: {path}")
        normalized.append((entry[0], entry[1]))
    return ConstructionTranspositionTable(normalized)


def _write_transpositions(
    path: Path,
    table: ConstructionTranspositionTable,
) -> None:
    _write_checkpoint_payload(
        path,
        {
            "schema_version": "construction-transposition-checkpoint-1",
            "input_sha256": _sha256({"kind": "transposition-table"}),
            "entries": [list(item) for item in table.snapshot()],
        },
    )


def _call_receipt(result: LLMCallResult) -> StageCallReceipt:
    usage = result.usage
    return StageCallReceipt(
        request_hash=result.request_hash,
        checkpoint_hit=False,
        provider_cache_hit=result.cache_hit,
        logical_calls=result.logical_calls,
        provider_attempts=result.provider_attempts,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        latency_ms=result.latency_ms,
    )


def _checkpoint_receipt(request_hash: str) -> StageCallReceipt:
    return StageCallReceipt(
        request_hash=request_hash,
        checkpoint_hit=True,
        provider_cache_hit=False,
        logical_calls=0,
        provider_attempts=0,
    )


def _render(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IncrementalFunctionalResult",
    "IncrementalProposer",
    "IncrementalProposerConfig",
    "IncrementalTopologyResult",
]
