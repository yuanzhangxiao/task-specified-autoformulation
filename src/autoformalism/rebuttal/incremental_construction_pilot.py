"""Public-only pilot for intent-routed incremental candidate construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.construction import (
    finalize_topology_draft,
    functional_draft_sha256,
    topology_draft_sha256,
)
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import IncrementalConstructionLLMClient
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluation,
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import FunctionalDraft, TopologyDraft
from autoformalism.schemas.base import NonEmptyText, StrictSchema
from autoformalism.search import (
    CandidateFeedbackEvidence,
    RoutedProposerFeedback,
    route_proposer_feedback,
)
from autoformalism.search.incremental_proposer import (
    IncrementalFunctionalResult,
    IncrementalProposer,
    IncrementalProposerConfig,
    IncrementalTopologyResult,
)
from autoformalism.staging import (
    StagedCandidateExpansion,
    topology_commitment_sha256,
)
from autoformalism.targets import (
    PublicTargetContract,
    PublicTargetEvaluation,
    evaluate_public_targets,
)

ConstructionAttemptStatus = Literal[
    "applied_incomplete",
    "complete",
    "rejected",
]


INTENT_SYSTEM_PROMPT = """You select one small scientific construction objective.
Use only the supplied public requirements and target channels. Return only the
required structured intent. Do not propose equations, topology, identifiers,
parameters, values, ranges, explanations, or summaries."""

TOPOLOGY_ACTION_SYSTEM_PROMPT = """You edit a runtime-maintained ODE topology.
Return only a small coherent transaction of typed topology actions. Use states
for dynamic memory and processes for instantaneous generated quantities. Add
only dependencies justified by the public task. The runtime owns observability,
mechanism tags, target kinds, parent identity, diffs, units, and validation.
Do not emit equations, parameter declarations, ranges, prose, or a full model."""

FUNCTIONAL_ACTION_SYSTEM_PROMPT = """You assign functions to an immutable ODE
topology. Return only localized typed function or latent-initial actions. Each
expression is an RHS contribution, must use exactly the sources declared for
that interaction, and must omit the outer sign already owned by the topology.
Declare only fitted parameter names and qualitative roles; never give ranges,
scopes, units, parent IDs, topology hashes, prose, or a full model. Use only the
restricted expression grammar exposed by the runtime."""


class IncrementalConstructionPilotConfig(StrictSchema):
    """Bounded topology/function action budget for one public pilot task."""

    schema_version: Literal["incremental-construction-pilot-config-1"] = (
        "incremental-construction-pilot-config-1"
    )
    topology_branch_count: int = Field(default=2, ge=1, le=8)
    function_children_per_topology: int = Field(default=2, ge=1, le=8)
    maximum_topology_action_steps: int = Field(default=4, ge=1, le=16)
    maximum_functional_action_steps: int = Field(default=6, ge=1, le=32)


class ConstructionAttemptRecord(StrictSchema):
    """One accepted or rejected action step retained as negative memory."""

    stage: Literal["topology", "functional_form"]
    topology_slot: int = Field(ge=0)
    function_slot: int | None = Field(default=None, ge=0)
    step_index: int = Field(ge=0)
    proposal_slot: int = Field(ge=0)
    parent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ConstructionAttemptStatus
    result: dict[str, object] | None = None
    failure_class: NonEmptyText | None = None
    error: NonEmptyText | None = None


class ConstructedCandidateArtifact(StrictSchema):
    """One complete candidate before numerical fitting or selection."""

    topology_slot: int = Field(ge=0)
    function_slot: int = Field(ge=0)
    topology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    functional_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology_draft: TopologyDraft
    functional_draft: FunctionalDraft
    expansion: StagedCandidateExpansion
    public_target_evaluation: PublicTargetEvaluation
    public_mechanism_evaluation: MechanismEvaluation


class IncrementalConstructionPilotResult(StrictSchema):
    """Complete public-only action ledger and constructed-candidate set."""

    schema_version: Literal["incremental-construction-pilot-result-1"] = (
        "incremental-construction-pilot-result-1"
    )
    status: Literal["complete", "incomplete"]
    test_data_opened: Literal[False] = False
    private_reference_opened: Literal[False] = False
    topology_branch_count_requested: int = Field(ge=1)
    complete_topology_count: int = Field(ge=0)
    complete_candidate_count: int = Field(ge=0)
    attempts: tuple[ConstructionAttemptRecord, ...]
    candidates: tuple[ConstructedCandidateArtifact, ...]


def build_public_construction_problem(
    *,
    public_prompt: str,
    context: ValidationContext,
    target_contract: PublicTargetContract,
    mechanism_spec: MechanismEvaluationSpec,
) -> str:
    """Render the runtime-owned Level-0 summary supplied to every LLM stage."""
    if not public_prompt.strip():
        raise ValueError("public prompt must not be empty")
    payload = {
        "schema_version": "public-construction-problem-1",
        "benchmark_prompt": public_prompt,
        "public_channels": {
            "targets": list(context.targets),
            "auxiliaries": list(context.auxiliaries),
            "forcing_channels": sorted(context.forcing_channels),
            "time_symbol": context.time_symbol,
        },
        "target_requirements": [
            item.model_dump(mode="json") for item in target_contract.targets
        ],
        "required_mechanisms": [
            item.model_dump(mode="json")
            for item in mechanism_spec.required_mechanisms
        ],
        "runtime_rules": {
            "target_mappings_cover_each_public_target_exactly_once": True,
            "generated_states_are_observed_only_by_direct_identity_mapping": True,
            "all_other_generated_states_are_latent": True,
            "test_data_available": False,
            "private_reference_available": False,
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def construct_public_candidates(
    *,
    client: IncrementalConstructionLLMClient,
    proposer_config: IncrementalProposerConfig,
    pilot_config: IncrementalConstructionPilotConfig,
    public_problem: str,
    context: ValidationContext,
    target_contract: PublicTargetContract,
    mechanism_spec: MechanismEvaluationSpec,
    output_path: Path | None = None,
) -> IncrementalConstructionPilotResult:
    """Construct a bounded topology/function pool without fitting or test data."""
    if not public_problem.strip():
        raise ValueError("public_problem must not be empty")
    requirement_ids = tuple(item.id for item in mechanism_spec.required_mechanisms)
    proposer = IncrementalProposer(client=client, config=proposer_config)
    attempts: list[ConstructionAttemptRecord] = []
    candidates: list[ConstructedCandidateArtifact] = []
    complete_topologies: dict[str, tuple[int, TopologyDraft]] = {}

    for topology_slot in range(pilot_config.topology_branch_count):
        draft = TopologyDraft()
        feedback = RoutedProposerFeedback()
        topology = None
        for step_index in range(pilot_config.maximum_topology_action_steps):
            proposal_slot = 1000 * topology_slot + step_index
            parent_sha256 = topology_draft_sha256(draft)
            try:
                result = proposer.revise_topology(
                    public_problem=public_problem,
                    context=context,
                    allowed_requirement_ids=requirement_ids,
                    feedback=feedback,
                    parent=draft,
                    proposal_slot=proposal_slot,
                )
            except (
                LLMProviderError,
                LLMResponseError,
                ModelValidationError,
                ValueError,
            ) as exc:
                attempts.append(
                    _failed_attempt(
                        stage="topology",
                        topology_slot=topology_slot,
                        function_slot=None,
                        step_index=step_index,
                        proposal_slot=proposal_slot,
                        parent_sha256=parent_sha256,
                        exc=exc,
                    )
                )
                feedback = _topology_failure_feedback(exc)
                _write_progress(
                    output_path,
                    pilot_config,
                    attempts,
                    candidates,
                    len(complete_topologies),
                )
                continue
            draft = result.application.draft
            try:
                topology = finalize_topology_draft(draft, context)
            except (ModelValidationError, ValueError) as exc:
                attempts.append(
                    _successful_topology_attempt(
                        result,
                        topology_slot=topology_slot,
                        step_index=step_index,
                        proposal_slot=proposal_slot,
                        parent_sha256=parent_sha256,
                        status="applied_incomplete",
                        completion_error=exc,
                    )
                )
                feedback = _topology_failure_feedback(exc)
                _write_progress(
                    output_path,
                    pilot_config,
                    attempts,
                    candidates,
                    len(complete_topologies),
                )
                continue
            attempts.append(
                _successful_topology_attempt(
                    result,
                    topology_slot=topology_slot,
                    step_index=step_index,
                    proposal_slot=proposal_slot,
                    parent_sha256=parent_sha256,
                    status="complete",
                )
            )
            break
        if topology is None:
            continue
        topology_sha256 = topology_draft_sha256(draft)
        if topology_sha256 in complete_topologies:
            continue
        complete_topologies[topology_sha256] = (topology_slot, draft)

        for function_slot in range(
            pilot_config.function_children_per_topology
        ):
            functional = FunctionalDraft(
                topology_commitment_sha256=(
                    topology_commitment_sha256(topology)
                )
            )
            feedback = RoutedProposerFeedback()
            complete: IncrementalFunctionalResult | None = None
            for step_index in range(
                pilot_config.maximum_functional_action_steps
            ):
                proposal_slot = (
                    100_000 * topology_slot
                    + 1000 * function_slot
                    + step_index
                )
                parent_sha256 = functional_draft_sha256(functional)
                try:
                    result = proposer.revise_functions(
                        public_problem=public_problem,
                        context=context,
                        allowed_requirement_ids=requirement_ids,
                        feedback=feedback,
                        topology=topology,
                        parent=functional,
                        proposal_slot=proposal_slot,
                    )
                except (
                    LLMProviderError,
                    LLMResponseError,
                    ModelValidationError,
                    ValueError,
                ) as exc:
                    attempts.append(
                        _failed_attempt(
                            stage="functional_form",
                            topology_slot=topology_slot,
                            function_slot=function_slot,
                            step_index=step_index,
                            proposal_slot=proposal_slot,
                            parent_sha256=parent_sha256,
                            exc=exc,
                        )
                    )
                    feedback = _functional_failure_feedback(exc)
                    _write_progress(
                        output_path,
                        pilot_config,
                        attempts,
                        candidates,
                        len(complete_topologies),
                    )
                    continue
                functional = result.application.draft
                attempts.append(
                    _successful_functional_attempt(
                        result,
                        topology_slot=topology_slot,
                        function_slot=function_slot,
                        step_index=step_index,
                        proposal_slot=proposal_slot,
                        parent_sha256=parent_sha256,
                    )
                )
                if result.expansion is not None:
                    complete = result
                    break
                feedback = _incomplete_function_feedback(result)
            if complete is None or complete.expansion is None:
                continue
            candidate = complete.expansion.candidate
            candidates.append(
                ConstructedCandidateArtifact(
                    topology_slot=topology_slot,
                    function_slot=function_slot,
                    topology_sha256=topology_sha256,
                    functional_sha256=functional_draft_sha256(functional),
                    topology_draft=draft,
                    functional_draft=functional,
                    expansion=complete.expansion,
                    public_target_evaluation=evaluate_public_targets(
                        candidate, target_contract
                    ),
                    public_mechanism_evaluation=evaluate_mechanisms(
                        candidate, mechanism_spec
                    ),
                )
            )
            _write_progress(
                output_path,
                pilot_config,
                attempts,
                candidates,
                len(complete_topologies),
            )

    result = _result(
        pilot_config,
        attempts,
        candidates,
        len(complete_topologies),
    )
    if output_path is not None:
        _atomic_write_json(output_path, result.model_dump(mode="json"))
    return result


def _successful_topology_attempt(
    result: IncrementalTopologyResult,
    *,
    topology_slot: int,
    step_index: int,
    proposal_slot: int,
    parent_sha256: str,
    status: Literal["applied_incomplete", "complete"],
    completion_error: Exception | None = None,
) -> ConstructionAttemptRecord:
    return ConstructionAttemptRecord(
        stage="topology",
        topology_slot=topology_slot,
        step_index=step_index,
        proposal_slot=proposal_slot,
        parent_sha256=parent_sha256,
        status=status,
        result=result.model_dump(mode="json"),
        failure_class=(
            "deterministic_topology_completeness"
            if completion_error is not None
            else None
        ),
        error=(
            f"{type(completion_error).__name__}: "
            f"{str(completion_error)[:2000]}"
            if completion_error is not None
            else None
        ),
    )


def _successful_functional_attempt(
    result: IncrementalFunctionalResult,
    *,
    topology_slot: int,
    function_slot: int,
    step_index: int,
    proposal_slot: int,
    parent_sha256: str,
) -> ConstructionAttemptRecord:
    return ConstructionAttemptRecord(
        stage="functional_form",
        topology_slot=topology_slot,
        function_slot=function_slot,
        step_index=step_index,
        proposal_slot=proposal_slot,
        parent_sha256=parent_sha256,
        status=(
            "complete" if result.expansion is not None else "applied_incomplete"
        ),
        result=result.model_dump(mode="json"),
    )


def _failed_attempt(
    *,
    stage: Literal["topology", "functional_form"],
    topology_slot: int,
    function_slot: int | None,
    step_index: int,
    proposal_slot: int,
    parent_sha256: str,
    exc: Exception,
) -> ConstructionAttemptRecord:
    return ConstructionAttemptRecord(
        stage=stage,
        topology_slot=topology_slot,
        function_slot=function_slot,
        step_index=step_index,
        proposal_slot=proposal_slot,
        parent_sha256=parent_sha256,
        status="rejected",
        failure_class=_failure_class(exc),
        error=f"{type(exc).__name__}: {str(exc)[:2000]}",
    )


def _failure_class(exc: Exception) -> str:
    if isinstance(exc, LLMProviderError):
        return "provider_transport"
    if isinstance(exc, LLMResponseError):
        return "provider_schema_or_post_schema_contract"
    if isinstance(exc, ModelValidationError):
        codes = {item.code for item in exc.diagnostics}
        if codes & {"SYNTAX_ERROR", "UNKNOWN_FUNCTION", "UNKNOWN_SYMBOL"}:
            return "restricted_expression_grammar"
        return "deterministic_model_contract"
    return "deterministic_action_contract"


def _topology_failure_feedback(exc: Exception) -> RoutedProposerFeedback:
    return route_proposer_feedback(
        CandidateFeedbackEvidence(
            graph_mechanism_failures=(
                f"Previous topology action failed: {str(exc)[:1200]}",
            )
        )
    )


def _functional_failure_feedback(exc: Exception) -> RoutedProposerFeedback:
    return route_proposer_feedback(
        CandidateFeedbackEvidence(
            deterministic_validation_failures=(
                f"Previous functional action failed: {str(exc)[:1200]}",
            )
        )
    )


def _incomplete_function_feedback(
    result: IncrementalFunctionalResult,
) -> RoutedProposerFeedback:
    compatibility = result.compatibility
    messages = []
    if compatibility.missing_interaction_ids:
        messages.append(
            "Assign functions for remaining interactions: "
            f"{list(compatibility.missing_interaction_ids)}"
        )
    if compatibility.missing_latent_initial_states:
        messages.append(
            "Initialize remaining latent states: "
            f"{list(compatibility.missing_latent_initial_states)}"
        )
    return route_proposer_feedback(
        CandidateFeedbackEvidence(
            deterministic_validation_failures=tuple(messages)
        )
    )


def _write_progress(
    output_path: Path | None,
    config: IncrementalConstructionPilotConfig,
    attempts: list[ConstructionAttemptRecord],
    candidates: list[ConstructedCandidateArtifact],
    complete_topology_count: int,
) -> None:
    if output_path is None:
        return
    _atomic_write_json(
        output_path,
        _result(
            config,
            attempts,
            candidates,
            complete_topology_count,
        ).model_dump(mode="json"),
    )


def _result(
    config: IncrementalConstructionPilotConfig,
    attempts: list[ConstructionAttemptRecord],
    candidates: list[ConstructedCandidateArtifact],
    complete_topology_count: int,
) -> IncrementalConstructionPilotResult:
    expected = (
        config.topology_branch_count * config.function_children_per_topology
    )
    return IncrementalConstructionPilotResult(
        status="complete" if len(candidates) == expected else "incomplete",
        topology_branch_count_requested=config.topology_branch_count,
        complete_topology_count=complete_topology_count,
        complete_candidate_count=len(candidates),
        attempts=tuple(attempts),
        candidates=tuple(candidates),
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "FUNCTIONAL_ACTION_SYSTEM_PROMPT",
    "INTENT_SYSTEM_PROMPT",
    "TOPOLOGY_ACTION_SYSTEM_PROMPT",
    "ConstructedCandidateArtifact",
    "ConstructionAttemptRecord",
    "IncrementalConstructionPilotConfig",
    "IncrementalConstructionPilotResult",
    "build_public_construction_problem",
    "construct_public_candidates",
]
