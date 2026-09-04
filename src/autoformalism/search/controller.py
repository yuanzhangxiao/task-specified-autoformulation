"""Judge-integrated, checkpointed iterative candidate search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.expressions import (
    ModelValidationError,
    compile_candidate,
    repair_protected_declarations,
    validate_fixed_latent_basis_parameterization,
    validate_gmm_parameterization,
    validate_profiled_latent_basis_parameterization,
)
from autoformalism.fitting import (
    EvaluationMetrics,
    FitConfig,
    FitResult,
    InitializationDiagnostic,
    OptimizationDiagnostic,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.llm import LLMClient
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import StagedLLMClient
from autoformalism.pruning import prune_candidate
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import (
    CandidateModel,
    JudgeAssessment,
    ScientificJudgeResult,
    TopologyCandidate,
    parse_judge_assessment,
)
from autoformalism.search.checkpoints import CheckpointStore
from autoformalism.search.feedback_routing import (
    CandidateFeedbackEvidence,
    FeedbackRoute,
    RoutedProposerFeedback,
    TargetValidationMetric,
    evidence_from_completed_candidate,
    route_proposer_feedback,
)
from autoformalism.search.hybrid_pair import (
    HybridPairJudgment,
    PairwiseScientificJudge,
)
from autoformalism.search.identity import CandidateIdentity, candidate_identity
from autoformalism.search.models import (
    CandidateRecord,
    FinalEvaluation,
    FrozenSelection,
    IncumbentChallenge,
    SearchConfig,
)
from autoformalism.targets import (
    PublicTargetContract,
    evaluate_public_targets,
)

if TYPE_CHECKING:
    from autoformalism.search.staged_proposer import StagedProposer

StageCallback = Callable[[str, int | None], None]

_SCIENTIFIC_JUDGE_CATEGORIES = (
    "mechanistic_coherence",
    "source_sink_balance_semantics",
    "dynamic_plausibility",
    "mechanism_coupling_task_sufficiency",
    "nonredundancy_accounting",
    "latent_state_complexity_justification",
)

_DETERMINISTIC_CERTIFICATIONS = (
    "response schema is valid",
    "every state has exactly one governing equation",
    "every expression symbol is declared or supplied",
    "target mappings exist",
    "algebraic definitions are acyclic",
    "only causally available public channels are used",
    "parameter declarations are valid",
    "restricted expressions are executable",
)


class SearchController:
    """Run one exploratory proposal per round with validation-only selection."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        context: Any,
        training: DatasetSplit,
        validation: DatasetSplit,
        test_loader: Callable[[FrozenSelection], DatasetSplit],
        config: SearchConfig,
        pairwise_judge: PairwiseScientificJudge | None = None,
        public_target_contract: PublicTargetContract | None = None,
        public_mechanism_spec: MechanismEvaluationSpec | None = None,
        stage_callback: StageCallback | None = None,
    ) -> None:
        if training.name is not SplitName.TRAIN:
            raise ValueError("controller training split must be train")
        if validation.name is not SplitName.VALIDATION:
            raise ValueError("controller validation split must be val")
        self._client = llm_client
        self._context = context
        self._training = training
        self._validation = validation
        self._test_loader = test_loader
        self._config = config
        self._pairwise_judge = pairwise_judge
        self._public_target_contract = public_target_contract
        self._public_mechanism_spec = public_mechanism_spec
        if public_target_contract is not None:
            contract_targets = {
                item.target_channel for item in public_target_contract.targets
            }
            if contract_targets != set(context.targets):
                raise ValueError(
                    "public target contract channels do not match validation context"
                )
            public_symbols = set(context.targets) | set(context.forcing_channels)
            unknown_dependencies = {
                symbol
                for target in public_target_contract.targets
                for dependency in target.required_dependencies
                for symbol in dependency.acceptable_symbols
                if symbol not in public_symbols
            }
            if unknown_dependencies:
                raise ValueError(
                    "public target contract references unavailable channels: "
                    f"{sorted(unknown_dependencies)}"
                )
        if (
            config.selection_policy == "incumbent_relative_hybrid"
            and pairwise_judge is None
        ):
            raise ValueError(
                "incumbent_relative_hybrid requires a pairwise scientific judge"
            )
        self._callback = stage_callback or (lambda _stage, _round: None)
        self._store = CheckpointStore(
            config.checkpoint_directory, self._run_fingerprint()
        )
        self._staged_proposer: StagedProposer | None = None
        if config.proposer_construction_mode == "staged_v2":
            from autoformalism.search.staged_proposer import (
                StagedProposer,
                StagedProposerConfig,
            )

            for method in ("propose_topology", "propose_functions"):
                if not callable(getattr(llm_client, method, None)):
                    raise TypeError(
                        "staged_v2 requires an LLM client with "
                        f"{method}()"
                    )
            assert config.staged_topology_system_prompt is not None
            assert config.staged_functional_system_prompt is not None
            self._staged_proposer = StagedProposer(
                client=cast(StagedLLMClient, llm_client),
                config=StagedProposerConfig(
                    checkpoint_directory=(
                        config.checkpoint_directory / "staged-provider"
                    ),
                    run_fingerprint=self._store.fingerprint,
                    topology_system_prompt=(
                        config.staged_topology_system_prompt
                    ),
                    functional_system_prompt=(
                        config.staged_functional_system_prompt
                    ),
                ),
            )

    def run(self) -> FinalEvaluation:
        """Run search, freeze validation selection, and optionally test once."""
        completed = self._completed_records()
        start_round = self._completed_round_count()
        if self._config.selection_policy == "incumbent_relative_hybrid":
            incumbent = _hybrid_incumbent(completed)
            best_seen = (
                float("inf")
                if incumbent is None
                else incumbent[0].pruned_fit.validation_metrics.normalized_mse
            )
        else:
            best_seen = min(
                (
                    item.pruned_fit.validation_metrics.normalized_mse
                    for item in completed
                ),
                default=float("inf"),
            )
        stagnant = _trailing_stagnation(completed, self._config)
        stopping_reason = "iteration_budget"

        if best_seen <= self._config.validation_mse_target:
            stopping_reason = "validation_target"
            return self._finalize(completed, stopping_reason)
        if stagnant >= self._config.stagnation_iterations:
            stopping_reason = "stagnation"
            return self._finalize(completed, stopping_reason)

        for round_index in range(start_round, self._config.maximum_iterations):
            beam = _beam(completed, self._config.beam_size, self._config)
            previous_selection = beam[0].structural_hash if beam else None
            record = self._run_round(round_index, beam, completed)
            if record is not None:
                completed.append(record)
                current_selection = _beam(completed, 1, self._config)[
                    0
                ].structural_hash
                selected_record = _beam(completed, 1, self._config)[0]
                current = selected_record.pruned_fit.validation_metrics.normalized_mse
                if current < best_seen - 1e-12:
                    best_seen = current
                if current_selection != previous_selection:
                    stagnant = 0
                else:
                    stagnant += 1
            if best_seen <= self._config.validation_mse_target:
                stopping_reason = "validation_target"
                break
            if stagnant >= self._config.stagnation_iterations:
                stopping_reason = "stagnation"
                break
        if not completed:
            raise RuntimeError("search produced no valid fitted candidates")
        return self._finalize(completed, stopping_reason)

    def _run_round(
        self,
        round_index: int,
        beam: Sequence[CandidateRecord],
        completed: Sequence[CandidateRecord],
    ) -> CandidateRecord | None:
        payload = self._store.load_round(round_index) or {
            "round_index": round_index,
            "stage": "new",
            "valid": True,
        }
        stage = payload["stage"]
        if stage == "complete":
            return _record_from_dict(payload["record"]) if payload["valid"] else None

        if stage == "new":
            try:
                if self._config.proposer_construction_mode == "staged_v2":
                    proposal, construction = self._propose_staged_candidate(
                        round_index,
                        beam,
                    )
                    payload["staged_proposal"] = construction
                else:
                    feedback = self._proposer_feedback(beam, round_index)
                    proposal_mode = _proposal_mode(
                        self._config, round_index, beam
                    )
                    request_payload: dict[str, object] = {
                        "round": round_index,
                        "proposal_mode": proposal_mode,
                        "beam_feedback": feedback,
                    }
                    if self._config.proposer_feedback_mode == "rich_v1":
                        request_payload["feedback_schema_version"] = (
                            "proposer-feedback-rich-1"
                        )
                        request_payload["failed_structure_memory"] = (
                            self._failed_structure_memory(round_index)
                        )
                    if proposal_mode == "incumbent_refinement":
                        request_payload["refinement_contract"] = (
                            _refinement_contract(beam[0])
                        )
                    proposal_arguments = {
                        "system_prompt": self._config.proposer_system_prompt,
                        "user_prompt": json.dumps(
                            request_payload, sort_keys=True
                        ),
                    }
                    if (
                        round_index == 0
                        and self._config.require_initial_proposer_cache_hit
                    ):
                        proposal = self._client.propose(
                            **proposal_arguments,
                            cache_only=True,
                        ).parsed
                    else:
                        proposal = self._client.propose(
                            **proposal_arguments
                        ).parsed
            except (LLMProviderError, LLMResponseError, ValueError) as exc:
                payload.update(
                    valid=False,
                    error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                    failure_class=(
                        "proposal_transport"
                        if isinstance(exc, (LLMProviderError, LLMResponseError))
                        else "staged_proposal_contract"
                    ),
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            raw_proposal = proposal
            proposal, repairs = repair_protected_declarations(
                raw_proposal, self._context
            )
            repair_messages = list(repairs)
            if (
                self._config.proposal_policy == "incumbent_refinement_v1"
                and beam
            ):
                expected_parent = beam[0].pruned_candidate.candidate_id
                if proposal.parent_candidate_id != expected_parent:
                    proposal = proposal.model_copy(
                        update={"parent_candidate_id": expected_parent}
                    )
                    repair_messages.append(
                        "bound refinement lineage to active incumbent: "
                        f"{expected_parent}"
                    )
            if repair_messages:
                payload["raw_candidate"] = raw_proposal.model_dump(mode="json")
                payload["deterministic_repairs"] = repair_messages
            payload["candidate"] = proposal.model_dump(mode="json")
            stage = self._save_stage(round_index, payload, "proposed")
        candidate = CandidateModel.model_validate(payload["candidate"])

        if stage == "proposed":
            permitted_parents = {
                identifier
                for item in beam
                for identifier in (
                    item.candidate.candidate_id,
                    item.pruned_candidate.candidate_id,
                )
            }
            permitted_parents.update(
                self._rejected_candidate_ids(round_index)
            )
            if (
                candidate.parent_candidate_id is not None
                and candidate.parent_candidate_id not in permitted_parents
            ):
                payload.update(
                    valid=False,
                    error="candidate parent is not present in the active lineage",
                    failure_class="lineage_contract",
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            try:
                compiled = compile_candidate(candidate, self._context)
                if (
                    self._config.fit_config.parameter_fit_strategy
                    == "exact_derivative_linear_ridge"
                ):
                    validate_gmm_parameterization(compiled.validated)
                elif (
                    self._config.fit_config.parameter_fit_strategy
                    == "fixed_latent_basis_linear_ridge"
                ):
                    validate_fixed_latent_basis_parameterization(
                        compiled.validated
                    )
                elif (
                    self._config.fit_config.parameter_fit_strategy
                    == "profiled_latent_basis_linear_ridge"
                ):
                    validate_profiled_latent_basis_parameterization(
                        compiled.validated
                    )
            except ModelValidationError as exc:
                payload.update(
                    valid=False,
                    error=str(exc),
                    failure_class="deterministic_contract",
                    deterministic_validation_diagnostics=[
                        {
                            "code": item.code,
                            "location": item.location,
                            "message": item.message,
                        }
                        for item in exc.diagnostics
                    ],
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            identity = candidate_identity(candidate)
            payload["candidate_identity"] = identity.model_dump(mode="json")
            structural_hash = identity.functional_sha256
            payload["structural_hash"] = structural_hash
            prior_failure = self._prior_failed_structure(
                round_index,
                identity,
            )
            if prior_failure is not None:
                payload.update(
                    valid=False,
                    error="previously failed structural duplicate",
                    failure_class="duplicate",
                    prior_structural_failure=prior_failure,
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            if self._public_target_contract is not None:
                target_evaluation = evaluate_public_targets(
                    candidate, self._public_target_contract
                )
                payload["public_target_evaluation"] = target_evaluation.model_dump(
                    mode="json"
                )
                if self._public_mechanism_spec is not None:
                    payload["public_mechanism_evaluation"] = evaluate_mechanisms(
                        candidate, self._public_mechanism_spec
                    ).model_dump(mode="json")
                if not target_evaluation.passed:
                    failed = [
                        item.predicate
                        for item in target_evaluation.predicates
                        if item.status == "failed"
                    ]
                    payload.update(
                        valid=False,
                        error=(
                            "PUBLIC_TARGET_CONTRACT_FAILED: "
                            + ", ".join(failed)
                        ),
                        failure_class="public_contract",
                        stage="complete",
                    )
                    self._store.save_round(round_index, payload)
                    self._callback("complete", round_index)
                    return None
            elif self._public_mechanism_spec is not None:
                payload["public_mechanism_evaluation"] = evaluate_mechanisms(
                    candidate, self._public_mechanism_spec
                ).model_dump(mode="json")
            if compiled.validated.warnings:
                payload["validation_warnings"] = [
                    {
                        "code": item.code,
                        "location": item.location,
                        "message": item.message,
                    }
                    for item in compiled.validated.warnings
                ]
            if any(
                structural_hash
                in {
                    _structural_hash(item.candidate),
                    _structural_hash(item.pruned_candidate),
                }
                for item in completed
            ):
                payload.update(
                    valid=False,
                    error="structural duplicate",
                    failure_class="duplicate",
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            stage = self._save_stage(round_index, payload, "validated")
        compiled = compile_candidate(candidate, self._context)

        if stage == "validated" and self._config.cheap_prefit_judge:
            prefit = self._judge(candidate, "pre_fit")
            payload["prefit_judge"] = prefit.model_dump(mode="json")
            stage = self._save_stage(round_index, payload, "prefit_judged")
        if stage == "validated":
            stage = "prefit_judged"

        if stage == "prefit_judged":
            fitted, fit_attempts = _fit_with_retry(
                compiled,
                self._training,
                self._validation,
                primary_config=self._config.fit_config,
                retry_config=self._config.fit_retry_config,
            )
            payload["fit_attempts"] = fit_attempts
            if not fitted.success:
                payload.update(
                    valid=False,
                    error="numerical fit failed",
                    failure_class="numerical_fit",
                    fit=_fit_to_dict(fitted),
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            payload["fit"] = _fit_to_dict(fitted)
            stage = self._save_stage(round_index, payload, "fitted")
        fitted = _fit_from_dict(payload["fit"])

        if stage == "fitted":
            postfit = self._judge(candidate, "post_fit")
            payload["postfit_judge"] = postfit.model_dump(mode="json")
            stage = self._save_stage(round_index, payload, "postfit_judged")
        postfit = parse_judge_assessment(payload["postfit_judge"])

        if stage == "postfit_judged":
            if self._config.apply_postfit_pruning:
                pruning = prune_candidate(
                    compiled,
                    self._training,
                    self._validation,
                    fit_config=self._config.fit_config,
                    pruning_config=self._config.pruning_config,
                    unpruned_fit=fitted,
                )
                selected_candidate = pruning.selected_candidate
                selected_fit = pruning.selected_fit
                removed_terms = pruning.selected_removed_terms
                removed_parameters = pruning.selected_removed_parameters
                contributions = dict(pruning.contribution_by_term)
                persistence_training_mse = pruning.persistence_training_mse
                persistence_validation_mse = pruning.persistence_validation_mse
                rejected_supports = sum(
                    not item.accepted for item in pruning.candidates
                )
            else:
                selected_candidate = candidate
                selected_fit = fitted
                removed_terms = ()
                removed_parameters = ()
                contributions = {}
                persistence_training_mse = None
                persistence_validation_mse = None
                rejected_supports = 0
            if self._public_target_contract is not None:
                pruned_target_evaluation = evaluate_public_targets(
                    selected_candidate, self._public_target_contract
                )
                payload["pruned_public_target_evaluation"] = (
                    pruned_target_evaluation.model_dump(mode="json")
                )
                if not pruned_target_evaluation.passed:
                    selected_candidate = candidate
                    selected_fit = fitted
                    removed_terms = ()
                    removed_parameters = ()
                    payload["pruning_target_contract_fallback"] = True
            payload["pruned_candidate"] = selected_candidate.model_dump(mode="json")
            payload["pruned_candidate_identity"] = candidate_identity(
                selected_candidate
            ).model_dump(mode="json")
            if self._public_mechanism_spec is not None:
                payload["pruned_public_mechanism_evaluation"] = (
                    evaluate_mechanisms(
                        selected_candidate, self._public_mechanism_spec
                    ).model_dump(mode="json")
                )
            payload["pruned_fit"] = _fit_to_dict(selected_fit)
            payload["pruning"] = {
                "applied": self._config.apply_postfit_pruning,
                "removed_terms": list(removed_terms),
                "removed_parameters": list(removed_parameters),
                "contributions": contributions,
                "persistence_training_mse": persistence_training_mse,
                "persistence_validation_mse": persistence_validation_mse,
                "rejected_supports": rejected_supports,
            }
            stage = self._save_stage(round_index, payload, "pruned")
        pruned_candidate = CandidateModel.model_validate(payload["pruned_candidate"])
        pruned_fit = _fit_from_dict(payload["pruned_fit"])

        if stage == "pruned":
            if self._config.apply_postfit_pruning:
                postpruning = self._judge(pruned_candidate, "post_pruning")
                payload["postpruning_judge"] = postpruning.model_dump(
                    mode="json"
                )
            else:
                payload["postpruning_judge"] = payload["postfit_judge"]
                payload["postpruning_judge_reused"] = True
            stage = self._save_stage(round_index, payload, "postpruning_judged")
        postpruning = parse_judge_assessment(payload["postpruning_judge"])

        if stage == "postpruning_judged":
            challenge = None
            if self._config.selection_policy == "incumbent_relative_hybrid":
                incumbent_state = _hybrid_incumbent(completed)
                if incumbent_state is not None:
                    incumbent, path_score = incumbent_state
                    assert self._pairwise_judge is not None
                    judgment = self._pairwise_judge.compare(
                        incumbent.pruned_candidate,
                        pruned_candidate,
                    )
                    challenge = _incumbent_challenge(
                        incumbent=incumbent,
                        challenger_hash=_structural_hash(pruned_candidate),
                        challenger_validation_mse=(
                            pruned_fit.validation_metrics.normalized_mse
                        ),
                        incumbent_path_score=path_score,
                        judgment=judgment,
                        science_weight=self._config.hybrid_science_weight,
                    )
                    payload["incumbent_challenge"] = challenge.model_dump(
                        mode="json"
                    )
                stage = self._save_stage(
                    round_index, payload, "incumbent_compared"
                )
            else:
                stage = "incumbent_compared"

        if stage == "incumbent_compared":
            challenge_payload = payload.get("incumbent_challenge")
            challenge = (
                None
                if challenge_payload is None
                else IncumbentChallenge.model_validate(challenge_payload)
            )
            record = CandidateRecord(
                round_index=round_index,
                candidate=candidate,
                parent_candidate_id=candidate.parent_candidate_id,
                structural_hash=_structural_hash(pruned_candidate),
                fit=fitted,
                postfit_judge=postfit,
                pruned_candidate=pruned_candidate,
                pruned_fit=pruned_fit,
                postpruning_judge=postpruning,
                pruning_removed_terms=tuple(payload["pruning"]["removed_terms"]),
                pruning_removed_parameters=tuple(
                    payload["pruning"]["removed_parameters"]
                ),
                pruning_contributions=dict(payload["pruning"]["contributions"]),
                incumbent_challenge=challenge,
            )
            payload.update(
                valid=True,
                record=_record_to_dict(record),
            )
            self._save_stage(round_index, payload, "complete")
            return record
        raise RuntimeError(f"unsupported checkpoint stage: {stage}")

    def _propose_staged_candidate(
        self,
        round_index: int,
        beam: Sequence[CandidateRecord],
    ) -> tuple[CandidateModel, dict[str, object]]:
        """Construct one candidate through checkpointed graph/function stages."""
        assert self._staged_proposer is not None
        assert self._config.staged_public_problem is not None
        feedback = self._staged_feedback(beam, round_index)
        incumbent_topology, parent_candidate_id = (
            self._staged_incumbent_topology(beam, round_index)
        )
        requires_topology_revision = any(
            item.route is FeedbackRoute.TOPOLOGY for item in feedback.items
        )
        if incumbent_topology is None:
            decision = "initial_topology_and_functions"
            fixed_topology = None
            revisable_topology = None
        elif requires_topology_revision:
            decision = "topology_and_function_revision"
            fixed_topology = None
            revisable_topology = incumbent_topology
        else:
            decision = "function_only_revision"
            fixed_topology = incumbent_topology
            revisable_topology = None
        result = self._staged_proposer.construct(
            public_problem=self._config.staged_public_problem,
            context=self._context,
            feedback=feedback,
            fixed_topology=fixed_topology,
            incumbent_topology=revisable_topology,
            cache_only=(
                True
                if round_index == 0
                and self._config.require_initial_proposer_cache_hit
                else None
            ),
        )
        candidate = result.expansion.candidate
        lineage_repair = None
        if candidate.parent_candidate_id != parent_candidate_id:
            candidate = candidate.model_copy(
                update={"parent_candidate_id": parent_candidate_id}
            )
            lineage_repair = (
                "runtime bound staged candidate lineage to "
                f"{parent_candidate_id or 'root'}"
            )
        construction = result.model_dump(mode="json")
        construction.update(
            {
                "construction_mode": "staged_v2",
                "revision_decision": decision,
                "requires_topology_revision": requires_topology_revision,
                "runtime_parent_candidate_id": parent_candidate_id,
                "lineage_repair": lineage_repair,
            }
        )
        return candidate, construction

    def _staged_feedback(
        self,
        beam: Sequence[CandidateRecord],
        round_index: int,
    ) -> RoutedProposerFeedback:
        """Collect bounded public evidence and route it by revision stage."""
        evidence: list[CandidateFeedbackEvidence] = []
        if beam:
            incumbent = beam[0]
            evidence.append(
                evidence_from_completed_candidate(
                    incumbent.pruned_candidate,
                    incumbent.pruned_fit,
                    (
                        incumbent.postpruning_judge
                        if self._config.use_judge
                        else None
                    ),
                    public_target_contract=self._public_target_contract,
                    public_mechanism_spec=self._public_mechanism_spec,
                )
            )
        rejected = self._latest_rejected_staged_payload(round_index)
        if rejected is not None:
            evidence.append(self._evidence_from_rejected_payload(rejected))
        return route_proposer_feedback(_merge_feedback_evidence(evidence))

    def _evidence_from_rejected_payload(
        self,
        payload: Mapping[str, Any],
    ) -> CandidateFeedbackEvidence:
        """Recover typed feedback from one checkpointed rejected candidate."""
        candidate_payload = payload.get("candidate")
        candidate = (
            CandidateModel.model_validate(candidate_payload)
            if isinstance(candidate_payload, Mapping)
            else None
        )
        fit_payload = payload.get("fit")
        if candidate is not None and isinstance(fit_payload, Mapping):
            base = evidence_from_completed_candidate(
                candidate,
                _fit_from_dict(dict(fit_payload)),
                None,
                public_target_contract=self._public_target_contract,
                public_mechanism_spec=self._public_mechanism_spec,
            )
        else:
            base = CandidateFeedbackEvidence()
        structural = (
            _structural_feedback_evidence(
                candidate,
                self._public_target_contract,
                self._public_mechanism_spec,
            )
            if candidate is not None
            else CandidateFeedbackEvidence()
        )
        diagnostics = tuple(
            _diagnostic_message(item)
            for item in payload.get("deterministic_validation_diagnostics", [])
            if isinstance(item, Mapping)
        )
        failure_class = str(payload.get("failure_class", ""))
        error = str(payload.get("error", "candidate was rejected"))
        base = _merge_feedback_evidence((base, structural))
        deterministic = list(base.deterministic_validation_failures)
        fit_failures = list(base.fit_failures)
        if diagnostics:
            deterministic.extend(diagnostics)
        elif failure_class in {
            "deterministic_contract",
            "duplicate",
            "public_contract",
            "staged_proposal_contract",
        }:
            deterministic.append(error)
        elif failure_class == "numerical_fit" and error not in fit_failures:
            fit_failures.append(error)
        return base.model_copy(
            update={
                "deterministic_validation_failures": tuple(
                    dict.fromkeys(deterministic)
                ),
                "fit_failures": tuple(dict.fromkeys(fit_failures)),
            }
        )

    def _staged_incumbent_topology(
        self,
        beam: Sequence[CandidateRecord],
        round_index: int,
    ) -> tuple[TopologyCandidate | None, str | None]:
        """Restore the topology belonging to the active or last failed parent."""
        wanted = beam[0].pruned_candidate.candidate_id if beam else None
        fallback: tuple[TopologyCandidate, str] | None = None
        for previous_round in range(round_index - 1, -1, -1):
            payload = self._store.load_round(previous_round)
            if payload is None or not isinstance(
                payload.get("staged_proposal"), Mapping
            ):
                continue
            candidate_payload = payload.get("candidate")
            topology_payload = payload["staged_proposal"].get("topology")
            if not isinstance(candidate_payload, Mapping) or not isinstance(
                topology_payload, Mapping
            ):
                continue
            candidate_id = str(candidate_payload.get("candidate_id", ""))
            topology = TopologyCandidate.model_validate(topology_payload)
            if wanted is not None and candidate_id == wanted:
                return topology, wanted
            if fallback is None and not payload.get("valid"):
                fallback = (topology, candidate_id)
        if wanted is not None:
            raise RuntimeError(
                "active staged incumbent has no checkpointed topology artifact"
            )
        return fallback if fallback is not None else (None, None)

    def _latest_rejected_staged_payload(
        self,
        round_index: int,
    ) -> dict[str, Any] | None:
        """Return the newest staged candidate rejection, if one exists."""
        for previous_round in range(round_index - 1, -1, -1):
            payload = self._store.load_round(previous_round)
            if (
                payload is not None
                and payload.get("stage") == "complete"
                and not payload.get("valid")
                and isinstance(payload.get("staged_proposal"), Mapping)
            ):
                return payload
        return None

    def _judge(self, candidate: CandidateModel, stage: str) -> JudgeAssessment:
        if (
            not self._config.use_judge
            or self._config.selection_policy == "incumbent_relative_hybrid"
        ):
            return ScientificJudgeResult.model_validate(
                {
                    "hard_red_flags": [],
                    "category_scores": dict.fromkeys(
                        _SCIENTIFIC_JUDGE_CATEGORIES, 0.0
                    ),
                    "missing_requirements": [],
                    "actionable_edits": [],
                }
            )
        try:
            return self._client.judge(
                system_prompt=self._config.judge_system_prompt,
                user_prompt=json.dumps(
                    {
                        "stage": stage,
                        "deterministic_certifications": list(
                            _DETERMINISTIC_CERTIFICATIONS
                        ),
                        "candidate": candidate.model_dump(mode="json"),
                    },
                    sort_keys=True,
                ),
            ).parsed
        except (LLMProviderError, LLMResponseError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:1000]}"
            return ScientificJudgeResult.model_validate(
                {
                    "hard_red_flags": [
                        {
                            "code": "judge_unavailable",
                            "evidence": message,
                        }
                    ],
                    "category_scores": {
                        name: {"score": 0.0, "justification": message}
                        for name in _SCIENTIFIC_JUDGE_CATEGORIES
                    },
                    "missing_requirements": [
                        "Judge feedback was unavailable; deterministic and "
                        "numerical checks remain authoritative."
                    ],
                    "actionable_edits": [],
                }
            )

    def _proposer_feedback(
        self,
        beam: Sequence[CandidateRecord],
        round_index: int,
    ) -> list[dict[str, Any]]:
        feedback: list[dict[str, Any]] = []
        for record in beam:
            fit = record.pruned_fit
            feedback.append(
                {
                    "candidate_id": record.pruned_candidate.candidate_id,
                    "eligible_parent": True,
                    "lineage_parent": record.parent_candidate_id,
                    "equations": {
                        item.state: item.rhs
                        for item in record.pruned_candidate.state_equations
                    },
                    "fitted_parameters": dict(fit.global_parameters),
                    "training_normalized_mse": (
                        fit.training_metrics.normalized_mse
                    ),
                    "validation_normalized_mse": (
                        fit.validation_metrics.normalized_mse
                    ),
                    "judge_category_scores": dict(
                        record.postpruning_judge.numeric_category_scores
                    ),
                    "judge_advisory_red_flags": [
                        item.model_dump(mode="json")
                        for item in record.postpruning_judge.hard_red_flags
                    ],
                    "judge_missing_requirements": list(
                        record.postpruning_judge.missing_requirements
                    ),
                    "judge_actionable_edits": [
                        item.model_dump(mode="json")
                        for item in record.postpruning_judge.actionable_edits
                    ],
                    "incumbent_relative_scientific_comparison": (
                        None
                        if record.incumbent_challenge is None
                        else _incumbent_challenge_feedback(
                            record.incumbent_challenge
                        )
                    ),
                    "numerical_failures": {
                        "training_trajectories": list(
                            fit.training_metrics.failed_trajectories
                        ),
                        "validation_trajectories": list(
                            fit.validation_metrics.failed_trajectories
                        ),
                        "optimizer_integration_failures": sum(
                            item.integration_failures for item in fit.diagnostics
                        ),
                    },
                    "pruning_diagnostics": {
                        "removed_terms": list(record.pruning_removed_terms),
                        "removed_parameters": list(
                            record.pruning_removed_parameters
                        ),
                        "normalized_contributions": record.pruning_contributions,
                    },
                    "structural_novelty_requirement": (
                        "Change at least one dependency, operator, state, or "
                        "algebraic mechanism. Renaming symbols is not novel."
                    ),
                }
            )
            if self._config.proposer_feedback_mode in {"structured", "rich_v1"}:
                feedback[-1]["deterministic_runtime"] = (
                    _deterministic_runtime_feedback(
                        record.pruned_candidate,
                        fit,
                        self._public_target_contract,
                        self._public_mechanism_spec,
                    )
                )
            if self._config.proposer_feedback_mode == "rich_v1":
                feedback[-1]["feedback_schema_version"] = (
                    "proposer-feedback-rich-1"
                )
                feedback[-1]["incumbent_snapshot"] = (
                    _candidate_refinement_snapshot(record.pruned_candidate)
                )
                feedback[-1]["per_target_error"] = {
                    "training_normalized_mse": dict(
                        fit.training_metrics.per_target_normalized_mse
                    ),
                    "validation_normalized_mse": dict(
                        fit.validation_metrics.per_target_normalized_mse
                    ),
                }
                feedback[-1]["soft_constraint_violations"] = {
                    "training": {
                        key: dict(value)
                        for key, value in (
                            fit.training_metrics.soft_constraint_violations.items()
                        )
                    },
                    "validation": {
                        key: dict(value)
                        for key, value in (
                            fit.validation_metrics.soft_constraint_violations.items()
                        )
                    },
                }
            if (
                not self._config.use_judge
                or self._config.selection_policy == "incumbent_relative_hybrid"
            ):
                for key in (
                    "judge_category_scores",
                    "judge_advisory_red_flags",
                    "judge_missing_requirements",
                    "judge_actionable_edits",
                ):
                    feedback[-1].pop(key)
            if not self._config.use_judge:
                feedback[-1].pop("incumbent_relative_scientific_comparison")
        remaining = self._config.beam_size - len(feedback)
        if remaining <= 0 and feedback:
            if self._config.selection_policy == "incumbent_relative_hybrid":
                recent_challenger = self._recent_rejected_hybrid_challenger(
                    beam,
                    round_index,
                )
                if recent_challenger is not None:
                    feedback[0]["recent_rejected_challenger"] = recent_challenger
            for rejected_round in range(round_index - 1, -1, -1):
                payload = self._store.load_round(rejected_round)
                if (
                    payload is None
                    or payload.get("stage") != "complete"
                    or payload.get("valid")
                    or not isinstance(payload.get("candidate"), dict)
                ):
                    continue
                candidate = CandidateModel.model_validate(payload["candidate"])
                error = str(payload.get("error", "candidate was rejected"))
                feedback[0]["recent_rejected_candidate"] = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_identity": payload.get("candidate_identity"),
                    "structural_hash": payload.get("structural_hash"),
                    "failure_class": payload.get("failure_class"),
                    "error": error,
                    "prior_structural_failure": payload.get(
                        "prior_structural_failure"
                    ),
                    "deterministic_validation_diagnostics": list(
                        payload.get("deterministic_validation_diagnostics", [])
                    ),
                    "public_target_evaluation": payload.get(
                        "public_target_evaluation"
                    ),
                    "public_mechanism_evaluation": payload.get(
                        "public_mechanism_evaluation"
                    ),
                    "equations": {
                        item.state: item.rhs
                        for item in candidate.state_equations
                    },
                    "required_edit": (
                        "If this was a structural duplicate, renaming states or "
                        "parameters is insufficient; change the dependency graph, "
                        "operators, state set, or algebraic mechanisms."
                    ),
                }
                if self._config.proposer_feedback_mode == "rich_v1":
                    feedback[0]["recent_rejected_candidate"][
                        "candidate_snapshot"
                    ] = _candidate_refinement_snapshot(candidate)
                if self._config.proposer_feedback_mode == "legacy":
                    feedback[0]["recent_rejected_candidate"].pop(
                        "deterministic_validation_diagnostics"
                    )
                    feedback[0]["recent_rejected_candidate"].pop(
                        "public_target_evaluation"
                    )
                    feedback[0]["recent_rejected_candidate"].pop(
                        "public_mechanism_evaluation"
                    )
                break
        if remaining > 0:
            for rejected_round in range(round_index - 1, -1, -1):
                payload = self._store.load_round(rejected_round)
                if (
                    payload is None
                    or payload.get("stage") != "complete"
                    or payload.get("valid")
                    or not isinstance(payload.get("candidate"), dict)
                ):
                    continue
                candidate = CandidateModel.model_validate(payload["candidate"])
                error = str(payload.get("error", "candidate was rejected"))
                rejected_fit = (
                    _fit_from_dict(payload["fit"])
                    if isinstance(payload.get("fit"), dict)
                    else None
                )
                rejected_fit_valid = bool(
                    rejected_fit is not None and rejected_fit.success
                )
                failure_messages = (
                    sorted(
                        {
                            message
                            for diagnostic in rejected_fit.diagnostics
                            for message in diagnostic.integration_failure_messages
                        }
                    )
                    if rejected_fit is not None
                    else []
                )
                feedback.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_identity": payload.get("candidate_identity"),
                        "structural_hash": payload.get("structural_hash"),
                        "failure_class": payload.get("failure_class"),
                        "eligible_parent": True,
                        "lineage_parent": candidate.parent_candidate_id,
                        "equations": {
                            item.state: item.rhs
                            for item in candidate.state_equations
                        },
                        "fitted_parameters": (
                            dict(rejected_fit.global_parameters)
                            if rejected_fit_valid and rejected_fit is not None
                            else None
                        ),
                        "parameter_estimates_valid": rejected_fit_valid,
                        "training_normalized_mse": (
                            rejected_fit.training_metrics.normalized_mse
                            if rejected_fit_valid and rejected_fit is not None
                            else None
                        ),
                        "validation_normalized_mse": (
                            rejected_fit.validation_metrics.normalized_mse
                            if rejected_fit_valid and rejected_fit is not None
                            else None
                        ),
                        "judge_category_scores": {},
                        "deterministic_runtime": (
                            {
                                "candidate_validation": "failed",
                                "validation_diagnostics": list(
                                    payload.get(
                                        "deterministic_validation_diagnostics",
                                        [],
                                    )
                                ),
                                "public_target_evaluation": payload.get(
                                    "public_target_evaluation"
                                ),
                                "public_mechanism_evaluation": payload.get(
                                    "public_mechanism_evaluation"
                                ),
                            }
                            if rejected_fit is None
                            else _deterministic_runtime_feedback(
                                candidate,
                                rejected_fit,
                                self._public_target_contract,
                                self._public_mechanism_spec,
                            )
                        ),
                        "numerical_failures": {
                            (
                                "deterministic_validation"
                                if rejected_fit is None
                                else "rejection"
                            ): [error],
                            "integration_messages": failure_messages,
                            "optimizer_integration_failures": (
                                sum(
                                    item.integration_failures
                                    for item in rejected_fit.diagnostics
                                )
                                if rejected_fit is not None
                                else 0
                            ),
                        },
                        "pruning_diagnostics": {},
                        "rejected_before_fit": rejected_fit is None,
                        "prior_structural_failure": payload.get(
                            "prior_structural_failure"
                        ),
                        "required_edit": (
                            "Address the recorded failure at its stated identity "
                            "level. Renaming is never a change; numerical-only "
                            "failures may instead revise executable bounds or "
                            "initialization without claiming scientific novelty."
                        ),
                    }
                )
                if self._config.proposer_feedback_mode == "legacy":
                    feedback[-1].pop("deterministic_runtime")
                elif self._config.proposer_feedback_mode == "rich_v1":
                    feedback[-1]["feedback_schema_version"] = (
                        "proposer-feedback-rich-1"
                    )
                    feedback[-1]["candidate_snapshot"] = (
                        _candidate_refinement_snapshot(candidate)
                    )
                remaining -= 1
                if remaining == 0:
                    break
        return feedback

    def _recent_rejected_hybrid_challenger(
        self,
        beam: Sequence[CandidateRecord],
        round_index: int,
    ) -> dict[str, object] | None:
        """Expose the latest losing valid challenge without making it a parent."""
        if not beam:
            return None
        incumbent_hash = beam[0].structural_hash
        for prior_round in range(round_index - 1, -1, -1):
            payload = self._store.load_round(prior_round)
            if (
                payload is None
                or payload.get("stage") != "complete"
                or not payload.get("valid")
                or not isinstance(payload.get("record"), dict)
            ):
                continue
            record = _record_from_dict(payload["record"])
            challenge = record.incumbent_challenge
            if challenge is None:
                continue
            if (
                challenge.selected_hash == incumbent_hash
                and record.structural_hash != incumbent_hash
            ):
                feedback: dict[str, object] = {
                    "candidate_id": record.pruned_candidate.candidate_id,
                    "eligible_parent": False,
                    "equations": {
                        item.state: item.rhs
                        for item in record.pruned_candidate.state_equations
                    },
                    "comparison": _incumbent_challenge_feedback(challenge),
                    "instruction": (
                        "Use the scientific and fit evidence to improve the "
                        "eligible incumbent; this rejected challenger is context, "
                        "not an eligible parent."
                    ),
                }
                if self._config.proposer_feedback_mode == "rich_v1":
                    feedback["candidate_snapshot"] = (
                        _candidate_refinement_snapshot(record.pruned_candidate)
                    )
                    feedback["per_target_error"] = {
                        "training_normalized_mse": dict(
                            record.pruned_fit.training_metrics.per_target_normalized_mse
                        ),
                        "validation_normalized_mse": dict(
                            record.pruned_fit.validation_metrics.per_target_normalized_mse
                        ),
                    }
                    feedback["deterministic_runtime"] = (
                        _deterministic_runtime_feedback(
                            record.pruned_candidate,
                            record.pruned_fit,
                            self._public_target_contract,
                            self._public_mechanism_spec,
                        )
                    )
                return feedback
            return None
        return None

    def _rejected_candidate_ids(self, round_index: int) -> set[str]:
        """Return checkpointed rejected candidates that may be refined."""
        identifiers: set[str] = set()
        for previous_round in range(round_index):
            payload = self._store.load_round(previous_round)
            if (
                payload is None
                or payload.get("stage") != "complete"
                or payload.get("valid")
                or not isinstance(payload.get("candidate"), dict)
            ):
                continue
            candidate = CandidateModel.model_validate(payload["candidate"])
            identifiers.add(candidate.candidate_id)
        return identifiers

    def _prior_failed_structure(
        self,
        round_index: int,
        identity: CandidateIdentity,
    ) -> dict[str, object] | None:
        """Return a prior failure that makes this retry non-informative.

        Public or deterministic structural failures apply to the
        alpha-invariant functional identity. Numerical failures apply only to
        the complete executable identity, so a scientifically unchanged model
        may still repair bounds or initial conditions and receive a new fit.
        """
        for previous_round in range(round_index - 1, -1, -1):
            payload = self._store.load_round(previous_round)
            if (
                payload is None
                or payload.get("stage") != "complete"
                or payload.get("valid")
                or not isinstance(payload.get("candidate"), dict)
            ):
                continue
            previous = CandidateModel.model_validate(payload["candidate"])
            previous_identity = _identity_from_payload(payload, previous)
            if previous_identity is None:
                continue
            failure_class = _failure_class_from_payload(payload)
            if failure_class in {"lineage_contract", "proposal_transport"}:
                continue
            inherited = payload.get("prior_structural_failure")
            inherited_match_level = (
                inherited.get("duplicate_match_level")
                if isinstance(inherited, Mapping)
                else None
            )
            if (
                failure_class == "numerical_fit"
                or inherited_match_level == "executable"
            ):
                match_level = "executable"
                matches = (
                    previous_identity.executable_sha256
                    == identity.executable_sha256
                )
            else:
                match_level = "functional"
                matches = (
                    previous_identity.functional_sha256
                    == identity.functional_sha256
                )
            if not matches:
                continue
            if isinstance(inherited, dict):
                return {
                    **inherited,
                    "duplicate_attempt_round": previous_round,
                    "duplicate_candidate_id": previous.candidate_id,
                }
            fit = payload.get("fit")
            fit_summary = None
            if isinstance(fit, dict):
                fit_success = bool(fit.get("success", False))
                training_metrics = fit.get("training_metrics", {})
                validation_metrics = fit.get("validation_metrics", {})
                fit_summary = {
                    "success": fit_success,
                    "message": fit.get("message"),
                    "parameter_estimates_valid": fit_success,
                    "global_parameters": (
                        dict(fit.get("global_parameters", {}))
                        if fit_success
                        else None
                    ),
                    "training_normalized_mse": (
                        training_metrics.get("normalized_mse")
                        if fit_success
                        else None
                    ),
                    "validation_normalized_mse": (
                        validation_metrics.get("normalized_mse")
                        if fit_success
                        else None
                    ),
                    "training_per_target_normalized_mse": (
                        dict(training_metrics.get("per_target_normalized_mse", {}))
                        if fit_success
                        else None
                    ),
                    "validation_per_target_normalized_mse": (
                        dict(
                            validation_metrics.get(
                                "per_target_normalized_mse", {}
                            )
                        )
                        if fit_success
                        else None
                    ),
                    "diagnostics": [
                        {
                            "backend": item.get("backend"),
                            "status": item.get("status"),
                            "message": item.get("message"),
                            "function_evaluations": item.get(
                                "function_evaluations"
                            ),
                            "integration_failures": item.get(
                                "integration_failures"
                            ),
                            "integration_failure_messages": list(
                                item.get("integration_failure_messages", [])
                            ),
                        }
                        for item in fit.get("diagnostics", [])
                    ],
                }
            return {
                "round_index": previous_round,
                "candidate_id": previous.candidate_id,
                "failure_class": failure_class,
                "duplicate_match_level": match_level,
                "candidate_identity": previous_identity.model_dump(mode="json"),
                "structural_hash": previous_identity.functional_sha256,
                "error": str(payload.get("error", "candidate was rejected")),
                "fit_attempts": list(payload.get("fit_attempts", [])),
                "fit": fit_summary,
                "equations": {
                    item.state: item.rhs
                    for item in previous.state_equations
                },
                "processes": {
                    item.name: item.expression for item in previous.processes
                },
                "instruction": (
                    "This retry matches a previously failed candidate at the "
                    f"{match_level} level and will not be fit again. "
                    + (
                        "Revise its scientific or dynamic structure."
                        if match_level == "functional"
                        else (
                            "Change executable numerical metadata or choose a "
                            "different predeclared fitting strategy."
                        )
                    )
                ),
            }
        return None

    def _failed_structure_memory(
        self,
        round_index: int,
        *,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Expose a bounded, deduplicated memory of expensive failed structures."""
        remembered: list[dict[str, object]] = []
        seen: set[tuple[int, str]] = set()
        for previous_round in range(round_index - 1, -1, -1):
            payload = self._store.load_round(previous_round)
            if (
                payload is None
                or payload.get("stage") != "complete"
                or payload.get("valid")
                or not isinstance(payload.get("candidate"), dict)
            ):
                continue
            candidate = CandidateModel.model_validate(payload["candidate"])
            identity = _identity_from_payload(payload, candidate)
            if identity is None:
                continue
            entry = self._prior_failed_structure(round_index, identity)
            if entry is not None:
                memory_key = (
                    int(entry["round_index"]),
                    str(entry["candidate_id"]),
                )
                if memory_key in seen:
                    continue
                seen.add(memory_key)
                remembered.append(entry)
            if len(remembered) == limit:
                break
        return remembered

    def _completed_records(self) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        for round_index in range(self._config.maximum_iterations):
            payload = self._store.load_round(round_index)
            if payload is None or payload.get("stage") != "complete":
                break
            if payload.get("valid"):
                records.append(_record_from_dict(payload["record"]))
        return records

    def _completed_round_count(self) -> int:
        count = 0
        for round_index in range(self._config.maximum_iterations):
            payload = self._store.load_round(round_index)
            if payload is None or payload.get("stage") != "complete":
                break
            count += 1
        return count

    def _finalize(
        self,
        completed: Sequence[CandidateRecord],
        stopping_reason: str,
    ) -> FinalEvaluation:
        existing = self._store.load_final()
        selected = _beam(completed, 1, self._config)[0]
        if self._config.selection_policy == "incumbent_relative_hybrid":
            incumbent_state = _hybrid_incumbent(completed)
            assert incumbent_state is not None
            objective = (-incumbent_state[1], 0.0, 0.0)
            incumbent_path_score = incumbent_state[1]
        else:
            objective = _selection_objectives(
                _unique_records(completed), self._config
            )[selected.structural_hash]
            incumbent_path_score = None
        frozen = FrozenSelection(
            selection_hash=selected.structural_hash,
            candidate=selected.pruned_candidate,
            validation_mse=selected.pruned_fit.validation_metrics.normalized_mse,
            round_index=selected.round_index,
            selection_policy=self._config.selection_policy,
            selection_objective=objective[0],
            normalized_log_validation=objective[1],
            normalized_judge_penalty=objective[2],
            judge_score=selected.postpruning_judge.aggregate_score,
            incumbent_path_score=incumbent_path_score,
            hybrid_science_weight=(
                self._config.hybrid_science_weight
                if self._config.selection_policy == "incumbent_relative_hybrid"
                else None
            ),
        )
        if existing is None:
            existing = {
                "stage": "frozen",
                "frozen": _frozen_to_dict(frozen),
                "stopping_reason": stopping_reason,
                "completed_iterations": len(completed),
            }
            self._store.save_final(existing)
            self._callback("frozen", None)
        frozen = _frozen_from_dict(existing["frozen"])
        compiled = compile_candidate(frozen.candidate, self._context)

        if existing["stage"] == "frozen":
            combined = _combined_training(self._training, self._validation)
            final_fit, final_fit_attempts = _fit_with_retry(
                compiled,
                combined,
                self._validation,
                primary_config=self._config.final_fit_config,
                retry_config=self._config.final_fit_retry_config,
                initial_global_parameters=selected.pruned_fit.global_parameters,
            )
            if not final_fit.success:
                raise RuntimeError("train-plus-validation final refit failed")
            existing["final_fit"] = _fit_to_dict(final_fit)
            existing["final_fit_attempts"] = final_fit_attempts
            existing["stage"] = "refitted"
            self._store.save_final(existing)
            self._callback("refitted", None)
        final_fit = _fit_from_dict(existing["final_fit"])

        if not self._config.evaluate_test:
            if existing["stage"] == "refitted":
                existing["stage"] = "development_complete"
                self._store.save_final(existing)
                self._callback("development_complete", None)
            return FinalEvaluation(
                frozen_selection=frozen,
                final_fit=final_fit,
                test_metrics=None,
                test_trajectory_initial_conditions=None,
                stopping_reason=existing["stopping_reason"],
                completed_iterations=int(existing["completed_iterations"]),
            )

        if existing["stage"] == "refitted":
            existing["stage"] = "test_started"
            self._store.save_final(existing)
            self._callback("test_started", None)

        if existing["stage"] == "test_started":
            self._store.claim_test_access()
            test = self._test_loader(frozen)
            if test.name is not SplitName.TEST:
                raise ValueError("deferred test loader returned a non-test split")
            local_initials, test_metrics = evaluate_fitted_candidate(
                compiled,
                test,
                global_parameters=final_fit.global_parameters,
                global_initial_conditions=final_fit.global_initial_conditions,
                target_scales=final_fit.target_scales,
                config=self._config.final_fit_config,
                fit_trajectory_initial_conditions=False,
            )
            existing["test_initials"] = {
                key: dict(value) for key, value in local_initials.items()
            }
            existing["test_metrics"] = _metrics_to_dict(test_metrics)
            existing["stage"] = "complete"
            self._store.save_final(existing)
            self._callback("test_evaluated", None)
        return FinalEvaluation(
            frozen_selection=frozen,
            final_fit=final_fit,
            test_metrics=_metrics_from_dict(existing["test_metrics"]),
            test_trajectory_initial_conditions={
                key: dict(value)
                for key, value in existing["test_initials"].items()
            },
            stopping_reason=existing["stopping_reason"],
            completed_iterations=int(existing["completed_iterations"]),
        )
    def _save_stage(
        self,
        round_index: int,
        payload: dict[str, Any],
        stage: str,
    ) -> str:
        payload["stage"] = stage
        self._store.save_round(round_index, payload)
        self._callback(stage, round_index)
        return stage

    def _run_fingerprint(self) -> str:
        payload = {
            "implementation": _implementation_fingerprint(),
            "config": self._config.model_dump(
                mode="json", exclude={"checkpoint_directory"}
            ),
            "pairwise_judge_fingerprint": (
                None
                if self._pairwise_judge is None
                else self._pairwise_judge.fingerprint
            ),
            "public_target_contract": (
                None
                if self._public_target_contract is None
                else self._public_target_contract.model_dump(mode="json")
            ),
            "public_mechanism_spec": (
                None
                if self._public_mechanism_spec is None
                else self._public_mechanism_spec.model_dump(mode="json")
            ),
            "context": self._context.model_dump(mode="json"),
            "splits": {
                "train": self._training.fingerprint,
                "validation": self._validation.fingerprint,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


def _merge_feedback_evidence(
    evidence: Sequence[CandidateFeedbackEvidence],
) -> CandidateFeedbackEvidence:
    """Merge evidence records deterministically without repeating messages."""
    if not evidence:
        return CandidateFeedbackEvidence()

    def messages(field: str) -> tuple[str, ...]:
        values = (
            value
            for item in evidence
            for value in getattr(item, field)
        )
        return tuple(dict.fromkeys(values))[:64]

    metrics: dict[str, float] = {}
    for item in evidence:
        for metric in item.validation_metrics:
            metrics[metric.target] = max(
                metric.normalized_mse,
                metrics.get(metric.target, 0.0),
            )
    return CandidateFeedbackEvidence(
        target_contract_failures=messages("target_contract_failures"),
        graph_mechanism_failures=messages("graph_mechanism_failures"),
        annotation_failures=messages("annotation_failures"),
        deterministic_validation_failures=messages(
            "deterministic_validation_failures"
        ),
        validation_metrics=tuple(
            TargetValidationMetric(target=target, normalized_mse=value)
            for target, value in sorted(metrics.items())
        ),
        fit_failures=messages("fit_failures"),
        integration_failures=messages("integration_failures"),
        scientific_missing_requirements=messages(
            "scientific_missing_requirements"
        ),
        scientific_actionable_edits=messages(
            "scientific_actionable_edits"
        ),
    )


def _structural_feedback_evidence(
    candidate: CandidateModel,
    public_target_contract: PublicTargetContract | None,
    public_mechanism_spec: MechanismEvaluationSpec | None,
) -> CandidateFeedbackEvidence:
    """Extract target and mechanism findings without requiring a fit result."""
    target_failures: list[str] = []
    if public_target_contract is not None:
        evaluation = evaluate_public_targets(candidate, public_target_contract)
        target_failures.extend(
            f"{item.target_channel}/{item.predicate}: {item.evidence}"
            for item in evaluation.predicates
            if item.status == "failed"
        )

    graph_failures: list[str] = []
    annotation_failures: list[str] = []
    if public_mechanism_spec is not None:
        evaluation = evaluate_mechanisms(candidate, public_mechanism_spec)
        for result in evaluation.mechanism_results:
            if result.status != "satisfied":
                graph_failures.append(_mechanism_feedback_message(result))
        for result in evaluation.annotation_results:
            if result.status != "satisfied":
                annotation_failures.append(_mechanism_feedback_message(result))
        annotation_failures.extend(
            f"{item.mechanism_id}: {item.evidence}; suggested components="
            f"{list(item.suggested_components)}"
            for item in evaluation.annotation_repairs
        )
    return CandidateFeedbackEvidence(
        target_contract_failures=tuple(target_failures),
        graph_mechanism_failures=tuple(graph_failures),
        annotation_failures=tuple(annotation_failures),
    )


def _mechanism_feedback_message(result: Any) -> str:
    evidence = [
        item.evidence
        for item in result.predicates
        if item.status != "satisfied"
    ]
    return f"{result.mechanism_id}/{result.status}: {'; '.join(evidence)}"


def _diagnostic_message(diagnostic: Mapping[str, Any]) -> str:
    code = str(diagnostic.get("code", "validation_error"))
    location = str(diagnostic.get("location", "candidate"))
    message = str(diagnostic.get("message", "candidate validation failed"))
    return f"{code} at {location}: {message}"


def _fit_with_retry(
    model: Any,
    training: DatasetSplit,
    validation: DatasetSplit,
    *,
    primary_config: FitConfig,
    retry_config: FitConfig | None,
    initial_global_parameters: Mapping[str, float] | None = None,
) -> tuple[FitResult, list[dict[str, Any]]]:
    """Fit once, then deterministically retry any rejected candidate if enabled.

    The caller checkpoints both the selected fit and the compact attempt ledger.
    Retrying never changes the candidate, splits, objective, or initial warm start;
    it only enlarges a predeclared numerical budget.
    """
    attempts: list[dict[str, Any]] = []
    settings = (primary_config, retry_config)
    final: FitResult | None = None
    for attempt_index, config in enumerate(settings):
        if config is None:
            continue
        fitted = fit_candidate(
            model,
            training,
            validation,
            config,
            initial_global_parameters=initial_global_parameters,
        )
        attempts.append(_fit_attempt_to_dict(attempt_index, config, fitted))
        final = fitted
        if fitted.success:
            break
    if final is None:
        raise AssertionError("a primary fit configuration is required")
    return final, attempts


def _fit_attempt_to_dict(
    attempt_index: int,
    config: FitConfig,
    fit: FitResult,
) -> dict[str, Any]:
    """Serialize compact, deterministic provenance for one fit attempt."""
    return {
        "attempt_index": attempt_index,
        "fit_config": config.model_dump(mode="json"),
        "success": fit.success,
        "message": fit.message,
        "diagnostic_statuses": [item.status for item in fit.diagnostics],
        "diagnostic_function_evaluations": [
            item.function_evaluations for item in fit.diagnostics
        ],
        "initialization_diagnostics": [
            asdict(item) for item in fit.initialization_diagnostics
        ],
    }


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "search" / "checkpoints.py",
        root / "search" / "hybrid_pair.py",
        root / "search" / "identity.py",
        root / "search" / "models.py",
        root / "search" / "feedback_routing.py",
        root / "search" / "staged_proposer.py",
        root / "judging" / "hybrid.py",
        root / "judging" / "prompts.py",
        root / "targets.py",
        root / "staging.py",
        root / "expressions" / "compiler.py",
        root / "fitting" / "casadi_initializer.py",
        root / "fitting" / "fitter.py",
        root / "fitting" / "models.py",
        root / "pruning" / "pruner.py",
        root / "schemas" / "candidate.py",
        root / "schemas" / "judge.py",
        root / "schemas" / "proposal.py",
        root / "schemas" / "staged.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _structural_hash(candidate: CandidateModel) -> str:
    """Return the backward-compatible functional-identity fingerprint."""

    return candidate_identity(candidate).functional_sha256


def _identity_from_payload(
    payload: Mapping[str, Any],
    candidate: CandidateModel,
) -> CandidateIdentity | None:
    """Read an identity or derive one when every expression is parseable."""

    stored = payload.get("candidate_identity")
    if isinstance(stored, Mapping):
        return CandidateIdentity.model_validate(stored)
    try:
        return candidate_identity(candidate)
    except SyntaxError:
        return None


def _failure_class_from_payload(payload: Mapping[str, Any]) -> str:
    """Classify old checkpoints that predate explicit failure classes."""

    stored = payload.get("failure_class")
    if isinstance(stored, str):
        return stored
    if payload.get("error") == "numerical fit failed":
        return "numerical_fit"
    fit = payload.get("fit")
    if isinstance(fit, Mapping) and not bool(fit.get("success", False)):
        return "numerical_fit"
    return "legacy_failure"


def _beam(
    records: Sequence[CandidateRecord], size: int, config: SearchConfig
) -> list[CandidateRecord]:
    if config.selection_policy == "incumbent_relative_hybrid":
        incumbent = _hybrid_incumbent(records)
        return [] if incumbent is None else [incumbent[0]]
    candidates = _unique_records(records)
    objectives = _selection_objectives(candidates, config)
    return sorted(
        candidates,
        key=lambda record: (
            objectives[record.structural_hash][0],
            *_validation_rank(record),
        ),
    )[:size]


def _hybrid_incumbent(
    records: Sequence[CandidateRecord],
) -> tuple[CandidateRecord, float] | None:
    """Replay checkpointed sequential challenges and return incumbent/path score."""
    ordered = sorted(records, key=lambda item: item.round_index)
    if not ordered:
        return None
    incumbent = ordered[0]
    path_score = 0.0
    if incumbent.incumbent_challenge is not None:
        raise ValueError("first hybrid-search record must seed the incumbent")
    for challenger in ordered[1:]:
        challenge = challenger.incumbent_challenge
        if challenge is None:
            raise ValueError("hybrid-search record is missing incumbent challenge")
        if challenge.incumbent_hash != incumbent.structural_hash:
            raise ValueError("hybrid-search challenge references a stale incumbent")
        if abs(challenge.incumbent_path_score_before - path_score) > 1e-12:
            raise ValueError("hybrid-search path score is inconsistent")
        if challenge.selected_hash == challenger.structural_hash:
            incumbent = challenger
        elif challenge.selected_hash != incumbent.structural_hash:
            raise ValueError("hybrid-search challenge selected an unknown candidate")
        path_score = challenge.incumbent_path_score_after
    return incumbent, path_score


def _fit_preference_for_challenger(
    incumbent_loss: float,
    challenger_loss: float,
) -> float:
    """Return a bounded, symmetric relative validation-fit improvement."""
    denominator = incumbent_loss + challenger_loss
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.clip((incumbent_loss - challenger_loss) / denominator, -1, 1))


def _deterministic_runtime_feedback(
    candidate: CandidateModel,
    fit: FitResult,
    public_target_contract: PublicTargetContract | None,
    public_mechanism_spec: MechanismEvaluationSpec | None,
) -> dict[str, object]:
    """Expose actionable certified facts without test data or hidden contracts."""
    target_evaluation = (
        None
        if public_target_contract is None
        else evaluate_public_targets(candidate, public_target_contract).model_dump(
            mode="json"
        )
    )
    mechanism_evaluation = (
        None
        if public_mechanism_spec is None
        else evaluate_mechanisms(candidate, public_mechanism_spec).model_dump(
            mode="json"
        )
    )
    lower_contacts = sorted(
        {
            parameter
            for item in fit.diagnostics
            for parameter in item.parameters_at_lower_bound
        }
    )
    upper_contacts = sorted(
        {
            parameter
            for item in fit.diagnostics
            for parameter in item.parameters_at_upper_bound
        }
    )
    return {
        "candidate_validation": "passed",
        "public_target_evaluation": target_evaluation,
        "public_mechanism_evaluation": mechanism_evaluation,
        "public_mechanism_feedback_contract": {
            "primary_scientific_signal": "graph_mechanism_compliance",
            "metadata_signal": "mechanism_annotation_compliance",
            "annotation_repair_policy": (
                "apply only unambiguous runtime suggestions and preserve repair "
                "provenance"
            ),
        },
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "backends": sorted({item.backend for item in fit.diagnostics}),
            "nonlinear_initializers": [
                {
                    "backend": item.backend,
                    "success": item.success,
                    "status": item.status,
                    "message": item.message,
                    "objective": item.objective,
                    "iterations": item.iterations,
                    "wall_seconds": item.wall_seconds,
                }
                for item in fit.initialization_diagnostics
            ],
            "optimizer_messages": [
                item.message for item in fit.diagnostics if item.message
            ],
            "function_evaluations": sum(
                item.function_evaluations for item in fit.diagnostics
            ),
            "integration_failures": sum(
                item.integration_failures for item in fit.diagnostics
            ),
            "integration_failure_messages": sorted(
                {
                    message
                    for item in fit.diagnostics
                    for message in item.integration_failure_messages
                }
            )[:10],
            "parameters_at_lower_bound": lower_contacts,
            "parameters_at_upper_bound": upper_contacts,
            "bound_contact_interpretation": (
                "advisory scale or identifiability diagnostic; not standalone "
                "evidence of structural invalidity"
            ),
            "training_failed_trajectories": list(
                fit.training_metrics.failed_trajectories
            ),
            "validation_failed_trajectories": list(
                fit.validation_metrics.failed_trajectories
            ),
        },
    }


def _candidate_refinement_snapshot(candidate: CandidateModel) -> dict[str, object]:
    """Render the complete bounded structure needed for an informed revision."""
    equations = {item.state: item.rhs for item in candidate.state_equations}
    initials = {
        item.state: item.model_dump(mode="json")
        for item in candidate.initial_conditions
    }
    constraints: dict[str, list[dict[str, object]]] = {}
    for constraint in candidate.constraints:
        constraints.setdefault(constraint.subject, []).append(
            constraint.model_dump(mode="json")
        )
    return {
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_candidate_id,
        "change_summary": candidate.change_summary,
        "states": [
            {
                **item.model_dump(mode="json"),
                "rhs": equations[item.name],
                "initial": initials.get(item.name),
                "constraints": constraints.get(item.name, []),
            }
            for item in candidate.states
        ],
        "algebraics": [
            {
                **item.model_dump(mode="json"),
                "constraints": constraints.get(item.name, []),
            }
            for item in candidate.processes
        ],
        "observation_mappings": [
            item.model_dump(mode="json") for item in candidate.observation_mappings
        ],
        "parameters": [
            item.model_dump(mode="json") for item in candidate.parameters
        ],
    }


def _proposal_mode(
    config: SearchConfig,
    round_index: int,
    beam: Sequence[CandidateRecord],
) -> str:
    """Choose a request mode without changing the shared round-zero request."""
    if (
        config.proposal_policy == "incumbent_refinement_v1"
        and round_index > 0
        and beam
    ):
        return "incumbent_refinement"
    if round_index > 0 and not beam:
        return "feedback_guided_recovery"
    return "exploratory"


def _refinement_contract(record: CandidateRecord) -> dict[str, object]:
    """Describe a relaxed, feedback-motivated refinement of one incumbent."""
    return {
        "schema_version": "incumbent-refinement-contract-1",
        "required_parent_candidate_id": record.pruned_candidate.candidate_id,
        "output_contract": "return_one_complete_candidate_not_a_patch",
        "edit_policy": (
            "Make the smallest coherent set of structural edits that addresses "
            "the supplied failures. Multiple related edits are allowed. Preserve "
            "validated mechanisms and equations unless changing them is necessary "
            "for the stated repair, and explain every intentional change in "
            "change_summary. Numeric parameter values are still fitted by runtime."
        ),
        "priority_order": [
            "deterministic_contract_failures",
            "integration_or_optimizer_failures",
            "worst_validation_target",
            "public_mechanism_predicate_failures",
            "question_level_scientific_feedback",
            "parsimony",
        ],
    }


def _incumbent_challenge_feedback(
    challenge: IncumbentChallenge,
) -> dict[str, object]:
    """Return bounded scientific feedback without transport/checkpoint internals."""
    judgment = challenge.judgment
    result = None if judgment is None else judgment.consensus_result
    return {
        "incumbent_candidate_hash": challenge.incumbent_hash,
        "challenger_candidate_hash": challenge.challenger_hash,
        "selected_candidate_hash": challenge.selected_hash,
        "fit_preference_for_challenger": challenge.fit_preference_for_challenger,
        "science_preference_for_challenger": (
            challenge.science_preference_for_challenger
        ),
        "combined_preference_for_challenger": (
            challenge.combined_preference_for_challenger
        ),
        "challenger_relative_score": challenge.challenger_relative_score,
        "symmetric_scientific_evidence_available": result is not None,
        "absolute_assessments": (
            []
            if result is None
            else [item.model_dump(mode="json") for item in result.absolute_assessments]
        ),
        "comparative_assessments": (
            []
            if result is None
            else [
                item.model_dump(mode="json")
                for item in result.comparative_assessments
            ]
        ),
        "orientation_disagreements": (
            {}
            if judgment is None
            else {
                "absolute": list(judgment.absolute_disagreements),
                "comparative": list(judgment.comparative_disagreements),
            }
        ),
    }


def _science_preference_for_challenger(
    decision_for_incumbent: float | None,
    *,
    decision_scale: float,
    tie_threshold: float,
) -> float | None:
    """Map the frozen hybrid decision to a bounded challenger preference.

    Values inside the judge's already-frozen tie interval contribute no science
    preference. Outside it, the remaining interval is linearly mapped to [-1, 1].
    """
    if decision_for_incumbent is None:
        return None
    challenger = -decision_for_incumbent
    magnitude = abs(challenger)
    if magnitude <= tie_threshold:
        return 0.0
    denominator = max(decision_scale - tie_threshold, np.finfo(float).eps)
    return float(
        np.sign(challenger) * min(1.0, (magnitude - tie_threshold) / denominator)
    )


def _incumbent_challenge(
    *,
    incumbent: CandidateRecord,
    challenger_hash: str,
    challenger_validation_mse: float,
    incumbent_path_score: float,
    judgment: HybridPairJudgment | None,
    science_weight: float,
) -> IncumbentChallenge:
    fit_delta = _fit_preference_for_challenger(
        incumbent.pruned_fit.validation_metrics.normalized_mse,
        challenger_validation_mse,
    )
    science_delta = (
        None
        if judgment is None
        else _science_preference_for_challenger(
            judgment.decision_value_for_incumbent,
            decision_scale=judgment.decision_scale,
            tie_threshold=judgment.tie_threshold,
        )
    )
    combined = (
        None
        if science_delta is None
        else (1.0 - science_weight) * fit_delta + science_weight * science_delta
    )
    challenger_selected = combined is not None and combined > 0.0
    selected_hash = (
        challenger_hash if challenger_selected else incumbent.structural_hash
    )
    path_after = incumbent_path_score + (max(0.0, combined) if combined else 0.0)
    return IncumbentChallenge(
        incumbent_hash=incumbent.structural_hash,
        challenger_hash=challenger_hash,
        fit_preference_for_challenger=fit_delta,
        science_preference_for_challenger=science_delta,
        combined_preference_for_challenger=combined,
        challenger_relative_score=(
            None if combined is None else (1.0 + combined) / 2.0
        ),
        incumbent_path_score_before=incumbent_path_score,
        incumbent_path_score_after=path_after,
        selected_hash=selected_hash,
        judgment=judgment,
    )


def _unique_records(records: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """Keep one deterministic representative of each structural hash."""
    unique: dict[str, CandidateRecord] = {}
    for record in records:
        existing = unique.get(record.structural_hash)
        if existing is None or _validation_rank(record) < _validation_rank(existing):
            unique[record.structural_hash] = record
    return list(unique.values())


def _validation_rank(record: CandidateRecord) -> tuple[float, float, int, str]:
    return (
        record.pruned_fit.validation_metrics.normalized_mse,
        -record.postpruning_judge.aggregate_score,
        sum(
            len(_additive_sources(item.rhs))
            for item in record.pruned_candidate.state_equations
        ),
        record.structural_hash,
    )


def _selection_objectives(
    records: Sequence[CandidateRecord], config: SearchConfig
) -> dict[str, tuple[float, float, float]]:
    """Return objective, normalized fit, and normalized judge cost by hash."""
    if not records:
        return {}
    losses = np.asarray(
        [record.pruned_fit.validation_metrics.normalized_mse for record in records],
        dtype=float,
    )
    scores = np.asarray(
        [record.postpruning_judge.aggregate_score for record in records],
        dtype=float,
    )
    loss_z = _robust_standardize(np.log(np.maximum(losses, np.finfo(float).tiny)))
    judge_z = _robust_standardize(-np.log(scores + config.judge_score_epsilon))
    if config.selection_policy == "validation_only":
        objectives = losses
    else:
        objectives = loss_z + config.judge_weight * judge_z
    return {
        record.structural_hash: (
            float(objectives[index]),
            float(loss_z[index]),
            float(judge_z[index]),
        )
        for index, record in enumerate(records)
    }


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    """Median/IQR standardize with deterministic zero-spread fallbacks."""
    median = float(np.median(values))
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    if scale <= np.finfo(float).eps:
        scale = float(np.ptp(values))
    if scale <= np.finfo(float).eps:
        scale = 1.0
    return (values - median) / scale


def _additive_sources(source: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in source.replace("-", "+-").split("+"))


def _trailing_stagnation(
    records: Sequence[CandidateRecord], config: SearchConfig
) -> int:
    selected_hash: str | None = None
    stagnant = 0
    prefix: list[CandidateRecord] = []
    for record in records:
        prefix.append(record)
        current_hash = _beam(prefix, 1, config)[0].structural_hash
        if current_hash != selected_hash:
            selected_hash = current_hash
            stagnant = 0
        else:
            stagnant += 1
    return stagnant


def _combined_training(
    training: DatasetSplit, validation: DatasetSplit
) -> DatasetSplit:
    fingerprint = hashlib.sha256(
        f"{training.fingerprint}:{validation.fingerprint}".encode()
    ).hexdigest()
    return DatasetSplit(
        SplitName.TRAIN,
        tuple(
            replace(trajectory, trajectory_id=f"train:{trajectory.trajectory_id}")
            for trajectory in training.trajectories
        )
        + tuple(
            replace(
                trajectory,
                trajectory_id=f"validation:{trajectory.trajectory_id}",
            )
            for trajectory in validation.trajectories
        ),
        fingerprint,
    )


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {
        "normalized_mse": metrics.normalized_mse,
        "per_target_normalized_mse": dict(metrics.per_target_normalized_mse),
        "failed_trajectories": list(metrics.failed_trajectories),
        "soft_constraint_violations": {
            key: dict(value)
            for key, value in metrics.soft_constraint_violations.items()
        },
    }


def _metrics_from_dict(payload: Mapping[str, Any]) -> EvaluationMetrics:
    return EvaluationMetrics(
        float(payload["normalized_mse"]),
        {
            key: float(value)
            for key, value in payload["per_target_normalized_mse"].items()
        },
        tuple(payload["failed_trajectories"]),
        {
            key: {metric: float(value) for metric, value in summary.items()}
            for key, summary in payload.get(
                "soft_constraint_violations", {}
            ).items()
        },
    )


def _fit_to_dict(fit: FitResult) -> dict[str, Any]:
    return {
        "success": bool(fit.success),
        "global_parameters": dict(fit.global_parameters),
        "global_initial_conditions": dict(fit.global_initial_conditions),
        "training_trajectory_initial_conditions": {
            key: dict(value)
            for key, value in fit.training_trajectory_initial_conditions.items()
        },
        "validation_trajectory_initial_conditions": {
            key: dict(value)
            for key, value in fit.validation_trajectory_initial_conditions.items()
        },
        "training_metrics": _metrics_to_dict(fit.training_metrics),
        "validation_metrics": _metrics_to_dict(fit.validation_metrics),
        "diagnostics": [asdict(item) for item in fit.diagnostics],
        "initialization_diagnostics": [
            asdict(item) for item in fit.initialization_diagnostics
        ],
        "best_start_index": fit.best_start_index,
        "target_scales": dict(fit.target_scales),
        "message": fit.message,
    }


def _fit_from_dict(payload: Mapping[str, Any]) -> FitResult:
    return FitResult(
        success=bool(payload["success"]),
        global_parameters=dict(payload["global_parameters"]),
        global_initial_conditions=dict(payload["global_initial_conditions"]),
        training_trajectory_initial_conditions={
            key: dict(value)
            for key, value in payload[
                "training_trajectory_initial_conditions"
            ].items()
        },
        validation_trajectory_initial_conditions={
            key: dict(value)
            for key, value in payload[
                "validation_trajectory_initial_conditions"
            ].items()
        },
        training_metrics=_metrics_from_dict(payload["training_metrics"]),
        validation_metrics=_metrics_from_dict(payload["validation_metrics"]),
        diagnostics=tuple(
            OptimizationDiagnostic(
                **{
                    **item,
                    "parameters_at_lower_bound": tuple(
                        item["parameters_at_lower_bound"]
                    ),
                    "parameters_at_upper_bound": tuple(
                        item["parameters_at_upper_bound"]
                    ),
                    "integration_failure_messages": tuple(
                        item.get("integration_failure_messages", ())
                    ),
                    "certified_parameter_transformations": tuple(
                        item.get("certified_parameter_transformations", ())
                    ),
                    "affine_parameters_outside_suggested_bounds": tuple(
                        item.get(
                            "affine_parameters_outside_suggested_bounds", ()
                        )
                    ),
                    "runtime_inferred_observed_states": tuple(
                        item.get("runtime_inferred_observed_states", ())
                    ),
                    "physical_outer_start_parameters": tuple(
                        item.get("physical_outer_start_parameters", ())
                    ),
                }
            )
            for item in payload["diagnostics"]
        ),
        initialization_diagnostics=tuple(
            InitializationDiagnostic(
                **{
                    **item,
                    "parameter_estimates": tuple(
                        item.get("parameter_estimates", ())
                    ),
                }
            )
            for item in payload.get("initialization_diagnostics", ())
        ),
        best_start_index=int(payload["best_start_index"]),
        target_scales=dict(payload["target_scales"]),
        message=payload["message"],
    )


def _record_to_dict(record: CandidateRecord) -> dict[str, Any]:
    return {
        "round_index": record.round_index,
        "candidate": record.candidate.model_dump(mode="json"),
        "parent_candidate_id": record.parent_candidate_id,
        "structural_hash": record.structural_hash,
        "fit": _fit_to_dict(record.fit),
        "postfit_judge": record.postfit_judge.model_dump(mode="json"),
        "pruned_candidate": record.pruned_candidate.model_dump(mode="json"),
        "pruned_fit": _fit_to_dict(record.pruned_fit),
        "postpruning_judge": record.postpruning_judge.model_dump(mode="json"),
        "pruning_removed_terms": list(record.pruning_removed_terms),
        "pruning_removed_parameters": list(record.pruning_removed_parameters),
        "pruning_contributions": record.pruning_contributions,
        "incumbent_challenge": (
            None
            if record.incumbent_challenge is None
            else record.incumbent_challenge.model_dump(mode="json")
        ),
    }


def _record_from_dict(payload: Mapping[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        round_index=int(payload["round_index"]),
        candidate=CandidateModel.model_validate(payload["candidate"]),
        parent_candidate_id=payload["parent_candidate_id"],
        structural_hash=payload["structural_hash"],
        fit=_fit_from_dict(payload["fit"]),
        postfit_judge=parse_judge_assessment(payload["postfit_judge"]),
        pruned_candidate=CandidateModel.model_validate(payload["pruned_candidate"]),
        pruned_fit=_fit_from_dict(payload["pruned_fit"]),
        postpruning_judge=parse_judge_assessment(
            payload["postpruning_judge"]
        ),
        pruning_removed_terms=tuple(payload["pruning_removed_terms"]),
        pruning_removed_parameters=tuple(payload["pruning_removed_parameters"]),
        pruning_contributions=dict(payload["pruning_contributions"]),
        incumbent_challenge=(
            None
            if payload.get("incumbent_challenge") is None
            else IncumbentChallenge.model_validate(payload["incumbent_challenge"])
        ),
    )


def _frozen_to_dict(frozen: FrozenSelection) -> dict[str, Any]:
    return {
        "selection_hash": frozen.selection_hash,
        "candidate": frozen.candidate.model_dump(mode="json"),
        "validation_mse": frozen.validation_mse,
        "round_index": frozen.round_index,
        "selection_policy": frozen.selection_policy,
        "selection_objective": frozen.selection_objective,
        "normalized_log_validation": frozen.normalized_log_validation,
        "normalized_judge_penalty": frozen.normalized_judge_penalty,
        "judge_score": frozen.judge_score,
        "incumbent_path_score": frozen.incumbent_path_score,
        "hybrid_science_weight": frozen.hybrid_science_weight,
    }


def _frozen_from_dict(payload: Mapping[str, Any]) -> FrozenSelection:
    return FrozenSelection(
        selection_hash=payload["selection_hash"],
        candidate=CandidateModel.model_validate(payload["candidate"]),
        validation_mse=float(payload["validation_mse"]),
        round_index=int(payload["round_index"]),
        selection_policy=payload.get("selection_policy", "validation_only"),
        selection_objective=float(
            payload.get("selection_objective", payload["validation_mse"])
        ),
        normalized_log_validation=float(
            payload.get("normalized_log_validation", 0.0)
        ),
        normalized_judge_penalty=float(
            payload.get("normalized_judge_penalty", 0.0)
        ),
        judge_score=float(payload.get("judge_score", 0.0)),
        incumbent_path_score=(
            None
            if payload.get("incumbent_path_score") is None
            else float(payload["incumbent_path_score"])
        ),
        hybrid_science_weight=(
            None
            if payload.get("hybrid_science_weight") is None
            else float(payload["hybrid_science_weight"])
        ),
    )
