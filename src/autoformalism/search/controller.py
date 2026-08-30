"""Judge-integrated, checkpointed iterative candidate search."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.expressions import (
    ModelValidationError,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.fitting import (
    EvaluationMetrics,
    FitResult,
    OptimizationDiagnostic,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.llm import LLMClient
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.pruning import prune_candidate
from autoformalism.schemas import (
    CandidateModel,
    JudgeAssessment,
    ScientificJudgeResult,
    parse_judge_assessment,
)
from autoformalism.search.checkpoints import CheckpointStore
from autoformalism.search.hybrid_pair import (
    HybridPairJudgment,
    PairwiseScientificJudge,
)
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
    "parameter declarations and bounds are valid",
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
            feedback = self._proposer_feedback(beam, round_index)
            try:
                proposal = self._client.propose(
                    system_prompt=self._config.proposer_system_prompt,
                    user_prompt=json.dumps(
                        {
                            "round": round_index,
                            "proposal_mode": "exploratory",
                            "beam_feedback": feedback,
                        },
                        sort_keys=True,
                    ),
                ).parsed
            except (LLMProviderError, LLMResponseError) as exc:
                payload.update(
                    valid=False,
                    error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            raw_proposal = proposal
            proposal, repairs = repair_protected_declarations(
                raw_proposal, self._context
            )
            if repairs:
                payload["raw_candidate"] = raw_proposal.model_dump(mode="json")
                payload["deterministic_repairs"] = list(repairs)
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
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            try:
                compiled = compile_candidate(candidate, self._context)
            except ModelValidationError as exc:
                payload.update(
                    valid=False,
                    error=str(exc),
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
                        stage="complete",
                    )
                    self._store.save_round(round_index, payload)
                    self._callback("complete", round_index)
                    return None
            if compiled.validated.warnings:
                payload["validation_warnings"] = [
                    {
                        "code": item.code,
                        "location": item.location,
                        "message": item.message,
                    }
                    for item in compiled.validated.warnings
                ]
            structural_hash = _structural_hash(candidate)
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
                    stage="complete",
                )
                self._store.save_round(round_index, payload)
                self._callback("complete", round_index)
                return None
            payload["structural_hash"] = structural_hash
            stage = self._save_stage(round_index, payload, "validated")
        compiled = compile_candidate(candidate, self._context)

        if stage == "validated" and self._config.cheap_prefit_judge:
            prefit = self._judge(candidate, "pre_fit")
            payload["prefit_judge"] = prefit.model_dump(mode="json")
            stage = self._save_stage(round_index, payload, "prefit_judged")
        if stage == "validated":
            stage = "prefit_judged"

        if stage == "prefit_judged":
            fitted = fit_candidate(
                compiled,
                self._training,
                self._validation,
                self._config.fit_config,
            )
            if not fitted.success:
                payload.update(
                    valid=False,
                    error="numerical fit failed",
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
            payload["pruned_fit"] = _fit_to_dict(selected_fit)
            payload["pruning"] = {
                "removed_terms": list(removed_terms),
                "removed_parameters": list(removed_parameters),
                "contributions": dict(pruning.contribution_by_term),
                "persistence_training_mse": pruning.persistence_training_mse,
                "persistence_validation_mse": pruning.persistence_validation_mse,
                "rejected_supports": sum(
                    not item.accepted for item in pruning.candidates
                ),
            }
            stage = self._save_stage(round_index, payload, "pruned")
        pruned_candidate = CandidateModel.model_validate(payload["pruned_candidate"])
        pruned_fit = _fit_from_dict(payload["pruned_fit"])

        if stage == "pruned":
            postpruning = self._judge(pruned_candidate, "post_pruning")
            payload["postpruning_judge"] = postpruning.model_dump(mode="json")
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
                    "error": error,
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
                        "eligible_parent": True,
                        "lineage_parent": candidate.parent_candidate_id,
                        "equations": {
                            item.state: item.rhs
                            for item in candidate.state_equations
                        },
                        "fitted_parameters": (
                            dict(rejected_fit.global_parameters)
                            if rejected_fit is not None
                            else {}
                        ),
                        "training_normalized_mse": (
                            rejected_fit.training_metrics.normalized_mse
                            if rejected_fit is not None
                            else None
                        ),
                        "validation_normalized_mse": (
                            rejected_fit.validation_metrics.normalized_mse
                            if rejected_fit is not None
                            else None
                        ),
                        "judge_category_scores": {},
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
                    }
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
                return {
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
            final_fit = fit_candidate(
                compiled,
                combined,
                self._validation,
                self._config.final_fit_config,
                initial_global_parameters=selected.pruned_fit.global_parameters,
            )
            if not final_fit.success and _fit_timed_out(final_fit):
                final_fit = fit_candidate(
                    compiled,
                    combined,
                    self._validation,
                    self._config.fit_config,
                    initial_global_parameters=selected.pruned_fit.global_parameters,
                )
            if not final_fit.success:
                raise RuntimeError("train-plus-validation final refit failed")
            existing["final_fit"] = _fit_to_dict(final_fit)
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
            "context": self._context.model_dump(mode="json"),
            "splits": {
                "train": self._training.fingerprint,
                "validation": self._validation.fingerprint,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


def _fit_timed_out(fit: FitResult) -> bool:
    """Identify a recoverable wall-clock exhaustion from fit diagnostics."""
    return any(diagnostic.status == -2 for diagnostic in fit.diagnostics)


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "search" / "checkpoints.py",
        root / "search" / "hybrid_pair.py",
        root / "search" / "models.py",
        root / "judging" / "hybrid.py",
        root / "judging" / "prompts.py",
        root / "targets.py",
        root / "expressions" / "compiler.py",
        root / "fitting" / "fitter.py",
        root / "pruning" / "pruner.py",
        root / "schemas" / "candidate.py",
        root / "schemas" / "judge.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _structural_hash(candidate: CandidateModel) -> str:
    state_map = {
        name: f"s{index}"
        for index, name in enumerate(sorted(item.name for item in candidate.states))
    }
    process_map = {
        name: f"q{index}"
        for index, name in enumerate(sorted(item.name for item in candidate.processes))
    }
    parameter_map = {
        name: f"p{index}"
        for index, name in enumerate(
            sorted(item.name for item in candidate.parameters)
        )
    }
    names = {**state_map, **process_map, **parameter_map}
    payload = {
        "states": sorted(
            (state_map[item.name], item.kind.value) for item in candidate.states
        ),
        "processes": sorted(
            (
                process_map[item.name],
                _canonical_expression(item.expression, names),
            )
            for item in candidate.processes
        ),
        "equations": sorted(
            (
                state_map[item.state],
                _canonical_expression(item.rhs, names),
            )
            for item in candidate.state_equations
        ),
        "observations": sorted(
            (
                item.channel,
                _canonical_expression(item.expression, names),
            )
            for item in candidate.observation_mappings
        ),
        "parameters": sorted(
            (parameter_map[item.name], item.scope.value)
            for item in candidate.parameters
        ),
        "initial_conditions": sorted(
            (state_map[item.state], item.scope.value)
            for item in candidate.initial_conditions
        ),
        "constraints": sorted(
            (
                names.get(item.subject, item.subject),
                item.kind.value,
                (
                    None
                    if item.bounds is None
                    else (item.bounds.lower, item.bounds.upper)
                ),
            )
            for item in candidate.constraints
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_expression(source: str, names: Mapping[str, str]) -> str:
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            return ast.copy_location(
                ast.Name(id=names.get(node.id, node.id), ctx=node.ctx),
                node,
            )

    parsed = ast.parse(source, mode="eval")
    renamed = Rename().visit(parsed)
    return ast.unparse(ast.fix_missing_locations(renamed))


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
                }
            )
            for item in payload["diagnostics"]
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
