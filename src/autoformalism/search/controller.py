"""Judge-integrated, checkpointed iterative candidate search."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
from autoformalism.search.models import (
    CandidateRecord,
    FinalEvaluation,
    FrozenSelection,
    SearchConfig,
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
        self._callback = stage_callback or (lambda _stage, _round: None)
        self._store = CheckpointStore(
            config.checkpoint_directory, self._run_fingerprint()
        )

    def run(self) -> FinalEvaluation:
        """Run search, freeze validation selection, and optionally test once."""
        completed = self._completed_records()
        start_round = self._completed_round_count()
        best_seen = min(
            (item.pruned_fit.validation_metrics.normalized_mse for item in completed),
            default=float("inf"),
        )
        stagnant = _trailing_stagnation(completed)
        stopping_reason = "iteration_budget"

        if best_seen <= self._config.validation_mse_target:
            stopping_reason = "validation_target"
            return self._finalize(completed, stopping_reason)
        if stagnant >= self._config.stagnation_iterations:
            stopping_reason = "stagnation"
            return self._finalize(completed, stopping_reason)

        for round_index in range(start_round, self._config.maximum_iterations):
            beam = _beam(completed, self._config.beam_size)
            record = self._run_round(round_index, beam, completed)
            if record is not None:
                completed.append(record)
                current = record.pruned_fit.validation_metrics.normalized_mse
                if current < best_seen - 1e-12:
                    best_seen = current
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
            payload["pruned_candidate"] = pruning.selected_candidate.model_dump(
                mode="json"
            )
            payload["pruned_fit"] = _fit_to_dict(pruning.selected_fit)
            payload["pruning"] = {
                "removed_terms": list(pruning.selected_removed_terms),
                "removed_parameters": list(pruning.selected_removed_parameters),
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
            )
            payload.update(
                valid=True,
                record=_record_to_dict(record),
            )
            self._save_stage(round_index, payload, "complete")
            return record
        raise RuntimeError(f"unsupported checkpoint stage: {stage}")

    def _judge(self, candidate: CandidateModel, stage: str) -> JudgeAssessment:
        if not self._config.use_judge:
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
            if not self._config.use_judge:
                for key in (
                    "judge_category_scores",
                    "judge_advisory_red_flags",
                    "judge_missing_requirements",
                    "judge_actionable_edits",
                ):
                    feedback[-1].pop(key)
        remaining = self._config.beam_size - len(feedback)
        if remaining <= 0 and feedback:
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
        selected = _beam(completed, 1)[0]
        frozen = FrozenSelection(
            selection_hash=selected.structural_hash,
            candidate=selected.pruned_candidate,
            validation_mse=selected.pruned_fit.validation_metrics.normalized_mse,
            round_index=selected.round_index,
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
    records: Sequence[CandidateRecord], size: int
) -> list[CandidateRecord]:
    unique: dict[str, CandidateRecord] = {}
    for record in records:
        existing = unique.get(record.structural_hash)
        if existing is None or _rank(record) < _rank(existing):
            unique[record.structural_hash] = record
    return sorted(unique.values(), key=_rank)[:size]


def _rank(record: CandidateRecord) -> tuple[float, float, int, str]:
    return (
        record.pruned_fit.validation_metrics.normalized_mse,
        -record.postpruning_judge.aggregate_score,
        sum(
            len(_additive_sources(item.rhs))
            for item in record.pruned_candidate.state_equations
        ),
        record.structural_hash,
    )


def _additive_sources(source: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in source.replace("-", "+-").split("+"))


def _trailing_stagnation(records: Sequence[CandidateRecord]) -> int:
    best = float("inf")
    stagnant = 0
    for record in records:
        value = record.pruned_fit.validation_metrics.normalized_mse
        if value < best - 1e-12:
            best = value
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
    )


def _frozen_to_dict(frozen: FrozenSelection) -> dict[str, Any]:
    return {
        "selection_hash": frozen.selection_hash,
        "candidate": frozen.candidate.model_dump(mode="json"),
        "validation_mse": frozen.validation_mse,
        "round_index": frozen.round_index,
    }


def _frozen_from_dict(payload: Mapping[str, Any]) -> FrozenSelection:
    return FrozenSelection(
        selection_hash=payload["selection_hash"],
        candidate=CandidateModel.model_validate(payload["candidate"]),
        validation_mse=float(payload["validation_mse"]),
        round_index=int(payload["round_index"]),
    )
