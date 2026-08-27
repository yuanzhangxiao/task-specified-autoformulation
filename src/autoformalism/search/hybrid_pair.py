"""Search-time paired hybrid judge with frozen symmetric response policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from autoformalism.judging import (
    HybridScoringConfig,
    atomic_candidate_context,
    atomic_findings_payload,
    atomic_role_compatibility_assessments,
    build_atomic_evidence_plan,
    candidate_claims,
    deterministic_pair_assessments,
    merge_atomic_assessments,
    question_consensus,
    require_deterministic_orientation_consensus,
    reverse_hybrid_result,
    reverse_paired_assessments,
    score_hybrid_pair,
    semantic_absolute_units,
    structural_facts,
)
from autoformalism.llm import LLMClient
from autoformalism.llm.exceptions import LLMError
from autoformalism.schemas import (
    AbsoluteCriterion,
    CandidateModel,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RequirementRegistry,
)
from autoformalism.schemas.base import StrictSchema

LEGACY_PAIRWISE_SEARCH_PROTOCOL_VERSION = (
    "incumbent-hybrid-question-consensus-1"
)
PAIRWISE_SEARCH_PROTOCOL_VERSION = (
    "incumbent-hybrid-question-consensus-2-fixed-denominator"
)
FAIL_CLOSED_TARGET_SEARCH_PROTOCOL_VERSION = (
    "incumbent-hybrid-question-consensus-3-fail-closed-target"
)
_REDUNDANT_ATOMIC_ROLE_UNITS = {
    (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate"),
    (AbsoluteCriterion.SINK_ROLES_CONSISTENT, "candidate"),
}
_FAIL_DOMINANT_TARGET_CRITERIA = frozenset(
    {AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT}
)


class HybridPairJudgment(StrictSchema):
    """Checkpoint-safe symmetric scientific comparison of two candidates."""

    protocol_version: str = PAIRWISE_SEARCH_PROTOCOL_VERSION
    incumbent_candidate_id: str
    challenger_candidate_id: str
    seed_attempt_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    decision_value_for_incumbent: float | None = Field(ge=-2.0, le=2.0)
    decision_scale: float = Field(gt=0.0)
    tie_threshold: float = Field(ge=0.0, le=1.0)
    comparative_indeterminate_policy: Literal[
        "exclude", "neutral_fixed_denominator"
    ] = "exclude"
    preferred: str
    orientation_values_for_incumbent: tuple[float | None, float | None]
    orientation_half_gap: float | None = Field(default=None, ge=0.0)
    absolute_disagreements: tuple[str, ...] = ()
    comparative_disagreements: tuple[str, ...] = ()
    consensus_result: HybridJudgeResult | None = None
    deterministic_assessments: tuple[PairedAbsoluteAssessment, ...]
    request_hashes: tuple[str, ...] = Field(default=(), max_length=4)
    prior_terminal_failures: tuple[str, ...] = ()


class PairwiseScientificJudge(Protocol):
    """Search controller dependency for incumbent-relative science judgments."""

    @property
    def fingerprint(self) -> str:
        """Return a stable protocol/client identity for checkpoint compatibility."""
        ...

    def compare(
        self,
        incumbent: CandidateModel,
        challenger: CandidateModel,
    ) -> HybridPairJudgment:
        """Return a complete judgment or an explicit bounded-failure outcome."""
        ...


@dataclass(frozen=True)
class _OrientationResult:
    result: HybridJudgeResult
    deterministic: tuple[PairedAbsoluteAssessment, ...]
    decision_value_for_a: float | None
    request_hashes: tuple[str, str]


class PairedHybridJudge:
    """Run the frozen two-stage judge in both orientations at one shared seed."""

    def __init__(
        self,
        *,
        seeded_clients: tuple[tuple[int, LLMClient], ...],
        requirements: RequirementRegistry,
        task_inputs: tuple[str, ...],
        system_prompt: str,
        atomic_system_prompt: str,
        scoring: HybridScoringConfig | None = None,
        repair_missing_atomic_units: bool = False,
        identity: str,
    ) -> None:
        if not seeded_clients:
            raise ValueError("paired hybrid judge requires at least one seed")
        if len({seed for seed, _client in seeded_clients}) != len(seeded_clients):
            raise ValueError("paired hybrid judge seeds must be distinct")
        self._seeded_clients = seeded_clients
        self._requirements = requirements
        self._task_inputs = task_inputs
        self._system_prompt = system_prompt
        self._atomic_system_prompt = atomic_system_prompt
        self._scoring = scoring or HybridScoringConfig()
        self._repair_missing_atomic_units = repair_missing_atomic_units
        if self._scoring.target_mapping_consensus == "fail_dominant":
            self._protocol_version = FAIL_CLOSED_TARGET_SEARCH_PROTOCOL_VERSION
        else:
            self._protocol_version = (
                PAIRWISE_SEARCH_PROTOCOL_VERSION
                if self._scoring.comparative_indeterminate_policy
                == "neutral_fixed_denominator"
                else LEGACY_PAIRWISE_SEARCH_PROTOCOL_VERSION
            )
        fingerprint_payload = {
            "protocol": self._protocol_version,
            "identity": identity,
            "seeds": [seed for seed, _client in seeded_clients],
            "requirements": requirements.model_dump(mode="json"),
            "task_inputs": task_inputs,
            "scoring": self._scoring.__dict__,
            "repair_missing_atomic_units": repair_missing_atomic_units,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "atomic_system_prompt_sha256": hashlib.sha256(
                atomic_system_prompt.encode()
            ).hexdigest(),
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode()
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def compare(
        self,
        incumbent: CandidateModel,
        challenger: CandidateModel,
    ) -> HybridPairJudgment:
        """Use one paired seed, replacing the whole pair once on terminal failure."""
        failures: list[str] = []
        for attempt_index, (seed, client) in enumerate(self._seeded_clients):
            try:
                forward = self._orientation(client, incumbent, challenger)
                reverse = self._orientation(client, challenger, incumbent)
            except LLMError as exc:
                failures.append(
                    f"seed={seed} {type(exc).__name__}: {str(exc)[:1000]}"
                )
                continue
            reverse_result = reverse_hybrid_result(reverse.result)
            consensus, absolute_disagreements, comparative_disagreements = (
                question_consensus(
                    forward.result,
                    reverse_result,
                    fail_dominant_absolute_criteria=(
                        _FAIL_DOMINANT_TARGET_CRITERIA
                        if self._scoring.target_mapping_consensus
                        == "fail_dominant"
                        else frozenset()
                    ),
                )
            )
            deterministic = require_deterministic_orientation_consensus(
                forward.deterministic,
                reverse_paired_assessments(reverse.deterministic),
            )
            score = score_hybrid_pair(
                consensus,
                deterministic,
                self._requirements,
                self._scoring,
            )
            forward_value = forward.decision_value_for_a
            reverse_value = (
                None
                if reverse.decision_value_for_a is None
                else -reverse.decision_value_for_a
            )
            half_gap = (
                None
                if forward_value is None or reverse_value is None
                else abs(forward_value - reverse_value) / 2.0
            )
            return HybridPairJudgment(
                protocol_version=self._protocol_version,
                incumbent_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                seed_attempt_index=attempt_index,
                seed=seed,
                decision_value_for_incumbent=score.decision_value,
                decision_scale=1.0 + self._scoring.comparative_weight,
                tie_threshold=self._scoring.tie_threshold,
                comparative_indeterminate_policy=(
                    self._scoring.comparative_indeterminate_policy
                ),
                preferred=score.preferred,
                orientation_values_for_incumbent=(forward_value, reverse_value),
                orientation_half_gap=half_gap,
                absolute_disagreements=absolute_disagreements,
                comparative_disagreements=comparative_disagreements,
                consensus_result=consensus,
                deterministic_assessments=deterministic,
                request_hashes=(
                    *forward.request_hashes,
                    *reverse.request_hashes,
                ),
                prior_terminal_failures=tuple(failures),
            )
        final_seed = self._seeded_clients[-1][0]
        return HybridPairJudgment(
            protocol_version=self._protocol_version,
            incumbent_candidate_id=incumbent.candidate_id,
            challenger_candidate_id=challenger.candidate_id,
            seed_attempt_index=len(self._seeded_clients) - 1,
            seed=final_seed,
            decision_value_for_incumbent=None,
            decision_scale=1.0 + self._scoring.comparative_weight,
            tie_threshold=self._scoring.tie_threshold,
            comparative_indeterminate_policy=(
                self._scoring.comparative_indeterminate_policy
            ),
            preferred="indeterminate",
            orientation_values_for_incumbent=(None, None),
            deterministic_assessments=(),
            prior_terminal_failures=tuple(failures),
        )

    def _orientation(
        self,
        client: LLMClient,
        candidate_a: CandidateModel,
        candidate_b: CandidateModel,
    ) -> _OrientationResult:
        deterministic = deterministic_pair_assessments(
            candidate_a,
            candidate_b,
            task_inputs=self._task_inputs,
        )
        atomic_plan = build_atomic_evidence_plan(candidate_a, candidate_b)
        atomic_request = {
            "public_requirement_registry": self._requirements.model_dump(
                mode="json"
            ),
            "candidate_unsigned_context": {
                "candidate_a": atomic_candidate_context(candidate_a),
                "candidate_b": atomic_candidate_context(candidate_b),
            },
            "atomic_evidence_plan": atomic_plan.prompt_payload(),
        }
        atomic_call = client.assess_atomic_evidence(
            system_prompt=self._atomic_system_prompt,
            user_prompt=json.dumps(atomic_request, sort_keys=True),
            expected_occurrence_ids=atomic_plan.occurrence_ids,
            expected_repeat_pair_ids=atomic_plan.repeat_pair_ids,
            repair_missing_units=self._repair_missing_atomic_units,
        )
        role_assessments = atomic_role_compatibility_assessments(
            atomic_call.parsed,
            atomic_plan,
        )
        request = {
            "public_requirement_registry": self._requirements.model_dump(
                mode="json"
            ),
            "requested_absolute_units": [
                {"criterion": criterion.value, "subject_id": subject}
                for criterion, subject in semantic_absolute_units(
                    self._requirements,
                    include_role_consistency=False,
                )
            ],
            "candidate_a": candidate_a.model_dump(mode="json"),
            "candidate_b": candidate_b.model_dump(mode="json"),
            "proposer_claims": {
                "candidate_a": [
                    item.model_dump(mode="json")
                    for item in candidate_claims(candidate_a)
                ],
                "candidate_b": [
                    item.model_dump(mode="json")
                    for item in candidate_claims(candidate_b)
                ],
            },
            "deterministic_structural_facts": {
                "candidate_a": structural_facts(
                    candidate_a, task_inputs=self._task_inputs
                ),
                "candidate_b": structural_facts(
                    candidate_b, task_inputs=self._task_inputs
                ),
            },
            "runtime_owned_absolute_assessments": [
                item.model_dump(mode="json") for item in deterministic
            ],
            "atomic_scientific_findings": atomic_findings_payload(
                atomic_call.parsed,
                atomic_plan,
                role_assessments,
            ),
        }
        expected_units = set(
            semantic_absolute_units(
                self._requirements,
                include_role_consistency=False,
            )
        )
        hybrid_call = client.assess_hybrid(
            system_prompt=self._system_prompt,
            user_prompt=json.dumps(request, sort_keys=True),
            expected_absolute_units=expected_units,
            redundant_absolute_units=_REDUNDANT_ATOMIC_ROLE_UNITS,
        )
        merged = merge_atomic_assessments(
            hybrid_call.parsed,
            atomic_call.parsed,
            atomic_plan,
            role_assessments,
        )
        merged.validate_expected_absolute_units(
            set(semantic_absolute_units(self._requirements))
        )
        score = score_hybrid_pair(
            merged,
            deterministic,
            self._requirements,
            self._scoring,
        )
        return _OrientationResult(
            result=merged,
            deterministic=deterministic,
            decision_value_for_a=score.decision_value,
            request_hashes=(atomic_call.request_hash, hybrid_call.request_hash),
        )
