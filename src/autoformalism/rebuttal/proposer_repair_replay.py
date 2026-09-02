"""Offline replay of frozen proposer text through lossless contract repairs."""

from __future__ import annotations

import hashlib
import json

from pydantic import ConfigDict

from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.schemas import (
    CandidateModel,
    ProposerCandidateV2,
    enrich_proposal_v2,
    normalize_proposer_candidate_v2_payload,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.targets import PublicTargetContract, evaluate_public_targets


class ProposerFirstAttemptReplay(StrictSchema):
    """Deterministic outcome for one frozen first-attempt response body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "phase-b-proposer-first-attempt-replay-1"
    raw_content_sha256: str
    raw_json_object: bool
    pre_schema_repairs: tuple[dict[str, object], ...] = ()
    schema_valid_after_repair: bool
    runtime_repairs: tuple[str, ...] = ()
    deterministic_valid: bool
    public_target_passed: bool
    candidate_sha256: str | None = None
    error_type: str | None = None
    error: str | None = None
    new_llm_calls_made: bool = False
    parameter_fitting_performed: bool = False
    scientific_judge_called: bool = False
    test_data_opened: bool = False


def replay_proposer_first_attempt(
    content: str,
    *,
    target_channels: tuple[str, ...],
    protected_parameter_names: tuple[str, ...],
    context: ValidationContext,
    target_contract: PublicTargetContract,
) -> tuple[ProposerFirstAttemptReplay, CandidateModel | None]:
    """Replay one response without sampling, fitting, judging, or test access."""
    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return (
            _failure(
                content_sha,
                raw_json_object=False,
                error=exc,
            ),
            None,
        )
    if not isinstance(payload, dict):
        return (
            _failure(
                content_sha,
                raw_json_object=False,
                error=ValueError("proposer response is not a JSON object"),
            ),
            None,
        )

    normalized, repairs = normalize_proposer_candidate_v2_payload(
        payload,
        protected_parameter_names=protected_parameter_names,
    )
    repair_records = tuple(item.as_json() for item in repairs)
    try:
        compact = ProposerCandidateV2.model_validate(normalized)
        candidate = enrich_proposal_v2(compact, target_channels)
    except ValueError as exc:
        return (
            _failure(
                content_sha,
                raw_json_object=True,
                error=exc,
                pre_schema_repairs=repair_records,
            ),
            None,
        )

    candidate, runtime_repairs = repair_protected_declarations(candidate, context)
    try:
        compile_candidate(candidate, context)
    except ModelValidationError as exc:
        return (
            _failure(
                content_sha,
                raw_json_object=True,
                error=exc,
                pre_schema_repairs=repair_records,
                schema_valid_after_repair=True,
                runtime_repairs=runtime_repairs,
            ),
            None,
        )
    target_result = evaluate_public_targets(candidate, target_contract)
    candidate_sha = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    result = ProposerFirstAttemptReplay(
        raw_content_sha256=content_sha,
        raw_json_object=True,
        pre_schema_repairs=repair_records,
        schema_valid_after_repair=True,
        runtime_repairs=runtime_repairs,
        deterministic_valid=True,
        public_target_passed=target_result.passed,
        candidate_sha256=candidate_sha,
    )
    return result, candidate


def _failure(
    content_sha: str,
    *,
    raw_json_object: bool,
    error: Exception,
    pre_schema_repairs: tuple[dict[str, object], ...] = (),
    schema_valid_after_repair: bool = False,
    runtime_repairs: tuple[str, ...] = (),
) -> ProposerFirstAttemptReplay:
    return ProposerFirstAttemptReplay(
        raw_content_sha256=content_sha,
        raw_json_object=raw_json_object,
        pre_schema_repairs=pre_schema_repairs,
        schema_valid_after_repair=schema_valid_after_repair,
        runtime_repairs=runtime_repairs,
        deterministic_valid=False,
        public_target_passed=False,
        error_type=type(error).__name__,
        error=str(error)[:4000],
    )
