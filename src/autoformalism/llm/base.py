"""Shared hashing, cache, retry, validation, and logging behavior."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel, ValidationError

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.llm.exceptions import (
    LLMCacheError,
    LLMCacheMissError,
    LLMProviderError,
    LLMResponseError,
    RepairDiagnosticCode,
)
from autoformalism.llm.models import (
    LLMCallResult,
    StructuredT,
    TokenUsage,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AtomicJudgeResult,
    CandidateModel,
    ComparativeJudgeResult,
    FunctionalCandidate,
    HybridJudgeResult,
    PairedTargetCompletenessJudgeResult,
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
    ProposerCandidateV2,
    ScientificJudgeResult,
    TargetCompletenessJudgeResult,
    TopologyCandidate,
    enrich_proposal_v2,
)

if TYPE_CHECKING:
    from autoformalism.expressions import ValidationContext


def _provider_request_count(raw_response: object) -> int:
    """Count physical requests represented by one provider response payload."""
    if not isinstance(raw_response, dict):
        return 1
    continuation = raw_response.get("_autoformalism_continuation")
    if not isinstance(continuation, dict):
        return 1
    count = continuation.get("request_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 1:
        return count
    return 1


@dataclass(frozen=True)
class ProviderResponse(Generic[StructuredT]):
    """Normalized result returned by a concrete provider adapter."""

    parsed: StructuredT
    raw_response: dict[str, object]
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    provider_attempts: int = 1


class CachedLLMClient(ABC):
    """Provider adapter with deterministic cache and append-only JSONL logs."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        cache_directory: Path,
        log_path: Path,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        jitter_fraction: float = 0.25,
        proposal_target_channels: tuple[str, ...] = (),
        proposal_protected_parameter_names: tuple[str, ...] = (),
        cache_only: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if initial_backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff durations must be nonnegative")
        if not 0.0 <= jitter_fraction <= 1.0:
            raise ValueError("jitter_fraction must be in [0, 1]")
        self._provider_name = provider_name
        self._model = model
        self._cache_directory = cache_directory
        self._log_path = log_path
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._jitter_fraction = jitter_fraction
        self._sleep = sleep
        self._random_value = random_value
        self._proposal_target_channels = proposal_target_channels
        self._proposal_protected_parameter_names = (
            proposal_protected_parameter_names
        )
        self._cache_only = cache_only

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cache_only: bool = False,
    ) -> LLMCallResult[CandidateModel]:
        """Request or restore a strict proposer candidate."""
        compact = self._structured_call(
            role="proposer",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ProposerCandidateV2,
            validate_parsed=lambda proposal: enrich_proposal_v2(
                proposal,
                self._proposal_target_channels,
            ),
            cache_only=cache_only,
        )
        return LLMCallResult(
            request_hash=compact.request_hash,
            parsed=enrich_proposal_v2(
                compact.parsed,
                self._proposal_target_channels,
            ),
            raw_response=compact.raw_response,
            cache_hit=compact.cache_hit,
            attempts=compact.attempts,
            latency_ms=compact.latency_ms,
            usage=compact.usage,
            actual_provider_attempts=compact.provider_attempts,
        )

    def propose_topology(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult[TopologyCandidate]:
        """Request a compact graph and enrich public metadata locally."""
        from autoformalism.staging import (
            enrich_topology_proposal,
            normalize_topology_proposal,
        )

        compact = self._structured_call(
            role="staged_topology_proposer_v3",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ProposedTopologyCandidate,
            normalize_parsed=lambda proposal: normalize_topology_proposal(
                proposal, context
            ),
            validate_parsed=lambda proposal: enrich_topology_proposal(
                proposal, context
            ),
            request_metadata={
                "validation_context": context.model_dump(mode="json")
            },
            cache_only=cache_only,
        )
        topology = enrich_topology_proposal(compact.parsed, context)
        return LLMCallResult(
            request_hash=compact.request_hash,
            parsed=topology,
            raw_response=compact.raw_response,
            cache_hit=compact.cache_hit,
            attempts=compact.attempts,
            latency_ms=compact.latency_ms,
            usage=compact.usage,
            actual_provider_attempts=compact.provider_attempts,
        )

    def propose_functions(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        topology: TopologyCandidate,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult[FunctionalCandidate]:
        """Request functions for one committed topology and enrich locally."""
        from autoformalism.staging import (
            enrich_functional_proposal,
            expand_staged_candidate,
            topology_commitment_sha256,
        )

        def validate(proposal: ProposedFunctionalCandidate) -> None:
            functional = enrich_functional_proposal(proposal, topology)
            expand_staged_candidate(topology, functional, context)

        compact = self._structured_call(
            role="staged_function_proposer_v2",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ProposedFunctionalCandidate,
            validate_parsed=validate,
            request_metadata={
                "topology_commitment_sha256": topology_commitment_sha256(
                    topology
                ),
                "validation_context": context.model_dump(mode="json"),
            },
            cache_only=cache_only,
        )
        functional = enrich_functional_proposal(compact.parsed, topology)
        return LLMCallResult(
            request_hash=compact.request_hash,
            parsed=functional,
            raw_response=compact.raw_response,
            cache_hit=compact.cache_hit,
            attempts=compact.attempts,
            latency_ms=compact.latency_ms,
            usage=compact.usage,
            actual_provider_attempts=compact.provider_attempts,
        )

    def judge(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[ScientificJudgeResult]:
        """Request or restore a strict scientific v2 judge result."""
        return self._structured_call(
            role="judge",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ScientificJudgeResult,
        )

    def compare(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[ComparativeJudgeResult]:
        """Request a strict blinded comparative calibration result."""
        return self._structured_call(
            role="comparative_judge",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ComparativeJudgeResult,
        )

    def assess_hybrid(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_absolute_units: set[tuple[AbsoluteCriterion, str]],
        redundant_absolute_units: set[tuple[AbsoluteCriterion, str]] | None = None,
    ) -> LLMCallResult[HybridJudgeResult]:
        """Request a strict provenance-aware hybrid calibration result."""
        redundant = redundant_absolute_units or set()

        def normalize(
            result: HybridJudgeResult,
        ) -> tuple[HybridJudgeResult, dict[str, object]]:
            normalized, removed = result.discard_redundant_absolute_units(
                expected=expected_absolute_units,
                redundant=redundant,
            )
            return normalized, {
                "redundant_absolute_units_removed": [
                    f"{criterion.value}:{subject}"
                    for criterion, subject in removed
                ],
                "redundant_absolute_unit_repair_count": len(removed),
            }

        return self._structured_call(
            role=(
                "hybrid_judge_atomic_repair_v1"
                if redundant_absolute_units is not None
                else "hybrid_judge"
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=HybridJudgeResult,
            normalize_parsed=(
                normalize if redundant_absolute_units is not None else None
            ),
            validate_parsed=lambda result: result.validate_expected_absolute_units(
                expected_absolute_units
            ),
        )

    def assess_atomic_evidence(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_occurrence_ids: set[str],
        expected_repeat_pair_ids: set[str],
        repair_missing_units: bool = False,
    ) -> LLMCallResult[AtomicJudgeResult]:
        """Request sign-blinded atomic evidence before pairwise assessment."""
        def normalize(
            result: AtomicJudgeResult,
        ) -> tuple[AtomicJudgeResult, dict[str, object]]:
            repaired, occurrences, repeats = (
                result.fill_missing_units_with_insufficient_information(
                    occurrence_ids=expected_occurrence_ids,
                    repeat_pair_ids=expected_repeat_pair_ids,
                )
            )
            return repaired, {
                "missing_occurrences_filled": list(occurrences),
                "missing_repeats_filled": list(repeats),
                "missing_occurrence_repair_count": len(occurrences),
                "missing_repeat_repair_count": len(repeats),
            }

        return self._structured_call(
            role=(
                "atomic_evidence_judge_missing_unit_repair_v1"
                if repair_missing_units
                else "atomic_evidence_judge"
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AtomicJudgeResult,
            normalize_after_exhausted=(
                normalize if repair_missing_units else None
            ),
            validate_parsed=lambda result: result.validate_expected_units(
                occurrence_ids=expected_occurrence_ids,
                repeat_pair_ids=expected_repeat_pair_ids,
            ),
        )

    def assess_target_completeness(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_target_ids: set[str],
    ) -> LLMCallResult[TargetCompletenessJudgeResult]:
        """Request one candidate's absolute public-target assessment."""
        return self._structured_call(
            role="target_completeness_judge_v1",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=TargetCompletenessJudgeResult,
            validate_parsed=lambda result: result.validate_expected_targets(
                expected_target_ids
            ),
        )

    def assess_paired_target_completeness(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_target_ids: set[str],
    ) -> LLMCallResult[PairedTargetCompletenessJudgeResult]:
        """Request independent public-target verdicts for a visible pair."""
        return self._structured_call(
            role="paired_target_completeness_judge_v1",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=PairedTargetCompletenessJudgeResult,
            validate_parsed=lambda result: result.validate_expected_targets(
                expected_target_ids
            ),
        )

    def _structured_call(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
        normalize_parsed: (
            Callable[[StructuredT], tuple[StructuredT, dict[str, object]]] | None
        ) = None,
        normalize_after_exhausted: (
            Callable[[StructuredT], tuple[StructuredT, dict[str, object]]] | None
        ) = None,
        validate_parsed: Callable[[StructuredT], object] | None = None,
        request_metadata: dict[str, object] | None = None,
        cache_only: bool = False,
    ) -> LLMCallResult[StructuredT]:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt must not be empty")
        request_hash = self.request_hash(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            request_metadata=request_metadata,
        )
        cached = self._load_cache(request_hash, response_model)
        if cached is not None:
            result = LLMCallResult(
                request_hash=request_hash,
                parsed=cached.parsed,
                raw_response=cached.raw_response,
                cache_hit=True,
                attempts=0,
                latency_ms=cached.latency_ms,
                usage=cached.usage,
                actual_provider_attempts=0,
            )
            self._log_success(role, result)
            return result
        if self._cache_only or cache_only:
            self._append_log(
                {
                    "event": "llm_cache_miss",
                    "provider": self._provider_name,
                    "model": self._model,
                    "role": role,
                    "request_hash": request_hash,
                    "cache_path": str(self._cache_path(request_hash)),
                    "per_call_cache_only": cache_only,
                }
            )
            raise LLMCacheMissError(
                "cache-only LLM request was not found: "
                f"{request_hash} ({self._cache_path(request_hash)})"
            )

        started = time.perf_counter()
        attempts = 0
        provider_attempts = 0
        attempt_user_prompt = user_prompt
        repair_diagnostic_codes: tuple[RepairDiagnosticCode, ...] = ()
        while attempts < self._max_attempts:
            attempts += 1
            current_provider_counted = False
            try:
                provider_response = self._call_provider(
                    role=role,
                    system_prompt=system_prompt,
                    user_prompt=attempt_user_prompt,
                    response_model=response_model,
                    attempt_number=attempts,
                    repair_diagnostic_codes=repair_diagnostic_codes,
                )
                provider_attempts += provider_response.provider_attempts
                current_provider_counted = True
                try:
                    if normalize_parsed is not None:
                        parsed, repair = normalize_parsed(
                            provider_response.parsed
                        )
                        raw_response = dict(provider_response.raw_response)
                        raw_response[
                            "_autoformalism_contract_repair"
                        ] = repair
                        provider_response = ProviderResponse(
                            parsed=parsed,
                            raw_response=raw_response,
                            usage=provider_response.usage,
                            latency_ms=provider_response.latency_ms,
                            provider_attempts=(
                                provider_response.provider_attempts
                            ),
                        )
                    if validate_parsed is not None:
                        validate_parsed(provider_response.parsed)
                except (ModelValidationError, ValueError) as exc:
                    if (
                        normalize_after_exhausted is not None
                        and attempts >= self._max_attempts
                    ):
                        parsed, repair = normalize_after_exhausted(
                            provider_response.parsed
                        )
                        try:
                            if validate_parsed is not None:
                                validate_parsed(parsed)
                        except (ModelValidationError, ValueError) as repaired_exc:
                            raise LLMResponseError(
                                "response failed post-schema validation: "
                                f"{repaired_exc}",
                                raw_response=provider_response.raw_response,
                                diagnostic_code=(
                                    RepairDiagnosticCode.POST_SCHEMA_VALIDATION
                                ),
                            ) from repaired_exc
                        raw_response = dict(provider_response.raw_response)
                        raw_response["_autoformalism_contract_repair"] = repair
                        provider_response = ProviderResponse(
                            parsed=parsed,
                            raw_response=raw_response,
                            usage=provider_response.usage,
                            latency_ms=provider_response.latency_ms,
                            provider_attempts=provider_response.provider_attempts,
                        )
                    else:
                        raise LLMResponseError(
                            "response failed post-schema validation: "
                            f"{exc}",
                            raw_response=provider_response.raw_response,
                            diagnostic_code=(
                                RepairDiagnosticCode.POST_SCHEMA_VALIDATION
                            ),
                        ) from exc
                break
            except (LLMProviderError, LLMResponseError) as exc:
                if not current_provider_counted:
                    provider_attempts += _provider_request_count(
                        getattr(exc, "raw_response", None)
                    )
                self._log_failure(role, request_hash, attempts, exc)
                if not exc.retryable or attempts >= self._max_attempts:
                    raise
                if isinstance(exc, LLMResponseError):
                    repair_diagnostic_codes += tuple(
                        item.code for item in exc.repair_diagnostics
                    )
                    attempt_user_prompt = (
                        f"{user_prompt}\n\n"
                        f"{exc.repair_prompt()}"
                    )
                self._sleep(self._backoff(attempts))
        else:  # pragma: no cover - loop exits by success or exception
            raise AssertionError("retry loop terminated unexpectedly")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = provider_response.latency_ms
        if latency_ms is None:
            latency_ms = elapsed_ms
        normalized = ProviderResponse(
            parsed=provider_response.parsed,
            raw_response=provider_response.raw_response,
            usage=provider_response.usage,
            latency_ms=latency_ms,
            provider_attempts=provider_response.provider_attempts,
        )
        self._write_cache(request_hash, normalized)
        result = LLMCallResult(
            request_hash=request_hash,
            parsed=normalized.parsed,
            raw_response=normalized.raw_response,
            cache_hit=False,
            attempts=attempts,
            latency_ms=normalized.latency_ms,
            usage=normalized.usage,
            actual_provider_attempts=provider_attempts,
        )
        self._log_success(role, result)
        return result

    def request_hash(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        request_metadata: dict[str, object] | None = None,
    ) -> str:
        """Hash all semantic request inputs using canonical JSON."""
        request = {
            "provider": self._provider_name,
            "model": self._model,
            "role": role,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_schema": response_model.model_json_schema(mode="validation"),
            "provider_options": self._hashable_provider_options(),
            "proposal_target_channels": self._proposal_target_channels,
            "proposal_protected_parameter_names": (
                self._proposal_protected_parameter_names
            ),
        }
        if request_metadata is not None:
            request["request_metadata"] = request_metadata
        canonical = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _backoff(self, failed_attempt: int) -> float:
        base = min(
            self._max_backoff_seconds,
            self._initial_backoff_seconds * (2 ** (failed_attempt - 1)),
        )
        jitter = base * self._jitter_fraction * self._random_value()
        return base + jitter

    def _cache_path(self, request_hash: str) -> Path:
        return self._cache_directory / f"{request_hash}.json"

    def _load_cache(
        self,
        request_hash: str,
        response_model: type[StructuredT],
    ) -> ProviderResponse[StructuredT] | None:
        path = self._cache_path(request_hash)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = response_model.model_validate(payload["parsed_response"])
            usage_payload = payload.get("usage")
            usage = TokenUsage(**usage_payload) if usage_payload is not None else None
            return ProviderResponse(
                parsed=parsed,
                raw_response=payload["raw_response"],
                usage=usage,
                latency_ms=payload.get("latency_ms"),
            )
        except (
            OSError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise LLMCacheError(f"invalid cache entry {path}: {exc}") from exc

    def _write_cache(
        self,
        request_hash: str,
        response: ProviderResponse[StructuredT],
    ) -> None:
        self._cache_directory.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(request_hash)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        payload = {
            "request_hash": request_hash,
            "provider": self._provider_name,
            "model": self._model,
            "parsed_response": response.parsed.model_dump(mode="json"),
            "raw_response": self._redact_secrets(response.raw_response),
            "usage": (
                None
                if response.usage is None
                else {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            ),
            "latency_ms": response.latency_ms,
        }
        try:
            temporary.write_text(
                f"{json.dumps(payload, sort_keys=True, ensure_ascii=False)}\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise LLMCacheError(f"could not write cache entry {path}: {exc}") from exc

    def _append_log(self, event: dict[str, Any]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        safe_event = self._redact_secrets(event)
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{json.dumps(safe_event, sort_keys=True, ensure_ascii=False)}\n"
                )
        except (OSError, TypeError, ValueError) as exc:
            raise LLMCacheError(
                f"could not append LLM log {self._log_path}: {exc}"
            ) from exc

    def _log_success(
        self,
        role: str,
        result: LLMCallResult[StructuredT],
    ) -> None:
        self._append_log(
            {
                "event": "llm_response",
                "provider": self._provider_name,
                "model": self._model,
                "role": role,
                "request_hash": result.request_hash,
                "cache_hit": result.cache_hit,
                "attempts": result.attempts,
                "logical_calls": result.logical_calls,
                "provider_attempts": result.provider_attempts,
                "repair_attempts": result.repair_attempts,
                "latency_ms": result.latency_ms,
                "usage": (
                    None
                    if result.usage is None
                    else {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                ),
                "raw_response": result.raw_response,
                "parsed_response": result.parsed.model_dump(mode="json"),
            }
        )

    def _log_failure(
        self,
        role: str,
        request_hash: str,
        attempt: int,
        error: LLMProviderError | LLMResponseError,
    ) -> None:
        self._append_log(
            {
                "event": "llm_failure",
                "provider": self._provider_name,
                "model": self._model,
                "role": role,
                "request_hash": request_hash,
                "attempt": attempt,
                "retryable": error.retryable,
                "failure_category": error.category.value,
                "error_type": type(error).__name__,
                "error": str(error),
                "repair_diagnostics": [
                    {"code": item.code.value, "message": item.message}
                    for item in getattr(error, "repair_diagnostics", ())
                ],
                "raw_response": getattr(error, "raw_response", None),
            }
        )

    @classmethod
    def _redact_secrets(cls, value: Any) -> Any:
        sensitive_fragments = (
            "api_key",
            "authorization",
            "bearer",
            "access_token",
            "refresh_token",
            "secret",
        )
        if isinstance(value, dict):
            return {
                key: (
                    "[REDACTED]"
                    if any(fragment in key.lower() for fragment in sensitive_fragments)
                    else cls._redact_secrets(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_secrets(item) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_secrets(item) for item in value]
        if isinstance(value, str):
            redacted = value
            for variable in (
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "OLLAMA_API_KEY",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            ):
                secret = os.environ.get(variable)
                if secret:
                    redacted = redacted.replace(secret, "[REDACTED]")
            return re.sub(
                r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
                "Bearer [REDACTED]",
                redacted,
            )
        return value

    def _parse_json(
        self,
        text: str,
        response_model: type[StructuredT],
    ) -> StructuredT:
        try:
            return response_model.model_validate_json(text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"response failed {response_model.__name__} validation: {exc}"
            ) from exc

    def _parse_single_embedded_json(
        self,
        text: str,
        response_model: type[StructuredT],
    ) -> StructuredT:
        """Validate the sole schema-matching JSON object embedded in text."""
        decoder = json.JSONDecoder()
        matches: list[StructuredT] = []
        for offset, character in enumerate(text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text, offset)
                matches.append(response_model.model_validate(payload))
            except (json.JSONDecodeError, ValidationError):
                continue
        if len(matches) != 1:
            raise LLMResponseError(
                "response did not contain exactly one embedded "
                f"{response_model.__name__} object; matches={len(matches)}"
            )
        return matches[0]

    def _hashable_provider_options(self) -> dict[str, object]:
        return {}

    @abstractmethod
    def _call_provider(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
        attempt_number: int,
        repair_diagnostic_codes: tuple[RepairDiagnosticCode, ...],
    ) -> ProviderResponse[StructuredT]:
        """Make one provider attempt and return a normalized response."""
