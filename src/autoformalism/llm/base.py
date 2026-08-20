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
from typing import Any, Generic

from pydantic import BaseModel, ValidationError

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
    CandidateModel,
    ComparativeJudgeResult,
    HybridJudgeResult,
    ProposerCandidateV2,
    ScientificJudgeResult,
    enrich_proposal_v2,
)


@dataclass(frozen=True)
class ProviderResponse(Generic[StructuredT]):
    """Normalized result returned by a concrete provider adapter."""

    parsed: StructuredT
    raw_response: dict[str, object]
    usage: TokenUsage | None = None
    latency_ms: float | None = None


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
        self._cache_only = cache_only

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
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
    ) -> LLMCallResult[HybridJudgeResult]:
        """Request a strict provenance-aware hybrid calibration result."""
        return self._structured_call(
            role="hybrid_judge",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=HybridJudgeResult,
            validate_parsed=lambda result: result.validate_expected_absolute_units(
                expected_absolute_units
            ),
        )

    def _structured_call(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
        validate_parsed: Callable[[StructuredT], object] | None = None,
    ) -> LLMCallResult[StructuredT]:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt must not be empty")
        request_hash = self.request_hash(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
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
            )
            self._log_success(role, result)
            return result
        if self._cache_only:
            self._append_log(
                {
                    "event": "llm_cache_miss",
                    "provider": self._provider_name,
                    "model": self._model,
                    "role": role,
                    "request_hash": request_hash,
                    "cache_path": str(self._cache_path(request_hash)),
                }
            )
            raise LLMCacheMissError(
                "cache-only LLM request was not found: "
                f"{request_hash} ({self._cache_path(request_hash)})"
            )

        started = time.perf_counter()
        attempts = 0
        attempt_user_prompt = user_prompt
        while attempts < self._max_attempts:
            attempts += 1
            try:
                provider_response = self._call_provider(
                    role=role,
                    system_prompt=system_prompt,
                    user_prompt=attempt_user_prompt,
                    response_model=response_model,
                )
                if validate_parsed is not None:
                    try:
                        validate_parsed(provider_response.parsed)
                    except ValueError as exc:
                        raise LLMResponseError(
                            f"response failed post-schema validation: {exc}",
                            raw_response=provider_response.raw_response,
                            diagnostic_code=(
                                RepairDiagnosticCode.POST_SCHEMA_VALIDATION
                            ),
                        ) from exc
                break
            except (LLMProviderError, LLMResponseError) as exc:
                self._log_failure(role, request_hash, attempts, exc)
                if not exc.retryable or attempts >= self._max_attempts:
                    raise
                if isinstance(exc, LLMResponseError):
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
        }
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
    ) -> ProviderResponse[StructuredT]:
        """Make one provider attempt and return a normalized response."""
