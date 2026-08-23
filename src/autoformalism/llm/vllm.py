"""Structured client for a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.config import VLLMReasoningEffort
from autoformalism.llm.exceptions import (
    LLMProviderError,
    LLMResponseError,
    RepairDiagnosticCode,
)
from autoformalism.llm.models import StructuredT, TokenUsage
from autoformalism.llm.ollama import _ollama_compatible_schema

VLLMTransport = Callable[[str, dict[str, object], float], dict[str, object]]


class VLLMClient(CachedLLMClient):
    """Use strict JSON-schema chat completions from a local vLLM server."""

    def __init__(
        self,
        *,
        model: str,
        cache_directory: Path,
        log_path: Path,
        base_url: str = "http://127.0.0.1:8000",
        reasoning_effort: VLLMReasoningEffort = VLLMReasoningEffort.LOW,
        timeout_seconds: float = 120.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int | None = None,
        transport: VLLMTransport | None = None,
        **retry_options: Any,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("vLLM base_url must use http or https")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if seed is not None and seed < 0:
            raise ValueError("seed must be nonnegative")
        super().__init__(
            provider_name="vllm",
            model=model,
            cache_directory=cache_directory,
            log_path=log_path,
            **retry_options,
        )
        self._base_url = base_url.rstrip("/")
        self._reasoning_effort = reasoning_effort
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._seed = seed
        self._transport = transport or self._http_transport

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            "base_url": self._base_url,
            "reasoning_effort": self._reasoning_effort.value,
            "temperature": self._temperature,
            "seed": self._seed,
            "max_output_tokens": self._max_output_tokens,
            "response_mode": "openai_chat_json_schema",
            "schema_compatibility": "bounded-compact-provider-schema-v4",
        }

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
        del role, repair_diagnostic_codes
        attempt_seed = (
            None if self._seed is None else self._seed + attempt_number - 1
        )
        body: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "reasoning_effort": self._reasoning_effort.value,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _ollama_compatible_schema(
                        response_model.model_json_schema(mode="validation")
                    ),
                },
            },
        }
        if attempt_seed is not None:
            body["seed"] = attempt_seed

        started = time.perf_counter()
        try:
            raw_response = self._transport(
                f"{self._base_url}/v1/chat/completions",
                body,
                self._timeout_seconds,
            )
        except LLMProviderError:
            raise
        except (OSError, TimeoutError) as exc:
            raise LLMProviderError(
                f"vLLM request failed ({type(exc).__name__}): {exc}",
                retryable=True,
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_response["_autoformalism_retry"] = {
            "attempt_number": attempt_number,
            "sampling_seed": attempt_seed,
            "format_mode": "vllm_openai_json_schema",
            "reasoning_effort": self._reasoning_effort.value,
            "embedded_json_extracted": False,
        }

        choices = raw_response.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else None
        )
        message = choice.get("message") if isinstance(choice, dict) else None
        finish_reason = (
            choice.get("finish_reason") if isinstance(choice, dict) else None
        )
        if not isinstance(message, dict):
            raise LLMResponseError(
                "vLLM response has no message object", raw_response=raw_response
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMResponseError(
                "vLLM response has no message.content string",
                raw_response=raw_response,
            )
        usage_payload = raw_response.get("usage")
        prompt_tokens = None
        output_tokens = None
        if isinstance(usage_payload, dict):
            prompt_tokens = _optional_int(usage_payload.get("prompt_tokens"))
            output_tokens = _optional_int(usage_payload.get("completion_tokens"))
        if not content.strip():
            thinking = (
                message.get("reasoning")
                or message.get("reasoning_content")
                or message.get("thinking")
            )
            thinking_characters = len(thinking) if isinstance(thinking, str) else 0
            raise LLMResponseError(
                "vLLM returned empty message.content "
                f"(finish_reason={finish_reason!r}, "
                f"completion_tokens={output_tokens!r}, "
                f"reasoning_present={bool(thinking_characters)}, "
                f"reasoning_characters={thinking_characters})",
                raw_response=raw_response,
                diagnostic_code=RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT,
            )
        try:
            parsed = self._parse_json(content, response_model)
        except LLMResponseError as exc:
            exc.raw_response = raw_response
            raise
        total_tokens = (
            None
            if prompt_tokens is None or output_tokens is None
            else prompt_tokens + output_tokens
        )
        usage = (
            None
            if prompt_tokens is None and output_tokens is None
            else TokenUsage(prompt_tokens, output_tokens, total_tokens)
        )
        return ProviderResponse(parsed, raw_response, usage, latency_ms)

    @staticmethod
    def _http_transport(
        url: str,
        body: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            detail = _http_error_detail(exc)
            raise LLMProviderError(
                f"vLLM HTTP {exc.code}: {exc.reason}{detail}",
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(
                f"vLLM connection failed: {exc.reason}", retryable=True
            ) from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"vLLM returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMResponseError("vLLM response must be a JSON object")
        return payload


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(4096).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        message = body.strip()
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = error.get("message")
            message = detail.strip() if isinstance(detail, str) else ""
        else:
            message = error.strip() if isinstance(error, str) else ""
    return f" ({message[:1000]})" if message else ""
