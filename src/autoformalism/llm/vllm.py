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
        continuation_max_output_tokens: int | None = None,
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
        if (
            continuation_max_output_tokens is not None
            and continuation_max_output_tokens < 128
        ):
            raise ValueError(
                "continuation_max_output_tokens must be at least 128"
            )
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
        self._continuation_max_output_tokens = continuation_max_output_tokens
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
            "continuation_max_output_tokens": (
                self._continuation_max_output_tokens
            ),
            "response_mode": (
                "openai_responses_json_schema_continue_once"
                if self._continuation_max_output_tokens is not None
                else "openai_chat_json_schema"
            ),
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
        if self._continuation_max_output_tokens is not None:
            return self._call_responses_provider(
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                attempt_number=attempt_number,
                repair_diagnostic_codes=repair_diagnostic_codes,
            )
        return self._call_chat_provider(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            attempt_number=attempt_number,
            repair_diagnostic_codes=repair_diagnostic_codes,
        )

    def _call_chat_provider(
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

    def _call_responses_provider(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
        attempt_number: int,
        repair_diagnostic_codes: tuple[RepairDiagnosticCode, ...],
    ) -> ProviderResponse[StructuredT]:
        """Continue one length-truncated Responses generation before restarting."""
        del role, repair_diagnostic_codes
        assert self._continuation_max_output_tokens is not None
        attempt_seed = (
            None if self._seed is None else self._seed + attempt_number - 1
        )
        schema = _ollama_compatible_schema(
            response_model.model_json_schema(mode="validation")
        )
        input_items: str | list[object] = user_prompt
        segments: list[dict[str, object]] = []
        visible_parts: list[str] = []
        started = time.perf_counter()
        for segment_index, segment_budget in enumerate(
            (self._max_output_tokens, self._continuation_max_output_tokens)
        ):
            body: dict[str, object] = {
                "model": self._model,
                "instructions": system_prompt,
                "input": input_items,
                "stream": False,
                "store": False,
                "reasoning": {"effort": self._reasoning_effort.value},
                "temperature": self._temperature,
                "max_output_tokens": segment_budget,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
            if attempt_seed is not None:
                body["seed"] = attempt_seed
            try:
                raw = self._transport(
                    f"{self._base_url}/v1/responses",
                    body,
                    self._timeout_seconds,
                )
            except LLMProviderError as exc:
                if segments:
                    exc.raw_response = _aggregate_responses_segments(
                        segments,
                        attempt_number=attempt_number,
                        sampling_seed=attempt_seed,
                        reasoning_effort=self._reasoning_effort.value,
                        request_count=len(segments) + 1,
                    )
                raise
            except (OSError, TimeoutError) as exc:
                error = LLMProviderError(
                    f"vLLM request failed ({type(exc).__name__}): {exc}",
                    retryable=True,
                )
                if segments:
                    error.raw_response = _aggregate_responses_segments(
                        segments,
                        attempt_number=attempt_number,
                        sampling_seed=attempt_seed,
                        reasoning_effort=self._reasoning_effort.value,
                        request_count=len(segments) + 1,
                    )
                raise error from exc
            segments.append(raw)
            visible_parts.extend(_responses_output_text(raw))
            if raw.get("status") != "incomplete":
                break
            if segment_index == 1:
                aggregate = _aggregate_responses_segments(
                    segments,
                    attempt_number=attempt_number,
                    sampling_seed=attempt_seed,
                    reasoning_effort=self._reasoning_effort.value,
                )
                raise LLMResponseError(
                    "vLLM Responses generation remained incomplete after one "
                    "continuation",
                    raw_response=aggregate,
                )
            output = raw.get("output")
            if not isinstance(output, list) or not output:
                aggregate = _aggregate_responses_segments(
                    segments,
                    attempt_number=attempt_number,
                    sampling_seed=attempt_seed,
                    reasoning_effort=self._reasoning_effort.value,
                )
                raise LLMResponseError(
                    "incomplete vLLM Responses generation has no resumable output",
                    raw_response=aggregate,
                )
            input_items = [
                {"role": "user", "content": user_prompt},
                *output,
            ]

        latency_ms = (time.perf_counter() - started) * 1000.0
        aggregate = _aggregate_responses_segments(
            segments,
            attempt_number=attempt_number,
            sampling_seed=attempt_seed,
            reasoning_effort=self._reasoning_effort.value,
        )
        content = "".join(visible_parts)
        if not content.strip():
            raise LLMResponseError(
                "vLLM Responses generation has no output_text content",
                raw_response=aggregate,
                diagnostic_code=RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT,
            )
        try:
            parsed = self._parse_json(content, response_model)
        except LLMResponseError as exc:
            exc.raw_response = aggregate
            raise
        usage = _aggregate_responses_usage(segments)
        return ProviderResponse(
            parsed,
            aggregate,
            usage,
            latency_ms,
            provider_attempts=len(segments),
        )

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


def _responses_output_text(raw: dict[str, object]) -> list[str]:
    """Extract visible text fragments from one Responses API segment."""
    fragments: list[str] = []
    output = raw.get("output")
    if not isinstance(output, list):
        return fragments
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                fragments.append(text)
    return fragments


def _aggregate_responses_segments(
    segments: list[dict[str, object]],
    *,
    attempt_number: int,
    sampling_seed: int | None,
    reasoning_effort: str,
    request_count: int | None = None,
) -> dict[str, object]:
    """Preserve every physical response while exposing the final response shape."""
    aggregate = dict(segments[-1]) if segments else {}
    aggregate["_autoformalism_retry"] = {
        "attempt_number": attempt_number,
        "sampling_seed": sampling_seed,
        "format_mode": "vllm_openai_responses_json_schema_continue_once",
        "reasoning_effort": reasoning_effort,
        "embedded_json_extracted": False,
    }
    aggregate["_autoformalism_continuation"] = {
        "request_count": request_count or len(segments),
        "continuation_request_count": max(
            0, (request_count or len(segments)) - 1
        ),
        "segments": segments,
    }
    return aggregate


def _aggregate_responses_usage(
    segments: list[dict[str, object]],
) -> TokenUsage | None:
    """Sum Responses API usage across the initial request and continuation."""
    input_tokens = 0
    output_tokens = 0
    observed_input = False
    observed_output = False
    for segment in segments:
        usage = segment.get("usage")
        if not isinstance(usage, dict):
            continue
        observed = _optional_int(usage.get("input_tokens"))
        if observed is not None:
            input_tokens += observed
            observed_input = True
        observed = _optional_int(usage.get("output_tokens"))
        if observed is not None:
            output_tokens += observed
            observed_output = True
    if not observed_input and not observed_output:
        return None
    normalized_input = input_tokens if observed_input else None
    normalized_output = output_tokens if observed_output else None
    total = (
        None
        if normalized_input is None or normalized_output is None
        else normalized_input + normalized_output
    )
    return TokenUsage(normalized_input, normalized_output, total)


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
