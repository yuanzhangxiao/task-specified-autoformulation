"""Local Ollama adapter for free/open-source model calls."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.config import OllamaResponseMode, OllamaThinking
from autoformalism.llm.exceptions import (
    LLMProviderError,
    LLMResponseError,
    RepairDiagnosticCode,
)
from autoformalism.llm.models import StructuredT, TokenUsage

OllamaTransport = Callable[[str, dict[str, object], float], dict[str, object]]
_STRUCTURED_TOOL_NAME = "submit_structured_response"
_TOOL_VERDICT_VALUES = frozenset(
    {
        "candidate_a",
        "candidate_b",
        "fail",
        "indeterminate",
        "not_applicable",
        "pass",
        "tie",
    }
)


class OllamaClient(CachedLLMClient):
    """Structured client for a local Ollama ``/api/chat`` endpoint."""

    def __init__(
        self,
        *,
        model: str,
        cache_directory: Path,
        log_path: Path,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int | None = None,
        thinking: OllamaThinking = OllamaThinking.AUTO,
        response_mode: OllamaResponseMode = OllamaResponseMode.JSON_SCHEMA,
        transport: OllamaTransport | None = None,
        **retry_options: Any,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must use http or https")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if seed is not None and seed < 0:
            raise ValueError("seed must be nonnegative")
        super().__init__(
            provider_name="ollama",
            model=model,
            cache_directory=cache_directory,
            log_path=log_path,
            **retry_options,
        )
        if (
            response_mode is OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK
            and self._max_attempts < 2
        ):
            raise ValueError(
                "json_schema_tool_fallback requires at least two attempts"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._seed = seed
        self._thinking = _resolve_thinking(model, thinking)
        self._response_mode = response_mode
        self._transport = transport or self._http_transport

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            "base_url": self._base_url,
            "temperature": self._temperature,
            "seed": self._seed,
            "max_output_tokens": self._max_output_tokens,
            "thinking": self._thinking,
            "response_mode": self._response_mode.value,
            "tool_fallback_attempt": (
                self._max_attempts
                if self._response_mode
                is OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK
                else None
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
        del role
        options: dict[str, object] = {
            "temperature": self._temperature,
            "num_predict": self._max_output_tokens,
        }
        attempt_seed = None
        if self._seed is not None:
            attempt_seed = self._seed + attempt_number - 1
            options["seed"] = attempt_seed
        use_tool_call = self._uses_tool_call(
            attempt_number=attempt_number,
            repair_diagnostic_codes=repair_diagnostic_codes,
        )
        use_tool_fallback = (
            use_tool_call
            and self._response_mode
            is OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK
        )
        use_openai_fallback = (
            not use_tool_call
            and RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT
            in repair_diagnostic_codes
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_schema = _ollama_compatible_schema(
            response_model.model_json_schema(mode="validation")
        )
        if use_tool_call:
            endpoint = f"{self._base_url}/api/chat"
            fallback_override = (
                "For this final fallback attempt, this tool requirement "
                "supersedes any earlier instruction to place JSON in ordinary "
                "final content. "
                if use_tool_fallback
                else ""
            )
            messages[-1]["content"] += (
                "\n\nProtocol completion requirement: "
                f"{fallback_override}call exactly the provided "
                f"{_STRUCTURED_TOOL_NAME} tool once with the complete assessment. "
                "Do not place the assessment in ordinary final text."
            )
            body = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": self._thinking,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": _STRUCTURED_TOOL_NAME,
                            "description": (
                                "Submit the complete response required by the "
                                "scientific evaluation protocol."
                            ),
                            "parameters": response_schema,
                        },
                    }
                ],
                "options": options,
            }
        elif use_openai_fallback:
            endpoint = f"{self._base_url}/v1/chat/completions"
            body: dict[str, object] = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "reasoning_effort": "none",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": response_schema,
                    },
                },
                "temperature": self._temperature,
                "max_tokens": self._max_output_tokens,
            }
            if attempt_seed is not None:
                body["seed"] = attempt_seed
        else:
            endpoint = f"{self._base_url}/api/chat"
            body = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": self._thinking,
                "format": response_schema,
                "options": options,
            }
        started = time.perf_counter()
        try:
            raw_response = self._transport(
                endpoint,
                body,
                self._timeout_seconds,
            )
        except LLMProviderError:
            raise
        except (OSError, TimeoutError) as exc:
            raise LLMProviderError(
                f"Ollama request failed ({type(exc).__name__}): {exc}",
                retryable=True,
            ) from exc
        raw_response["_autoformalism_retry"] = {
            "attempt_number": attempt_number,
            "sampling_seed": attempt_seed,
            "format_mode": (
                "native_tool_call_fallback"
                if use_tool_fallback
                else "native_tool_call"
                if use_tool_call
                else "openai_json_schema_no_reasoning"
                if use_openai_fallback
                else "native_json_schema"
            ),
            "embedded_json_extracted": False,
        }
        latency_ms = (time.perf_counter() - started) * 1000.0
        message: object = raw_response.get("message")
        done_reason = raw_response.get("done_reason")
        prompt_tokens = self._optional_int(raw_response.get("prompt_eval_count"))
        output_tokens = self._optional_int(raw_response.get("eval_count"))
        if use_openai_fallback:
            choices = raw_response.get("choices")
            choice = (
                choices[0]
                if isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                else None
            )
            message = choice.get("message") if isinstance(choice, dict) else None
            done_reason = (
                choice.get("finish_reason") if isinstance(choice, dict) else None
            )
            usage_payload = raw_response.get("usage")
            if isinstance(usage_payload, dict):
                prompt_tokens = self._optional_int(
                    usage_payload.get("prompt_tokens")
                )
                output_tokens = self._optional_int(
                    usage_payload.get("completion_tokens")
                )
        if not isinstance(message, dict):
            raise LLMResponseError("Ollama response has no message object")
        if use_tool_call:
            parsed = self._parse_tool_response(
                message,
                response_model,
                raw_response=raw_response,
                done_reason=done_reason,
                output_tokens=output_tokens,
            )
            content = ""
        else:
            if not isinstance(message.get("content"), str):
                raise LLMResponseError(
                    "Ollama response has no message.content string"
                )
            content = message["content"]
        if not use_tool_call and not content.strip():
            thinking = (
                message.get("thinking")
                or message.get("reasoning_content")
                or message.get("reasoning")
            )
            thinking_characters = len(thinking) if isinstance(thinking, str) else 0
            raise LLMResponseError(
                "Ollama returned empty message.content "
                f"(done_reason={done_reason!r}, "
                f"eval_count={output_tokens!r}, "
                f"thinking_present={bool(thinking_characters)}, "
                f"thinking_characters={thinking_characters})",
                raw_response=raw_response,
                diagnostic_code=RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT,
            )
        if not use_tool_call:
            try:
                parsed = self._parse_json(content, response_model)
            except LLMResponseError as exc:
                if not use_openai_fallback:
                    exc.raw_response = raw_response
                    raise
                try:
                    parsed = self._parse_single_embedded_json(
                        content, response_model
                    )
                except LLMResponseError as embedded_error:
                    embedded_error.raw_response = raw_response
                    raise embedded_error from exc
                raw_response["_autoformalism_retry"][
                    "embedded_json_extracted"
                ] = True
        provider_duration = raw_response.get("total_duration")
        if isinstance(provider_duration, int | float):
            latency_ms = float(provider_duration) / 1_000_000.0
        usage = None
        if prompt_tokens is not None or output_tokens is not None:
            total = (
                None
                if prompt_tokens is None or output_tokens is None
                else prompt_tokens + output_tokens
            )
            usage = TokenUsage(prompt_tokens, output_tokens, total)
        return ProviderResponse(parsed, raw_response, usage, latency_ms)

    def _uses_tool_call(
        self,
        *,
        attempt_number: int,
        repair_diagnostic_codes: tuple[RepairDiagnosticCode, ...],
    ) -> bool:
        """Use tools always in tool mode or once after exhausted empty JSON."""
        if self._response_mode is OllamaResponseMode.TOOL_CALL:
            return True
        return (
            self._response_mode
            is OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK
            and attempt_number == self._max_attempts
            and RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT
            in repair_diagnostic_codes
        )

    def _parse_tool_response(
        self,
        message: dict[str, object],
        response_model: type[StructuredT],
        *,
        raw_response: dict[str, object],
        done_reason: object,
        output_tokens: int | None,
    ) -> StructuredT:
        """Validate exactly one required tool call without parsing reasoning."""
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            thinking = message.get("thinking")
            thinking_characters = len(thinking) if isinstance(thinking, str) else 0
            raise LLMResponseError(
                "Ollama did not return exactly one structured tool call "
                f"(count={len(tool_calls) if isinstance(tool_calls, list) else 0}, "
                f"done_reason={done_reason!r}, eval_count={output_tokens!r}, "
                f"thinking_present={bool(thinking_characters)}, "
                f"thinking_characters={thinking_characters})",
                raw_response=raw_response,
                diagnostic_code=RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT,
            )
        tool_call = tool_calls[0]
        function = (
            tool_call.get("function") if isinstance(tool_call, dict) else None
        )
        if not isinstance(function, dict):
            raise LLMResponseError(
                "Ollama structured tool call has no function object",
                raw_response=raw_response,
            )
        if function.get("name") != _STRUCTURED_TOOL_NAME:
            raise LLMResponseError(
                "Ollama called an unexpected structured-output tool",
                raw_response=raw_response,
            )
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            normalized, repair_count = _repair_tool_argument_keys(arguments)
            encoded = json.dumps(normalized)
        elif isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                normalized, repair_count = _repair_tool_argument_keys(decoded)
                encoded = json.dumps(normalized)
            else:
                repair_count = 0
                encoded = arguments
        else:
            raise LLMResponseError(
                "Ollama structured tool arguments must be an object or JSON string",
                raw_response=raw_response,
            )
        if repair_count:
            retry = raw_response.get("_autoformalism_retry")
            if isinstance(retry, dict):
                retry["tool_argument_key_repairs"] = repair_count
        try:
            return self._parse_json(encoded, response_model)
        except LLMResponseError as exc:
            exc.raw_response = raw_response
            raise

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

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
                f"Ollama HTTP {exc.code}: {exc.reason}{detail}",
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(
                f"Ollama connection failed: {exc.reason}",
                retryable=True,
            ) from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMResponseError("Ollama response must be a JSON object")
        return payload


def _repair_tool_argument_keys(value: object) -> tuple[object, int]:
    """Repair one observed, unambiguous Ollama verdict-key corruption.

    The repair is intentionally narrower than fuzzy key matching. It applies only
    to an evidence-bearing assessment object, only when the correct key is absent,
    and only when the value is an allowed verdict literal. Local Pydantic and
    expected-unit validation still own the complete trusted boundary.
    """
    if isinstance(value, list):
        repaired_items = []
        repair_count = 0
        for item in value:
            repaired, count = _repair_tool_argument_keys(item)
            repaired_items.append(repaired)
            repair_count += count
        return repaired_items, repair_count
    if not isinstance(value, dict):
        return value, 0

    repair_alias = (
        "ver verdict" in value
        and "verdict" not in value
        and "evidence" in value
        and isinstance(value["ver verdict"], str)
        and value["ver verdict"] in _TOOL_VERDICT_VALUES
    )
    repaired_object: dict[str, object] = {}
    repair_count = 0
    for key, item in value.items():
        repaired_key = "verdict" if repair_alias and key == "ver verdict" else key
        repaired_item, count = _repair_tool_argument_keys(item)
        repaired_object[repaired_key] = repaired_item
        repair_count += count
    return repaired_object, repair_count + int(repair_alias)


def _resolve_thinking(
    model: str,
    thinking: OllamaThinking,
) -> bool | str:
    """Resolve the API value without attempting to disable GPT-OSS reasoning."""
    model_name = model.rsplit("/", 1)[-1].lower()
    requires_thinking = model_name.startswith("gpt-oss")
    if thinking is OllamaThinking.AUTO:
        return "low" if requires_thinking else False
    if thinking is OllamaThinking.OFF:
        if requires_thinking:
            raise ValueError(
                "GPT-OSS models require Ollama thinking level low, medium, or high"
            )
        return False
    return thinking.value


_OLLAMA_OMITTED_SCHEMA_KEYWORDS = {
    "default",
    "description",
    "examples",
    "title",
}


def _ollama_compatible_schema(value: object) -> object:
    """Produce a compact validation schema suitable for Ollama grammars.

    Ollama/llama.cpp converts JSON Schema length limits into bounded grammar
    repetitions. Pydantic's defensive limits (for example, 10,000-character
    strings) can exceed the grammar parser's own safety limit. Pydantic also
    emits titles, descriptions, examples, and defaults that add request tokens
    without changing the accepted JSON shape. The complete Pydantic model still
    validates the parsed response after generation, so removing provider-side
    annotations and upper size limits does not bypass local validation.
    """
    if isinstance(value, dict):
        compact: dict[str, object] = {}
        for key, item in value.items():
            if key in _OLLAMA_OMITTED_SCHEMA_KEYWORDS:
                continue
            if key == "maxLength" and isinstance(item, int):
                compact[key] = min(item, 512)
            elif key == "maxItems" and isinstance(item, int):
                compact[key] = min(item, 32)
            else:
                compact[key] = _ollama_compatible_schema(item)
        properties = compact.get("properties")
        if (
            isinstance(properties, dict)
            and {"candidate_id", "states", "equations"} <= properties.keys()
        ):
            required = compact.get("required")
            required_items = list(required) if isinstance(required, list) else []
            if "equations" not in required_items:
                required_items.append("equations")
            compact["required"] = required_items
            equations = properties.get("equations")
            if isinstance(equations, dict):
                equations["minItems"] = 1
        return compact
    if isinstance(value, list):
        return [_ollama_compatible_schema(item) for item in value]
    return value


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a bounded Ollama error message when the server supplies one."""
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
        message = error.strip() if isinstance(error, str) else ""
    if not message:
        return ""
    return f" ({message[:1000]})"
