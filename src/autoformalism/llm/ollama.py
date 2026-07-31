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
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import StructuredT, TokenUsage

OllamaTransport = Callable[[str, dict[str, object], float], dict[str, object]]


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
        temperature: float = 0.0,
        transport: OllamaTransport | None = None,
        **retry_options: Any,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must use http or https")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        super().__init__(
            provider_name="ollama",
            model=model,
            cache_directory=cache_directory,
            log_path=log_path,
            **retry_options,
        )
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._transport = transport or self._http_transport

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            "base_url": self._base_url,
            "temperature": self._temperature,
        }

    def _call_provider(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
    ) -> ProviderResponse[StructuredT]:
        del role
        body: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": response_model.model_json_schema(mode="validation"),
            "options": {"temperature": self._temperature},
        }
        started = time.perf_counter()
        try:
            raw_response = self._transport(
                f"{self._base_url}/api/chat",
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
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = raw_response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMResponseError("Ollama response has no message.content string")
        parsed = self._parse_json(message["content"], response_model)
        provider_duration = raw_response.get("total_duration")
        if isinstance(provider_duration, int | float):
            latency_ms = float(provider_duration) / 1_000_000.0
        prompt_tokens = self._optional_int(raw_response.get("prompt_eval_count"))
        output_tokens = self._optional_int(raw_response.get("eval_count"))
        usage = None
        if prompt_tokens is not None or output_tokens is not None:
            total = (
                None
                if prompt_tokens is None or output_tokens is None
                else prompt_tokens + output_tokens
            )
            usage = TokenUsage(prompt_tokens, output_tokens, total)
        return ProviderResponse(parsed, raw_response, usage, latency_ms)

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
            raise LLMProviderError(
                f"Ollama HTTP {exc.code}: {exc.reason}",
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

