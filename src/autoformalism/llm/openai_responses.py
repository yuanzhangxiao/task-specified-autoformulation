"""OpenAI Responses API adapter with Pydantic structured outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import StructuredT, TokenUsage


class OpenAIResponsesClient(CachedLLMClient):
    """Structured client using ``responses.parse`` from the OpenAI SDK."""

    def __init__(
        self,
        *,
        model: str,
        cache_directory: Path,
        log_path: Path,
        max_output_tokens: int = 2048,
        sdk_client: Any | None = None,
        **retry_options: Any,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model=model,
            cache_directory=cache_directory,
            log_path=log_path,
            **retry_options,
        )
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(max_retries=0)
        self._sdk_client = sdk_client
        self._max_output_tokens = max_output_tokens

    def _hashable_provider_options(self) -> dict[str, object]:
        return {"max_output_tokens": self._max_output_tokens}

    def _call_provider(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredT],
    ) -> ProviderResponse[StructuredT]:
        del role
        try:
            response = self._sdk_client.responses.parse(
                model=self._model,
                max_output_tokens=self._max_output_tokens,
                input=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=response_model,
            )
        except Exception as exc:
            self._raise_normalized_error(exc)

        status = getattr(response, "status", None)
        if status not in (None, "completed"):
            raise LLMResponseError(f"OpenAI response did not complete: {status}")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            output_text = getattr(response, "output_text", None)
            if not isinstance(output_text, str) or not output_text.strip():
                raise LLMResponseError("OpenAI response contained no structured output")
            parsed = self._parse_json(output_text, response_model)
        elif not isinstance(parsed, response_model):
            try:
                parsed = response_model.model_validate(parsed)
            except ValidationError as exc:
                raise LLMResponseError(
                    f"OpenAI parsed output failed validation: {exc}"
                ) from exc

        raw_response = self._raw_response(response)
        return ProviderResponse(
            parsed=parsed,
            raw_response=raw_response,
            usage=self._usage(response),
        )

    @staticmethod
    def _raw_response(response: Any) -> dict[str, object]:
        if hasattr(response, "model_dump"):
            try:
                value = response.model_dump(mode="json", warnings=False)
            except TypeError:
                value = response.model_dump(mode="json")
            if isinstance(value, dict):
                return value
        return {
            "id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "output_text": getattr(response, "output_text", None),
        }

    @staticmethod
    def _usage(response: Any) -> TokenUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def _raise_normalized_error(error: Exception) -> None:
        if isinstance(error, ValidationError):
            raise LLMResponseError(
                f"OpenAI structured output failed validation: {error}"
            ) from error
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            retryable_types = (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:  # pragma: no cover - dependency is pinned
            retryable_types = ()
        retryable = isinstance(error, retryable_types)
        raise LLMProviderError(
            f"OpenAI request failed ({type(error).__name__}): {error}",
            retryable=retryable,
        ) from error
