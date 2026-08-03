"""Google Gemini structured-output adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import StructuredT, TokenUsage


class GeminiClient(CachedLLMClient):
    """Structured client using the official Google Gen AI SDK."""

    def __init__(
        self,
        *,
        model: str,
        cache_directory: Path,
        log_path: Path,
        timeout_seconds: float = 120.0,
        max_output_tokens: int = 2048,
        sdk_client: Any | None = None,
        **retry_options: Any,
    ) -> None:
        super().__init__(
            provider_name="gemini",
            model=model,
            cache_directory=cache_directory,
            log_path=log_path,
            **retry_options,
        )
        if sdk_client is None:
            from google import genai

            sdk_client = genai.Client(
                http_options={"timeout": int(timeout_seconds * 1000)}
            )
        self._sdk_client = sdk_client
        self._max_output_tokens = max_output_tokens

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            "api": "generate_content",
            "max_output_tokens": self._max_output_tokens,
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
        try:
            response = self._sdk_client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": self._max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_json_schema": response_model.model_json_schema(
                        mode="validation"
                    ),
                },
            )
        except Exception as exc:
            self._raise_normalized_error(exc)

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Gemini response contained no structured output")
        try:
            parsed = response_model.model_validate_json(text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Gemini structured output failed validation: {exc}",
                raw_response=self._raw_response(response),
            ) from exc
        return ProviderResponse(
            parsed=parsed,
            raw_response=self._raw_response(response),
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
        return {"text": getattr(response, "text", None)}

    @staticmethod
    def _usage(response: Any) -> TokenUsage | None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    @staticmethod
    def _raise_normalized_error(error: Exception) -> None:
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(error, "code", None)
        retryable = status in {408, 429, 500, 502, 503, 504} or isinstance(
            error, (ConnectionError, TimeoutError)
        )
        raise LLMProviderError(
            f"Gemini request failed ({type(error).__name__}): {error}",
            retryable=retryable,
        ) from error
