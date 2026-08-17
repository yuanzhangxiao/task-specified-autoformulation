"""Google Gemini structured-output adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.models import StructuredT, TokenUsage

_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "const",
    "default",
    "maxLength",
    "minLength",
    "pattern",
}


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
        if sdk_client is None and not self._cache_only:
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
                    "response_json_schema": _gemini_provider_schema(response_model),
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


def _gemini_compatible_schema(value: object) -> object:
    """Remove JSON Schema keywords unsupported by Gemini structured output.

    The complete Pydantic model still validates every response locally, so this
    provider compatibility projection does not weaken the trusted boundary.
    """
    if isinstance(value, dict):
        return {
            key: _gemini_compatible_schema(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(value, list):
        return [_gemini_compatible_schema(item) for item in value]
    return value


def _gemini_provider_schema(response_model: type[StructuredT]) -> object:
    """Return a low-complexity Gemini schema for the two production contracts."""
    if response_model.__name__ == "ProposerCandidateV2":
        return _compact_proposer_schema()
    if response_model.__name__ == "ScientificJudgeResult":
        return _compact_scientific_judge_schema()
    return _gemini_compatible_schema(
        response_model.model_json_schema(mode="validation")
    )


def _nullable(schema: dict[str, object]) -> dict[str, object]:
    return {"anyOf": [schema, {"type": "null"}]}


def _compact_proposer_schema() -> dict[str, object]:
    text = {"type": "string"}
    number = {"type": "number"}
    string_list = {"type": "array", "items": text}
    value_range = {
        "type": "object",
        "properties": {"lower": number, "upper": number},
        "required": ["lower", "upper"],
        "additionalProperties": False,
    }
    initial = {
        "type": "object",
        "properties": {
            "fixed_value": _nullable(number),
            "expression": _nullable(text),
        },
        "additionalProperties": False,
    }
    state = {
        "type": "object",
        "properties": {
            "name": text,
            "kind": {"type": "string", "enum": ["observed", "latent"]},
            "rhs": text,
            "observed_channel": _nullable(text),
            "initial": _nullable(initial),
            "mechanisms": string_list,
        },
        "required": ["name", "kind", "rhs"],
        "additionalProperties": False,
    }
    algebraic = {
        "type": "object",
        "properties": {
            "name": text,
            "expression": text,
            "mechanisms": string_list,
        },
        "required": ["name", "expression"],
        "additionalProperties": False,
    }
    parameter = {
        "type": "object",
        "properties": {
            "name": text,
            "bounds": value_range,
            "initialization_range": _nullable(value_range),
            "scope": {"type": "string", "enum": ["global"]},
        },
        "required": ["name", "bounds"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2"]},
            "candidate_id": text,
            "parent_candidate_id": _nullable(text),
            "change_summary": text,
            "states": {"type": "array", "items": state, "minItems": 1},
            "algebraics": {"type": "array", "items": algebraic},
            "parameters": {"type": "array", "items": parameter},
        },
        "required": ["candidate_id", "states"],
        "additionalProperties": False,
    }


def _compact_scientific_judge_schema() -> dict[str, object]:
    text = {"type": "string"}
    score = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    category = {
        "type": "object",
        "properties": {"score": score, "justification": text},
        "required": ["score"],
        "additionalProperties": False,
    }
    categories = (
        "mechanistic_coherence",
        "source_sink_balance_semantics",
        "dynamic_plausibility",
        "mechanism_coupling_task_sufficiency",
        "nonredundancy_accounting",
        "latent_state_complexity_justification",
    )
    red_flag = {
        "type": "object",
        "properties": {"code": text, "description": text, "evidence": text},
        "required": ["code", "evidence"],
        "additionalProperties": False,
    }
    edit = {
        "type": "object",
        "properties": {
            "target": text,
            "instruction": text,
            "priority": {
                "type": "string",
                "enum": ["required", "recommended", "optional"],
            },
        },
        "required": ["target", "instruction", "priority"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2"]},
            "hard_red_flags": {"type": "array", "items": red_flag},
            "category_scores": {
                "type": "object",
                "properties": dict.fromkeys(categories, category),
                "required": list(categories),
                "additionalProperties": False,
            },
            "missing_requirements": {"type": "array", "items": text},
            "actionable_edits": {"type": "array", "items": edit},
        },
        "required": ["category_scores"],
        "additionalProperties": False,
    }
