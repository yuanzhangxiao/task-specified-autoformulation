"""Typed LLM provider selection and client factory."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from autoformalism.llm.models import LLMClient
from autoformalism.schemas.base import StrictSchema


class LLMProvider(str, Enum):
    """Supported paid/hosted and free/local provider choices."""

    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VLLM = "vllm"


class OllamaThinking(str, Enum):
    """Supported Ollama thinking controls with a model-aware default."""

    AUTO = "auto"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OllamaResponseMode(str, Enum):
    """Ollama transport used to obtain a validated structured response."""

    JSON_SCHEMA = "json_schema"
    JSON_SCHEMA_NATIVE_RETRY = "json_schema_native_retry"
    JSON_SCHEMA_OPENAI_THINKING_RETRY = "json_schema_openai_thinking_retry"
    JSON_SCHEMA_TOOL_FALLBACK = "json_schema_tool_fallback"
    TOOL_CALL = "tool_call"


class VLLMReasoningEffort(str, Enum):
    """GPT-OSS reasoning effort exposed by vLLM chat completions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMConfig(StrictSchema):
    """Provider-neutral settings; API keys are intentionally absent."""

    provider: LLMProvider
    model: str = Field(min_length=1, max_length=256)
    cache_directory: Path
    log_path: Path
    max_attempts: int = Field(default=3, ge=1, le=11)
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)
    jitter_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_thinking: OllamaThinking = OllamaThinking.AUTO
    ollama_response_mode: OllamaResponseMode = OllamaResponseMode.JSON_SCHEMA
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_seed: int | None = Field(default=None, ge=0)
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_reasoning_effort: VLLMReasoningEffort = VLLMReasoningEffort.LOW
    vllm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    vllm_seed: int | None = Field(default=None, ge=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    proposal_target_channels: tuple[str, ...] = ()
    proposal_protected_parameter_names: tuple[str, ...] = ()
    cache_only: bool = False


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Construct the selected client without reading or logging API keys."""
    retry_options = {
        "max_attempts": config.max_attempts,
        "initial_backoff_seconds": config.initial_backoff_seconds,
        "max_backoff_seconds": config.max_backoff_seconds,
        "jitter_fraction": config.jitter_fraction,
        "proposal_target_channels": config.proposal_target_channels,
        "proposal_protected_parameter_names": (
            config.proposal_protected_parameter_names
        ),
        "cache_only": config.cache_only,
    }
    if config.provider is LLMProvider.OPENAI:
        from autoformalism.llm.openai_responses import OpenAIResponsesClient

        return OpenAIResponsesClient(
            model=config.model,
            cache_directory=config.cache_directory,
            log_path=config.log_path,
            max_output_tokens=config.max_output_tokens,
            **retry_options,
        )
    if config.provider is LLMProvider.GEMINI:
        from autoformalism.llm.gemini import GeminiClient

        return GeminiClient(
            model=config.model,
            cache_directory=config.cache_directory,
            log_path=config.log_path,
            timeout_seconds=config.timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            **retry_options,
        )
    if config.provider is LLMProvider.VLLM:
        from autoformalism.llm.vllm import VLLMClient

        return VLLMClient(
            model=config.model,
            cache_directory=config.cache_directory,
            log_path=config.log_path,
            base_url=config.vllm_base_url,
            reasoning_effort=config.vllm_reasoning_effort,
            temperature=config.vllm_temperature,
            seed=config.vllm_seed,
            timeout_seconds=config.timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            **retry_options,
        )
    from autoformalism.llm.ollama import OllamaClient

    return OllamaClient(
        model=config.model,
        cache_directory=config.cache_directory,
        log_path=config.log_path,
        base_url=config.ollama_base_url,
        thinking=config.ollama_thinking,
        response_mode=config.ollama_response_mode,
        temperature=config.ollama_temperature,
        seed=config.ollama_seed,
        timeout_seconds=config.timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        **retry_options,
    )
