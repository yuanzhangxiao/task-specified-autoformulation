"""Bounded frontier-agent baseline over public train/validation files."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import Field, ValidationError, model_validator

from autoformalism.data import BenchmarkSpec, DevelopmentDataset, TrainingScaler
from autoformalism.expressions import (
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.fitting import (
    EvaluationMetrics,
    FitConfig,
    FitResult,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.llm.gemini import _gemini_provider_schema
from autoformalism.schemas import (
    CandidateModel,
    ParameterScope,
    ProposerCandidateV2,
    enrich_proposal_v2,
)
from autoformalism.schemas.base import (
    FiniteFloat,
    Identifier,
    NonEmptyText,
    StrictSchema,
)


class RawAgentProvider(str, Enum):
    """Hosted providers supported by the raw-data agent baseline."""

    OPENAI = "openai"
    GEMINI = "gemini"


class RawAgentOutputContract(str, Enum):
    """Whether the hosted agent returns a structure or an instantiated model."""

    STRUCTURE_ONLY = "structure_only"
    FITTED_MODEL = "fitted_model"


class RawAgentFittedParameter(StrictSchema):
    """One named numeric value in the agent's final fitted model."""

    name: Identifier
    value: FiniteFloat


class RawAgentFittedModel(StrictSchema):
    """Complete agent-selected structure and its train-fitted parameter vector."""

    schema_version: Literal["1"] = "1"
    candidate: ProposerCandidateV2
    fitted_parameters: tuple[RawAgentFittedParameter, ...] = Field(max_length=256)
    fit_method_summary: NonEmptyText

    @model_validator(mode="after")
    def parameter_values_match_candidate(self) -> RawAgentFittedModel:
        """Require one finite value for every and only global parameter."""
        expected = {item.name for item in self.candidate.parameters}
        names = [item.name for item in self.fitted_parameters]
        if len(names) != len(set(names)):
            raise ValueError("fitted parameter names must be unique")
        actual = set(names)
        if actual != expected:
            raise ValueError(
                "fitted parameter names differ; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        values = {item.name: item.value for item in self.fitted_parameters}
        for item in self.candidate.parameters:
            if item.scope is not ParameterScope.GLOBAL:
                raise ValueError(
                    "full fitted-model contract supports only global parameters: "
                    f"{item.name}"
                )
            value = values[item.name]
            if (
                item.bounds is not None
                and not item.bounds.lower <= value <= item.bounds.upper
            ):
                raise ValueError(
                    f"fitted value for {item.name} lies outside declared bounds"
                )
        return self

    @property
    def fitted_parameter_values(self) -> dict[str, float]:
        """Return the validated vector in runtime mapping form."""
        return {item.name: float(item.value) for item in self.fitted_parameters}


class RawAgentConfig(StrictSchema):
    """Frozen provider and budget settings for one independent repetition."""

    provider: RawAgentProvider
    model: str = Field(min_length=1, max_length=256)
    repetition: int = Field(ge=0)
    reasoning_effort: str = Field(default="high", min_length=1, max_length=32)
    timeout_seconds: float = Field(default=1200.0, gt=0.0, le=3600.0)
    max_output_tokens: int = Field(default=30000, ge=1024, le=128000)
    max_tool_calls: int = Field(default=12, ge=1, le=64)
    max_attempts: int = Field(default=2, ge=1, le=4)
    output_contract: RawAgentOutputContract = RawAgentOutputContract.STRUCTURE_ONLY


class RawAgentUsage(StrictSchema):
    """Provider-reported token accounting, when available."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class RawAgentArtifact(StrictSchema):
    """Checkpointable result of one paid raw-data agent call."""

    schema_version: str = "raw-data-agent-result-2"
    request_hash: str
    provider: RawAgentProvider
    model: str
    repetition: int
    response_id: str | None = None
    latency_seconds: float = Field(ge=0.0)
    tool_call_count: int = Field(ge=0)
    requested_max_tool_calls: int | None = Field(default=None, ge=0)
    provider_reported_max_tool_calls: int | None = Field(default=None, ge=0)
    tool_call_limit_exceeded: bool = False
    usage: RawAgentUsage | None = None
    compact_candidate: ProposerCandidateV2
    candidate: CandidateModel
    output_contract: RawAgentOutputContract = RawAgentOutputContract.STRUCTURE_ONLY
    fitted_parameter_values: dict[str, FiniteFloat] | None = None
    fit_method_summary: NonEmptyText | None = None
    raw_response_sha256: str

    @model_validator(mode="after")
    def fitted_payload_matches_contract(self) -> RawAgentArtifact:
        """Keep old structure artifacts valid and fitted artifacts complete."""
        fitted = self.output_contract is RawAgentOutputContract.FITTED_MODEL
        if fitted != (self.fitted_parameter_values is not None):
            raise ValueError("fitted parameter values must match output contract")
        if fitted != (self.fit_method_summary is not None):
            raise ValueError("fit method summary must match output contract")
        if fitted:
            assert self.fitted_parameter_values is not None
            expected = {item.name for item in self.compact_candidate.parameters}
            actual = set(self.fitted_parameter_values)
            if actual != expected:
                raise ValueError(
                    "artifact fitted parameter names differ; "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            for item in self.compact_candidate.parameters:
                if item.scope is not ParameterScope.GLOBAL:
                    raise ValueError(
                        "full fitted-model artifact supports only global "
                        f"parameters: {item.name}"
                    )
                value = self.fitted_parameter_values[item.name]
                if (
                    item.bounds is not None
                    and not item.bounds.lower <= value <= item.bounds.upper
                ):
                    raise ValueError(
                        f"artifact fitted value for {item.name} lies outside bounds"
                    )
        return self


@dataclass(frozen=True)
class RawAgentInputs:
    """Public files and typed runtime contract shown to the agent."""

    benchmark_id: str
    tier: str
    public_prompt: str
    train_path: Path
    validation_path: Path
    targets: tuple[str, ...]
    auxiliaries: tuple[str, ...]
    external_inputs: tuple[str, ...]
    fixed_covariates: tuple[str, ...]
    lagged_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        for path in (self.train_path, self.validation_path):
            if not path.is_file():
                raise ValueError(f"agent input is not a file: {path}")
        if self.train_path.resolve() == self.validation_path.resolve():
            raise ValueError("agent train and validation paths must differ")
        if not self.public_prompt.strip():
            raise ValueError("public prompt must not be empty")
        if not self.targets:
            raise ValueError("at least one target is required")

    @property
    def file_hashes(self) -> dict[str, str]:
        """Return content hashes without storing raw tabular data in metadata."""
        return {
            "train.csv": _sha256_file(self.train_path),
            "validation.csv": _sha256_file(self.validation_path),
        }


@dataclass(frozen=True)
class _ProviderResult:
    parsed: ProposerCandidateV2
    raw_response: dict[str, object]
    response_id: str | None
    usage: RawAgentUsage | None
    tool_call_count: int
    fitted_parameter_values: dict[str, float] | None = None
    fit_method_summary: str | None = None


def provider_reported_max_tool_calls(
    raw_response: Mapping[str, object],
) -> int | None:
    """Return the provider-echoed built-in tool limit when it is available."""
    value = raw_response.get("max_tool_calls")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def raw_agent_validation_context(
    dataset: DevelopmentDataset,
    spec: BenchmarkSpec,
) -> ValidationContext:
    """Build the exact public runtime context used by raw-agent evaluation."""
    bounds = _raw_agent_forcing_bounds(
        dataset,
        include_targets=spec.one_step_target_history,
    )
    return ValidationContext(
        targets=dataset.roles.targets,
        auxiliaries=dataset.roles.auxiliaries,
        external_inputs=tuple(name for name in spec.external_inputs if name in bounds),
        fixed_covariates=tuple(
            name for name in spec.fixed_covariates if name in bounds
        ),
        lagged_targets=dataset.roles.targets if spec.one_step_target_history else (),
        forcing_bounds=bounds,
    )


def _raw_agent_forcing_bounds(
    dataset: DevelopmentDataset,
    *,
    include_targets: bool,
) -> dict[str, tuple[float, float]]:
    collected: dict[str, list[np.ndarray[Any, Any]]] = {}
    for split in (dataset.train, dataset.validation):
        for trajectory in split.trajectories:
            channels: dict[str, Any] = {
                **trajectory.auxiliaries,
                **trajectory.external_inputs,
                **trajectory.fixed_covariates,
            }
            if include_targets:
                channels.update(trajectory.targets)
            for name, raw in channels.items():
                try:
                    values = np.asarray(raw, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if len(values) and np.isfinite(values).all():
                    collected.setdefault(name, []).append(values)
    return {
        name: (
            float(min(np.min(values) for values in arrays)),
            float(max(np.max(values) for values in arrays)),
        )
        for name, arrays in collected.items()
    }


class _AgentAdapter(Protocol):
    def call(
        self,
        *,
        config: RawAgentConfig,
        inputs: RawAgentInputs,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult: ...


def _provider_schema(
    config: RawAgentConfig,
) -> type[ProposerCandidateV2] | type[RawAgentFittedModel]:
    return (
        RawAgentFittedModel
        if config.output_contract is RawAgentOutputContract.FITTED_MODEL
        else ProposerCandidateV2
    )


def _normalize_provider_output(
    parsed: object,
    config: RawAgentConfig,
) -> tuple[ProposerCandidateV2, dict[str, float] | None, str | None]:
    if config.output_contract is RawAgentOutputContract.FITTED_MODEL:
        if not isinstance(parsed, RawAgentFittedModel):
            raise RuntimeError("provider response contained no fitted model")
        return (
            parsed.candidate,
            dict(parsed.fitted_parameter_values),
            str(parsed.fit_method_summary),
        )
    if not isinstance(parsed, ProposerCandidateV2):
        raise RuntimeError("provider response contained no candidate")
    return parsed, None, None

    def repair(
        self,
        *,
        config: RawAgentConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult: ...


class OpenAIRawDataAgent:
    """OpenAI Responses adapter using a provider-hosted Python container."""

    def __init__(self, sdk_client: Any | None = None) -> None:
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(max_retries=0)
        self._client = sdk_client

    def call(
        self,
        *,
        config: RawAgentConfig,
        inputs: RawAgentInputs,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult:
        uploaded: list[Any] = []
        try:
            with ExitStack() as stack:
                for path in (inputs.train_path, inputs.validation_path):
                    handle = stack.enter_context(path.open("rb"))
                    uploaded.append(
                        self._client.files.create(file=handle, purpose="user_data")
                    )
            response = self._client.responses.parse(
                model=config.model,
                instructions=system_prompt,
                input=user_prompt,
                tools=[
                    {
                        "type": "code_interpreter",
                        "container": {
                            "type": "auto",
                            "file_ids": [item.id for item in uploaded],
                        },
                    }
                ],
                tool_choice="auto",
                max_tool_calls=config.max_tool_calls,
                max_output_tokens=config.max_output_tokens,
                reasoning={"effort": config.reasoning_effort},
                text_format=_provider_schema(config),
                store=False,
                timeout=config.timeout_seconds,
            )
        finally:
            for item in uploaded:
                with suppress(Exception):
                    self._client.files.delete(item.id)
        status = getattr(response, "status", None)
        if status not in (None, "completed"):
            raise RuntimeError(f"OpenAI response did not complete: {status}")
        parsed: object = getattr(response, "output_parsed", None)
        expected_schema = _provider_schema(config)
        if not isinstance(parsed, expected_schema):
            text = getattr(response, "output_text", None)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("OpenAI response contained no structured model")
            try:
                parsed = expected_schema.model_validate_json(text)
            except ValidationError as exc:
                raise RuntimeError(
                    f"OpenAI model response failed validation: {exc}"
                ) from exc
        candidate, fitted_values, fit_summary = _normalize_provider_output(
            parsed, config
        )
        raw = _raw_response(response)
        return _ProviderResult(
            parsed=candidate,
            raw_response=raw,
            response_id=_optional_text(getattr(response, "id", None)),
            usage=_openai_usage(response),
            tool_call_count=_openai_processed_tool_call_count(response),
            fitted_parameter_values=fitted_values,
            fit_method_summary=fit_summary,
        )

    def repair(
        self,
        *,
        config: RawAgentConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult:
        """Repair only the typed runtime contract, without data or tools."""
        response = self._client.responses.parse(
            model=config.model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=config.max_output_tokens,
            reasoning={"effort": config.reasoning_effort},
            text_format=_provider_schema(config),
            store=False,
            timeout=config.timeout_seconds,
        )
        status = getattr(response, "status", None)
        if status not in (None, "completed"):
            raise RuntimeError(f"OpenAI repair response did not complete: {status}")
        parsed = getattr(response, "output_parsed", None)
        expected_schema = _provider_schema(config)
        if not isinstance(parsed, expected_schema):
            text = getattr(response, "output_text", None)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("OpenAI repair response contained no model")
            try:
                parsed = expected_schema.model_validate_json(text)
            except ValidationError as exc:
                raise RuntimeError(
                    f"OpenAI repaired model failed validation: {exc}"
                ) from exc
        candidate, fitted_values, fit_summary = _normalize_provider_output(
            parsed, config
        )
        return _ProviderResult(
            parsed=candidate,
            raw_response=_raw_response(response),
            response_id=_optional_text(getattr(response, "id", None)),
            usage=_openai_usage(response),
            tool_call_count=0,
            fitted_parameter_values=fitted_values,
            fit_method_summary=fit_summary,
        )


class GeminiRawDataAgent:
    """Gemini adapter using Files API inputs and hosted code execution."""

    def __init__(
        self,
        sdk_client: Any | None = None,
        *,
        timeout_seconds: float = 1200.0,
    ) -> None:
        if sdk_client is None:
            from google import genai

            sdk_client = genai.Client(
                http_options={"timeout": int(timeout_seconds * 1000)}
            )
        self._client = sdk_client

    def call(
        self,
        *,
        config: RawAgentConfig,
        inputs: RawAgentInputs,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult:
        uploaded: list[Any] = []
        try:
            for path in (inputs.train_path, inputs.validation_path):
                uploaded.append(self._client.files.upload(file=str(path)))
            response = self._client.models.generate_content(
                model=config.model,
                contents=[user_prompt, *uploaded],
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": config.max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_json_schema": _gemini_provider_schema(
                        _provider_schema(config)
                    ),
                    "tools": [{"code_execution": {}}],
                    "seed": config.repetition,
                    "thinking_config": {
                        "thinking_level": config.reasoning_effort.upper()
                    },
                },
            )
        finally:
            for item in uploaded:
                name = getattr(item, "name", None)
                if not name:
                    continue
                with suppress(Exception):
                    self._client.files.delete(name=name)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini response contained no candidate")
        try:
            parsed = _provider_schema(config).model_validate_json(text)
        except ValidationError as exc:
            raise RuntimeError(f"Gemini model failed validation: {exc}") from exc
        candidate, fitted_values, fit_summary = _normalize_provider_output(
            parsed, config
        )
        return _ProviderResult(
            parsed=candidate,
            raw_response=_raw_response(response),
            response_id=_optional_text(
                getattr(response, "response_id", None)
                or getattr(response, "id", None)
            ),
            usage=_gemini_usage(response),
            tool_call_count=_gemini_tool_call_count(response),
            fitted_parameter_values=fitted_values,
            fit_method_summary=fit_summary,
        )

    def repair(
        self,
        *,
        config: RawAgentConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> _ProviderResult:
        """Repair only the typed runtime contract, without data or tools."""
        response = self._client.models.generate_content(
            model=config.model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": config.max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": _gemini_provider_schema(
                    _provider_schema(config)
                ),
                "seed": config.repetition + 10_000,
                "thinking_config": {
                    "thinking_level": config.reasoning_effort.upper()
                },
            },
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini repair response contained no candidate")
        try:
            parsed = _provider_schema(config).model_validate_json(text)
        except ValidationError as exc:
            raise RuntimeError(
                f"Gemini repaired model failed validation: {exc}"
            ) from exc
        candidate, fitted_values, fit_summary = _normalize_provider_output(
            parsed, config
        )
        return _ProviderResult(
            parsed=candidate,
            raw_response=_raw_response(response),
            response_id=_optional_text(
                getattr(response, "response_id", None)
                or getattr(response, "id", None)
            ),
            usage=_gemini_usage(response),
            tool_call_count=0,
            fitted_parameter_values=fitted_values,
            fit_method_summary=fit_summary,
        )


def run_raw_data_agent(
    *,
    config: RawAgentConfig,
    inputs: RawAgentInputs,
    output_directory: Path,
    adapter: _AgentAdapter | None = None,
) -> RawAgentArtifact:
    """Run or restore one bounded, content-addressed raw-data agent call."""
    output_directory.mkdir(parents=True, exist_ok=True)
    system_prompt = raw_agent_system_prompt(inputs, config.output_contract)
    user_prompt = raw_agent_user_prompt(inputs, config.repetition)
    request_hash = raw_agent_request_hash(config, inputs, system_prompt, user_prompt)
    cache_path = output_directory / "cache" / f"{request_hash}.json"
    artifact_path = output_directory / "agent_result.json"
    if artifact_path.is_file():
        existing = RawAgentArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        if existing.request_hash != request_hash:
            raise ValueError("agent checkpoint does not match the requested run")
        return existing
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        artifact = RawAgentArtifact.model_validate(cached["artifact"])
        _atomic_json(artifact_path, artifact.model_dump(mode="json"))
        _append_event(
            output_directory / "events.jsonl",
            {"event": "raw_agent_cache_hit", "request_hash": request_hash},
        )
        return artifact

    selected_adapter = adapter or _adapter(config)
    started = time.monotonic()
    error: Exception | None = None
    provider_result: _ProviderResult | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            provider_result = selected_adapter.call(
                config=config,
                inputs=inputs,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            break
        except Exception as exc:
            error = exc
            _append_event(
                output_directory / "events.jsonl",
                {
                    "event": "raw_agent_failure",
                    "request_hash": request_hash,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": _redacted_error(exc),
                },
            )
            if attempt < config.max_attempts:
                time.sleep(min(2**attempt, 10))
    if provider_result is None:
        assert error is not None
        raise error

    candidate = enrich_proposal_v2(provider_result.parsed, inputs.targets)
    raw_encoded = json.dumps(
        provider_result.raw_response, sort_keys=True, default=str
    ).encode("utf-8")
    artifact = RawAgentArtifact(
        request_hash=request_hash,
        provider=config.provider,
        model=config.model,
        repetition=config.repetition,
        response_id=provider_result.response_id,
        latency_seconds=time.monotonic() - started,
        tool_call_count=provider_result.tool_call_count,
        requested_max_tool_calls=config.max_tool_calls,
        provider_reported_max_tool_calls=provider_reported_max_tool_calls(
            provider_result.raw_response
        ),
        tool_call_limit_exceeded=(
            provider_result.tool_call_count > config.max_tool_calls
        ),
        usage=provider_result.usage,
        compact_candidate=provider_result.parsed,
        candidate=candidate,
        output_contract=config.output_contract,
        fitted_parameter_values=getattr(
            provider_result, "fitted_parameter_values", None
        ),
        fit_method_summary=getattr(provider_result, "fit_method_summary", None),
        raw_response_sha256=hashlib.sha256(raw_encoded).hexdigest(),
    )
    cache_payload = {
        "schema_version": "raw-data-agent-cache-1",
        "artifact": artifact.model_dump(mode="json"),
        "raw_response": provider_result.raw_response,
    }
    _atomic_json(cache_path, cache_payload)
    _atomic_json(artifact_path, artifact.model_dump(mode="json"))
    _append_event(
        output_directory / "events.jsonl",
        {
            "event": "raw_agent_response",
            "request_hash": request_hash,
            "provider": config.provider.value,
            "model": config.model,
            "tool_call_count": provider_result.tool_call_count,
            "requested_max_tool_calls": config.max_tool_calls,
            "provider_reported_max_tool_calls": (
                artifact.provider_reported_max_tool_calls
            ),
            "tool_call_limit_exceeded": artifact.tool_call_limit_exceeded,
            "usage": (
                None
                if provider_result.usage is None
                else provider_result.usage.model_dump(mode="json")
            ),
        },
    )
    return artifact


def repair_raw_data_agent_candidate(
    *,
    config: RawAgentConfig,
    inputs: RawAgentInputs,
    original: RawAgentArtifact,
    diagnostics: str,
    repair_index: int,
    output_directory: Path,
    adapter: _AgentAdapter | None = None,
) -> RawAgentArtifact:
    """Run or restore one diagnostics-only candidate contract repair."""
    if repair_index < 1:
        raise ValueError("repair_index must be positive")
    system_prompt = (
        "Repair a proposed scientific model only enough to satisfy its typed "
        "syntax and deterministic runtime contract. Preserve its scientific "
        "structure, equations, mechanisms, target mappings, and parameter "
        "bounds and fitted parameter values whenever the diagnostic does not "
        "require a change. Do not add "
        "new scientific mechanisms. Latent initial expressions may use only "
        "the explicitly supplied forcing channels and time; they may never use "
        "parameters, other latent states, or unavailable observations. If such "
        "an initializer cannot be made causal without changing the science, "
        "replace it with a finite fixed_value and remove declarations made "
        "unused by that repair. "
        + (
            "Return exactly one RawAgentFittedModel with a value for every "
            "remaining parameter."
            if config.output_contract is RawAgentOutputContract.FITTED_MODEL
            else "Return exactly one ProposerCandidateV2."
        )
    )
    user_prompt = (
        "Public symbol contract:\n"
        f"targets={list(inputs.targets)}\n"
        f"auxiliaries={list(inputs.auxiliaries)}\n"
        f"external_inputs={list(inputs.external_inputs)}\n"
        f"fixed_covariates={list(inputs.fixed_covariates)}\n"
        f"lagged_targets={list(inputs.lagged_targets)}\n\n"
        "Deterministic validator diagnostics:\n"
        f"{diagnostics[:8000]}\n\n"
        "Model to repair:\n"
        + json.dumps(
            (
                {
                    "schema_version": "1",
                    "candidate": original.compact_candidate.model_dump(mode="json"),
                    "fitted_parameters": [
                        {"name": name, "value": value}
                        for name, value in (
                            original.fitted_parameter_values or {}
                        ).items()
                    ],
                    "fit_method_summary": original.fit_method_summary,
                }
                if original.output_contract is RawAgentOutputContract.FITTED_MODEL
                else original.compact_candidate.model_dump(mode="json")
            ),
            indent=2,
            sort_keys=True,
        )
    )
    request_payload = {
        "schema_version": "raw-data-agent-repair-request-1",
        "config": config.model_dump(mode="json"),
        "original_request_hash": original.request_hash,
        "repair_index": repair_index,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_schema": _provider_schema(config).model_json_schema(
            mode="validation"
        ),
    }
    request_hash = hashlib.sha256(
        json.dumps(
            request_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    legacy_artifact_path = (
        output_directory / f"repair_result_{repair_index:02d}.json"
    )
    artifact_path = output_directory / (
        f"repair_result_{repair_index:02d}_{request_hash[:16]}.json"
    )
    cache_path = output_directory / "cache" / f"{request_hash}.json"
    if legacy_artifact_path.is_file():
        restored = RawAgentArtifact.model_validate_json(
            legacy_artifact_path.read_text(encoding="utf-8")
        )
        if restored.request_hash == request_hash:
            return restored
    if artifact_path.is_file():
        restored = RawAgentArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        if restored.request_hash != request_hash:
            raise ValueError("repair checkpoint does not match the requested repair")
        return restored
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        restored = RawAgentArtifact.model_validate(payload["artifact"])
        _atomic_json(artifact_path, restored.model_dump(mode="json"))
        return restored

    selected_adapter = adapter or _adapter(config)
    started = time.monotonic()
    result = selected_adapter.repair(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    candidate = enrich_proposal_v2(result.parsed, inputs.targets)
    raw_encoded = json.dumps(
        result.raw_response, sort_keys=True, default=str
    ).encode("utf-8")
    artifact = RawAgentArtifact(
        request_hash=request_hash,
        provider=config.provider,
        model=config.model,
        repetition=config.repetition,
        response_id=result.response_id,
        latency_seconds=time.monotonic() - started,
        tool_call_count=0,
        requested_max_tool_calls=0,
        tool_call_limit_exceeded=False,
        usage=result.usage,
        compact_candidate=result.parsed,
        candidate=candidate,
        output_contract=config.output_contract,
        fitted_parameter_values=getattr(result, "fitted_parameter_values", None),
        fit_method_summary=getattr(result, "fit_method_summary", None),
        raw_response_sha256=hashlib.sha256(raw_encoded).hexdigest(),
    )
    _atomic_json(
        cache_path,
        {
            "schema_version": "raw-data-agent-repair-cache-1",
            "artifact": artifact.model_dump(mode="json"),
            "raw_response": result.raw_response,
        },
    )
    _atomic_json(artifact_path, artifact.model_dump(mode="json"))
    if not legacy_artifact_path.exists():
        _atomic_json(
            legacy_artifact_path,
            artifact.model_dump(mode="json"),
        )
    _append_event(
        output_directory / "events.jsonl",
        {
            "event": "raw_agent_contract_repair",
            "request_hash": request_hash,
            "repair_index": repair_index,
            "diagnostics_sha256": hashlib.sha256(
                diagnostics.encode("utf-8")
            ).hexdigest(),
        },
    )
    return artifact


def evaluate_raw_agent_candidate(
    *,
    artifact: RawAgentArtifact,
    dataset: DevelopmentDataset,
    context: ValidationContext,
    fit_config: FitConfig,
) -> tuple[CandidateModel, tuple[str, ...], FitResult, tuple[dict[str, str], ...]]:
    """Compile and refit an agent structure without pruning or test access."""
    candidate, repairs = repair_protected_declarations(artifact.candidate, context)
    compiled = compile_candidate(candidate, context)
    fit = fit_candidate(compiled, dataset.train, dataset.validation, fit_config)
    warnings = tuple(
        {
            "code": item.code,
            "location": item.location,
            "message": item.message,
        }
        for item in compiled.validated.warnings
    )
    return candidate, repairs, fit, warnings


def evaluate_raw_agent_fitted_model(
    *,
    artifact: RawAgentArtifact,
    dataset: DevelopmentDataset,
    context: ValidationContext,
    simulation_config: FitConfig | None = None,
) -> tuple[
    CandidateModel,
    tuple[str, ...],
    EvaluationMetrics,
    EvaluationMetrics,
    tuple[dict[str, str], ...],
]:
    """Simulate the agent's exact fitted model without parameter optimization."""
    if artifact.output_contract is not RawAgentOutputContract.FITTED_MODEL:
        raise ValueError("artifact does not contain a fitted model")
    if artifact.fitted_parameter_values is None:  # defensive model invariant.
        raise ValueError("fitted model has no parameter values")
    candidate, repairs = repair_protected_declarations(artifact.candidate, context)
    compiled = compile_candidate(candidate, context)
    expected = {item.name for item in compiled.validated.candidate.parameters}
    actual = set(artifact.fitted_parameter_values)
    if not expected.issubset(actual):
        raise ValueError(
            "compiled parameter names are missing fitted values; "
            f"missing={sorted(expected - actual)}"
        )
    settings = simulation_config or FitConfig(
        integration_backend="solve_ivp",
        allow_derivative_regression=False,
    )
    scaler = TrainingScaler().fit(dataset.train)
    target_scales = {
        channel: scaler.scales[f"target:{channel}"].standard_deviation
        for channel in context.targets
    }
    # Lossless declaration repair may discard a supplied but unused parameter.
    # Never pass such extras into the executable model; the repair ledger retains
    # their provenance while this mapping is exactly the vector being simulated.
    parameters = {
        name: artifact.fitted_parameter_values[name] for name in sorted(expected)
    }
    _, training_metrics = evaluate_fitted_candidate(
        compiled,
        dataset.train,
        global_parameters=parameters,
        global_initial_conditions={},
        target_scales=target_scales,
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    _, validation_metrics = evaluate_fitted_candidate(
        compiled,
        dataset.validation,
        global_parameters=parameters,
        global_initial_conditions={},
        target_scales=target_scales,
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    warnings = tuple(
        {
            "code": item.code,
            "location": item.location,
            "message": item.message,
        }
        for item in compiled.validated.warnings
    )
    return candidate, repairs, training_metrics, validation_metrics, warnings


def evaluation_metrics_payload(metrics: EvaluationMetrics) -> dict[str, object]:
    """Convert fixed-model rollout metrics into a compact JSON artifact."""
    return {
        "normalized_mse": metrics.normalized_mse,
        "per_target_normalized_mse": dict(metrics.per_target_normalized_mse),
        "failed_trajectories": list(metrics.failed_trajectories),
        "soft_constraint_violations": {
            key: dict(value)
            for key, value in metrics.soft_constraint_violations.items()
        },
    }


def fit_result_payload(fit: FitResult) -> dict[str, object]:
    """Convert a fit result into a compact JSON artifact."""
    return {
        "success": fit.success,
        "message": fit.message,
        "global_parameters": dict(fit.global_parameters),
        "global_initial_conditions": dict(fit.global_initial_conditions),
        "training_normalized_mse": fit.training_metrics.normalized_mse,
        "validation_normalized_mse": fit.validation_metrics.normalized_mse,
        "training_per_target_normalized_mse": dict(
            fit.training_metrics.per_target_normalized_mse
        ),
        "validation_per_target_normalized_mse": dict(
            fit.validation_metrics.per_target_normalized_mse
        ),
        "training_failed_trajectories": list(
            fit.training_metrics.failed_trajectories
        ),
        "validation_failed_trajectories": list(
            fit.validation_metrics.failed_trajectories
        ),
        "diagnostics": [asdict(item) for item in fit.diagnostics],
        "best_start_index": fit.best_start_index,
        "target_scales": dict(fit.target_scales),
    }


def raw_agent_system_prompt(
    inputs: RawAgentInputs,
    output_contract: RawAgentOutputContract = RawAgentOutputContract.STRUCTURE_ONLY,
) -> str:
    """Render the provider-neutral, benchmark-general agent instructions."""
    del inputs
    fitted_instruction = (
        " Fit every continuous parameter using train only. Return the fitted "
        "numeric value for every parameter, along with a concise description "
        "of the fitting procedure, inside exactly one RawAgentFittedModel. "
        "Validation may guide structure selection but must not be used to fit "
        "continuous values. The returned parameter vector is the final model: "
        "the evaluator will simulate it without parameter optimization."
        if output_contract is RawAgentOutputContract.FITTED_MODEL
        else " Return exactly one schema-valid ProposerCandidateV2."
    )
    return (
        "You are the complete frontier-agent baseline for scientific model "
        "discovery. Use the hosted python tool to inspect the attached public "
        "train.csv and validation.csv files, plot or summarize trajectories, "
        "fit and compare candidate structures. The public task prompt is "
        "authoritative. Train may be used to fit continuous parameters; "
        "validation may be used to select structure. No test data, private "
        "equations, web search, or external data are available. Do not encode "
        "data as lookup tables, per-time parameters, residual series, or future "
        "target values. The final candidate must be an explicit causal "
        "continuous-time model using only safe analytic expressions. Every "
        "state requires one rhs. Observed states require observed_channel and "
        "omit initial; latent states omit observed_channel and require exactly "
        "one fixed_value or causal analytic initialization expression. Put "
        "instantaneous formulas in algebraics. Every parameter must have finite "
        "lower/upper bounds and must be used. Target mappings are inferred from "
        "an observed_channel or same-named state/algebraic, so every target must "
        "match exactly one such component. Use ** for powers; supported common "
        "functions include exp, log, sqrt, tanh, sin, cos, abs, min, and max. "
        "Do not return analysis outside the structured response."
        + fitted_instruction
    )


def raw_agent_user_prompt(inputs: RawAgentInputs, repetition: int) -> str:
    """Combine public task text, split identities, and the runtime symbol contract."""
    def names(values: tuple[str, ...]) -> str:
        return ", ".join(values) if values else "(none)"

    return "\n\n".join(
        (
            f"Independent repetition ID: {repetition}. It has no scientific meaning.",
            "Attached files:\n- train.csv: development training split\n"
            "- validation.csv: development validation split",
            "Exact runtime symbol contract:\n"
            f"- target channels to generate: {names(inputs.targets)}\n"
            f"- supplied auxiliaries: {names(inputs.auxiliaries)}\n"
            f"- external inputs: {names(inputs.external_inputs)}\n"
            f"- fixed numeric covariates: {names(inputs.fixed_covariates)}\n"
            f"- causally available lagged targets: {names(inputs.lagged_targets)}\n"
            "- time symbol: t\n"
            "Only supplied auxiliaries, external inputs, fixed covariates, and "
            "explicitly listed lagged targets may appear as undeclared symbols.",
            "Public benchmark prompt:\n" + inputs.public_prompt,
            "Use the python tool before returning the final structured candidate.",
        )
    )


def raw_agent_request_hash(
    config: RawAgentConfig,
    inputs: RawAgentInputs,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Hash every semantic request input without embedding raw tables."""
    payload = {
        "schema_version": "raw-data-agent-request-1",
        "config": config.model_dump(mode="json"),
        "benchmark_id": inputs.benchmark_id,
        "tier": inputs.tier,
        "file_hashes": inputs.file_hashes,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_schema": ProposerCandidateV2.model_json_schema(mode="validation"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _adapter(config: RawAgentConfig) -> _AgentAdapter:
    if config.provider is RawAgentProvider.OPENAI:
        return OpenAIRawDataAgent()
    return GeminiRawDataAgent(timeout_seconds=config.timeout_seconds)


def _openai_usage(response: Any) -> RawAgentUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return RawAgentUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _gemini_usage(response: Any) -> RawAgentUsage | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return RawAgentUsage(
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        total_tokens=getattr(usage, "total_token_count", None),
    )


def _openai_processed_tool_call_count(response: Any) -> int:
    """Count processed calls while excluding nonterminal response records."""
    nonterminal = {"in_progress", "interpreting"}
    return sum(
        getattr(item, "type", None) == "code_interpreter_call"
        and getattr(item, "status", None) not in nonterminal
        for item in (getattr(response, "output", None) or [])
    )


def _gemini_tool_call_count(response: Any) -> int:
    count = 0
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "executable_code", None) is not None:
                count += 1
    return count


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
        "text": getattr(response, "text", None),
    }


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _redacted_error(error: Exception) -> str:
    rendered = str(error)
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
    return rendered[:4000]
