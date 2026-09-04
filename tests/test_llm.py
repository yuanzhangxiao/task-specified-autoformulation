"""Offline tests for the provider-neutral LLM abstraction."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.base import CachedLLMClient, ProviderResponse
from autoformalism.llm.config import (
    LLMConfig,
    LLMProvider,
    OllamaResponseMode,
    OllamaThinking,
    create_llm_client,
)
from autoformalism.llm.exceptions import (
    LLMCacheMissError,
    LLMProviderError,
    LLMResponseError,
    RepairDiagnosticCode,
)
from autoformalism.llm.gemini import (
    GeminiClient,
    _gemini_compatible_schema,
    _gemini_provider_schema,
)
from autoformalism.llm.mock import MockLLMClient
from autoformalism.llm.models import StructuredT, TokenUsage
from autoformalism.llm.ollama import OllamaClient, _ollama_compatible_schema
from autoformalism.llm.openai_responses import OpenAIResponsesClient
from autoformalism.schemas import (
    AbsoluteCriterion,
    AtomicJudgeResult,
    CandidateModel,
    HybridJudgeResult,
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
    ProposerCandidateV2,
    ScientificJudgeResult,
    enrich_proposal_v2,
)


def _proposal() -> ProposerCandidateV2:
    return ProposerCandidateV2.model_validate(
        {
            "schema_version": "2",
            "candidate_id": "candidate_1",
            "change_summary": "Initial candidate.",
            "states": [
                {
                    "name": "x",
                    "kind": "latent",
                    "rhs": "-k * x",
                    "initial": {"expression": "target"},
                }
            ],
            "algebraics": [],
            "parameters": [
                {
                    "name": "k",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
            ],
        }
    )


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "schema_version": "1",
            "candidate_id": "candidate_1",
            "parent_candidate_id": None,
            "change_summary": "Initial candidate.",
            "states": [
                {
                    "name": "x",
                    "kind": "latent",
                    "unit": "relative",
                    "description": "Latent response.",
                }
            ],
            "processes": [],
            "state_equations": [{"state": "x", "rhs": "-k * x"}],
            "observation_mappings": [
                {"channel": "target", "expression": "x", "unit": "relative"}
            ],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                    "unit": "1/time",
                    "description": "Decay rate.",
                }
            ],
            "initial_conditions": [
                {
                    "state": "x",
                    "scope": "trajectory_specific",
                    "initialization_range": {"lower": 0.0, "upper": 2.0},
                }
            ],
            "constraints": [],
        }
    )


def _judge() -> ScientificJudgeResult:
    return ScientificJudgeResult.model_validate(
        {
            "schema_version": "2",
            "hard_red_flags": [],
            "category_scores": {
                "mechanistic_coherence": 1.0,
                "source_sink_balance_semantics": 1.0,
                "dynamic_plausibility": 1.0,
                "mechanism_coupling_task_sufficiency": 1.0,
                "nonredundancy_accounting": 1.0,
                "latent_state_complexity_justification": 1.0,
            },
            "missing_requirements": [],
            "actionable_edits": [],
        }
    )


def _hybrid_judge() -> HybridJudgeResult:
    return HybridJudgeResult.model_validate(
        {
            "schema_version": "hybrid-1",
            "absolute_assessments": [
                {
                    "criterion": "source_roles_consistent",
                    "subject_id": "candidate",
                    "candidate_a": {
                        "verdict": "pass",
                        "evidence": "Candidate A source evidence.",
                    },
                    "candidate_b": {
                        "verdict": "fail",
                        "evidence": "Candidate B source evidence.",
                    },
                }
            ],
            "comparative_assessments": [
                {
                    "criterion": criterion,
                    "verdict": "candidate_a",
                    "evidence": "Direct comparative evidence.",
                }
                for criterion in (
                    "parsimony_while_task_sufficient",
                    "fewer_unsupported_assumptions",
                    "mechanistic_interpretability",
                )
            ],
        }
    )


class StubCachedClient(CachedLLMClient):
    """Test adapter with controlled failures and responses."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        failures: int = 0,
        sleep: Any = lambda _delay: None,
        random_value: Any = lambda: 0.0,
        cache_only: bool = False,
        proposal_target_channels: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            provider_name="stub",
            model="stub-model",
            cache_directory=tmp_path / "cache",
            log_path=tmp_path / "events.jsonl",
            max_attempts=3,
            initial_backoff_seconds=1.0,
            jitter_fraction=0.5,
            sleep=sleep,
            random_value=random_value,
            cache_only=cache_only,
            proposal_target_channels=proposal_target_channels,
        )
        self.failures = failures
        self.provider_calls = 0
        self.provider_attempt_numbers: list[int] = []

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
        del system_prompt, user_prompt, repair_diagnostic_codes
        self.provider_calls += 1
        self.provider_attempt_numbers.append(attempt_number)
        if self.provider_calls <= self.failures:
            raise LLMProviderError("temporary failure", retryable=True)
        parsed = _proposal() if role == "proposer" else _judge()
        validated = response_model.model_validate(parsed.model_dump(mode="json"))
        return ProviderResponse(
            parsed=validated,
            raw_response={
                "raw": parsed.model_dump(mode="json"),
                "api_key": "SECRET_SHOULD_NOT_BE_LOGGED",
            },
            usage=TokenUsage(10, 20, 30),
            latency_ms=12.5,
        )


def test_mock_has_separate_proposer_and_judge_methods() -> None:
    client = MockLLMClient(
        proposer_responses=[_candidate()],
        judge_responses=[_judge()],
    )

    candidate = client.propose(system_prompt="system", user_prompt="candidate")
    judge = client.judge(system_prompt="system", user_prompt="judge")

    assert candidate.parsed.candidate_id == "candidate_1"
    assert judge.parsed.aggregate_score == 1.0
    assert [call["role"] for call in client.calls] == ["proposer", "judge"]


def test_cached_client_has_separate_staged_proposer_calls(tmp_path: Path) -> None:
    topology_proposal = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "graph_0",
            "states": [{"name": "x"}],
            "interactions": [
                {
                    "interaction_id": "decay",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                    "polarity": "subtractive",
                }
            ],
            "observation_mappings": [{"channel": "target", "source": "x"}],
        }
    )
    functional_proposal = ProposedFunctionalCandidate.model_validate(
        {
            "candidate_id": "functions_0",
            "interaction_functions": [
                {"interaction_id": "decay", "expression": "rate * x"}
            ],
            "parameters": [{"name": "rate", "role": "rate"}],
        }
    )

    class StagedClient(StubCachedClient):
        def _call_provider(self, **kwargs: Any) -> ProviderResponse[Any]:
            self.provider_calls += 1
            response = (
                topology_proposal
                if kwargs["role"] == "staged_topology_proposer_v1"
                else functional_proposal
            )
            return ProviderResponse(parsed=response, raw_response={})

    context = ValidationContext(targets=("target",))
    client = StagedClient(tmp_path)
    topology = client.propose_topology(
        system_prompt="topology system",
        user_prompt="topology request",
        context=context,
    )
    functional = client.propose_functions(
        system_prompt="functional system",
        user_prompt="functional request",
        topology=topology.parsed,
        context=context,
    )
    cached_topology = client.propose_topology(
        system_prompt="topology system",
        user_prompt="topology request",
        context=context,
    )

    assert client.provider_calls == 2
    assert topology.parsed.states[0].kind.value == "observed"
    assert functional.parsed.parameters[0].scope.value == "global"
    assert cached_topology.cache_hit is True


def test_cached_hybrid_client_repairs_only_redundant_atomic_role_units(
    tmp_path: Path,
) -> None:
    payload = _hybrid_judge().model_dump(mode="json")
    payload["absolute_assessments"].extend(
        [
            {
                "criterion": "sink_roles_consistent",
                "subject_id": "candidate",
                "candidate_a": {
                    "verdict": "pass",
                    "evidence": "Redundant sink evidence A.",
                },
                "candidate_b": {
                    "verdict": "pass",
                    "evidence": "Redundant sink evidence B.",
                },
            },
            {
                "criterion": "proposer_claims_supported",
                "subject_id": "candidate",
                "candidate_a": {
                    "verdict": "pass",
                    "evidence": "Requested claim evidence A.",
                },
                "candidate_b": {
                    "verdict": "pass",
                    "evidence": "Requested claim evidence B.",
                },
            },
        ]
    )
    response = HybridJudgeResult.model_validate(payload)

    class HybridClient(StubCachedClient):
        def _call_provider(self, **kwargs: Any) -> ProviderResponse[Any]:
            self.provider_calls += 1
            assert kwargs["role"] == "hybrid_judge_atomic_repair_v1"
            return ProviderResponse(parsed=response, raw_response={"provider": True})

    client = HybridClient(tmp_path)
    redundant = {
        (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate"),
        (AbsoluteCriterion.SINK_ROLES_CONSISTENT, "candidate"),
    }
    expected = {(AbsoluteCriterion.PROPOSER_CLAIMS_SUPPORTED, "candidate")}

    first = client.assess_hybrid(
        system_prompt="system",
        user_prompt="atomic hybrid request",
        expected_absolute_units=expected,
        redundant_absolute_units=redundant,
    )
    second = client.assess_hybrid(
        system_prompt="system",
        user_prompt="atomic hybrid request",
        expected_absolute_units=expected,
        redundant_absolute_units=redundant,
    )

    assert client.provider_calls == 1
    assert first.attempts == 1
    assert second.cache_hit is True
    assert {
        (item.criterion, item.subject_id)
        for item in first.parsed.absolute_assessments
    } == expected
    assert first.raw_response["_autoformalism_contract_repair"] == {
        "redundant_absolute_units_removed": [
            "sink_roles_consistent:candidate",
            "source_roles_consistent:candidate",
        ],
        "redundant_absolute_unit_repair_count": 2,
    }


def test_cached_atomic_client_repairs_missing_units_only_after_retries(
    tmp_path: Path,
) -> None:
    empty = AtomicJudgeResult.model_validate(
        {
            "signed_occurrence_assessments": [],
            "repeated_contribution_assessments": [],
        }
    )

    class AtomicClient(StubCachedClient):
        def _call_provider(self, **kwargs: Any) -> ProviderResponse[Any]:
            self.provider_calls += 1
            self.provider_attempt_numbers.append(kwargs["attempt_number"])
            assert kwargs["role"] == (
                "atomic_evidence_judge_missing_unit_repair_v1"
            )
            return ProviderResponse(parsed=empty, raw_response={"provider": True})

    client = AtomicClient(tmp_path)

    result = client.assess_atomic_evidence(
        system_prompt="system",
        user_prompt="atomic",
        expected_occurrence_ids={"occurrence_a"},
        expected_repeat_pair_ids=set(),
        repair_missing_units=True,
    )

    assert client.provider_calls == 3
    assert client.provider_attempt_numbers == [1, 2, 3]
    assert result.attempts == 3
    assert result.parsed.signed_occurrence_assessments[0].expected_direction.value == (
        "insufficient_public_information"
    )
    assert result.raw_response["_autoformalism_contract_repair"][
        "missing_occurrence_repair_count"
    ] == 1


def test_request_hash_is_stable_and_materially_sensitive(tmp_path: Path) -> None:
    client = StubCachedClient(tmp_path)

    first = client.request_hash(
        role="proposer",
        system_prompt="system",
        user_prompt="user",
        response_model=CandidateModel,
    )
    again = client.request_hash(
        role="proposer",
        system_prompt="system",
        user_prompt="user",
        response_model=CandidateModel,
    )
    changed = client.request_hash(
        role="proposer",
        system_prompt="system",
        user_prompt="different",
        response_model=CandidateModel,
    )

    assert first == again
    assert first != changed


def test_embedded_json_repair_rejects_ambiguous_valid_objects(
    tmp_path: Path,
) -> None:
    client = StubCachedClient(tmp_path)
    encoded = _judge().model_dump_json()

    parsed = client._parse_single_embedded_json(
        f"Final object follows:\n{encoded}", ScientificJudgeResult
    )
    assert parsed == _judge()
    with pytest.raises(LLMResponseError, match="matches=2"):
        client._parse_single_embedded_json(
            f"{encoded}\n{encoded}", ScientificJudgeResult
        )


def test_disk_cache_avoids_second_provider_call_and_logs_raw_parsed(
    tmp_path: Path,
) -> None:
    first_client = StubCachedClient(tmp_path)
    first = first_client.propose(system_prompt="system", user_prompt="user")
    second_client = StubCachedClient(tmp_path)
    second = second_client.propose(system_prompt="system", user_prompt="user")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second_client.provider_calls == 0
    assert second.parsed == first.parsed
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["raw_response"]["api_key"] == "[REDACTED]"
    assert events[0]["parsed_response"]["candidate_id"] == "candidate_1"
    assert events[0]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    assert events[0]["latency_ms"] == 12.5
    assert "SECRET_SHOULD_NOT_BE_LOGGED" not in (tmp_path / "events.jsonl").read_text(
        encoding="utf-8"
    )


def test_cache_only_client_hits_cache_without_provider_call(tmp_path: Path) -> None:
    populated = StubCachedClient(tmp_path)
    expected = populated.propose(system_prompt="system", user_prompt="user")
    replay = StubCachedClient(tmp_path, cache_only=True)

    restored = replay.propose(system_prompt="system", user_prompt="user")

    assert restored.parsed == expected.parsed
    assert restored.cache_hit is True
    assert replay.provider_calls == 0


def test_cache_only_client_fails_closed_on_miss(tmp_path: Path) -> None:
    client = StubCachedClient(tmp_path, cache_only=True)

    with pytest.raises(LLMCacheMissError, match="cache-only LLM request"):
        client.propose(system_prompt="system", user_prompt="missing")

    assert client.provider_calls == 0
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "llm_cache_miss"


def test_per_call_cache_only_fails_closed_without_changing_client(
    tmp_path: Path,
) -> None:
    client = StubCachedClient(tmp_path)

    with pytest.raises(LLMCacheMissError, match="cache-only LLM request"):
        client.propose(
            system_prompt="system",
            user_prompt="missing",
            cache_only=True,
        )

    assert client.provider_calls == 0
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "llm_cache_miss"
    assert event["per_call_cache_only"] is True


def test_per_call_cache_only_restores_populated_entry(tmp_path: Path) -> None:
    populated = StubCachedClient(tmp_path)
    expected = populated.propose(system_prompt="system", user_prompt="user")
    replay = StubCachedClient(tmp_path)

    restored = replay.propose(
        system_prompt="system",
        user_prompt="user",
        cache_only=True,
    )

    assert restored.parsed == expected.parsed
    assert restored.cache_hit is True
    assert replay.provider_calls == 0


def test_cache_only_factory_does_not_require_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_llm_client(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="offline-model",
            cache_directory=tmp_path / "cache",
            log_path=tmp_path / "events.jsonl",
            cache_only=True,
        )
    )

    assert isinstance(client, OpenAIResponsesClient)


def test_retry_uses_exponential_backoff_and_jitter(tmp_path: Path) -> None:
    delays: list[float] = []
    client = StubCachedClient(
        tmp_path,
        failures=2,
        sleep=delays.append,
        random_value=lambda: 0.5,
    )

    result = client.propose(system_prompt="system", user_prompt="user")

    assert result.attempts == 3
    assert result.logical_calls == 1
    assert result.provider_attempts == 3
    assert result.repair_attempts == 2
    assert client.provider_attempt_numbers == [1, 2, 3]
    assert delays == [1.25, 2.5]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "llm_failure",
        "llm_failure",
        "llm_response",
    ]


def test_invalid_structured_response_retries_with_repair_feedback(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    class RepairClient(StubCachedClient):
        def _call_provider(self, **kwargs: Any) -> ProviderResponse[Any]:
            prompts.append(kwargs["user_prompt"])
            if len(prompts) == 1:
                raise LLMResponseError(
                    "duplicate declaration body_weight_kg",
                    raw_response={"message": {"content": "invalid"}},
                )
            return super()._call_provider(**kwargs)

    client = RepairClient(tmp_path)

    result = client.propose(system_prompt="system", user_prompt="original request")

    assert result.attempts == 2
    assert prompts[0] == "original request"
    assert "Repair only the executable response contract" in prompts[1]
    assert "duplicate declaration body_weight_kg" in prompts[1]
    failure = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert failure["raw_response"] == {"message": {"content": "invalid"}}
    assert failure["failure_category"] == "repairable_contract"
    assert failure["repair_diagnostics"] == [
        {
            "code": "response_validation",
            "message": "duplicate declaration body_weight_kg",
        }
    ]


def test_invalid_proposal_enrichment_retries_with_repair_feedback(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    class RepairClient(StubCachedClient):
        def _call_provider(self, **kwargs: Any) -> ProviderResponse[Any]:
            prompts.append(kwargs["user_prompt"])
            proposal = _proposal()
            if len(prompts) > 1:
                payload = proposal.model_dump(mode="json")
                payload["algebraics"] = [
                    {
                        "name": "target",
                        "expression": "x",
                        "constraints": [],
                        "mechanisms": [],
                    }
                ]
                proposal = ProposerCandidateV2.model_validate(payload)
            return ProviderResponse(
                parsed=proposal,
                raw_response={"message": {"content": proposal.model_dump_json()}},
            )

    client = RepairClient(tmp_path, proposal_target_channels=("target",))

    result = client.propose(system_prompt="system", user_prompt="original request")

    assert result.attempts == 2
    assert result.parsed.observation_mappings[0].channel == "target"
    assert "Repair only the executable response contract" in prompts[1]
    assert "target target must match exactly one" in prompts[1]
    assert (
        RepairDiagnosticCode.POST_SCHEMA_VALIDATION.value
        in json.loads(
            (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )["repair_diagnostics"][0]["code"]
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "llm_failure",
        "llm_response",
    ]


class FakeOpenAIResponse:
    status = "completed"
    output_parsed = _proposal()
    output_text = output_parsed.model_dump_json()

    class Usage:
        input_tokens = 3
        output_tokens = 4
        total_tokens = 7

    usage = Usage()

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"id": "response_1", "status": self.status, "output": []}


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return FakeOpenAIResponse()


class FakeSDK:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_openai_uses_responses_parse_with_pydantic_schema(tmp_path: Path) -> None:
    sdk = FakeSDK()
    client = OpenAIResponsesClient(
        model="test-model",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        sdk_client=sdk,
    )

    result = client.propose(system_prompt="system", user_prompt="user")

    assert result.parsed == enrich_proposal_v2(_proposal())
    assert result.usage == TokenUsage(3, 4, 7)
    assert sdk.responses.calls[0]["text_format"] is ProposerCandidateV2
    assert sdk.responses.calls[0]["model"] == "test-model"
    assert sdk.responses.calls[0]["max_output_tokens"] == 2048
    assert sdk.responses.calls[0]["input"] == [
        {"role": "developer", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_openai_retries_sdk_pydantic_validation_failure(tmp_path: Path) -> None:
    sdk = FakeSDK()
    original_parse = sdk.responses.parse
    calls = 0

    def fail_once(**kwargs: object) -> FakeOpenAIResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            ProposerCandidateV2.model_validate({"schema_version": "2"})
        return original_parse(**kwargs)

    sdk.responses.parse = fail_once  # type: ignore[method-assign]
    client = OpenAIResponsesClient(
        model="test-model",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        sdk_client=sdk,
        max_attempts=2,
        sleep=lambda _seconds: None,
    )

    result = client.propose(system_prompt="system", user_prompt="user")

    assert result.attempts == 2
    assert calls == 2
    repair_input = sdk.responses.calls[0]["input"]
    assert isinstance(repair_input, list)
    assert "Repair only the executable response contract" in repair_input[1][
        "content"
    ]
    failure = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert failure["error_type"] == "LLMResponseError"
    assert failure["retryable"] is True


class FakeGeminiResponse:
    text = _proposal().model_dump_json()

    class UsageMetadata:
        prompt_token_count = 5
        candidates_token_count = 6
        total_token_count = 11

    usage_metadata = UsageMetadata()

    def model_dump(self, *, mode: str, warnings: bool) -> dict[str, object]:
        assert mode == "json"
        assert warnings is False
        return {"text": self.text}


class FakeGeminiModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> FakeGeminiResponse:
        self.calls.append(kwargs)
        return FakeGeminiResponse()


class FakeGeminiSDK:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


def test_gemini_sends_json_schema_and_parses_response(tmp_path: Path) -> None:
    sdk = FakeGeminiSDK()
    client = GeminiClient(
        model="gemini-3.6-flash",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        sdk_client=sdk,
    )

    result = client.propose(system_prompt="system", user_prompt="user")

    assert result.parsed == enrich_proposal_v2(_proposal())
    assert result.usage == TokenUsage(5, 6, 11)
    call = sdk.models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["contents"] == "user"
    config = call["config"]
    assert isinstance(config, dict)
    assert config["system_instruction"] == "system"
    assert config["max_output_tokens"] == 2048
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == _gemini_provider_schema(
        ProposerCandidateV2
    )


def test_gemini_production_schemas_are_flat_and_locally_compatible() -> None:
    for model in (ProposerCandidateV2, ScientificJudgeResult):
        schema = _gemini_provider_schema(model)
        encoded = json.dumps(schema)
        assert "$defs" not in encoded
        assert "$ref" not in encoded
        assert "pattern" not in encoded
        assert schema["additionalProperties"] is False


def test_gemini_schema_removes_only_unsupported_validation_keywords() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-z]+$",
                "minLength": 1,
                "maxLength": 20,
                "default": "x",
                "description": "A name.",
            },
            "version": {"type": "string", "const": "2"},
        },
        "required": ["name", "version"],
        "additionalProperties": False,
    }

    compact = _gemini_compatible_schema(schema)

    encoded = json.dumps(compact)
    for keyword in ("pattern", "minLength", "maxLength", "default", "const"):
        assert f'"{keyword}"' not in encoded
    assert compact["additionalProperties"] is False
    assert compact["properties"]["name"]["description"] == "A name."


def test_ollama_sends_json_schema_and_parses_response(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object], float]] = []

    def transport(
        url: str,
        body: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body, timeout))
        return {
            "message": {"role": "assistant", "content": _judge().model_dump_json()},
            "prompt_eval_count": 5,
            "eval_count": 6,
            "total_duration": 7_000_000,
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        temperature=0.2,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.parsed == _judge()
    assert result.usage == TokenUsage(5, 6, 11)
    assert result.latency_ms == 7.0
    assert calls[0][0] == "http://127.0.0.1:11434/api/chat"
    assert calls[0][1]["format"] == _ollama_compatible_schema(
        ScientificJudgeResult.model_json_schema(mode="validation")
    )
    assert calls[0][1]["stream"] is False
    assert calls[0][1]["think"] == "low"
    assert calls[0][1]["options"]["num_predict"] == 2048
    assert calls[0][1]["options"]["temperature"] == 0.2
    assert calls[0][1]["options"]["seed"] == 17


def test_ollama_sampling_options_change_request_hash(tmp_path: Path) -> None:
    common = {
        "model": "gpt-oss:20b",
        "cache_directory": tmp_path / "cache",
        "log_path": tmp_path / "events.jsonl",
    }
    first = OllamaClient(**common, temperature=0.2, seed=1)
    second = OllamaClient(**common, temperature=0.2, seed=2)

    first_hash = first.request_hash(
        role="proposer",
        system_prompt="system",
        user_prompt="user",
        response_model=ProposerCandidateV2,
    )
    second_hash = second.request_hash(
        role="proposer",
        system_prompt="system",
        user_prompt="user",
        response_model=ProposerCandidateV2,
    )

    assert first_hash != second_hash


def test_ollama_tool_call_transport_validates_arguments_and_ignores_thinking(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        _url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(body)
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Correct but deliberately not parsed.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_structured_response",
                            "arguments": _judge().model_dump(mode="json"),
                        }
                    }
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.TOOL_CALL,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.parsed == _judge()
    assert result.usage == TokenUsage(10, 20, 30)
    assert "format" not in calls[0]
    assert calls[0]["tools"][0]["function"]["parameters"] == (
        _ollama_compatible_schema(
            ScientificJudgeResult.model_json_schema(mode="validation")
        )
    )
    assert "call exactly" in calls[0]["messages"][1]["content"]
    assert result.raw_response["_autoformalism_retry"]["format_mode"] == (
        "native_tool_call"
    )


def test_ollama_tool_call_transport_fails_without_required_call(
    tmp_path: Path,
) -> None:
    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.TOOL_CALL,
        max_attempts=1,
        transport=lambda _url, _body, _timeout: {
            "message": {
                "content": "",
                "thinking": _judge().model_dump_json(),
            }
        },
    )

    with pytest.raises(LLMResponseError, match="exactly one structured tool call"):
        client.judge(system_prompt="system", user_prompt="user")


def test_ollama_repairs_exact_tool_verdict_key_corruption(tmp_path: Path) -> None:
    expected = _hybrid_judge()
    arguments = expected.model_dump(mode="json")
    candidate_a = arguments["absolute_assessments"][0]["candidate_a"]
    candidate_a["ver verdict"] = candidate_a.pop("verdict")
    raw_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_structured_response",
                        "arguments": arguments,
                    }
                }
            ],
        }
    }
    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.TOOL_CALL,
        max_attempts=1,
        transport=lambda _url, _body, _timeout: raw_response,
    )

    result = client.assess_hybrid(
        system_prompt="system",
        user_prompt="user",
        expected_absolute_units={
            (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate")
        },
    )

    assert result.parsed == expected
    assert "ver verdict" in arguments["absolute_assessments"][0]["candidate_a"]
    assert result.raw_response["_autoformalism_retry"][
        "tool_argument_key_repairs"
    ] == 1


def test_ollama_rejects_ambiguous_tool_verdict_key_collision(
    tmp_path: Path,
) -> None:
    arguments = _hybrid_judge().model_dump(mode="json")
    candidate_a = arguments["absolute_assessments"][0]["candidate_a"]
    candidate_a["ver verdict"] = "fail"
    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.TOOL_CALL,
        max_attempts=1,
        transport=lambda _url, _body, _timeout: {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_structured_response",
                            "arguments": arguments,
                        }
                    }
                ],
            }
        },
    )

    with pytest.raises(LLMResponseError, match="Extra inputs are not permitted"):
        client.assess_hybrid(
            system_prompt="system",
            user_prompt="user",
            expected_absolute_units={
                (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate")
            },
        )


def test_ollama_json_schema_uses_tool_only_on_final_empty_content_attempt(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(
        url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body))
        if len(calls) == 1:
            return {
                "done_reason": "stop",
                "message": {"content": "", "thinking": "No final JSON."},
            }
        if len(calls) == 2:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": ""},
                    }
                ]
            }
        return {
            "message": {
                "content": "",
                "thinking": "Validated assessment.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_structured_response",
                            "arguments": _judge().model_dump(mode="json"),
                        }
                    }
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
        max_attempts=3,
        initial_backoff_seconds=0.0,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.parsed == _judge()
    assert result.attempts == 3
    assert calls[0][0].endswith("/api/chat")
    assert "format" in calls[0][1]
    assert "tools" not in calls[0][1]
    assert calls[1][0].endswith("/v1/chat/completions")
    assert calls[1][1]["reasoning_effort"] == "none"
    assert calls[2][0].endswith("/api/chat")
    assert "format" not in calls[2][1]
    assert len(calls[2][1]["tools"]) == 1
    assert calls[0][1]["options"]["seed"] == 17
    assert calls[1][1]["seed"] == 18
    assert calls[2][1]["options"]["seed"] == 19
    assert "supersedes any earlier instruction" in (
        calls[2][1]["messages"][1]["content"]
    )
    assert result.raw_response["_autoformalism_retry"] == {
        "attempt_number": 3,
        "sampling_seed": 19,
        "format_mode": "native_tool_call_fallback",
        "embedded_json_extracted": False,
    }


def test_ollama_json_schema_tool_fallback_requires_empty_content_diagnostic(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        _url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(body)
        return {"message": {"content": "{}"}}

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        transport=transport,
    )

    with pytest.raises(LLMResponseError, match="ScientificJudgeResult validation"):
        client.judge(system_prompt="system", user_prompt="user")

    assert len(calls) == 2
    assert all("format" in call for call in calls)
    assert all("tools" not in call for call in calls)


def test_ollama_json_schema_tool_fallback_requires_two_attempts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires at least two attempts"):
        OllamaClient(
            model="gpt-oss:20b",
            cache_directory=tmp_path / "cache",
            log_path=tmp_path / "events.jsonl",
            response_mode=OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
            max_attempts=1,
        )


def test_ollama_response_mode_changes_request_hash(tmp_path: Path) -> None:
    common = {
        "model": "gpt-oss:20b",
        "cache_directory": tmp_path / "cache",
        "log_path": tmp_path / "events.jsonl",
    }
    schema_client = OllamaClient(
        **common, response_mode=OllamaResponseMode.JSON_SCHEMA
    )
    native_retry_client = OllamaClient(
        **common,
        response_mode=OllamaResponseMode.JSON_SCHEMA_NATIVE_RETRY,
    )
    openai_thinking_retry_client = OllamaClient(
        **common,
        response_mode=OllamaResponseMode.JSON_SCHEMA_OPENAI_THINKING_RETRY,
    )
    tool_client = OllamaClient(**common, response_mode=OllamaResponseMode.TOOL_CALL)
    fallback_client = OllamaClient(
        **common,
        response_mode=OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
    )
    later_fallback_client = OllamaClient(
        **common,
        response_mode=OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
        max_attempts=4,
    )

    schema_hash = schema_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )
    native_retry_hash = native_retry_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )
    openai_thinking_retry_hash = openai_thinking_retry_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )
    tool_hash = tool_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )
    fallback_hash = fallback_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )
    later_fallback_hash = later_fallback_client.request_hash(
        role="judge",
        system_prompt="system",
        user_prompt="user",
        response_model=ScientificJudgeResult,
    )

    assert len(
        {
            schema_hash,
            native_retry_hash,
            openai_thinking_retry_hash,
            fallback_hash,
            later_fallback_hash,
            tool_hash,
        }
    ) == 6


def test_ollama_auto_disables_thinking_for_non_gpt_oss(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        _url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(body)
        return {"message": {"content": _judge().model_dump_json()}}

    client = OllamaClient(
        model="qwen3:8b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        transport=transport,
    )

    client.judge(system_prompt="system", user_prompt="user")

    assert calls[0]["think"] is False


def test_ollama_rejects_disabling_gpt_oss_thinking(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GPT-OSS models require"):
        OllamaClient(
            model="gpt-oss:20b",
            cache_directory=tmp_path / "cache",
            log_path=tmp_path / "events.jsonl",
            thinking=OllamaThinking.OFF,
        )


def test_ollama_empty_content_reports_metadata_without_parsing_thinking(
    tmp_path: Path,
) -> None:
    thinking_candidate = _proposal().model_dump_json()
    raw_response = {
        "done": True,
        "done_reason": "stop",
        "eval_count": 357,
        "message": {"content": "", "thinking": thinking_candidate},
    }
    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        transport=lambda _url, _body, _timeout: raw_response,
    )

    with pytest.raises(
        LLMResponseError,
        match=(
            r"empty message.content .*done_reason='stop'.*eval_count=357.*"
            r"thinking_present=True"
        ),
    ) as caught:
        client.propose(system_prompt="system", user_prompt="user")

    assert caught.value.raw_response is raw_response
    assert caught.value.repair_diagnostics[0].code is (
        RepairDiagnosticCode.EMPTY_PROVIDER_CONTENT
    )


def test_ollama_repair_attempts_use_deterministic_fallback_seeds(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(
        url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body))
        if len(calls) == 1:
            return {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "", "thinking": "No final answer."},
            }
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "I will now return the requested object.\n"
                            + _judge().model_dump_json()
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 13},
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=2,
        initial_backoff_seconds=0.0,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.attempts == 2
    assert calls[0][0].endswith("/api/chat")
    assert calls[1][0].endswith("/v1/chat/completions")
    assert calls[0][1]["options"]["seed"] == 17
    assert calls[1][1]["seed"] == 18
    assert calls[1][1]["reasoning_effort"] == "none"
    assert calls[1][1]["response_format"]["type"] == "json_schema"
    assert calls[1][1]["response_format"]["json_schema"]["strict"] is True
    assert "final response content" in calls[1][1]["messages"][1]["content"]
    assert result.usage == TokenUsage(8, 13, 21)
    assert result.raw_response["_autoformalism_retry"] == {
        "attempt_number": 2,
        "sampling_seed": 18,
        "format_mode": "openai_json_schema_no_reasoning",
        "embedded_json_extracted": True,
    }


def test_ollama_native_json_retry_preserves_thinking_and_schema(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(
        url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body))
        if len(calls) == 1:
            return {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "", "thinking": "No final answer."},
            }
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": _judge().model_dump_json(), "thinking": "Done."},
            "prompt_eval_count": 8,
            "eval_count": 13,
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.JSON_SCHEMA_NATIVE_RETRY,
        thinking=OllamaThinking.LOW,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.attempts == 2
    assert all(url.endswith("/api/chat") for url, _body in calls)
    assert all(body["think"] == "low" for _url, body in calls)
    assert all("format" in body for _url, body in calls)
    assert all("tools" not in body for _url, body in calls)
    assert calls[0][1]["options"]["seed"] == 17
    assert calls[1][1]["options"]["seed"] == 18
    assert "final response content" in calls[1][1]["messages"][1]["content"]
    assert result.usage == TokenUsage(8, 13, 21)
    assert result.raw_response["_autoformalism_retry"] == {
        "attempt_number": 2,
        "sampling_seed": 18,
        "format_mode": "native_json_schema_retry",
        "embedded_json_extracted": False,
    }


def test_ollama_openai_retry_preserves_configured_thinking(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(
        url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body))
        if len(calls) == 1:
            return {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "", "thinking": "No final answer."},
            }
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": _judge().model_dump_json()},
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 13},
        }

    client = OllamaClient(
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        response_mode=OllamaResponseMode.JSON_SCHEMA_OPENAI_THINKING_RETRY,
        thinking=OllamaThinking.LOW,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.attempts == 2
    assert calls[0][0].endswith("/api/chat")
    assert calls[0][1]["think"] == "low"
    assert calls[0][1]["options"]["seed"] == 17
    assert calls[1][0].endswith("/v1/chat/completions")
    assert calls[1][1]["reasoning_effort"] == "low"
    assert calls[1][1]["seed"] == 18
    assert calls[1][1]["response_format"]["type"] == "json_schema"
    assert "final response content" in calls[1][1]["messages"][1]["content"]
    assert result.usage == TokenUsage(8, 13, 21)
    assert result.raw_response["_autoformalism_retry"] == {
        "attempt_number": 2,
        "sampling_seed": 18,
        "format_mode": "openai_json_schema_thinking_retry",
        "embedded_json_extracted": False,
    }


def test_ollama_schema_is_compact_but_preserves_validation_structure(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        _url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(body)
        return {"message": {"content": _proposal().model_dump_json()}}

    client = OllamaClient(
        model="qwen3:8b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        transport=transport,
    )
    client.propose(system_prompt="system", user_prompt="user")

    encoded_schema = json.dumps(calls[0]["format"])
    schema = calls[0]["format"]

    def values_for(key: str, value: object) -> list[int]:
        if isinstance(value, dict):
            found = [value[key]] if isinstance(value.get(key), int) else []
            return found + [
                item for child in value.values() for item in values_for(key, child)
            ]
        if isinstance(value, list):
            return [item for child in value for item in values_for(key, child)]
        return []

    assert max(values_for("maxLength", schema)) <= 512
    assert max(values_for("maxItems", schema)) <= 32
    # A property may itself be named ``description``; schema annotations are gone.
    assert "Complete machine-readable proposer candidate" not in encoded_schema
    assert '"title":' not in encoded_schema
    assert '"default":' not in encoded_schema
    assert "minLength" in encoded_schema
    assert '"required"' in encoded_schema
    assert '"additionalProperties": false' in encoded_schema
    assert '"pattern"' in encoded_schema
    assert "states" in schema["required"]
    assert "rhs" in schema["$defs"]["ProposedStateV2"]["required"]


def test_ollama_http_error_includes_server_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/chat",
        400,
        "Bad Request",
        {},
        __import__("io").BytesIO(b'{"error":"failed to parse grammar"}'),
    )

    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    client = OllamaClient(
        model="qwen3:8b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
    )

    with pytest.raises(LLMProviderError, match="failed to parse grammar"):
        client.propose(system_prompt="system", user_prompt="user")


def test_nonretryable_error_is_not_retried(tmp_path: Path) -> None:
    client = StubCachedClient(tmp_path)

    def fail(**_kwargs: object) -> ProviderResponse[Any]:
        raise LLMProviderError("bad request", retryable=False)

    client._call_provider = fail  # type: ignore[method-assign]
    with pytest.raises(LLMProviderError, match="bad request"):
        client.propose(system_prompt="system", user_prompt="user")
    assert client.provider_calls == 0


def test_failure_log_redacts_environment_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-test-never-log-this"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    client = StubCachedClient(tmp_path)

    def fail(**_kwargs: object) -> ProviderResponse[Any]:
        raise LLMProviderError(
            f"Authorization: Bearer {secret}",
            retryable=False,
        )

    client._call_provider = fail  # type: ignore[method-assign]
    with pytest.raises(LLMProviderError):
        client.propose(system_prompt="system", user_prompt="user")

    logged = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert secret not in logged
    assert "Bearer [REDACTED]" in logged


def test_failure_log_redacts_gemini_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "gemini-test-never-log-this"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    client = StubCachedClient(tmp_path)

    def fail(**_kwargs: object) -> ProviderResponse[Any]:
        raise LLMProviderError(f"API key: {secret}", retryable=False)

    client._call_provider = fail  # type: ignore[method-assign]
    with pytest.raises(LLMProviderError):
        client.propose(system_prompt="system", user_prompt="user")

    logged = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert secret not in logged


def test_config_selects_free_local_ollama_without_api_key(tmp_path: Path) -> None:
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="gpt-oss:20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
    )

    client = create_llm_client(config)

    assert isinstance(client, OllamaClient)
    assert "api_key" not in config.model_dump()
