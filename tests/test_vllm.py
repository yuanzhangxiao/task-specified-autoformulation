"""Offline tests for the local vLLM structured-response adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.llm import (
    LLMConfig,
    LLMProvider,
    VLLMClient,
    VLLMReasoningEffort,
    create_llm_client,
)
from autoformalism.llm.exceptions import LLMResponseError
from autoformalism.llm.models import TokenUsage
from autoformalism.schemas import ScientificJudgeResult


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


def _proposal_payload() -> dict[str, object]:
    return {
        "schema_version": "2",
        "candidate_id": "candidate_1",
        "change_summary": "Test candidate.",
        "states": [
            {
                "name": "target",
                "kind": "observed",
                "observed_channel": "target",
                "rhs": "-k * target + aux",
            }
        ],
        "algebraics": [],
        "parameters": [
            {
                "name": "k",
                "bounds": {"lower": 0.0, "upper": 2.0},
            }
        ],
    }


def test_vllm_losslessly_repairs_proposer_parameters_before_schema_validation(
    tmp_path: Path,
) -> None:
    payload = _proposal_payload()
    parameter = payload["parameters"][0]
    payload["parameters"] = [
        parameter,
        dict(parameter),
        {"name": "aux", "bounds": {"lower": 0.0, "upper": 1.0}},
        {"name": "aux", "bounds": {"lower": -1.0, "upper": 1.0}},
    ]
    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        proposal_target_channels=("target",),
        proposal_protected_parameter_names=("target", "aux"),
        transport=lambda _url, _body, _timeout: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(payload)},
                }
            ]
        },
    )

    result = client.propose(system_prompt="system", user_prompt="user")

    assert [item.name for item in result.parsed.parameters] == ["k"]
    repair = result.raw_response["_autoformalism_pre_schema_repair"]
    assert repair["repair_count"] == 2
    assert repair["repairs"] == [
        {
            "code": "removed_protected_parameter",
            "parameter_name": "aux",
            "removed_count": 2,
        },
        {
            "code": "removed_exact_duplicate_parameter",
            "parameter_name": "k",
            "removed_count": 1,
        },
    ]


def test_vllm_rejects_conflicting_learnable_parameter_duplicates(
    tmp_path: Path,
) -> None:
    payload = _proposal_payload()
    payload["parameters"] = [
        {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}},
        {"name": "k", "bounds": {"lower": 0.0, "upper": 3.0}},
    ]
    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        proposal_target_channels=("target",),
        proposal_protected_parameter_names=("target", "aux"),
        transport=lambda _url, _body, _timeout: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(payload)},
                }
            ]
        },
    )

    with pytest.raises(LLMResponseError, match="duplicate parameter"):
        client.propose(system_prompt="system", user_prompt="user")


def test_vllm_client_sends_strict_schema_reasoning_and_seed(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(
        url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append((url, body))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": _judge().model_dump_json(),
                        "reasoning": "Checked the requested criteria.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 13},
        }

    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        base_url="http://127.0.0.1:9000/",
        reasoning_effort=VLLMReasoningEffort.HIGH,
        temperature=0.2,
        seed=9000,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert calls[0][0] == "http://127.0.0.1:9000/v1/chat/completions"
    request = calls[0][1]
    assert request["reasoning_effort"] == "high"
    assert request["temperature"] == 0.2
    assert request["seed"] == 9000
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert result.parsed == _judge()
    assert result.usage == TokenUsage(8, 13, 21)
    assert result.raw_response["_autoformalism_retry"] == {
        "attempt_number": 1,
        "sampling_seed": 9000,
        "format_mode": "vllm_openai_json_schema",
        "reasoning_effort": "high",
        "embedded_json_extracted": False,
    }


def test_vllm_empty_final_content_retries_with_next_seed(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        _url: str,
        body: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(body)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "reasoning": "No final JSON."},
                    }
                ],
                "usage": {"completion_tokens": 17},
            }
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": _judge().model_dump_json()},
                }
            ]
        }

    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=2,
        initial_backoff_seconds=0.0,
        seed=17,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.attempts == 2
    assert [call["seed"] for call in calls] == [17, 18]
    assert all(call["reasoning_effort"] == "low" for call in calls)
    assert "final response content" in calls[1]["messages"][1]["content"]


def test_vllm_terminal_empty_content_reports_reasoning_metadata(
    tmp_path: Path,
) -> None:
    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        transport=lambda _url, _body, _timeout: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "", "reasoning": "Finished analysis."},
                }
            ]
        },
    )

    with pytest.raises(
        LLMResponseError,
        match=r"empty message.content .*reasoning_present=True",
    ):
        client.judge(system_prompt="system", user_prompt="user")


def test_vllm_responses_continues_incomplete_reasoning_once(
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
                "status": "incomplete",
                "output": [
                    {
                        "id": "reasoning_1",
                        "type": "reasoning",
                        "status": "incomplete",
                        "summary": [
                            {"type": "summary_text", "text": "Planning."}
                        ],
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 128},
            }
        return {
            "status": "completed",
            "output": [
                {
                    "id": "message_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": _judge().model_dump_json(),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 136, "output_tokens": 64},
        }

    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        max_output_tokens=128,
        continuation_max_output_tokens=256,
        reasoning_effort=VLLMReasoningEffort.MEDIUM,
        seed=42,
        transport=transport,
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert [item[0] for item in calls] == [
        "http://127.0.0.1:8000/v1/responses",
        "http://127.0.0.1:8000/v1/responses",
    ]
    assert calls[0][1]["max_output_tokens"] == 128
    assert calls[1][1]["max_output_tokens"] == 256
    assert calls[0][1]["reasoning"] == {"effort": "medium"}
    continued_input = calls[1][1]["input"]
    assert isinstance(continued_input, list)
    assert continued_input[-1]["status"] == "incomplete"
    assert result.parsed == _judge()
    assert result.usage == TokenUsage(144, 192, 336)
    assert result.attempts == 1
    assert result.provider_attempts == 2
    assert result.raw_response["_autoformalism_continuation"][
        "continuation_request_count"
    ] == 1


def test_vllm_responses_joins_partial_json_with_continuation(
    tmp_path: Path,
) -> None:
    serialized = _judge().model_dump_json()
    split = len(serialized) // 2
    responses = [
        {
            "status": "incomplete",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {"type": "output_text", "text": serialized[:split]}
                    ],
                }
            ],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": serialized[split:]}
                    ],
                }
            ],
        },
    ]
    client = VLLMClient(
        model="openai/gpt-oss-20b",
        cache_directory=tmp_path / "cache",
        log_path=tmp_path / "events.jsonl",
        max_attempts=1,
        continuation_max_output_tokens=256,
        transport=lambda _url, _body, _timeout: responses.pop(0),
    )

    result = client.judge(system_prompt="system", user_prompt="user")

    assert result.parsed == _judge()
    assert result.provider_attempts == 2


def test_vllm_factory_uses_provider_specific_configuration(tmp_path: Path) -> None:
    client = create_llm_client(
        LLMConfig(
            provider=LLMProvider.VLLM,
            model="openai/gpt-oss-20b",
            cache_directory=tmp_path / "cache",
            log_path=tmp_path / "events.jsonl",
            vllm_base_url="http://127.0.0.1:9000",
            vllm_reasoning_effort=VLLMReasoningEffort.HIGH,
            vllm_temperature=0.2,
            vllm_seed=9000,
        )
    )

    assert isinstance(client, VLLMClient)
    options = client._hashable_provider_options()
    assert options["base_url"] == "http://127.0.0.1:9000"
    assert options["reasoning_effort"] == "high"
    assert options["seed"] == 9000
