"""Frozen, cached, no-fit validation of the staged sign-contract probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_sign_contract_campaign import (
    freeze_sign_contract_campaign,
    run_sign_contract_campaign,
)


def _response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _config(path: Path) -> Path:
    payload = {
        "protocol": "scientific-staged-sign-contract-1",
        "purpose": "test",
        "platform": "aces-h100x1",
        "serving_image_sha256": "a" * 64,
        "model_settings": {
            "model": "openai/gpt-oss-20b",
            "model_revision": "revision",
            "reasoning_effort": "low",
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "timeout_seconds": 1,
            "maximum_requests": 12,
            "maximum_total_tokens": 65536,
            "attempts_per_step": 2,
        },
        "served_context_tokens": 16384,
        "seeds": [0],
        "wall_seconds": 60,
        "shutdown_margin_seconds": 30,
    }
    path.write_text(json.dumps(payload))
    return path


def test_campaign_freezes_runs_and_replays_exact_cached_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = freeze_sign_contract_campaign(_config(tmp_path / "config.json"), plan_path)
    assert len(plan["tasks"]) == 1
    assert not plan["parameter_fitting_performed"]
    calls: list[dict[str, object]] = []

    def transport(
        url: str, body: dict[str, object], timeout: float
    ) -> dict[str, object]:
        del url, timeout
        messages = body["messages"]
        assert isinstance(messages, list)
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        calls.append(payload)
        role = payload["selected_term"]["scientific_role"]
        if role.startswith("direct input"):
            reply = {
                "expression": "drive_gain*u",
                "parameters": [{"name": "drive_gain", "role": "coefficient"}],
            }
        elif role.startswith("self-relaxation"):
            reply = {
                "expression": "x/tau",
                "parameters": [{"name": "tau", "role": "time_constant"}],
            }
        elif role.startswith("unknown outer magnitude"):
            reply = {
                "expression": "contrast_gain*(u-threshold*x)",
                "parameters": [
                    {"name": "contrast_gain", "role": "coefficient"},
                    {"name": "threshold", "role": "coefficient"},
                ],
            }
        else:
            reply = {
                "expression": "baseline",
                "parameters": [{"name": "baseline", "role": "offset"}],
            }
        return _response(reply)

    import autoformalism.rebuttal.staged_sign_contract_campaign as campaign

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    output = tmp_path / "results"
    first = run_sign_contract_campaign(plan_path, output, "http://localhost:8000")
    assert first["status"] == "complete"
    assert first["complete_model_rate"] == 1.0
    assert first["matched_offset_domain_control_pass_rate"] == 1.0
    assert len(calls) == 4
    result = json.loads((output / "sign_contract_seed0" / "result.json").read_text())
    parameters = {item["name"]: item for item in result["candidate"]["parameters"]}
    assert parameters["drive_gain"]["domain"] == "nonnegative"
    assert parameters["contrast_gain"]["domain"] == "nonnegative"
    assert parameters["threshold"]["domain"] == "real"
    assert parameters["baseline"]["domain"] == "real"
    assert parameters["tau"]["domain"] == "positive"

    second = run_sign_contract_campaign(plan_path, output, "http://localhost:8000")
    assert second == first
    assert len(calls) == 4


def test_freeze_refuses_to_overwrite_a_different_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        freeze_sign_contract_campaign(_config(tmp_path / "config.json"), plan_path)
