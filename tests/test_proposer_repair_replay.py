"""Tests for no-call replay of conservative proposer contract repairs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.proposer_repair_replay import (
    replay_proposer_first_attempt,
)
from autoformalism.targets import PublicTargetContract
from scripts import replay_phase_b_proposer_first_attempt_repairs as replay_script


def _contract() -> PublicTargetContract:
    return PublicTargetContract.model_validate(
        {
            "schema_version": "public-target-contract-1",
            "benchmark_id": "fixture",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "source": "public_prompt",
            "targets": [
                {
                    "target_channel": "target",
                    "public_requirement": "generate target",
                    "required_dependencies": [],
                }
            ],
        }
    )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "2",
        "candidate_id": "candidate_1",
        "change_summary": "Replay fixture.",
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
            {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}}
        ],
    }


def test_replay_repairs_protected_and_exact_duplicate_parameters() -> None:
    payload = _payload()
    parameter = payload["parameters"][0]
    payload["parameters"] = [
        parameter,
        dict(parameter),
        {"name": "aux", "bounds": {"lower": 0.0, "upper": 1.0}},
        {"name": "aux", "bounds": {"lower": -2.0, "upper": 2.0}},
    ]

    replay, candidate = replay_proposer_first_attempt(
        json.dumps(payload),
        target_channels=("target",),
        protected_parameter_names=("target", "aux"),
        context=ValidationContext(targets=("target",), auxiliaries=("aux",)),
        target_contract=_contract(),
    )

    assert replay.schema_valid_after_repair is True
    assert replay.deterministic_valid is True
    assert replay.public_target_passed is True
    assert [item.name for item in candidate.parameters] == ["k"]
    assert replay.pre_schema_repairs == (
        {
            "code": "removed_protected_parameter",
            "parameter_name": "aux",
            "removed_count": 2,
        },
        {
            "code": "removed_legacy_parameter_ranges",
            "parameter_name": "aux",
            "removed_count": 2,
        },
        {
            "code": "removed_legacy_parameter_ranges",
            "parameter_name": "k",
            "removed_count": 2,
        },
        {
            "code": "removed_exact_duplicate_parameter",
            "parameter_name": "k",
            "removed_count": 1,
        },
    )


def test_replay_ignores_obsolete_range_only_parameter_differences() -> None:
    payload = _payload()
    payload["parameters"] = [
        {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}},
        {"name": "k", "bounds": {"lower": 0.0, "upper": 3.0}},
    ]

    replay, candidate = replay_proposer_first_attempt(
        json.dumps(payload),
        target_channels=("target",),
        protected_parameter_names=("target", "aux"),
        context=ValidationContext(targets=("target",), auxiliaries=("aux",)),
        target_contract=_contract(),
    )

    assert candidate is not None
    assert replay.schema_valid_after_repair is True
    assert replay.deterministic_valid is True
    assert [item.name for item in candidate.parameters] == ["k"]
    assert {item["code"] for item in replay.pre_schema_repairs} == {
        "removed_exact_duplicate_parameter",
        "removed_legacy_parameter_ranges",
    }


def test_frozen_replay_selects_low_condition_without_new_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "experiment"
    frozen = experiment / "frozen"
    frozen.mkdir(parents=True)
    source_plan = json.loads(
        Path("configs/phase_b_proposer_reasoning_calibration_v3.json").read_text(
            encoding="utf-8"
        )
    )
    source_plan["cells"] = [
        {
            "benchmark_id": "fixture",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "public_target_contract_sha256": "1" * 64,
        }
    ]
    source_plan["repetitions"] = [0]
    source_plan_path = frozen / "plan.json"
    source_plan_path.write_text(json.dumps(source_plan) + "\n", encoding="utf-8")
    (frozen / "freeze_manifest.json").write_text("{}\n", encoding="utf-8")
    analysis = experiment / "analysis"
    analysis.mkdir()
    (analysis / "proposer_transport_calibration.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "phase-b-proposer-transport-calibration-analysis-1"
                ),
                "status": "fail",
                "operating_points": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_plan = {
        "schema_version": "phase-b-proposer-repair-replay-plan-1",
        "status": "frozen_before_offline_replay",
        "source_plan_sha256": replay_script._sha256(source_plan_path),
        "conditions": [
            {"reasoning_effort": "low", "max_output_tokens": 16384},
            {"reasoning_effort": "medium", "max_output_tokens": 24576},
        ],
        "selection_rule": (
            "first_declared_condition_with_complete_replay_validity"
        ),
        "repair_policy": (
            "remove_protected_parameters_and_exact_duplicate_parameters_only"
        ),
        "new_llm_calls_permitted": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
    }
    replay_plan_path = tmp_path / "replay-plan.json"
    replay_plan_path.write_text(json.dumps(replay_plan) + "\n", encoding="utf-8")
    payload = _payload()
    parameter = payload["parameters"][0]
    payload["parameters"] = [parameter, dict(parameter)]
    for effort, budget, event_name in (
        ("low", 16384, "llm_failure"),
        ("medium", 24576, "llm_response"),
    ):
        request_hash = effort * 8
        result_dir = experiment / "results" / "task_000" / f"effort_{effort}"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"budget_{budget:06d}.json").write_text(
            json.dumps(
                {
                    "plan_sha256": replay_script._sha256(source_plan_path),
                    "task_index": 0,
                    "benchmark_id": "fixture",
                    "tier": "easy",
                    "repetition": 0,
                    "model": "openai/gpt-oss-120b",
                    "reasoning_effort": effort,
                    "max_output_tokens": budget,
                    "request_hash": request_hash,
                    "response_success": True,
                    "first_attempt_response_success": event_name == "llm_response",
                    "provider_attempt_count": 1,
                    "length_exhausted_attempt_count": 0,
                    "reasoning_character_count": 0,
                    "deterministic_valid": True,
                    "public_target_passed": True,
                    "test_data_opened": False,
                    "scientific_judge_called": False,
                    "parameter_fitting_performed": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        event_dir = (
            experiment
            / "conditions"
            / "task_000"
            / f"effort_{effort}"
            / f"budget_{budget:06d}"
        )
        event_dir.mkdir(parents=True)
        event = {
            "event": event_name,
            "request_hash": request_hash,
            "raw_response": {
                "choices": [
                    {"message": {"content": json.dumps(payload)}}
                ]
            },
        }
        if event_name == "llm_failure":
            event["attempt"] = 1
        else:
            event["attempts"] = 1
        (event_dir / "proposer_events.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8"
        )

    context = ValidationContext(targets=("target",), auxiliaries=("aux",))
    dataset = SimpleNamespace(
        roles=SimpleNamespace(targets=("target",))
    )
    monkeypatch.setattr(
        replay_script,
        "_public_context",
        lambda **_kwargs: (dataset, context, _contract()),
    )
    output = tmp_path / "replay"

    manifest = replay_script.replay_frozen_first_attempts(
        replay_plan_path,
        experiment,
        data_root=tmp_path / "public",
        target_contract_root=tmp_path / "contracts",
        output_root=output,
    )

    assert manifest["status"] == "pass"
    assert manifest["selected_reasoning_effort"] == "low"
    assert manifest["selected_max_output_tokens"] == 16384
    assert manifest["new_llm_calls_made"] is False
    assert (output / "finalists/low_016384/task_000.json").is_file()
    assert replay_script.replay_frozen_first_attempts(
        replay_plan_path,
        experiment,
        data_root=tmp_path / "public",
        target_contract_root=tmp_path / "contracts",
        output_root=output,
    ) == manifest
