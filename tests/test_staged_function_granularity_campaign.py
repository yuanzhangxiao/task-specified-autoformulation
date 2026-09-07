"""Frozen matched campaign for atomic and same-LHS function generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.rebuttal.staged_function_granularity_campaign import (
    FunctionGranularityCampaignConfig,
    freeze_granularity_campaign,
    run_granularity_campaign,
    summarize_granularity,
)
from autoformalism.schemas.staged_topology import ModelingLimits
from autoformalism.staged_topology import content_hash


def _provider_response(payload: dict) -> dict:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _transport(calls: list[dict]):
    def transport(url, body, timeout):
        del url, timeout
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_state" in payload:
            return _provider_response({"initial": {"fixed_value": 0.0}})
        if "selected_equation" in payload:
            functions = [
                {"expression": "-".join(term["sources"]), "parameters": []}
                for term in payload["selected_equation"]["terms"]
            ]
            return _provider_response({"functions": functions})
        term = payload["selected_term"]
        return _provider_response(
            {"expression": "-".join(term["sources"]), "parameters": []}
        )

    return transport


def _write_campaign_inputs(root: Path) -> tuple[Path, Path]:
    fixture = function_diagnostic("driven_memory", ModelingLimits())
    source = fixture["source"]
    source.update(
        public_structure_checks_passed=True,
        test_data_opened=False,
        private_reference_opened=False,
    )
    reviewed = "synthetic_seed0"
    source_config = json.loads(
        Path("configs/staged_function_probe_v2.json").read_text()
    )
    source_config.update(
        public_cells=["synthetic"],
        seeds=[0],
        diagnostic_fixtures=[],
        selected_topologies=[
            {"task_id": reviewed, "result_sha256": content_hash(source)}
        ],
    )
    source_plan = {
        "config": source_config,
        "tasks": [
            {
                "task_id": f"{reviewed}_functions_seed0",
                "kind": "benchmark",
                "seed": 0,
                "brief": fixture["brief"],
                "context": fixture["context"],
                "source": source,
                "reviewed_source_task": reviewed,
            }
        ],
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    source_plan["plan_sha256"] = content_hash(source_plan)
    source_path = root / "source.json"
    source_path.write_text(json.dumps(source_plan))
    config = {
        **source_config,
        "protocol": "scientific-staged-function-granularity-1",
        "purpose": "unit test",
        "source_function_plan_sha256": source_plan["plan_sha256"],
        "generation_granularities": ["atomic_interaction", "equation_batch"],
    }
    config.pop("source_plan_sha256")
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, source_path


def test_config_requires_exact_matched_arms() -> None:
    config = json.loads(Path("configs/staged_function_granularity_v1.json").read_text())
    config["generation_granularities"] = [
        "atomic_interaction",
        "atomic_interaction",
    ]
    with pytest.raises(ValueError, match="both prespecified arms"):
        FunctionGranularityCampaignConfig.model_validate(config)


def test_freeze_run_resume_and_missing_rows(tmp_path: Path, monkeypatch) -> None:
    import autoformalism.rebuttal.staged_function_granularity_campaign as campaign

    config, source = _write_campaign_inputs(tmp_path)
    plan = freeze_granularity_campaign(config, source, tmp_path / "plan.json")
    assert len(plan["tasks"]) == 2
    assert {task["generation_granularity"] for task in plan["tasks"]} == {
        "atomic_interaction",
        "equation_batch",
    }
    calls: list[dict] = []
    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(
            **kwargs,
            transport=_transport(calls),
        ),
    )
    summary = run_granularity_campaign(
        tmp_path / "plan.json",
        tmp_path / "results",
        "http://localhost:8000",
    )
    assert summary["status"] == "complete"
    assert [group["complete_models"] for group in summary["groups"]] == [1, 1]
    assert all(
        group["function_first_attempt_step_success_rate"] == 1.0
        for group in summary["groups"]
    )
    assert all(
        group["latent_initial_first_attempt_step_success_rate"] == 1.0
        for group in summary["groups"]
    )
    assert not summary["automatic_winner_defined"]
    assert run_granularity_campaign(
        tmp_path / "plan.json",
        tmp_path / "results",
        "http://localhost:8000",
    ) == summary
    assert len(calls) == 6
    assert sum("selected_term" in call for call in calls) == 2
    assert sum("selected_equation" in call for call in calls) == 2
    assert sum("selected_state" in call for call in calls) == 2

    partial = summarize_granularity(
        plan,
        [
            json.loads(
                (
                    tmp_path
                    / "results"
                    / plan["tasks"][0]["task_id"]
                    / "terminal.json"
                ).read_text()
            )
        ],
    )
    assert partial["status"] == "incomplete"
    assert partial["planned_tasks"] == 2
    assert partial["terminal_results"] == 1
    assert len(partial["rows"]) == 2


def test_freeze_rejects_source_plan_or_selection_mismatch(tmp_path: Path) -> None:
    config, source = _write_campaign_inputs(tmp_path)
    payload = json.loads(config.read_text())
    payload["source_function_plan_sha256"] = "0" * 64
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source function plan digest mismatch"):
        freeze_granularity_campaign(config, source, tmp_path / "plan.json")
