"""Frozen campaign for equation batches with focused atomic repair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.rebuttal.staged_function_hybrid_campaign import (
    HybridFunctionCampaignConfig,
    freeze_hybrid_campaign,
    run_hybrid_campaign,
    summarize_hybrid,
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
        functions = [
            {"expression": "-".join(term["sources"]), "parameters": []}
            for term in payload["selected_equation"]["terms"]
        ]
        return _provider_response({"functions": functions})

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
        "protocol": "scientific-staged-function-hybrid-repair-1",
        "purpose": "unit test",
        "source_function_plan_sha256": source_plan["plan_sha256"],
        "selected_topology": source_config["selected_topologies"][0],
        "generation_granularity": "equation_batch_atomic_repair",
    }
    config.pop("source_plan_sha256")
    config.pop("selected_topologies")
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, source_path


def test_config_requires_one_targeted_public_cell() -> None:
    config = json.loads(
        Path("configs/staged_function_hybrid_repair_v1.json").read_text()
    )
    config["public_cells"].append("another_cell")
    with pytest.raises(ValueError, match="exactly one public cell"):
        HybridFunctionCampaignConfig.model_validate(config)


def test_freeze_run_resume_and_missing_rows(tmp_path: Path, monkeypatch) -> None:
    import autoformalism.rebuttal.staged_function_hybrid_campaign as campaign

    config, source = _write_campaign_inputs(tmp_path)
    plan = freeze_hybrid_campaign(config, source, tmp_path / "plan.json")
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["generation_granularity"] == (
        "equation_batch_atomic_repair"
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(
            **kwargs,
            transport=_transport(calls),
        ),
    )
    summary = run_hybrid_campaign(
        tmp_path / "plan.json",
        tmp_path / "results",
        "http://localhost:8000",
    )
    assert summary["status"] == "complete"
    assert summary["complete_models"] == 1
    assert summary["batch_first_attempt_step_success_rate"] == 1.0
    assert summary["latent_initial_first_attempt_step_success_rate"] == 1.0
    assert summary["batch_term_acceptance_rate"] == 1.0
    assert summary["atomic_repair_activation_count"] == 0
    assert summary["models_with_cross_lhs_shared_parameters"] == 0
    assert not summary["automatic_winner_defined"]
    assert (
        run_hybrid_campaign(
            tmp_path / "plan.json",
            tmp_path / "results",
            "http://localhost:8000",
        )
        == summary
    )
    assert len(calls) == 3
    assert sum("selected_equation" in call for call in calls) == 2
    assert sum("selected_state" in call for call in calls) == 1

    partial = summarize_hybrid(plan, [])
    assert partial["status"] == "incomplete"
    assert partial["planned_tasks"] == 1
    assert partial["terminal_results"] == 0
    assert len(partial["rows"]) == 1


def test_freeze_rejects_source_plan_or_selection_mismatch(tmp_path: Path) -> None:
    config, source = _write_campaign_inputs(tmp_path)
    payload = json.loads(config.read_text())
    payload["source_function_plan_sha256"] = "0" * 64
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source function plan digest mismatch"):
        freeze_hybrid_campaign(config, source, tmp_path / "plan.json")


def test_v2_freeze_carries_certified_role_policy(tmp_path: Path) -> None:
    config, source = _write_campaign_inputs(tmp_path)
    payload = json.loads(config.read_text())
    payload.update(
        protocol="scientific-staged-function-hybrid-repair-2",
        function_repair_policy="certified_outer_gain",
    )
    config.write_text(json.dumps(payload))
    plan = freeze_hybrid_campaign(config, source, tmp_path / "plan.json")
    assert plan["schema_version"] == ("scientific-staged-function-hybrid-repair-plan-2")
    assert plan["tasks"][0]["function_repair_policy"] == ("certified_outer_gain")
    payload["function_repair_policy"] = "legacy"
    with pytest.raises(ValueError, match="requires function repair policy"):
        HybridFunctionCampaignConfig.model_validate(payload)
