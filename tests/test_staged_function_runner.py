"""Function-stage repair, deterministic resume, source provenance and draining."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    strict_provider_schema,
)
from autoformalism.rebuttal.staged_function_campaign import (
    freeze_function_campaign,
    function_diagnostic,
    run_function_campaign,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.staged_functions import LatentInitialReply
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import content_hash


def response(payload):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def function_transport(calls, repair=False):
    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if repair and len(calls) == 1:
            return response({"expression": "typo", "parameters": []})
        if "runtime_diagnostics" in payload:
            assert (
                payload["runtime_diagnostics"]["rejected_response"]["expression"]
                == "typo"
            )
            assert payload["accepted_functions"] == []
        if "selected_state" in payload:
            return response({"initial": {"fixed_value": 0.0}})
        selected = payload["selected_term"]
        expression = "z-x" if selected["lhs"] == "x" else "u-z"
        return response({"expression": expression, "parameters": []})

    return transport


def test_function_repair_then_complete_candidate_and_exact_replay(
    tmp_path: Path,
) -> None:
    task = function_diagnostic("driven_memory", ModelingLimits())
    calls = []

    def run():
        client = StagedTopologyClient(
            settings=StagedModelSettings(),
            base_url="http://localhost:8000",
            directory=tmp_path / "calls",
            namespace="functions",
            seed=0,
            transport=function_transport(calls, repair=True),
        )
        return run_staged_functions(
            PublicScientificBrief.model_validate(task["brief"]),
            ValidationContext.model_validate(task["context"]),
            task["source"],
            client,
            tmp_path,
        )

    result = run()
    assert result["complete_model"]
    assert result["physical_requests"] == 4
    assert len(result["draft"]["latent_initials"]) == 1
    assert sum(not event["accepted"] for event in result["events"]) == 1
    assert run() == result
    assert len(calls) == 4
    assert not result["parameter_fitting_performed"]


def test_function_drain_saves_partial_state_and_resumes_without_reissuing(
    tmp_path: Path,
) -> None:
    task = function_diagnostic("driven_memory", ModelingLimits())
    calls = []

    def run(can_start):
        client = StagedTopologyClient(
            settings=StagedModelSettings(),
            base_url="http://localhost:8000",
            directory=tmp_path / "calls",
            namespace="drain",
            seed=0,
            transport=function_transport(calls),
            can_start=can_start,
        )
        return run_staged_functions(
            PublicScientificBrief.model_validate(task["brief"]),
            ValidationContext.model_validate(task["context"]),
            task["source"],
            client,
            tmp_path,
        )

    with pytest.raises(DeferredCall):
        run(lambda: not calls)
    assert (
        len(json.loads((tmp_path / "progress.json").read_text())["accepted_functions"])
        == 1
    )
    assert not (tmp_path / "result.json").exists()
    assert run(lambda: True)["complete_model"]
    assert len(calls) == 3


def test_function_handoff_rejects_changed_topology_before_provider(
    tmp_path: Path,
) -> None:
    task = function_diagnostic("driven_memory", ModelingLimits())
    task["source"]["equations"][0]["terms"][0]["scientific_role"] = "changed hypothesis"
    calls = []
    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://localhost:8000",
        directory=tmp_path,
        namespace="mismatch",
        seed=0,
        transport=function_transport(calls),
    )
    with pytest.raises(ValueError, match="differs from its scientific declarations"):
        run_staged_functions(
            PublicScientificBrief.model_validate(task["brief"]),
            ValidationContext.model_validate(task["context"]),
            task["source"],
            client,
            tmp_path,
        )
    assert calls == []


def test_initial_schema_has_disjoint_strict_modes() -> None:
    schema = strict_provider_schema(LatentInitialReply.model_json_schema())
    assert schema["$defs"]["FixedInitial"]["required"] == ["fixed_value"]
    assert schema["$defs"]["AnalyticInitial"]["required"] == ["expression"]
    assert schema["$defs"]["FixedInitial"]["additionalProperties"] is False


def test_function_campaign_verifies_frozen_handoff_and_replays(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_function_campaign as campaign

    fixture = function_diagnostic("driven_memory", ModelingLimits())
    config = json.loads(Path("configs/staged_function_probe_v1.json").read_text())
    task_id = config["selected_topologies"][0]["task_id"]
    source = {**fixture["source"], "public_structure_checks_passed": True}
    original = {
        "task_id": task_id,
        "kind": "benchmark",
        "brief": fixture["brief"],
        "context": fixture["context"],
    }
    parent = {"tasks": [original]}
    parent["plan_sha256"] = content_hash(parent)
    config.update(
        source_plan_sha256=parent["plan_sha256"],
        seeds=[0],
        diagnostic_fixtures=[],
        selected_topologies=[
            {"task_id": task_id, "result_sha256": content_hash(source)}
        ],
    )
    root = tmp_path / "source" / task_id
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps(source))
    terminal = {
        "identity": content_hash([parent["plan_sha256"], original]),
        "result": source,
    }
    (root / "terminal.json").write_text(json.dumps(terminal))
    (tmp_path / "parent.json").write_text(json.dumps(parent))
    (tmp_path / "config.json").write_text(json.dumps(config))
    plan = freeze_function_campaign(
        tmp_path / "config.json",
        tmp_path / "parent.json",
        tmp_path / "source",
        tmp_path / "plan.json",
    )
    assert plan["runtime_source_sha256"] == runtime_source_hash()
    assert len(plan["tasks"]) == 1
    calls = []
    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(
            **kwargs, transport=function_transport(calls)
        ),
    )
    first = run_function_campaign(
        tmp_path / "plan.json", tmp_path / "out", "http://localhost:8000"
    )
    assert first["selected_topology_models"] == 1
    assert (
        run_function_campaign(
            tmp_path / "plan.json", tmp_path / "out", "http://localhost:8000"
        )
        == first
    )
    assert len(calls) == 3
    bad = copy.deepcopy(source)
    monkeypatch.setattr(campaign, "function_launcher_hash", lambda: "0" * 64)
    with pytest.raises(ValueError, match="launcher differs"):
        run_function_campaign(
            tmp_path / "plan.json", tmp_path / "out", "http://localhost:8000"
        )
    bad["inventory"][0]["scientific_role"] = "tampered"
    (root / "result.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="result or terminal identity differs"):
        freeze_function_campaign(
            tmp_path / "config.json",
            tmp_path / "parent.json",
            tmp_path / "source",
            tmp_path / "another.json",
        )
