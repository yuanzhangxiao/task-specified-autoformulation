"""Integration tests exercising real prompts, schemas, repair, and call replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
)
from autoformalism.rebuttal.staged_topology_campaign import (
    diagnostic_task,
    public_validation_context,
    summarize_tasks,
)
from autoformalism.schemas.staged_topology import (
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.staged_topology_runner import run_staged_topology


def provider_response(payload):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100, "completion_tokens": 30, "prompt_tokens": 70},
    }


def task():
    return diagnostic_task("driven_memory", ModelingLimits())


def test_complete_construction_repair_and_exact_resume(tmp_path: Path) -> None:
    fixture = task()
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_lhs" not in payload:
            return provider_response({"variables": fixture["initial_inventory"]})
        if (
            payload["selected_lhs"]["name"] == "x"
            and "runtime_diagnostics" not in payload
        ):
            return provider_response(
                {
                    "terms": [
                        {
                            "sources": ["typo"],
                            "outer_sign": "add",
                            "scientific_role": "memory",
                        }
                    ],
                    "inventory_revision": None,
                }
            )
        if "runtime_diagnostics" in payload:
            assert payload["runtime_diagnostics"]["rejected_response"]["terms"][0][
                "sources"
            ] == ["typo"]
        sources = ["z"] if payload["selected_lhs"]["name"] == "x" else ["u", "z"]
        return provider_response(
            {
                "terms": [
                    {
                        "sources": sources,
                        "outer_sign": "add",
                        "scientific_role": "response",
                    }
                ],
                "inventory_revision": None,
            }
        )

    def run():
        client = StagedTopologyClient(
            settings=StagedModelSettings(),
            base_url="http://localhost:8000",
            directory=tmp_path / "calls",
            namespace="fixture",
            seed=0,
            transport=transport,
        )
        return run_staged_topology(
            PublicScientificBrief.model_validate(fixture["brief"]),
            ValidationContext.model_validate(fixture["context"]),
            client,
            tmp_path / "result",
        )

    first = run()
    assert first["complete_topology"] and first["public_structure_checks_passed"]
    assert len(calls) == 5
    assert sum(not item["accepted"] for item in first["events"]) == 1
    assert run() == first
    assert len(calls) == 5


def test_revision_request_is_preserved_without_automatic_routing(
    tmp_path: Path,
) -> None:
    fixture = task()
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        return provider_response(
            {
                "terms": [],
                "inventory_revision": {
                    "variable": {
                        "name": "other",
                        "definition": "differential",
                        "scientific_role": "additional memory",
                    },
                    "reason": "needed to represent another timescale",
                },
            }
        )

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://localhost:8000",
        directory=tmp_path / "calls",
        namespace="fixture",
        seed=0,
        transport=transport,
    )
    result = run_staged_topology(
        PublicScientificBrief.model_validate(fixture["brief"]),
        ValidationContext.model_validate(fixture["context"]),
        client,
        tmp_path / "result",
        initial_inventory=tuple(
            ScientificVariable.model_validate(item)
            for item in fixture["initial_inventory"]
        ),
    )
    assert len(calls) == 1
    assert result["status"] == "inventory_revision_requested"
    assert result["inventory_revision"]["variable"]["name"] == "other"
    assert not result["complete_topology"]


def test_drain_preserves_partial_calls_without_terminal_failure(tmp_path: Path) -> None:
    fixture = task()
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        return provider_response({"variables": fixture["initial_inventory"]})

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://localhost:8000",
        directory=tmp_path / "calls",
        namespace="fixture",
        seed=0,
        transport=transport,
        can_start=lambda: not calls,
    )
    with pytest.raises(DeferredCall):
        run_staged_topology(
            PublicScientificBrief.model_validate(fixture["brief"]),
            ValidationContext.model_validate(fixture["context"]),
            client,
            tmp_path / "result",
        )
    assert (tmp_path / "result/progress.json").exists()
    assert not (tmp_path / "result/result.json").exists()


def test_no_op_revision_is_repaired_locally_and_replayed(tmp_path: Path) -> None:
    fixture = task()
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in fixture["initial_inventory"]
    )
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if len(calls) == 1:
            return provider_response(
                {
                    "terms": [],
                    "inventory_revision": {
                        "variable": next(
                            item
                            for item in payload["frozen_inventory"]
                            if item["name"] == payload["selected_lhs"]["name"]
                        ),
                        "reason": "a clearance parameter is missing",
                    },
                }
            )
        if len(calls) == 2:
            diagnostic = payload["runtime_diagnostics"]
            assert "definition unchanged" in diagnostic["error"]
            assert (
                "clearance parameter"
                in diagnostic["rejected_response"]["inventory_revision"]["reason"]
            )
        sources = ["z", "x"] if payload["selected_lhs"]["name"] == "x" else ["u", "z"]
        return provider_response(
            {
                "terms": [
                    {
                        "sources": sources,
                        "outer_sign": "add",
                        "scientific_role": "response",
                    }
                ],
                "inventory_revision": None,
            }
        )

    def run():
        client = StagedTopologyClient(
            settings=StagedModelSettings(),
            base_url="http://localhost:8000",
            directory=tmp_path / "calls",
            namespace="no-op-repair",
            seed=0,
            transport=transport,
        )
        return run_staged_topology(
            PublicScientificBrief.model_validate(fixture["brief"]),
            ValidationContext.model_validate(fixture["context"]),
            client,
            tmp_path / "result",
            initial_inventory=inventory,
        )

    result = run()
    assert result["status"] == "complete"
    assert result["public_structure_checks_passed"]
    assert result["inventory_revision"] is None
    assert result["inventory"] == [item.model_dump(mode="json") for item in inventory]
    assert sum(not event["accepted"] for event in result["events"]) == 1
    assert result["physical_requests"] == 3
    assert run() == result
    assert len(calls) == 3


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_malformed_and_truncated_visible_text_reaches_repair(
    tmp_path: Path, finish: str
) -> None:
    fixture = task()
    partial = '{"variables":['
    seen = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        seen.append(payload)
        if len(seen) == 1:
            return {
                "choices": [{"finish_reason": finish, "message": {"content": partial}}]
            }
        if len(seen) == 2:
            assert payload["runtime_diagnostics"]["rejected_response"] == partial
        return provider_response({"variables": fixture["initial_inventory"]})

    client = StagedTopologyClient(
        settings=StagedModelSettings(maximum_requests=3),
        base_url="http://localhost:8000",
        directory=tmp_path / "calls",
        namespace="fixture",
        seed=0,
        transport=transport,
    )
    result = run_staged_topology(
        PublicScientificBrief.model_validate(fixture["brief"]),
        ValidationContext.model_validate(fixture["context"]),
        client,
        tmp_path / "result",
    )
    assert len(seen) == 3
    assert result["status"] == "failed"
    assert result["unmeasured_requests"] == 1


def test_campaign_drains_and_replays_terminal_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_topology_campaign as campaign
    from autoformalism.staged_topology import content_hash

    fixture = task()
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        sources = ["z"] if payload["selected_lhs"]["name"] == "x" else ["u", "z"]
        return provider_response(
            {
                "terms": [
                    {
                        "sources": sources,
                        "outer_sign": "add",
                        "scientific_role": "response",
                    }
                ],
                "inventory_revision": None,
            }
        )

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    config = json.loads(Path("configs/staged_topology_probe_v1.json").read_text())
    plan = {
        "tasks": [fixture],
        "config": config,
        "runtime_source_sha256": campaign.runtime_source_hash(),
    }
    plan["plan_sha256"] = content_hash(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    first = campaign.run_campaign(plan_path, tmp_path / "out", "http://localhost:8000")
    assert first["finished"] == 1
    assert len(calls) == 2
    assert (
        campaign.run_campaign(plan_path, tmp_path / "out", "http://localhost:8000")
        == first
    )
    assert len(calls) == 2
    plan["config"]["purpose"] = "tampered"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="digest mismatch"):
        campaign.run_campaign(plan_path, tmp_path / "out", "http://localhost:8000")


def test_public_context_does_not_need_trajectory_files() -> None:
    context = public_validation_context("phase_b_dalla_man_t2_canonical_named_easy")
    assert context.targets == ("Gp", "I", "U")
    assert "meal_event_g" in context.external_inputs
    assert not context.lagged_targets


def test_diagnostics_never_inflate_benchmark_success() -> None:
    result = {
        "status": "complete",
        "complete_topology": True,
        "public_structure_checks_passed": True,
        "physical_requests": 2,
        "budget_charge": 200,
        "observed_total_tokens": 200,
        "unmeasured_requests": 0,
        "provider_seconds": 1,
    }
    summary = summarize_tasks(
        {"tasks": ["d", "b"], "plan_sha256": "x"},
        [{"task_id": "d", "kind": "diagnostic", "result": result}],
    )
    assert summary["finished"] == 1
    assert summary["benchmark_topologies"] == 0
    assert summary["benchmark_requirements_passed"] == 0
