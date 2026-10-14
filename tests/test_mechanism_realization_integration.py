"""Real requests, stage transitions, provenance and resume for realization records."""

import json

import pytest

from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_focus_review as review
from autoformalism.search import construction_realizations as prompts
from autoformalism.search import mechanism_realizations as realizations
from scripts import jetstream_construction_campaign as jetstream
from tests.test_construction_comparison import source_fixture
from tests.test_construction_focus import execute
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_jetstream_construction_campaign import fixture
from tests.test_phase_c_construction_baseline import tokenize


def test_declaration_recovery_then_exact_memory_repair_and_cached_resume(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert "mechanism_bindings" not in json.dumps(body)
        assert "remove_bindings" not in json.dumps(body)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable("y"), variable("m")]
            if len(calls) > 1:
                edit["mechanism_realizations"] = [
                    {
                        "requirement_id": "memory",
                        "components": ["m", "y"],
                        "memory_states": ["m"],
                    }
                ]
        elif p["stage"] == "equations":
            edit["equations"] = [equation("m", "u"), equation("y", "u")]
        elif p["stage"] == "repair":
            edit["equations"] = [equation("y", "m")]
        return response(edit)

    result = execute(tmp_path, send, policy=prompts.POLICY)
    assert result["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == [
        "variables",
        "variables",
        "shared_laws",
        "equations",
        "repair",
    ]
    assert not result["before_repair"]["assessment"]["eligible"]
    assert result["draft"]["mechanism_realizations"][0]["components"] == ["m", "y"]
    check = result["assessment"]["mechanism_realization_checks"][0]
    assert check["behavior_status"] == "unverified"
    assert result["assessment"]["mechanism_realization_handoff"]["topology_hash"]
    event = sealed_read(tmp_path / "construction/events/001.json")
    assert event["repair_receipt"]["committed_edits"]["mechanism_realizations"]
    n = len(calls)
    assert execute(tmp_path, send, policy=prompts.POLICY) == result
    assert len(calls) == n


def test_new_campaign_reports_and_preview_retain_schema_and_records(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "study"
    plan = campaign.freeze(
        tmp_path / "new", root, study="mechanism_realization_confirmation", seeds=[0]
    )
    assert len(plan["tasks"]) == 8
    assert plan["config"]["repair_requests"] == 10
    assert campaign.verify(root) == plan
    bodies = []

    def send(url, body, timeout):
        bodies.append(body)
        p = json.loads(body["messages"][1]["content"])
        targets = [
            v["name"]
            for v in p["public_brief"]["public_variables"]
            if v["data_role"] == "target"
        ]
        driver = next(
            v["name"]
            for v in p["public_brief"]["public_variables"]
            if v["data_role"] == "external_input"
        )
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable(n) for n in [*targets, "m"]]
            edit["mechanism_realizations"] = [
                {
                    "requirement_id": r["id"],
                    "components": ["m", *r["targets"]],
                    "memory_states": ["m"] if r["requires_dynamic_memory"] else [],
                }
                for r in p["public_brief"]["requirements"]
            ]
        elif p["stage"] == "equations":
            edit["equations"] = [equation(n, "m", n) for n in targets] + [
                equation("m", driver, "m")
            ]
        return response(edit)

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete"
    campaign.report(root, plan)
    assert (
        json.loads((root / "VARIABLES.json").read_text())["completed_variable_stages"]
        == 8
    )
    assert (
        json.loads((root / "SHARED_PROCESSES.json").read_text())[
            "completed_shared_stages"
        ]
        == 8
    )
    task = plan["tasks"][0]
    event = sealed_read(
        root / "results" / task["task_id"] / "construction/events/000.json"
    )
    preview = review.preview(
        plan["cells"][task["benchmark_id"]], task, event, bodies[0], prompts.POLICY
    )
    assert preview["after"] == bodies[0]
    summary = review.render(root, tmp_path / "review", prompts.POLICY)
    assert summary["requests"] == len(bodies)
    assert "new realization schema" in summary["size_unit"]
    text = (tmp_path / "review/PROMPTS.html").read_text()
    assert "Changes from protocol-clarity prompts" in text
    assert "mechanism_realizations" in text
    count = len(bodies)
    for task in plan["tasks"]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    assert len(bodies) == count


def test_jetstream_preparation_freezes_new_policy_and_identical_budget(tmp_path):
    args = fixture(tmp_path)
    plan = jetstream.prepare(*args, study="mechanism_realization_confirmation")
    assert len(plan["tasks"]) == 40
    assert plan["config"]["bookkeeping_policy"] == prompts.POLICY
    assert all(
        t["task_id"].endswith("_minimal_mechanism_realizations") for t in plan["tasks"]
    )
    assert plan["config"]["model_settings"]["maximum_requests"] == 133
    jetstream.validate_settings(plan["config"])
    assert jetstream.verify(args[-1]) == plan
    assert "mechanism_bindings" not in realizations.Draft().model_dump()


def test_zero_call_checkpoint_uses_explicit_schema_but_rejects_unlogged_edits(tmp_path):
    path = tmp_path / "construction/before_repair.json"
    checkpoint = {
        "event_count": 0,
        "draft": realizations.Draft().model_dump(mode="json"),
        "cost": _cost([]),
    }
    sealed_write(path, checkpoint)
    assert campaign.checked_records(tmp_path, "identity") == []
    checkpoint["draft"]["variables"] = [variable("y")]
    invalid_root = tmp_path / "unlogged"
    sealed_write(invalid_root / "construction/before_repair.json", checkpoint)
    with pytest.raises(ValueError, match="transaction prefix"):
        campaign.checked_records(invalid_root, "identity")
