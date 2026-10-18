"""Equality retest starts preserve recovered provenance and use a new allowance."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_equality_retest as equality
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_provider_recovery import (  # noqa: F401
    choices,
    failed_source,
    live,
    prepared,
    recovered,
    saved_setup,
    source,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def recovered_final(recovered):  # noqa: F811
    """Exercise all seven remaining calls, whose event indices are 003 through 009."""
    root, plan, task = recovered
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        reply["remove_variables"] = [f"not_defined_{len(calls)}"]
        return response(reply)

    result = execute(root, plan, task, send)
    assert result["event_count"] == 7
    assert result["physical_request_offset"] == 3
    return root, plan, task, result


def _rewrite(path, change):
    """Reseal an intentional fixture alteration to test its semantic binding."""
    value = sealed_read(path)
    value.pop("artifact_sha256")
    change(value)
    path.unlink()
    return sealed_write(path, value)


def test_freeze_checks_offset_chain_and_keeps_final_science(tmp_path, recovered_final):
    root, historical, task, result = recovered_final
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    destination = tmp_path / "equality"
    plan = equality.freeze(root, destination, [task["task_id"]])
    assert equality.freeze(root, destination, [task["task_id"]]) == plan
    assert campaign.verify(destination) == plan
    assert plan["protocol"] == live_campaign.PROTOCOL_V4
    assert plan["actionable_policy"] == api.POLICY_V4
    assert plan["budget_policy"] == equality.BUDGET_POLICY
    assert plan["settings"] == historical["settings"]
    assert plan["maximum_calls_per_arm"] == 10
    assert plan["maximum_tokens_per_arm"] == 262144
    assert plan["test_data_opened"] is False
    assert plan["optimizer_calls"] == 0
    assert plan["automatic_followup"] is False
    assert "provider_recovery_policy" not in plan
    assert "inherited_provider_outcomes" not in plan
    start = plan["starts"][task["task_id"]]
    original = historical["starts"][task["source_task"]]
    for field in ("draft", "initializer_handoff", "initial_information_contract"):
        assert start[field] == result[field]
    assert start["previous_conflicts"] == audit._saved_conflicts(original, result)
    assert start["historical_cost"] == result["cost"]
    assert start["historical_cost"]["physical_requests"] == 10
    assert start["predecessor_historical_cost"] == original["historical_cost"]
    assert start["source_physical_request_offset"] == 3
    final_event = sealed_read(root / "results" / task["task_id"] / "events/009.json")
    assert start["source_final_event_sha256"] == final_event["artifact_sha256"]
    assert len(start["source_event_sha256"]) == 7
    assert start["last_edit"]["rejected_reply"]["remove_variables"] == ["not_defined_7"]
    problem = campaign.Problem(plan["cells"][task["benchmark_id"]], start["task"])
    session = live_campaign.Session(problem, start, policy=api.POLICY_V4)
    assert session.stage(api.Draft.model_validate(start["draft"])) == "repair"
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_archived_source_identity_need_not_equal_current_code(
    tmp_path, recovered_final, monkeypatch
):
    root, parent, task, _ = recovered_final
    monkeypatch.setattr(campaign, "source_identity", lambda: "new-v4-source")
    assert parent["source_identity"] != campaign.source_identity()
    plan = equality.freeze(root, tmp_path / "equality", [task["task_id"]])
    assert plan["source_identity"] == "new-v4-source"
    assert plan["source_plan_sha256"] == parent["artifact_sha256"]


def test_new_episode_starts_at_zero_and_replays(tmp_path, recovered_final):
    root, _, task, _ = recovered_final
    plan = equality.freeze(root, tmp_path / "inputs", [task["task_id"]])
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        packets.append(packet)
        reply["remove_variables"] = [f"new_unknown_{len(packets)}"]
        return response(reply)

    destination = tmp_path / "run"
    new_task = plan["tasks"][0]
    result = execute(destination, plan, new_task, send)
    budget = packets[0]["feedback"]["repair_budget"]
    assert budget["charged_before_call"] == 0
    assert budget["current_call"] == 1
    assert budget["maximum_global_calls"] == 10
    assert budget["maximum_global_repair_tokens"] == 262144
    assert result["cost"]["physical_requests"] == len(packets) == 10
    assert "inherited_cost" not in result
    assert (destination / "results" / new_task["task_id"] / "events/000.json").exists()
    assert execute(
        destination, plan, new_task, lambda *args: pytest.fail("uncached generation")
    ) == result


@pytest.mark.parametrize(
    "names", [[], ["missing"], ["duplicate"] * 2, ["a", "b", "c"], ["../outside"]]
)
def test_freeze_refuses_invalid_selection(tmp_path, recovered_final, names):
    with pytest.raises(ValueError):
        equality.freeze(recovered_final[0], tmp_path / "invalid", names)


def test_freeze_refuses_writing_inside_source(recovered_final):
    root, _, task, _ = recovered_final
    for destination in (root, root / "new"):
        with pytest.raises(ValueError, match="outside"):
            equality.freeze(root, destination, [task["task_id"]])


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "source_draft_sha256",
        "physical_request_offset",
        "event_count",
        "cost",
        "draft",
        "test_data_opened",
        "optimizer_calls",
    ],
)
def test_freeze_refuses_result_provenance_changes(tmp_path, recovered_final, change):
    root, _, task, _ = recovered_final
    path = root / "results" / task["task_id"] / "result.json"

    def alter(value):
        if change in {"identity", "source_draft_sha256"}:
            value[change] = "different"
        elif change in {"physical_request_offset", "event_count", "optimizer_calls"}:
            value[change] += 1
        elif change == "cost":
            value[change]["budget_charge"] = 0
        elif change == "test_data_opened":
            value[change] = True
        else:
            value[change] = {}

    _rewrite(path, alter)
    with pytest.raises(ValueError):
        equality.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize("change", ["before", "index", "record_sha256", "missing"])
def test_freeze_checks_intermediate_events(tmp_path, recovered_final, change):
    root, _, task, _ = recovered_final
    path = root / "results" / task["task_id"] / "events/005.json"
    if change == "missing":
        path.unlink()
    else:
        _rewrite(
            path,
            lambda value: value.update({change: {} if change == "before" else 0}),
        )
    with pytest.raises(ValueError, match="event"):
        equality.freeze(root, tmp_path / "invalid", [task["task_id"]])


def test_freeze_refuses_missing_or_changed_calls(tmp_path, recovered_final):
    root, _, task, _ = recovered_final
    folder = root / "results" / task["task_id"]
    event = sealed_read(folder / "events/005.json")
    path = folder / "calls" / f"{event['request_hash']}.json"
    original = path.read_text()
    record = json.loads(original)
    record["budget_charge"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="response predecessor"):
        equality.freeze(root, tmp_path / "invalid", [task["task_id"]])
    path.unlink()
    with pytest.raises(ValueError, match="call coverage"):
        equality.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize("field", ["test_data_opened", "optimizer_calls"])
def test_freeze_refuses_non_public_plan(tmp_path, recovered_final, field):
    root, _, task, _ = recovered_final
    _rewrite(root / "plan.json", lambda value: value.update({field: 1}))
    with pytest.raises(ValueError, match="public-only"):
        equality.freeze(root, tmp_path / "invalid", [task["task_id"]])


def test_final_incomplete_response_keeps_error_without_promoting_reasoning():
    event = {
        "accepted": False,
        "error": "incomplete provider response: length",
        "normalizations": [],
        "initialization_before": {},
        "initialization_after": {},
        "consecutive_unchanged": 1,
        "structural_assessment": {"errors": []},
    }
    record = deepcopy(response({}))
    record["choices"][0]["finish_reason"] = "length"
    last, repetition = equality._last_feedback(
        event, {"status": "responded", "raw_response": record}
    )
    assert last["rejected_reply"] is None
    assert repetition["previous_reply"] is None
    assert last["error"] == event["error"]
