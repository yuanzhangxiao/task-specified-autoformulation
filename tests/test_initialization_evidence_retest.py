"""V5 freezing binds completed V4 science, provider provenance and separate costs."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import initialization_equality_retest as equality
from autoformalism.research import initialization_evidence_retest as evidence
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from autoformalism.staged_topology import content_hash
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_equality_retest import (  # noqa: F401
    _rewrite,
    choices,
    failed_source,
    live,
    prepared,
    recovered,
    recovered_final,
    saved_setup,
    source,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def equality_final(tmp_path, recovered_final):  # noqa: F811
    """Produce a completed V4 episode with three recorded rejected no-op replies."""
    historical, _, task, _ = recovered_final
    root = tmp_path / "equality"
    plan = equality.freeze(historical, root, [task["task_id"]])
    task = plan["tasks"][0]

    def send(url, body, timeout):
        _, reply = packet_and_reply(body)
        reply["remove_variables"] = ["not_defined"]
        return response(reply)

    result = execute(root, plan, task, send)
    assert result["event_count"] == 3
    assert result["initialization_status"]["complete"]
    return root, plan, task, result


def test_freeze_preserves_final_science_history_and_sources(tmp_path, equality_final):
    root, historical, task, result = equality_final
    before = {str(path): path.read_bytes() for path in root.rglob("*.json")}
    output = tmp_path / "evidence"
    plan = evidence.freeze(root, output, [task["task_id"]])
    assert evidence.freeze(root, output, [task["task_id"]]) == plan
    assert campaign.verify(output) == plan
    assert plan["protocol"] == live_campaign.PROTOCOL_V5
    assert plan["actionable_policy"] == api.POLICY_V5
    assert plan["budget_policy"] == equality.BUDGET_POLICY
    assert plan["settings"] == historical["settings"]
    assert plan["maximum_calls_per_arm"] == 10
    assert plan["maximum_tokens_per_arm"] == 262144
    assert plan["test_data_opened"] is False
    assert plan["optimizer_calls"] == 0
    assert plan["automatic_followup"] is False
    assert "controlled causal comparison" in plan["selection"]
    assert "provider_recovery_policy" not in plan
    start = plan["starts"][task["task_id"]]
    original = historical["starts"][task["source_task"]]
    for field in ("draft", "initializer_handoff", "initial_information_contract"):
        assert start[field] == result[field]
    assert start["historical_cost"] == result["cost"]
    assert start["historical_cost"]["physical_requests"] == 3
    assert start["predecessor_historical_cost"] == {
        "historical_cost": original["historical_cost"],
        "predecessor_historical_cost": original["predecessor_historical_cost"],
    }
    assert (
        start["predecessor_provenance"]["source_result_sha256"]
        == (original["source_result_sha256"])
    )
    final = sealed_read(root / "results" / task["task_id"] / "events/002.json")
    assert start["source_final_event_sha256"] == final["artifact_sha256"]
    assert len(start["source_event_sha256"]) == 3
    assert start["last_edit"]["rejected_reply"]["remove_variables"] == ["not_defined"]
    old_witnesses = original["previous_conflicts"]["tracked_conflicts"]
    retained = {
        content_hash(question)
        for question in start["previous_conflicts"]["tracked_conflicts"]
    }
    assert {content_hash(q) for q in old_witnesses} <= retained
    assert before == {str(path): path.read_bytes() for path in root.rglob("*.json")}


def test_archived_code_identity_is_not_current_source_requirement(
    tmp_path, equality_final, monkeypatch
):
    root, parent, task, _ = equality_final
    monkeypatch.setattr(campaign, "source_identity", lambda: "current-v5-code")
    assert parent["source_identity"] != campaign.source_identity()
    plan = evidence.freeze(root, tmp_path / "inputs", [task["task_id"]])
    assert plan["source_identity"] == "current-v5-code"
    assert plan["source_plan_sha256"] == parent["artifact_sha256"]


def test_new_episode_uses_new_budget_and_exact_cached_replay(tmp_path, equality_final):
    root, _, task, _ = equality_final
    plan = evidence.freeze(root, tmp_path / "inputs", [task["task_id"]])
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        packets.append(packet)
        reply["remove_variables"] = ["same_missing_variable"]
        return response(reply)

    output = tmp_path / "new-run"
    new_task = plan["tasks"][0]
    result = execute(output, plan, new_task, send)
    budget = packets[0]["feedback"]["repair_budget"]
    assert budget["charged_before_call"] == 0
    assert budget["current_call"] == 1
    assert budget["maximum_global_calls"] == 10
    assert budget["maximum_global_repair_tokens"] == 262144
    assert result["cost"]["physical_requests"] == len(packets) == 3
    assert result["cost"]["budget_charge"] == 300
    assert "inherited_cost" not in result
    assert (output / "results" / new_task["task_id"] / "events/000.json").exists()
    assert (
        execute(
            output, plan, new_task, lambda *args: pytest.fail("uncached provider call")
        )
        == result
    )


@pytest.mark.parametrize(
    "names", [[], ["missing"], ["duplicate"] * 2, ["a", "b", "c"], ["../outside"]]
)
def test_freeze_rejects_invalid_selection(tmp_path, equality_final, names):
    with pytest.raises(ValueError):
        evidence.freeze(equality_final[0], tmp_path / "invalid", names)


def test_freeze_rejects_destination_within_source(equality_final):
    root, _, task, _ = equality_final
    for output in (root, root / "new"):
        with pytest.raises(ValueError, match="outside"):
            evidence.freeze(root, output, [task["task_id"]])


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "wrong"),
        ("question_identity_policy", "wrong"),
        ("initial_time_policy", "wrong"),
        ("witness_policy", "wrong"),
        ("budget_policy", "wrong"),
        ("test_data_opened", True),
        ("optimizer_calls", 1),
        ("automatic_followup", True),
        ("maximum_calls_per_arm", 11),
        ("provider_recovery_policy", "unexpected"),
    ],
)
def test_freeze_rejects_changed_scientific_boundaries(
    tmp_path, equality_final, field, value
):
    root, _, task, _ = equality_final
    _rewrite(root / "plan.json", lambda parent: parent.update({field: value}))
    with pytest.raises(ValueError, match="public-only"):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "source_draft_sha256",
        "draft",
        "assessment",
        "scientific_state",
        "initializer_handoff",
        "initial_information_contract",
        "initialization_status",
        "interaction_source",
        "cost",
        "status",
        "stop_reason",
        "test_data_opened",
        "optimizer_calls",
    ],
)
def test_freeze_rejects_final_result_mismatch(tmp_path, equality_final, field):
    root, _, task, _ = equality_final
    path = root / "results" / task["task_id"] / "result.json"
    _rewrite(path, lambda result: result.update({field: "wrong"}))
    with pytest.raises(ValueError):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize("field", ["before", "initialization_before", "index"])
def test_freeze_checks_intermediate_chain(tmp_path, equality_final, field):
    root, _, task, _ = equality_final
    path = root / "results" / task["task_id"] / "events/001.json"
    _rewrite(path, lambda event: event.update({field: "different"}))
    with pytest.raises(ValueError, match="predecessor"):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize("kind", ["event", "call"])
def test_freeze_requires_complete_event_and_call_coverage(
    tmp_path, equality_final, kind
):
    root, _, task, _ = equality_final
    directory = root / "results" / task["task_id"]
    event_path = directory / "events/001.json"
    event = sealed_read(event_path)
    path = (
        event_path
        if kind == "event"
        else (directory / "calls" / f"{event['request_hash']}.json")
    )
    path.unlink()
    with pytest.raises(ValueError, match=f"{kind} coverage"):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])


@pytest.mark.parametrize(
    "change", ["settings", "namespace", "hash", "charge", "usage", "status", "length"]
)
def test_freeze_rejects_invalid_cached_calls(tmp_path, equality_final, change):
    root, _, task, _ = equality_final
    directory = root / "results" / task["task_id"]
    event = sealed_read(directory / "events/001.json")
    path = directory / "calls" / f"{event['request_hash']}.json"
    record = json.loads(path.read_text())
    if change == "settings":
        record["request"]["settings"]["maximum_requests"] = 11
    elif change == "namespace":
        record["request"]["namespace"] = "other"
    elif change == "hash":
        record["request_hash"] = "other"
    elif change == "charge":
        record["budget_charge"] += 1
    elif change == "usage":
        record["raw_response"]["usage"]["total_tokens"] += 1
    elif change == "status":
        record["status"] = "provider_failure"
    else:
        record["raw_response"]["choices"][0]["finish_reason"] = "length"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])


def test_prior_qualified_witnesses_cannot_be_erased_by_final_tracker():
    prior = {"question_id": "qualified", "target": "U", "witnesses": [{"times": [0]}]}
    original = {
        "diagnostics": {"evidence_sha256": "same", "advisory_questions": []},
        "previous_conflicts": {"evidence_sha256": "same", "tracked_conflicts": [prior]},
    }
    result = {"scientific_state": {"evidence_sha256": "same", "tracked_conflicts": []}}
    assert evidence._conflicts(original, result)["tracked_conflicts"] == [prior]
    changed = deepcopy(original)
    changed["previous_conflicts"]["evidence_sha256"] = "different"
    with pytest.raises(ValueError, match="witness evidence"):
        evidence._conflicts(changed, result)


def test_response_validation_requires_matching_settings_and_integer_charges():
    record = {
        "status": "responded",
        "request": {
            "settings": {"maximum_requests": 10},
            "budget_reservation_policy": "serving-tokens-plus-output-1",
        },
        "raw_response": response({}),
        "observed_total_tokens": 100,
        "budget_charge": 100,
        "budget_charge_basis": "provider_total_tokens",
    }
    evidence._call(record, record["request"]["settings"])
    with pytest.raises(ValueError, match="settings"):
        evidence._call(record, {"maximum_requests": 11})
    record["observed_total_tokens"] = True
    with pytest.raises(ValueError, match="charge"):
        evidence._call(record, record["request"]["settings"])


def test_historical_budget_settings_cannot_expand(tmp_path, equality_final):
    root, _, task, _ = equality_final
    _rewrite(
        root / "plan.json",
        lambda parent: parent["settings"].update(maximum_total_tokens=300000),
    )
    with pytest.raises(ValueError, match="allowance"):
        evidence.freeze(root, tmp_path / "invalid", [task["task_id"]])
