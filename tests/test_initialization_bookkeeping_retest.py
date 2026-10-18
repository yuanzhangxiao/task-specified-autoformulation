"""V6 preserves V5 science and every charge while excluding truncated edit text."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import initialization_bookkeeping_retest as bookkeeping
from autoformalism.research import initialization_equality_retest as equality
from autoformalism.research import initialization_evidence_retest as evidence
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from autoformalism.staged_topology import content_hash
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_evidence_retest import (  # noqa: F401
    _rewrite,
    choices,
    equality_final,
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
def evidence_final(tmp_path, equality_final):  # noqa: F811
    """Record eight usable V5 replies and two charged output-length failures."""
    historical, _, task, _ = equality_final
    root = tmp_path / "evidence"
    plan = evidence.freeze(historical, root, [task["task_id"]])
    task = plan["tasks"][0]
    replies = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        index = len(replies)
        replies.append(reply)
        if index < 7:
            reply["remove_variables"] = [f"not_defined_{index}"]
        elif index == 7:
            reply["initialization"] = packet["initialization"]["retained_handoff"]
        raw = response(reply)
        if index >= 8:
            raw["choices"][0]["finish_reason"] = "length"
            raw["choices"][0]["message"]["content"] = '{"incomplete":'
        return raw

    result = execute(root, plan, task, send)
    assert result["event_count"] == 10
    assert result["cost"]["budget_charge"] == 1000
    assert result["stop_reason"] == "three_repeated_repairs"
    return root, plan, task, result


def test_freeze_retains_final_science_all_costs_and_complete_reply_provenance(
    tmp_path, evidence_final
):
    root, historical, task, result = evidence_final
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    output = tmp_path / "bookkeeping"
    plan = bookkeeping.freeze(root, output, [task["task_id"]])
    assert bookkeeping.freeze(root, output, [task["task_id"]]) == plan
    assert campaign.verify(output) == plan
    assert plan["protocol"] == live_campaign.PROTOCOL_V6
    assert plan["actionable_policy"] == api.POLICY_V6
    assert plan["budget_policy"] == equality.BUDGET_POLICY
    assert plan["settings"] == historical["settings"]
    assert plan["maximum_calls_per_arm"] == 10
    assert plan["maximum_tokens_per_arm"] == 262144
    assert plan["test_data_opened"] is False
    assert plan["optimizer_calls"] == 0
    assert plan["automatic_followup"] is False
    assert "controlled causal comparison" in plan["selection"]
    assert "provider_recovery_policy" not in plan
    assert "inherited_provider_outcomes" not in plan
    start = plan["starts"][task["task_id"]]
    original = historical["starts"][task["source_task"]]
    for key in ("draft", "initializer_handoff", "initial_information_contract"):
        assert start[key] == result[key]
    assert start["historical_cost"] == result["cost"]
    assert start["historical_cost"]["physical_requests"] == 10
    assert start["predecessor_historical_cost"] == {
        "historical_cost": original["historical_cost"],
        "predecessor_historical_cost": original["predecessor_historical_cost"],
    }
    retained = {
        content_hash(q) for q in start["previous_conflicts"]["tracked_conflicts"]
    }
    assert {
        content_hash(q) for q in original["previous_conflicts"]["tracked_conflicts"]
    } <= retained
    assert start["source_recorded_stop_reason"] == "three_repeated_repairs"
    assert len(start["source_event_sha256"]) == 10
    availability = start["source_response_availability"]
    assert sum(row["budget_charge"] for row in availability) == 1000
    assert [
        r["event_index"] for r in availability if not r["scientific_reply_available"]
    ] == [
        8,
        9,
    ]
    assert availability[-1]["recorded_consecutive_unchanged"] == 3
    feedback = start["source_feedback"]
    assert feedback["selected_event_index"] == 7
    assert feedback["unavailable_event_indices"] == [8, 9]
    assert feedback["last_edit_carried"] is False
    assert "availability-classification defect" in feedback["interpretation"]
    assert start["last_edit"] is None
    assert start["repetition_feedback"] is None
    assert before == {str(p): p.read_bytes() for p in root.rglob("*.json")}


def test_new_budget_starts_at_zero_and_cached_resume_is_exact(tmp_path, evidence_final):
    root, _, task, _ = evidence_final
    output = tmp_path / "bookkeeping"
    plan = bookkeeping.freeze(root, output, [task["task_id"]])
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        packets.append(packet)
        reply["remove_variables"] = ["missing_in_new_episode"]
        return response(reply)

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
        execute(output, plan, new_task, lambda *args: pytest.fail("live call"))
        == result
    )


def test_archived_source_code_identity_is_not_current_code_requirement(
    tmp_path, evidence_final, monkeypatch
):
    root, parent, task, _ = evidence_final
    monkeypatch.setattr(campaign, "source_identity", lambda: "new-v6-code")
    plan = bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])
    assert plan["source_identity"] == "new-v6-code"
    assert plan["source_plan_sha256"] == parent["artifact_sha256"]


@pytest.mark.parametrize(
    "names", [[], ["missing"], ["a"] * 2, ["a", "b", "c"], ["../x"]]
)
def test_explicit_selection_is_bounded(tmp_path, evidence_final, names):
    with pytest.raises(ValueError):
        bookkeeping.freeze(evidence_final[0], tmp_path / "new", names)


def test_source_cannot_be_destination(evidence_final):
    root, _, task, _ = evidence_final
    for destination in (root, root / "nested"):
        with pytest.raises(ValueError, match="outside"):
            bookkeeping.freeze(root, destination, [task["task_id"]])


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "wrong"),
        ("actionable_policy", "wrong"),
        ("question_identity_policy", "wrong"),
        ("witness_policy", "wrong"),
        ("budget_policy", "wrong"),
        ("test_data_opened", True),
        ("optimizer_calls", 1),
        ("automatic_followup", True),
        ("maximum_calls_per_arm", 11),
        ("provider_recovery_policy", "unexpected"),
    ],
)
def test_source_scientific_boundaries_remain_frozen(
    tmp_path, evidence_final, field, value
):
    root, _, task, _ = evidence_final
    _rewrite(root / "plan.json", lambda p: p.update({field: value}))
    with pytest.raises(ValueError, match="public-only"):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "draft",
        "initializer_handoff",
        "scientific_state",
        "cost",
        "stop_reason",
    ],
)
def test_final_result_must_match_saved_provenance(tmp_path, evidence_final, field):
    root, _, task, _ = evidence_final
    _rewrite(
        root / "results" / task["task_id"] / "result.json",
        lambda r: r.update({field: "wrong"}),
    )
    with pytest.raises(ValueError):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


@pytest.mark.parametrize("field", ["before", "initialization_before", "index"])
def test_truncated_event_binds_to_actual_prior_state(tmp_path, evidence_final, field):
    root, _, task, _ = evidence_final
    path = root / "results" / task["task_id"] / "events/008.json"

    def corrupt(event):
        event[field] = "wrong"
        # Matching each other cannot substitute for matching the preceding event.
        if field in {"before", "initialization_before"}:
            event[field.replace("before", "after")] = "wrong"

    _rewrite(path, corrupt)
    with pytest.raises(ValueError, match="predecessor"):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


@pytest.mark.parametrize(
    "field,value",
    [("accepted", True), ("error", "different"), ("decisions", [{"invented": True}])],
)
def test_truncated_text_cannot_supply_an_edit(tmp_path, evidence_final, field, value):
    root, _, task, _ = evidence_final
    _rewrite(
        root / "results" / task["task_id"] / "events/008.json",
        lambda e: e.update({field: value}),
    )
    with pytest.raises(ValueError, match="truncated"):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


@pytest.mark.parametrize("kind", ["event", "call"])
def test_even_unavailable_calls_require_exact_coverage(tmp_path, evidence_final, kind):
    root, _, task, _ = evidence_final
    directory = root / "results" / task["task_id"]
    path = directory / "events/009.json"
    if kind == "call":
        event = sealed_read(path)
        path = directory / "calls" / f"{event['request_hash']}.json"
    path.unlink()
    with pytest.raises(ValueError, match=f"{kind} coverage"):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


@pytest.mark.parametrize("change", ["settings", "namespace", "hash", "charge", "usage"])
def test_truncated_calls_keep_provider_identity_and_charges(
    tmp_path, evidence_final, change
):
    root, _, task, _ = evidence_final
    directory = root / "results" / task["task_id"]
    event = sealed_read(directory / "events/008.json")
    path = directory / "calls" / f"{event['request_hash']}.json"
    record = json.loads(path.read_text())
    if change == "settings":
        record["request"]["settings"]["maximum_requests"] = 11
    elif change == "namespace":
        record["request"]["namespace"] = "different"
    elif change == "hash":
        record["request_hash"] = "different"
    elif change == "charge":
        record["budget_charge"] = 0
    else:
        record["raw_response"]["usage"]["total_tokens"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        bookkeeping.freeze(root, tmp_path / "new", [task["task_id"]])


def test_latest_rejected_feedback_skips_unavailable_text(evidence_final):
    root, parent, task, result = evidence_final
    directory = root / "results" / task["task_id"]
    original = parent["starts"][task["source_task"]]
    events, records, availability = bookkeeping._episode(
        directory, parent, task, original, result
    )
    last, repetition, receipt = bookkeeping._feedback(
        events[:7] + events[8:], records, availability[:7]
    )
    assert receipt["selected_event_index"] == 6
    assert last["rejected_reply"]["remove_variables"] == ["not_defined_6"]
    assert repetition["previous_reply"] == last["rejected_reply"]
    # No complete response means neutral feedback, never a fabricated last edit.
    unavailable = [dict(row, scientific_reply_available=False) for row in availability]
    last, repetition, receipt = bookkeeping._feedback(events, records, unavailable)
    assert last is None and repetition is None
    assert receipt["selected_event_index"] is None
    assert receipt["unavailable_event_indices"] == list(range(10))


def test_complete_json_marked_length_is_still_unavailable(evidence_final):
    root, plan, task, _ = evidence_final
    directory = root / "results" / task["task_id"]
    event = sealed_read(directory / "events/007.json")
    record = json.loads(
        (directory / "calls" / f"{event['request_hash']}.json").read_text()
    )
    assert isinstance(bookkeeping._call(record, plan["settings"]), dict)
    truncated = deepcopy(record)
    truncated["raw_response"]["choices"][0]["finish_reason"] = "length"
    assert bookkeeping._call(truncated, plan["settings"]) is None
    truncated["budget_charge"] = 0
    with pytest.raises(ValueError, match="charge"):
        bookkeeping._call(truncated, plan["settings"])
