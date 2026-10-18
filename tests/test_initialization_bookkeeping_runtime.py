"""V6 response availability, atomic edits, and exact saved-request resume."""

import json
from copy import deepcopy

import pytest

from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_evidence_runtime import (  # noqa: F401
    choices,
    equality_plan,
    events,
    evidence_plan,
    live,
    prepared,
    saved_setup,
    source,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def bookkeeping_plan(evidence_plan):  # noqa: F811
    plan, task = deepcopy(evidence_plan)
    plan.update(
        protocol=campaign.PROTOCOL_INITIALIZATION_V6, actionable_policy=api.POLICY_V6
    )
    return plan, task


@pytest.mark.parametrize(
    "kind",
    [
        "length",
        "no_content",
        "invalid_json",
        "no_choices",
        "null_choice",
        "null_message",
    ],
)
def test_unavailable_final_response_is_not_a_scientific_attempt(
    tmp_path, bookkeeping_plan, kind
):
    plan, task = bookkeeping_plan
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        packets.append(packet)
        result = response(reply)
        if len(packets) == 2:
            if kind == "length":
                result["choices"][0]["finish_reason"] = "length"
            elif kind == "no_content":
                result["choices"][0]["message"]["content"] = ""
            elif kind == "invalid_json":
                result["choices"][0]["message"]["content"] = "{"
            elif kind == "null_choice":
                result["choices"] = [None]
            elif kind == "null_message":
                result["choices"][0]["message"] = None
            else:
                result["choices"] = []
        return result

    result = execute(tmp_path, plan, task, send)
    rows = events(tmp_path, task)
    assert len(packets) == result["cost"]["physical_requests"] == 2
    assert result["status"] == result["stop_reason"] == "provider_unavailable"
    assert rows[1]["response_error"]
    assert rows[1]["transport_status"] == "responded"
    assert not rows[1]["scientific_reply_received"]
    assert rows[1]["consecutive_unchanged"] == rows[0]["consecutive_unchanged"] == 1
    assert rows[1]["before"] == rows[1]["after"] == rows[0]["after"]
    assert rows[1]["initialization_before"] == rows[1]["initialization_after"]
    assert result["cost"]["observed_tokens"] > 0
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("retry")) == result
    live_campaign.preview(plan, tmp_path / "preview")
    preview = json.loads((tmp_path / "preview/request-00.json").read_text())
    assert preview["packets"]["repair"] == packets[0]
    assert "inherited initializer record automatically" in packets[0]["editing_rules"]
    assert "Keep intended records, remove records" not in packets[0]["editing_rules"]


def test_complete_json_with_invalid_schema_still_gets_repair_feedback(
    tmp_path, bookkeeping_plan
):
    plan, task = bookkeeping_plan
    result = execute(
        tmp_path, plan, task, lambda *args: response({"invented_control": True})
    )
    rows = events(tmp_path, task)
    assert result["status"] != "provider_unavailable"
    assert all(row["error"] for row in rows)
    assert all(not row["accepted"] for row in rows)
    assert [row["consecutive_unchanged"] for row in rows] == [1, 2, 3]


@pytest.mark.parametrize("explicit", [False, True])
def test_retyping_cleans_only_inherited_initializer_in_atomic_transaction(
    tmp_path, bookkeeping_plan, explicit
):
    from tests.test_construction_ledger import variable

    plan, task = bookkeeping_plan
    start = plan["starts"][task["source_task"]]
    assert start["initializer_handoff"]["records"][0]["state"] == "z"

    def send(url, body, timeout):
        _, reply = packet_and_reply(body, complete=False)
        declaration = variable("z")
        declaration["definition"] = "algebraic"
        reply["variables"] = [declaration]
        if explicit:
            reply["initialization"] = start["initializer_handoff"]
        return response(reply)

    execute(tmp_path, plan, task, send)
    first = events(tmp_path, task)[0]
    assert first["accepted"] is not explicit
    if explicit:
        assert "stale" in first["error"]
        assert first["before"] == first["after"]
        assert first["initialization_before"] == first["initialization_after"]
    else:
        assert not first["initialization_after"]["records"]
        assert first["initialization_before"]["records"]
        assert {
            "code": "inherited_initializers_removed",
            "states": ["z"],
            "reason": "explicit variable removal or retyping to algebraic",
        } in first["normalizations"]


def test_redundant_equation_does_not_reset_rejected_initializer_attempt(
    tmp_path, bookkeeping_plan
):
    plan, task = bookkeeping_plan
    start = plan["starts"][task["source_task"]]
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        handoff = deepcopy(start["initializer_handoff"])
        handoff["records"][0]["rule"] = {"mode": "observed_initial", "arguments": []}
        reply["initialization"] = handoff
        if len(calls) % 2:
            reply["equations"] = start["draft"]["equations"]
        calls.append(packet)
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    assert result["cost"]["physical_requests"] == 3
    rows = events(tmp_path, task)
    assert [row["consecutive_unchanged"] for row in rows] == [1, 2, 3]
    assert all(not row["accepted"] for row in rows)
    assert all(row["before"] == row["after"] for row in rows)
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("retry")) == result
