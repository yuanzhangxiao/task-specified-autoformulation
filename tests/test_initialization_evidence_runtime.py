"""Versioned evidence presentation and semantic initializer repetition on replay."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_equality_runtime import (  # noqa: F401
    choices,
    equality_plan,
    live,
    prepared,
    saved_setup,
    source,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def evidence_plan(equality_plan):  # noqa: F811
    plan, task = deepcopy(equality_plan)
    plan.update(
        protocol=campaign.PROTOCOL_INITIALIZATION_V5,
        actionable_policy=api.POLICY_V5,
        budget_policy="separate-new-retest-budget-1",
    )
    return plan, task


def events(root, task):
    return [
        sealed_read(path)
        for path in sorted((root / "results" / task["task_id"] / "events").glob("*"))
    ]


@pytest.mark.parametrize("version, expected", [(4, 10), (5, 3)])
def test_null_and_redundant_initializer_stop_semantically_only_in_new_policy(
    tmp_path, equality_plan, version, expected  # noqa: F811
):
    plan, task = deepcopy(equality_plan)
    if version == 5:
        plan.update(
            protocol=campaign.PROTOCOL_INITIALIZATION_V5,
            actionable_policy=api.POLICY_V5,
        )
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        if len(calls) % 2 == 0:
            reply["initialization"] = packet["initialization"]["retained_handoff"]
        calls.append(packet)
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    assert result["cost"]["physical_requests"] == expected
    recorded = events(tmp_path, task)
    assert all(event["semantic_model_unchanged"] for event in recorded)
    assert [event["consecutive_unchanged"] for event in recorded] == (
        [1, 2, 3] if version == 5 else [1] * 10
    )
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("new call")) == result
    if version == 5:
        assert result["stop_reason"] == "three_repeated_repairs"
        live_campaign.preview(plan, tmp_path / "preview")
        shown = json.loads((tmp_path / "preview/request-00.json").read_text())
        assert shown["packets"]["repair"] == calls[0]


def test_different_rejected_initializer_alternatives_keep_separate_attempts(
    tmp_path, evidence_plan
):
    plan, task = evidence_plan
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        initializer = deepcopy(packet["initialization"]["retained_handoff"])
        initializer["records"][0]["rule"] = (
            {"mode": "observed_initial", "arguments": []}
            if len(calls) % 2 == 0
            else {"mode": "common_map", "arguments": ["unlisted"]}
        )
        reply["initialization"] = initializer
        calls.append(packet)
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    recorded = events(tmp_path, task)
    assert len(recorded) == 10
    assert all(not event["accepted"] for event in recorded)
    assert all(event["semantic_model_unchanged"] for event in recorded)
    assert all(event["consecutive_unchanged"] == 1 for event in recorded)
    assert result["initializer_handoff"] == recorded[0]["initialization_before"]
    assert "unsupported_observed" in json.dumps(calls[1])
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("new call")) == result


def test_accepted_initializer_change_is_progress(tmp_path, evidence_plan):
    plan, task = evidence_plan
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=False)
        if not calls:
            initializer = deepcopy(packet["initialization"]["retained_handoff"])
            initializer["records"][0]["rule"] = {
                "mode": "common_map",
                "arguments": ["u"],
            }
            reply["initialization"] = initializer
        calls.append(packet)
        return response(reply)

    execute(tmp_path, plan, task, send)
    first = events(tmp_path, task)[0]
    assert first["accepted"]
    assert not first["semantic_model_unchanged"]
    assert first["consecutive_unchanged"] == 0
    assert first["initialization_after"] != first["initialization_before"]


def test_provider_failure_does_not_become_scientific_repetition(
    tmp_path, evidence_plan
):
    plan, task = evidence_plan

    def fail(*args):
        raise OSError("offline simulated provider failure")

    result = execute(tmp_path, plan, task, fail)
    assert result["status"] == "provider_unavailable"
    assert result["cost"]["physical_requests"] == 1
    event = events(tmp_path, task)[0]
    assert event["consecutive_unchanged"] == 0
    assert not event["scientific_reply_received"]
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("retry")) == result
