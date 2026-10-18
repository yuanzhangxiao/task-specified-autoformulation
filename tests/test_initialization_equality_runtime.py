"""Versioned equality feedback preserves replay and provider-failure boundaries."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_focused_retest as focus
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_focused_retest import (  # noqa: F401
    choices,
    live,
    prepared,
    saved_setup,
    source,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def equality_plan(tmp_path, source):  # noqa: F811
    parent, _, task, _, _ = source
    plan = deepcopy(focus.freeze(parent, tmp_path / "inputs", [task["task_id"]]))
    plan.update(
        protocol=campaign.PROTOCOL_INITIALIZATION_V4,
        actionable_policy=api.POLICY_V4,
        question_identity_policy=training.MATHEMATICAL_QUESTION_IDENTITY,
    )
    return plan, plan["tasks"][0]


def test_equality_requests_match_preview_and_resume_without_calls(
    tmp_path, equality_plan
):
    plan, task = equality_plan
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        packets.append(packet)
        assert packet["prompt_policy"] == api.POLICY_V4
        assert list(packet).index("feedback") < list(packet).index("public_brief")
        assert body["messages"][1]["content"] == campaign.request_user(packet)
        return response(reply)

    root = tmp_path / "run"
    result = execute(root, plan, task, send)
    assert result["cost"]["physical_requests"] <= campaign.MAX_CALLS
    assert execute(root, plan, task, lambda *a: pytest.fail("uncached call")) == result
    live_campaign.preview(plan, tmp_path / "preview")
    import json

    preview = json.loads((tmp_path / "preview/request-00.json").read_text())
    assert preview["packets"]["repair"] == packets[0]


def test_equality_provider_failure_is_terminal_without_scientific_stagnation(
    tmp_path, equality_plan
):
    plan, task = equality_plan
    calls = []

    def fail(*args):
        calls.append(args)
        raise OSError("provider unavailable")

    result = execute(tmp_path, plan, task, fail)
    assert len(calls) == 1
    assert result["status"] == result["stop_reason"] == "provider_unavailable"
    assert result["cost"]["physical_requests"] == 1
    event = sealed_read(tmp_path / "results" / task["task_id"] / "events/000.json")
    assert event["consecutive_unchanged"] == 0
    assert event["error"] is None
    assert event["before"] == event["after"]
    assert not event["scientific_reply_received"]
    assert execute(tmp_path, plan, task, lambda *a: pytest.fail("retry")) == result
    report = campaign.report(tmp_path, plan)
    assert report["groups"]["actionable"]["status_counts"] == {
        "provider_unavailable": 1
    }
