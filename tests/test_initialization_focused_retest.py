"""Focused repair keeps scientific predecessors and resume accounting explicit."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import initialization_focused_retest as focus
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_qualification_v2 import (  # noqa: F401
    choices,
    live,
    prepared,
    saved_setup,
    v2,
)
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def source(tmp_path, v2, monkeypatch):  # noqa: F811
    plan, task, problem = deepcopy(v2)
    monkeypatch.setattr(campaign, "source_identity", lambda: "runtime-fixture")
    parent = tmp_path / "source"
    plan.pop("artifact_sha256")
    plan.update(test_data_opened=False, initialization_calls_within_total_budget=3)
    plan = sealed_write(parent / "plan.json", plan)

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        if packet["stage"] == "initialization":
            reply["initialization"] = choices("z").model_dump(mode="json")
            reply["stage_complete"] = False
        else:
            # Preserve an actual rejected record for first focused feedback.
            reply["remove_variables"] = ["not_defined"]
        return response(reply)

    result = execute(parent, plan, task, send)
    assert result["initialization_status"]["complete"]
    return parent, plan, task, result, problem


def test_focused_freeze_keeps_final_choices_witnesses_and_rejected_receipt(
    tmp_path, source
):
    parent, historical, task, result, problem = source
    original = {str(p): p.read_bytes() for p in parent.rglob("*.json")}
    root = tmp_path / "focused"
    plan = focus.freeze(parent, root, [task["task_id"]])
    assert campaign.verify(root) == plan
    assert focus.freeze(parent, root, [task["task_id"]]) == plan
    start = plan["starts"][task["task_id"]]
    assert start["draft"] == result["draft"]
    assert start["initializer_handoff"] == result["initializer_handoff"]
    assert start["last_edit"]["rejected_reply"]["remove_variables"] == ["not_defined"]
    assert start["historical_cost"] == result["cost"]
    assert plan["settings"] == historical["settings"]
    session = live_campaign.Session(problem, start, policy=api.POLICY_V3)
    assert session.stage(api.Draft.model_validate(start["draft"])) == "repair"
    assert original == {str(p): p.read_bytes() for p in parent.rglob("*.json")}


def test_focused_actual_request_is_ordered_versioned_and_replay_exact(tmp_path, source):
    parent, _, task, _, _ = source
    plan = focus.freeze(parent, tmp_path / "inputs", [task["task_id"]])
    packets = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        packets.append(packet)
        assert "algebraic equations" in body["messages"][0]["content"]
        assert packet["prompt_policy"] == api.POLICY_V3
        assert list(packet).index("feedback") < list(packet).index("public_brief")
        assert body["messages"][1]["content"] == campaign.request_user(packet)
        return response(reply)

    output = tmp_path / "run"
    result = execute(output, plan, plan["tasks"][0], send)
    assert packets[0]["stage"] == "repair"
    assert result["cost"]["physical_requests"] <= campaign.MAX_CALLS
    assert (
        execute(
            output,
            plan,
            plan["tasks"][0],
            lambda *args: pytest.fail("uncached request"),
        )
        == result
    )
    live_campaign.preview(plan, tmp_path / "preview")
    shown = json.loads((tmp_path / "preview/request-00.json").read_text())
    assert shown["packets"]["repair"] == packets[0]


@pytest.mark.parametrize(
    "names", [[], ["missing"], ["actionable"] * 2, ["a", "b", "c", "d"], ["../outside"]]
)
def test_focused_freeze_rejects_invalid_selection(tmp_path, source, names):
    with pytest.raises(ValueError):
        focus.freeze(source[0], tmp_path / "invalid", names)


def test_focused_freeze_rejects_mismatched_final_event(tmp_path, source):
    parent, _, task, result, _ = source
    path = (
        parent
        / "results"
        / task["task_id"]
        / "events"
        / f"{result['event_count'] - 1:03d}.json"
    )
    event = json.loads(path.read_text())
    event.pop("artifact_sha256")
    event["identity"] = "another-task"
    path.unlink()
    sealed_write(path, event)
    with pytest.raises(ValueError, match="final event"):
        focus.freeze(parent, tmp_path / "invalid", [task["task_id"]])
