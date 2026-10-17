"""Provider-only recovery preserves original scientific state and bounded costs."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_focused_retest as focus
from autoformalism.research import initialization_provider_recovery as recovery
from autoformalism.research import saved_topology_repair as campaign
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
def failed_source(tmp_path, source):  # noqa: F811
    """Create the original v3 terminal failure using its unchanged controller."""
    parent, _, task, _, _ = source
    root = tmp_path / "failed-focused"
    plan = focus.freeze(parent, root, [task["task_id"]])

    def fail(*args):
        raise OSError("provider renderer unavailable")

    result = execute(root, plan, plan["tasks"][0], fail)
    assert result["stop_reason"] == "three_repeated_repairs"
    sealed_write(
        root / "execution.json",
        {
            "protocol": "aces-saved-initialization-execution-1",
            "plan_sha256": plan["artifact_sha256"],
            "platform": "aces-h100x1",
            "model": plan["settings"]["model"],
            "model_revision": plan["settings"]["model_revision"],
            "served_context_tokens": 32768,
            "test_data_opened": False,
            "automatic_followup": False,
        },
    )
    return root, plan, result


@pytest.fixture
def recovered(tmp_path, failed_source):
    """Freeze a fresh namespace with inherited provider outcomes embedded."""
    root = tmp_path / "recovery"
    plan = recovery.freeze(failed_source[0], root)
    return root, plan, plan["tasks"][0]


def test_freeze_preserves_scientific_starts_and_original_artifacts(
    tmp_path, failed_source
):
    original, parent, result = failed_source
    before = {str(p): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    root = tmp_path / "recovery"
    plan = recovery.freeze(original, root)
    assert recovery.freeze(original, root) == plan
    assert campaign.verify(root) == plan
    assert plan["starts"] == parent["starts"]
    assert plan["tasks"] == parent["tasks"]
    assert plan["settings"] == parent["settings"]
    task = plan["tasks"][0]
    inherited = plan["inherited_provider_outcomes"][task["task_id"]]
    assert inherited["cost"] == result["cost"]
    assert inherited["physical_request_offset"] == 3
    assert campaign.namespace(plan, task) != campaign.namespace(parent, task)
    assert before == {
        str(p): p.read_bytes() for p in original.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="outside"):
        recovery.freeze(original, original / "recovery")


@pytest.mark.parametrize(
    "change", ["response", "uncertain", "draft", "handoff", "cost"]
)
def test_recovery_refuses_substantive_or_ambiguous_sources(recovered, change):
    _, plan, task = recovered
    plan = deepcopy(plan)
    proof = plan["inherited_provider_outcomes"][task["task_id"]]
    if change == "response":
        proof["calls"][0]["raw_response"] = {}
    elif change == "uncertain":
        proof["calls"][0]["status"] = "uncertain"
    elif change in {"draft", "handoff"}:
        field = "draft" if change == "draft" else "initializer_handoff"
        proof["result"][field] = {}
    else:
        proof["cost"]["budget_charge"] = 0
    with pytest.raises(ValueError):
        recovery.validate_plan(plan)


@pytest.mark.parametrize("change", ["start", "cap", "offset", "roster", "policy"])
def test_recovery_plan_cannot_reset_budget_or_change_science(recovered, change):
    _, plan, task = recovered
    plan = deepcopy(plan)
    if change == "start":
        plan["starts"][task["source_task"]]["last_edit"] = None
    elif change == "cap":
        plan["settings"]["maximum_requests"] += 1
    elif change == "offset":
        plan["inherited_provider_outcomes"][task["task_id"]][
            "physical_request_offset"
        ] = 0
    elif change == "roster":
        plan["tasks"] = []
    else:
        plan["provider_recovery_policy"] = "unknown"
    with pytest.raises(ValueError):
        recovery.validate_plan(plan)


def test_failure_stops_once_preserves_scientific_receipt_and_replays(recovered):
    root, plan, task = recovered
    packets = []

    def fail(url, body, timeout):
        packets.append(json.loads(body["messages"][1]["content"]))
        raise OSError("new provider failure")

    result = execute(root, plan, task, fail)
    assert len(packets) == 1
    assert result["status"] == result["stop_reason"] == "provider_unavailable"
    assert result["provider_availability"] == "unavailable"
    assert result["cost"]["physical_requests"] == 4
    assert result["new_cost"]["physical_requests"] == 1
    assert result["inherited_cost"]["physical_requests"] == 3
    assert result["cost"]["budget_charge"] == (
        result["inherited_cost"]["budget_charge"] + result["new_cost"]["budget_charge"]
    )
    start = plan["starts"][task["source_task"]]
    assert result["draft"] == start["draft"]
    assert result["initializer_handoff"] == start["initializer_handoff"]
    event = sealed_read(root / "results" / task["task_id"] / "events/003.json")
    assert event["retained_last_edit"] == start["last_edit"]
    assert event["retained_repetition_feedback"] == start["repetition_feedback"]
    assert event["consecutive_unchanged"] == 0
    assert event["error"] is None
    assert not event["scientific_reply_received"]
    assert event["transport_status"] == "provider_failure"
    budget = packets[0]["feedback"]["repair_budget"]
    assert budget["charged_before_call"] == result["inherited_cost"]["budget_charge"]
    assert budget["active_token_ceiling"] == campaign.MAX_TOKENS
    assert execute(root, plan, task, lambda *args: pytest.fail("new call")) == result
    report = campaign.report(root, plan)
    assert report["groups"]["actionable"]["status_counts"] == {
        "provider_unavailable": 1
    }
    assert report["inherited_cost"]["physical_requests"] == 3
    assert report["new_cost"]["physical_requests"] == 1
    assert report["cumulative_cost"] == result["cost"]
    assert "Availability" in (root / "SUMMARY.md").read_text()


def test_recovery_spends_only_seven_new_attempts_and_replays(recovered):
    root, plan, task = recovered
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        reply["remove_variables"] = [f"unknown_{len(calls)}"]
        return response(reply)

    result = execute(root, plan, task, send)
    assert len(calls) == result["new_cost"]["physical_requests"] == 7
    assert result["cost"]["physical_requests"] == 10
    assert result["stop_reason"] == "repair_budget_exhausted"
    paths = sorted((root / "results" / task["task_id"] / "events").glob("*.json"))
    assert [p.stem for p in paths] == [f"{index:03d}" for index in range(3, 10)]
    budgets = [packet["feedback"]["repair_budget"] for packet in calls]
    inherited = result["inherited_cost"]["budget_charge"]
    assert [b["charged_before_call"] for b in budgets] == [
        inherited + 100 * i for i in range(7)
    ]
    assert execute(root, plan, task, lambda *args: pytest.fail("new call")) == result


def test_recovery_transport_uses_only_remaining_token_cap(recovered):
    root, plan, task = recovered
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        reply["remove_variables"] = [f"unknown_{len(calls)}"]
        result = response(reply)
        # A reported charge near the remaining ceiling blocks the next request.
        result["usage"]["total_tokens"] = 250000
        return result

    result = execute(root, plan, task, send)
    assert len(calls) == 1
    assert result["stop_reason"].startswith("preflight_stop:")
    path = root / "results" / task["task_id"] / "calls/budget_preflight"
    receipts = [json.loads(p.read_text()) for p in path.glob("*.json")]
    assert {r["active_request_ceiling"] for r in receipts} == {7}
    assert {r["active_token_ceiling"] for r in receipts} == {
        campaign.MAX_TOKENS - result["inherited_cost"]["budget_charge"]
    }
    assert any(r["status"] == "blocked" for r in receipts)
    assert execute(root, plan, task, lambda *args: pytest.fail("new call")) == result


def test_pending_recovery_reports_inherited_cost(recovered):
    root, plan, _ = recovered
    report = campaign.report(root, plan)
    assert report["physical_requests"] == 3
    assert report["new_cost"]["physical_requests"] == 0
    assert report["groups"]["actionable"]["status_counts"] == {"pending": 1}
