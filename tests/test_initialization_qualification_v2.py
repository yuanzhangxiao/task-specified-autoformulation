"""End-to-end qualification of the versioned initialization runtime fixes."""

from copy import deepcopy

import pytest

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_initialized_repair as api
from tests.test_explicit_variable_bindings import response
from tests.test_initialized_topology_repair import (  # noqa: F401
    choices,
    events,
    live,
    packet_and_reply,
    prepared,
    saved_setup,
)
from tests.test_saved_topology_repair import execute


@pytest.fixture
def v2(live):  # noqa: F811
    plan, task, problem = live
    plan.update(
        protocol=campaign.PROTOCOL_INITIALIZATION_V2,
        actionable_policy=api.POLICY_V2,
        budget_reservation_policy=MEASURED_RESERVATION,
        witness_policy="all-seen-witnesses-1",
    )
    plan["starts"]["saved"]["initial_information_contract"].update(
        protocol="recorded-t0-initial-information-2", measured_initial_bindings=[]
    )
    return plan, task, problem


def test_v2_rejected_initializer_keeps_valid_intent_in_feedback_and_resumes(
    tmp_path, v2
):
    plan, task, _ = v2
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        if len(calls) == 1:
            reply["initialization"] = choices("y", "z").model_dump(mode="json")
        elif len(calls) == 2:
            assert packet["initialization"]["editable_states"] == ["z"]
            receipt = packet["feedback"]["rejected_initialization"]
            assert not receipt["transaction_committed"]
            assert (
                receipt["valid_intended_editable_records"]
                == choices("z").model_dump(mode="json")["records"]
            )
            assert receipt["invalid_records"][0]["state"] == "y"
            reply["initialization"] = choices("z").model_dump(mode="json")
        return response(reply)

    with pytest.raises(DeferredCall):
        execute(tmp_path, plan, task, send, can_start=lambda: len(calls) < 1)
    first = events(tmp_path)[0]
    assert not first["accepted"]
    assert first["initialization_after"] == choices().model_dump(mode="json")
    result = execute(tmp_path, plan, task, send)
    assert result["initialization_status"]["complete"]
    assert result["initializer_handoff"] == choices("z").model_dump(mode="json")
    assert (
        execute(tmp_path, plan, task, lambda *args: pytest.fail("uncached generation"))
        == result
    )


def test_malformed_rejected_record_does_not_crash_next_feedback(tmp_path, v2):
    plan, task, _ = v2
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        if len(calls) == 1:
            reply["initialization"] = {"records": "wrong type"}
        elif len(calls) == 2:
            assert packet["feedback"]["rejected_initialization"]["schema_error"]
            reply["initialization"] = choices("z").model_dump(mode="json")
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    assert result["initialization_status"]["complete"]


def test_session_rejects_mixed_contract_and_prompt_policy(live, v2):  # noqa: F811
    # Imported fixtures can share mutable inputs; build both explicit variants.
    _, _, problem = v2
    start = deepcopy(v2[0]["starts"]["saved"])
    with pytest.raises(ValueError, match="policy and information contract"):
        live_campaign.Session(problem, start, policy=api.POLICY)
    start["initial_information_contract"].pop("measured_initial_bindings")
    start["initial_information_contract"]["protocol"] = (
        "recorded-t0-initial-information-1"
    )
    with pytest.raises(ValueError, match="policy and information contract"):
        live_campaign.Session(problem, start, policy=api.POLICY_V2)


def test_refine_retains_starts_settings_seeds_and_preserves_historical_plan(
    tmp_path,
    live,  # noqa: F811
    monkeypatch,
):
    plan, _, _ = live
    monkeypatch.setattr(campaign, "source_identity", lambda: "qualified-runtime")
    plan = deepcopy(plan)
    plan.pop("artifact_sha256")
    plan.update(test_data_opened=False, initialization_calls_within_total_budget=3)
    source, output = tmp_path / "original", tmp_path / "qualified"
    parent = sealed_write(source / "plan.json", plan)
    original = (source / "plan.json").read_bytes()
    frozen = live_campaign.refine(source, output)
    assert live_campaign.refine(source, output) == frozen
    assert (source / "plan.json").read_bytes() == original
    assert frozen["tasks"] == parent["tasks"]
    assert frozen["settings"] == parent["settings"]
    assert frozen["starts"]["saved"]["draft"] == parent["starts"]["saved"]["draft"]
    assert frozen["historical_initialization_plan_sha256"] == parent["artifact_sha256"]
    assert campaign.verify(output) == frozen
    with pytest.raises(ValueError, match="unknown starting draft"):
        live_campaign.refine(source, tmp_path / "invalid", {"absent": []})
    with pytest.raises(ValueError, match="outside"):
        live_campaign.refine(source, source)
