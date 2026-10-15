"""Repair guidance preserves science choices and cached continuation semantics."""

import json

from autoformalism.search import construction_evidence as original
from autoformalism.search import construction_evidence_repair as repaired
from tests.test_construction_ledger import equation, variable
from tests.test_evidence_feedback_schedule import execute
from tests.test_explicit_variable_bindings import response


def test_initial_http_requests_are_identical_between_policies(tmp_path):
    """Later repair improvements must not confound the initial construction."""
    requests = {}
    for policy in (original.POLICY, repaired.POLICY):
        requests[policy] = []

        def send(url, body, timeout, *, policy=policy):
            packet = json.loads(body["messages"][1]["content"])
            edit = {**packet["response_template"], "stage_complete": True}
            if packet["stage"] != "repair":
                requests[policy].append(body)
            if packet["stage"] == "variables":
                edit["variables"] = [variable("y")]
            elif packet["stage"] == "equations":
                edit["equations"] = [equation("y", "u")]
            elif packet["stage"] == "repair":
                edit["compatibility_decisions"] = [
                    {
                        "question_id": q["question_id"],
                        "disposition": "retain_approximation",
                        "explanation": "This fixture checks transport equality.",
                    }
                    for q in packet["compatibility_questions"]
                ]
            return response(edit)

        execute(tmp_path / policy, send, policy=policy)
    assert len(requests[original.POLICY]) == 3
    assert requests[original.POLICY] == requests[repaired.POLICY]


def test_answer_then_prose_edit_keeps_answer_but_requires_explicit_completion(tmp_path):
    """An accepted explanation survives prose; false remains a real continuation."""
    packets = []
    repair_packets = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        packets.append(packet)
        edit = {**packet["response_template"], "stage_complete": True}
        if packet["stage"] == "variables":
            edit["variables"] = [variable("y")]
        elif packet["stage"] == "equations":
            edit["equations"] = [equation("y", "u")]
        elif packet["stage"] == "repair":
            repair_packets.append(packet)
            if len(repair_packets) == 1:
                assert packet["compatibility_questions"]
                edit["compatibility_decisions"] = [
                    {
                        "question_id": q["question_id"],
                        "disposition": "retain_approximation",
                        "explanation": "Retain this restricted approximation for now.",
                    }
                    for q in packet["compatibility_questions"]
                ]
                changed = equation("y", "u")
                changed["terms"][0]["scientific_role"] = "Changed description only."
                edit["equations"] = [changed]
                edit["stage_complete"] = False
            else:
                assert not packet.get("compatibility_questions")
                # The runtime must explicitly give the requested extra turn.
                assert len(repair_packets) == 2
                work = packet["feedback"]["repair_work"]
                assert work["mode"] == "completion_confirmation"
                assert work["previous_accepted_stage_complete"] is False
        return response(edit)

    result = execute(tmp_path, send, policy=repaired.POLICY)
    assert len(repair_packets) == 2
    assert result["status"] == "topology_complete"
    advice = result["assessment"]["training_compatibility"]
    assert advice["review_status"] == "addressed"
    assert advice["scientific_correctness"] == "unverified"
    assert advice["advisory_questions"]  # Acknowledging does not erase the conflict.
    assert advice["question_identity_policy"] == "diagnostic-semantics-2"
    count = len(packets)
    assert execute(tmp_path, send, policy=repaired.POLICY) == result
    assert len(packets) == count


def test_revise_without_edits_does_not_resolve_conditional_conflict(tmp_path):
    """Readiness remains separate from a still-unfulfilled revision decision."""
    repairs = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        if packet["stage"] == "variables":
            edit["variables"] = [variable("y")]
        elif packet["stage"] == "equations":
            edit["equations"] = [equation("y", "u")]
        elif packet["stage"] == "repair":
            repairs.append(packet)
            assert len(repairs) <= 2
            edit["compatibility_decisions"] = [
                {
                    "question_id": q["question_id"],
                    "disposition": "revise"
                    if len(repairs) == 1
                    else "retain_approximation",
                    "explanation": "An intention alone is not a scientific edit.",
                }
                for q in packet["compatibility_questions"]
            ]
        return response(edit)

    result = execute(tmp_path, send, policy=repaired.POLICY)
    assert len(repairs) == 2
    assert result["status"] == "topology_complete"
    assert [q["question_id"] for q in repairs[0]["compatibility_questions"]] == [
        q["question_id"] for q in repairs[1]["compatibility_questions"]
    ]
