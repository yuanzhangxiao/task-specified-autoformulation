"""Three-policy repair routing preserves complete patches, budgets and evidence."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.research import construction_search_policy as engine
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as base
from tests.test_construction_revision_edits import brief, problem


def patch(expression="a*mem*Gp", *, name=None, **changes):
    value = {
        "decision": "revise",
        "hypothesis": "Change a dependency relevant to the displayed witness.",
        "evidence_ids": ["public_task"],
        "expected_observable_effect": "Allow different conditional responses.",
        "edits": [
            {
                "component": "q",
                "kind": "algebraic",
                "sources": [],
                "expression": expression,
            }
        ],
    }
    if name:
        value["new_states"] = [
            {
                "name": name,
                "description": "An additional history coordinate",
                "rhs": f"I-{name}",
                "initial": {"mode": "value", "scope": "trajectory"},
            }
        ]
    return {**value, **changes}


def option(index, payload):
    return {
        "option_id": f"option_{index}",
        "affected_target": "I",
        "premise_changed": f"Different complete dynamical hypothesis {index}",
        "patch": payload,
    }


@pytest.fixture
def parent(monkeypatch):
    value = {
        "problem": problem().model_dump(mode="json"),
        "public_brief": brief().model_dump(mode="json"),
        "scientific_aliases": {name: name for name in ("I", "Gp")},
    }

    def assessment(parent, candidate):
        # A deterministic synthetic gate isolates control-flow behavior; real
        # scientific certificates are covered by the gate's own test modules.
        expression = candidate.base_candidate.processes[0].expression
        status = "deferred" if "mem*Gp" in expression else "conflict"
        return {
            "status": status,
            "equality_status": status,
            "child_problem_sha256": content_hash(candidate.model_dump(mode="json")),
            "conflicts": []
            if status != "conflict"
            else [
                {"target": "I", "property": "synthetic_witness", "status": "conflict"}
            ],
        }

    monkeypatch.setattr(base, "_assessment", assessment)
    return value


class Client:
    """Keep exact synthetic prompts for replay without any network operation."""

    def __init__(self, payloads):
        self.payloads, self.calls = deepcopy(payloads), []

    def call(self, *, system, user, response_model, step, attempt):
        fields = {
            "system": system,
            "user": user,
            "schema": response_model.model_json_schema(),
            "step": step,
            "attempt": attempt,
        }
        self.calls.append(fields)
        if attempt >= len(self.payloads):
            raise DeferredCall("saved prefix ends here")
        payload = self.payloads[attempt]
        return {
            "request_hash": content_hash(fields),
            "status": "responded",
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(payload)},
                    }
                ],
                "usage": {"total_tokens": 31},
            },
        }


def run(parent, payloads, policy, *, seed=0):
    client = Client(payloads)
    result = engine.run_episode(
        parent, client, StagedModelSettings(maximum_requests=3), policy, seed=seed
    )
    return result, client


def test_renamed_failed_family_branches_after_validation(parent):
    before = deepcopy(parent)
    values = [patch("a*mem+X", name="X"), patch("a*mem+Z", name="Z"), patch()]
    result, client = run(parent, values, "sequential_restart")
    assert result["generation_calls"] == 3
    assert result["eligible_candidate_ids"] == ["call2-option0"]
    first, second, third = result["candidates"]
    assert first["status"] == second["status"] == "rejected"
    assert first["family_identity"]["sha256"] == second["family_identity"]["sha256"]
    assert second["repeated_unchanged_failure"]
    assert second["repeats_candidate_ids"] == ["call0-option0"]
    assert third["status"] == "eligible"
    packets = [json.loads(row["user"]) for row in client.calls]
    assert (
        packets[1]["repair_feedback"]["complete_intended_patch"]["new_states"][0][
            "name"
        ]
        == "X"
    )
    assert "intended_trial_model" in packets[1]["repair_feedback"]
    assert (
        packets[1]["repair_feedback"]["status"] == "rejected_in_full_parent_unchanged"
    )
    assert packets[2]["search_instruction"]["mode"] == "fresh_alternative"
    assert "repair_feedback" not in packets[2]
    assert len(packets[2]["attempted_repairs"]) == 2
    assert all(
        packet["current_model"] == packets[0]["current_model"] for packet in packets
    )
    assert parent == before
    assert not result["model_promoted"] and result["optimizer_calls"] == 0


def test_exact_replay_is_deterministic(parent):
    values = [patch("a*mem+X", name="X"), patch("a*mem+Z", name="Z"), patch()]
    first, first_client = run(parent, values, "sequential_restart")
    second, second_client = run(parent, values, "sequential_restart")
    assert first == second
    assert first_client.calls == second_client.calls


def test_independent_breadth_has_no_history_and_preserves_all_eligible(parent):
    result, client = run(parent, [patch(), patch(), patch()], "independent")
    assert result["generation_calls"] == 3
    assert len(result["eligible_candidate_ids"]) == 3
    for call in client.calls:
        packet = json.loads(call["user"])
        assert "attempted_repairs" not in packet
        assert "repair_feedback" not in packet
    assert all(not row["repeated_unchanged_failure"] for row in result["candidates"])


def test_options_repair_two_distinct_families_once_each(parent):
    batch = {
        "options": [
            option(0, patch("a*mem+X", name="X")),
            option(1, patch("a*mem+X*X", name="X")),
            option(2, patch("a*mem+X*X*X", name="X")),
        ],
        "fewer_options_reason": "",
    }
    result, client = run(parent, [batch, patch(), patch()], "options_shallow", seed=3)
    assert result["generation_calls"] == 3
    assert result["candidate_count"] == 5
    assert len(result["eligible_candidate_ids"]) == 2
    branch_ids = [row["branch_id"] for row in result["attempts"][1:]]
    assert len(set(branch_ids)) == 2
    assert all(branch.startswith("call0-option") for branch in branch_ids)
    assert client.calls[0]["schema"] == engine.OptionsReply.model_json_schema()
    assert (
        client.calls[1]["schema"] == engine.StructuralRevisionReply.model_json_schema()
    )
    assert [call["attempt"] for call in client.calls] == [0, 1, 2]
    again, again_client = run(
        parent, [batch, patch(), patch()], "options_shallow", seed=3
    )
    assert again == result and again_client.calls == client.calls


def test_options_do_not_spend_two_repairs_on_renamed_same_family(parent):
    batch = {
        "options": [
            option(0, patch("a*mem+X", name="X")),
            option(1, patch("a*mem+Z", name="Z")),
            option(2, patch()),
        ],
        "fewer_options_reason": "",
    }
    result, _ = run(parent, [batch, patch()], "options_shallow")
    assert result["generation_calls"] == 2
    assert result["candidate_count"] == 4
    assert len(result["eligible_candidate_ids"]) == 2


def test_bad_option_does_not_discard_valid_sibling(parent):
    batch = {
        "options": [{"bad_field": "bad"}, option(1, patch())],
        "fewer_options_reason": "Only one coherent full alternative remains.",
    }
    result, _ = run(parent, [batch], "options_shallow")
    assert result["generation_calls"] == 1
    assert result["eligible_candidate_ids"] == ["call0-option1"]
    assert result["candidates"][0]["diagnostic"]["code"] == "OPTION_OR_RESPONSE_FORMAT"


def test_valid_wrapper_with_invalid_patch_gets_bounded_repair(parent):
    batch = {
        "options": [option(0, {"decision": "revise"})],
        "fewer_options_reason": "Only one candidate.",
    }
    result, _ = run(parent, [batch, patch()], "options_shallow")
    assert result["generation_calls"] == 2
    assert result["eligible_candidate_ids"] == ["call1-option0"]


def test_fewer_options_needs_reason_and_excess_options_are_bounded(parent):
    for batch in (
        {"options": [], "fewer_options_reason": ""},
        {"options": [option(i, patch()) for i in range(4)], "fewer_options_reason": ""},
    ):
        result, _ = run(parent, [batch], "options_shallow")
        assert result["generation_calls"] == 1
        assert result["eligible_candidate_ids"] == []
        assert result["attempts"][0]["parse_error"] is not None
    result, _ = run(
        parent,
        [{"options": [], "fewer_options_reason": "No defensible option"}],
        "options_shallow",
    )
    assert result["candidate_count"] == 0


def test_schema_correction_is_progress_before_repetition(parent):
    bad = patch(evidence_ids=["made_up_evidence"])
    first = engine.assess_candidate(parent, bad)
    second = engine.assess_candidate(parent, patch(), candidate_id="fixed")
    assert first["patch_sha256"] == second["patch_sha256"]
    assert first["status"] == "rejected" and second["status"] == "eligible"
    assert not engine.annotate_repetition(second, [first])["repeated_unchanged_failure"]


def test_changed_failure_is_not_an_unchanged_repeat(parent):
    first = engine.assess_candidate(parent, patch("a*mem+X", name="X"))
    second = deepcopy(first)
    second["id"] = "later"
    second["diagnostic_sha256"] = "different-validity-stage"
    assert not engine.annotate_repetition(second, [first])["repeated_unchanged_failure"]


def test_latest_complete_patch_and_earlier_history_are_both_visible(parent):
    first = engine.assess_candidate(
        parent, patch("a*mem+X", name="X"), candidate_id="first"
    )
    second = engine.assess_candidate(
        parent, patch("a*mem+Y*Y", name="Y"), candidate_id="second"
    )
    packet = json.loads(
        engine.request(
            parent, "sequential_restart", history=[first, second], intended=second
        )
    )
    assert len(packet["attempted_repairs"]) == 2
    feedback = packet["repair_feedback"]
    assert feedback["complete_intended_patch"]["new_states"][0]["name"] == "Y"
    assert "omitting it restores" in feedback["instruction"]
    assert "last_reply" not in feedback
    assert (
        "complete_intended_patch" in packet["training_consistency"]["model_reference"]
    )


def test_untrusted_malformed_reply_does_not_break_ledger(parent):
    for value in (None, [], {"edits": None, "new_states": 7, "hypothesis": 8}):
        row = engine.assess_candidate(parent, value)
        assert row["status"] == "rejected"
        assert len(engine.ledger_view([row])) == 1


def test_wider_revision_ends_sequential_branch_without_model(parent):
    result, _ = run(
        parent,
        [patch(decision="requires_wider_revision", edits=[])],
        "sequential_restart",
    )
    assert result["generation_calls"] == 1
    assert result["candidates"][0]["status"] == "requires_wider_revision"
    assert result["candidates"][0]["problem"] is None


def test_deferred_client_propagates_for_durable_resume(parent):
    with pytest.raises(DeferredCall):
        run(parent, [patch("a*mem")], "sequential_restart")


def test_policy_and_global_call_limit_are_fixed(parent):
    with pytest.raises(ValueError, match="unknown"):
        run(parent, [], "invented")
    with pytest.raises(ValueError, match="three-call"):
        engine.run_episode(
            parent, Client([]), StagedModelSettings(maximum_requests=4), "independent"
        )


def test_omitted_prior_correction_is_visible_but_never_merged(parent):
    first_reply = patch("a*mem+X", name="X")
    first_reply["edits"].append(
        {"component": "dep", "kind": "dynamic", "sources": [], "expression": "-2*dep"}
    )
    first = engine.assess_candidate(
        parent, first_reply, candidate_id="first", branch_id="branch"
    )
    second = engine.assess_candidate(
        parent, patch("a*mem+X", name="X"), candidate_id="second", branch_id="branch"
    )
    second = engine.annotate_repetition(second, [first])
    assert second["omitted_previous_actions"] == [
        {"field": "edits", "previous_action": first["normalized_reply"]["edits"][1]}
    ]
    packet = json.loads(
        engine.request(
            parent, "sequential_restart", history=[first, second], intended=second
        )
    )
    reminders = packet["repair_feedback"]["previous_trial_actions_now_omitted"]
    assert reminders["actions"][0]["previous_action"]["expression"] == "-2*dep"
    assert "not assumed" in reminders["instruction"]
    equations = second["problem"]["base_candidate"]["state_equations"]
    assert next(row["rhs"] for row in equations if row["state"] == "dep") == "-dep"
    assert not any(
        row["component"] == "dep"
        for row in packet["repair_feedback"]["complete_intended_patch"]["edits"]
    )


def test_unhashable_malformed_actions_cannot_crash_history(parent):
    first = engine.assess_candidate(
        parent, patch("a*mem"), candidate_id="first", branch_id="branch"
    )
    second = engine.assess_candidate(
        parent,
        patch(edits=[{"component": []}]),
        candidate_id="second",
        branch_id="branch",
    )
    result = engine.annotate_repetition(second, [first])
    assert result["status"] == "rejected"


def test_provider_truncation_is_unavailable_not_scientific_rejection(parent):
    class Truncated(Client):
        def call(self, **kwargs):
            record = super().call(**kwargs)
            record["raw_response"]["choices"][0]["finish_reason"] = "length"
            return record

    client = Truncated([patch(), patch(), patch()])
    result = engine.run_episode(
        parent, client, StagedModelSettings(maximum_requests=3), "independent"
    )
    assert result["generation_calls"] == 3
    assert all(row["status"] == "unavailable" for row in result["candidates"])
    assert all(
        row["diagnostic"]["code"] == "RESPONSE_SCHEMA_OR_AVAILABILITY"
        for row in result["candidates"]
    )
