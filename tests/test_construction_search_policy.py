"""Three-policy repair routing preserves complete patches, budgets and evidence."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.research import construction_branch_patch as branch
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

    def assessment(parent, candidate, **kwargs):
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


def run(
    parent,
    payloads,
    policy,
    *,
    seed=0,
    options_count=3,
    maximum_calls=3,
    branch_delta=False,
    common_zero_forcing=False,
    action_feedback=False,
):
    client = Client(payloads)
    result = engine.run_episode(
        parent,
        client,
        StagedModelSettings(maximum_requests=maximum_calls),
        policy,
        seed=seed,
        options_count=options_count,
        maximum_calls=maximum_calls,
        branch_delta=branch_delta,
        common_zero_forcing=common_zero_forcing,
        action_feedback=action_feedback,
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


def test_five_options_repair_four_distinct_branches_once_each(parent):
    batch = {
        "options": [
            option(index, patch("a*mem+" + "*".join(["X"] * (index + 1)), name="X"))
            for index in range(5)
        ]
    }
    payloads = [batch, patch(), patch(), patch(), patch()]
    result, client = run(
        parent, payloads, "options_shallow", seed=3, options_count=5, maximum_calls=5
    )
    assert result["generation_calls"] == result["maximum_calls"] == 5
    assert result["options_count"] == 5 and result["candidate_count"] == 9
    assert len(result["eligible_candidate_ids"]) == 4
    branches = [row["branch_id"] for row in result["attempts"][1:]]
    assert len(set(branches)) == 4
    assert all(branch.startswith("call0-option") for branch in branches)
    assert (
        client.calls[0]["schema"] == engine.options_reply_model(5).model_json_schema()
    )
    assert all(
        call["schema"] == engine.StructuralRevisionReply.model_json_schema()
        for call in client.calls[1:]
    )
    assert [row["attempt"] for row in client.calls] == list(range(5))
    assert all(
        json.loads(row["user"])["search_instruction"]["maximum_episode_calls"] == 5
        for row in client.calls
    )
    again, replay = run(
        parent, payloads, "options_shallow", seed=3, options_count=5, maximum_calls=5
    )
    assert again == result and replay.calls == client.calls


def test_five_options_with_three_calls_repairs_at_most_two_branches(parent):
    batch = {
        "options": [
            option(index, patch("a*mem+" + "*".join(["X"] * (index + 1)), name="X"))
            for index in range(5)
        ]
    }
    result, _ = run(
        parent, [batch, patch(), patch()], "options_shallow", options_count=5
    )
    assert result["generation_calls"] == 3 and result["candidate_count"] == 7
    assert result["maximum_calls"] == 3


def test_five_options_do_not_repair_private_renames_as_distinct_families(parent):
    batch = {
        "options": [
            option(i, patch(f"a*mem+{name}", name=name))
            for i, name in enumerate(["X", "Y", "Z", "W", "V"])
        ]
    }
    result, _ = run(
        parent, [batch, patch()], "options_shallow", options_count=5, maximum_calls=5
    )
    assert result["generation_calls"] == 2 and result["candidate_count"] == 6


@pytest.mark.parametrize("count", [0, 1, 3, 4, 6])
def test_five_option_count_is_mandatory_without_free_retry(parent, count):
    batch = {"options": [option(i, patch()) for i in range(count)]}
    result, client = run(
        parent, [batch], "options_shallow", options_count=5, maximum_calls=5
    )
    assert result["generation_calls"] == len(client.calls) == 1
    assert result["eligible_candidate_ids"] == []
    error = result["attempts"][0]["parse_error"]
    assert (
        error["code"] == "OPTION_OR_RESPONSE_FORMAT"
        and "exactly five" in error["message"]
    )
    with pytest.raises(ValueError):
        engine.options_reply_model(5).model_validate(batch)


def test_five_options_isolates_malformed_siblings(parent):
    batch = {
        "options": [{"bad_field": "bad"}, *[option(i, patch()) for i in range(1, 5)]]
    }
    result, _ = run(
        parent, [batch], "options_shallow", options_count=5, maximum_calls=5
    )
    assert result["generation_calls"] == 1
    assert result["eligible_candidate_ids"] == [f"call0-option{i}" for i in range(1, 5)]
    assert result["candidates"][0]["diagnostic"]["code"] == "OPTION_OR_RESPONSE_FORMAT"


def test_five_option_prompt_and_schema_match_the_declared_budget(parent):
    packet = json.loads(
        engine.request(
            parent, "options_shallow", batch=True, options_count=5, maximum_calls=5
        )
    )
    assert "exactly five" in packet["search_instruction"]["instruction"]
    assert "All five share" in packet["search_instruction"]["instruction"]
    assert packet["search_instruction"]["maximum_episode_calls"] == 5
    schema = engine.options_reply_model(5).model_json_schema()["properties"]["options"]
    assert schema["minItems"] == schema["maxItems"] == 5
    assert engine.options_reply_model(3) is engine.OptionsReply


@pytest.mark.parametrize("field", ["options_count", "maximum_calls"])
@pytest.mark.parametrize("value", [1, 4, 6, True, 3.0, "5", None])
def test_search_width_and_call_budget_are_bounded_before_generation(
    parent, field, value
):
    limits = {"options_count": 3, "maximum_calls": 3, field: value}
    client = Client([])
    with pytest.raises(ValueError, match=field):
        engine.run_episode(
            parent,
            client,
            StagedModelSettings(maximum_requests=3),
            "options_shallow",
            **limits,
        )
    assert client.calls == []
    with pytest.raises(ValueError, match=field):
        engine.request(parent, "options_shallow", batch=True, **limits)


def test_five_call_limit_must_match_durable_client_budget(parent):
    client = Client([])
    with pytest.raises(ValueError, match="five-call"):
        engine.run_episode(
            parent,
            client,
            StagedModelSettings(maximum_requests=3),
            "options_shallow",
            options_count=5,
            maximum_calls=5,
        )
    assert client.calls == []


def test_five_call_resume_replays_the_exact_saved_prefix(parent):
    batch = {
        "options": [
            option(index, patch("a*mem+" + "*".join(["X"] * (index + 1)), name="X"))
            for index in range(5)
        ]
    }
    prefix = Client([batch, patch()])
    with pytest.raises(DeferredCall):
        engine.run_episode(
            parent,
            prefix,
            StagedModelSettings(maximum_requests=5),
            "options_shallow",
            options_count=5,
            maximum_calls=5,
            seed=2,
        )
    result, full = run(
        parent,
        [batch, patch(), patch(), patch(), patch()],
        "options_shallow",
        options_count=5,
        maximum_calls=5,
        seed=2,
    )
    assert prefix.calls == full.calls[: len(prefix.calls)]
    assert result["generation_calls"] == 5


def test_options_do_not_spend_two_repairs_on_renamed_same_family(parent):
    batch = {
        "options": [
            option(0, patch("a*mem+X", name="X")),
            option(1, patch("a*mem+Z", name="Z")),
            option(2, patch()),
        ],
    }
    result, _ = run(parent, [batch, patch()], "options_shallow")
    assert result["generation_calls"] == 2
    assert result["candidate_count"] == 4
    assert len(result["eligible_candidate_ids"]) == 2


def test_bad_option_does_not_discard_valid_sibling(parent):
    batch = {
        "options": [{"bad_field": "bad"}, option(1, patch()), option(2, patch())],
    }
    result, _ = run(parent, [batch], "options_shallow")
    assert result["generation_calls"] == 1
    assert result["eligible_candidate_ids"] == ["call0-option1", "call0-option2"]
    assert result["candidates"][0]["diagnostic"]["code"] == "OPTION_OR_RESPONSE_FORMAT"


def test_valid_wrapper_with_invalid_patch_gets_bounded_repair(parent):
    batch = {
        "options": [
            option(0, {"decision": "revise"}),
            option(1, patch()),
            option(2, patch()),
        ],
    }
    result, _ = run(parent, [batch, patch()], "options_shallow")
    assert result["generation_calls"] == 2
    assert result["eligible_candidate_ids"] == [
        "call0-option1",
        "call0-option2",
        "call1-option0",
    ]


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_exactly_three_options_mandatory_without_budget_reset(parent, count):
    batch = {"options": [option(i, patch()) for i in range(count)]}
    result, _ = run(parent, [batch], "options_shallow")
    assert result["generation_calls"] == 1
    assert result["eligible_candidate_ids"] == []
    assert result["attempts"][0]["parse_error"]["code"] == "OPTION_OR_RESPONSE_FORMAT"
    assert "exactly three" in result["attempts"][0]["parse_error"]["message"]
    with pytest.raises(ValueError):
        engine.OptionsReply.model_validate(batch)


def test_old_fewer_options_escape_is_rejected(parent):
    batch = {"options": [option(0, patch())], "fewer_options_reason": "One suffices"}
    result, _ = run(parent, [batch], "options_shallow")
    assert result["eligible_candidate_ids"] == []
    assert result["attempts"][0]["parse_error"] is not None


def test_all_three_options_must_be_repair_proposals(parent):
    batch = {
        "options": [
            option(0, patch(decision="retain", edits=[])),
            option(1, patch()),
            option(2, patch()),
        ]
    }
    result, _ = run(parent, [batch], "options_shallow")
    assert result["candidates"][0]["option_wrapper_valid"] is False
    assert "decision=revise" in result["candidates"][0]["diagnostic"]["message"]
    assert len(result["eligible_candidate_ids"]) == 2


def test_prompt_distinguishes_complete_changes_from_whole_model(parent):
    packet = json.loads(engine.request(parent, "options_shallow", batch=True))
    assert "exactly three" in packet["search_instruction"]["instruction"]
    assert "List only changes" in packet["editing_scope"]["patch_semantics"]
    assert "A+B" in engine.SYSTEM and "A+C" in engine.SYSTEM
    schema = engine.OptionsReply.model_json_schema()["properties"]["options"]
    assert schema["minItems"] == schema["maxItems"] == 3


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


def branch_reply(candidate, changes=None, *, mode="delta", **extra):
    return {
        "mode": mode,
        **branch.reference(candidate),
        "patch": None if mode == "withdraw" else changes or {},
        **extra,
    }


@pytest.mark.parametrize("batch,fresh", [(True, False), (False, True), (False, False)])
def test_action_feedback_preserves_initial_and_fresh_requests(parent, batch, fresh):
    kwargs = {
        "batch": batch,
        "fresh": fresh,
        "branch_delta": True,
        "options_count": 5,
        "maximum_calls": 5,
        "common_zero_forcing": True,
    }
    previous = engine.request(parent, "options_shallow", **kwargs)
    enabled = engine.request(parent, "options_shallow", action_feedback=True, **kwargs)
    assert enabled == previous


def test_action_feedback_requires_explicit_branch_contract(parent):
    with pytest.raises(ValueError, match="action_feedback requires branch_delta"):
        engine.request(parent, "options_shallow", action_feedback=True)
    with pytest.raises(ValueError, match="action_feedback requires branch_delta"):
        run(parent, [], "options_shallow", action_feedback=True)


def test_action_feedback_changes_only_followup_advice_and_preserves_gate_results(
    parent,
):
    raw = patch(
        "a*mem+b",
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "expression": "a*mem+b",
                "sources": [],
            },
            {"component": "b", "kind": "dynamic", "expression": "0", "sources": []},
        ],
        new_parameters=[{"name": "b", "role": "coefficient"}],
    )
    initial = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    assert initial["diagnostic"]["code"] == "COMPONENT_IDENTITY"
    correction = branch_reply(initial, drop_actions=[{"field": "edits", "key": "b"}])
    intermediate = engine._assess_branch_reply(
        parent,
        correction,
        initial,
        common_zero_forcing=False,
        candidate_id="call1-option0",
    )
    final = branch_reply(intermediate, patch("a*mem*Gp+b"))
    replies = [raw, correction, final]
    previous, old_client = run(parent, replies, "sequential_restart", branch_delta=True)
    current, client = run(
        parent,
        replies,
        "sequential_restart",
        branch_delta=True,
        action_feedback=True,
    )
    assert client.calls[0] == old_client.calls[0]
    assert client.calls[1]["system"] == old_client.calls[1]["system"]
    assert client.calls[1]["schema"] == old_client.calls[1]["schema"]
    assert client.calls[1]["user"] != old_client.calls[1]["user"]
    packet = json.loads(client.calls[1]["user"])
    advice = packet["repair_feedback"]["action_feedback"]
    assert advice["policy"] == engine.ACTION_FEEDBACK_POLICY
    assert advice["action_choices"][0]["choices"][0]["reply_fragment"] == {
        "drop_actions": [{"field": "edits", "key": "b"}]
    }
    assert current["action_feedback_policy"] == engine.ACTION_FEEDBACK_POLICY
    assert "action_feedback_policy" not in previous
    assert (
        current["eligible_candidate_ids"]
        == previous["eligible_candidate_ids"]
        == ["call2-option0"]
    )
    for new, old in zip(current["candidates"], previous["candidates"], strict=True):
        for key in ("status", "diagnostic", "problem", "composed_patch"):
            assert new.get(key) == old.get(key)
    assert (
        current["candidates"][1]["diagnostic"]["code"]
        == "QUALIFIED_MODEL_DATA_CONFLICT"
    )
    assert (
        current["candidates"][1]["composed_patch"]["new_parameters"]
        == raw["new_parameters"]
    )
    replay, replay_client = run(
        parent,
        replies,
        "sequential_restart",
        branch_delta=True,
        action_feedback=True,
    )
    assert replay == current and replay_client.calls == client.calls


def test_branch_delta_citation_only_preserves_trial_and_scopes_feedback(parent):
    raw = patch(name="Z", evidence_ids=["unsupported_label"])
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    assert candidate["diagnostic"]["code"] == "UNKNOWN_EVIDENCE"
    correction = branch_reply(candidate, {"evidence_ids": ["public_task"]})
    result, client = run(
        parent, [raw, correction], "sequential_restart", branch_delta=True
    )
    assert result["eligible_candidate_ids"] == ["call1-option0"]
    child = result["candidates"][1]
    assert child["composed_patch"]["new_states"] == raw["new_states"]
    assert child["composed_patch"]["edits"] == raw["edits"]
    assert child["submitted_reply"] == correction
    packet = json.loads(client.calls[1]["user"])
    assert packet["training_consistency"]["evidence_scope"] == "unchanged_parent_only"
    assert (
        packet["training_consistency"]["trial_assessment"]
        == "trial_not_scientifically_assessed"
    )
    assert (
        "citation metadata"
        in packet["repair_feedback"]["guidance"]["specific_feedback"]
    )
    assert packet["repair_feedback"]["complete_intended_patch"] == raw
    assert (
        client.calls[1]["schema"] == engine.repair_reply_model(True).model_json_schema()
    )
    assert (
        client.calls[0]["schema"] == engine.StructuralRevisionReply.model_json_schema()
    )
    assert all(call["system"] == engine.system_for(True) for call in client.calls)
    assert "Every patch must contain all changes" not in engine.system_for(True)
    assert not result["model_promoted"]
    replay, replay_client = run(
        parent, [raw, correction], "sequential_restart", branch_delta=True
    )
    assert replay == result and replay_client.calls == client.calls


def test_branch_delta_retains_two_prior_parameters_on_nonlinear_repair(parent):
    raw = patch(
        "a*mem+b*X",
        name="X",
        new_parameters=[
            {"name": "b", "role": "coefficient"},
            {"name": "c", "role": "rate"},
        ],
    )
    raw["new_states"][0]["rhs"] = "c*I-X"
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    assert candidate["diagnostic"]["code"] == "QUALIFIED_MODEL_DATA_CONFLICT"
    correction = branch_reply(
        candidate,
        {
            "edits": [
                {"component": "q", "kind": "algebraic", "expression": "a*mem*Gp+b*I*X"}
            ]
        },
    )
    result, client = run(
        parent, [raw, correction], "sequential_restart", branch_delta=True
    )
    child = result["candidates"][1]
    assert child["status"] == "eligible"
    assert child["composed_patch"]["new_parameters"] == raw["new_parameters"]
    assert child["composed_patch"]["new_states"] == raw["new_states"]
    assert child["omitted_previous_actions"] == []
    packet = json.loads(client.calls[1]["user"])
    literal = packet["attempted_repairs"][0]["literal_attempted_changes"]
    assert literal["edits"][0]["expression"] == "a*mem+b*X"
    assert literal["new_states"][0]["rhs"] == "c*I-X"
    assert "description" not in literal["new_states"][0]
    assert literal["new_states"][0]["initial"]["scope"] == "trajectory"
    assert (
        packet["training_consistency"]["trial_assessment"]
        == "trial_has_qualified_conflict"
    )


def test_branch_delta_explicit_drop_still_runs_full_validation(parent):
    raw = patch(
        "a*mem+b*X", name="X", new_parameters=[{"name": "b", "role": "coefficient"}]
    )
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    dropped = branch_reply(
        candidate, drop_actions=[{"field": "new_parameters", "key": "b"}]
    )
    child = engine._assess_branch_reply(
        parent,
        dropped,
        candidate,
        common_zero_forcing=False,
        candidate_id="call1-option0",
    )
    assert child["status"] == "rejected"
    assert child["composed_patch"]["new_parameters"] == []
    assert child["composed_patch"]["edits"] == raw["edits"]
    assert child["diagnostic"]["code"] != "QUALIFIED_MODEL_DATA_CONFLICT"


def test_branch_composition_rejection_retains_pending_patch_without_adoption(parent):
    raw = patch("a*mem+X", name="X")
    initial = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    wrong = branch_reply(
        initial, {"evidence_ids": ["public_task"]}, base_candidate_id="sibling"
    )
    failed = engine._assess_branch_reply(
        parent,
        wrong,
        initial,
        common_zero_forcing=False,
        candidate_id="call1-option0",
        branch_id=initial["branch_id"],
    )
    assert failed["diagnostic"]["code"] == "BRANCH_PATCH_COMPOSITION"
    assert failed["composed_patch"] == raw
    fixed = branch_reply(failed, patch(), mode="replace")
    result, _ = run(
        parent, [raw, wrong, fixed], "sequential_restart", branch_delta=True
    )
    assert result["eligible_candidate_ids"] == ["call2-option0"]
    assert result["candidates"][1]["problem"] is None


def test_branch_withdraw_is_not_parent_acceptance(parent):
    raw = patch("a*mem+X", name="X")
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    result, _ = run(
        parent,
        [raw, branch_reply(candidate, mode="withdraw")],
        "sequential_restart",
        branch_delta=True,
    )
    assert result["generation_calls"] == 2
    assert result["candidates"][-1]["status"] == "withdrawn"
    assert result["eligible_candidate_ids"] == []


def test_branch_delta_is_not_interpreted_under_historical_contract(parent):
    raw = patch("a*mem+X", name="X")
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    correction = branch_reply(candidate, patch(), mode="replace")
    result, client = run(parent, [raw, correction, patch()], "sequential_restart")
    assert result["candidates"][1]["status"] == "rejected"
    assert "composed_patch" not in result["candidates"][1]
    assert "branch_delta_policy" not in result
    assert all(row["system"] == engine.SYSTEM for row in client.calls)


def test_branch_cached_prefix_replay_preserves_composition(parent):
    raw = patch(name="Z", evidence_ids=["unsupported_label"])
    initial = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    corrected = branch_reply(initial, {"evidence_ids": ["public_task"]})
    prefix = Client([raw])
    with pytest.raises(DeferredCall):
        engine.run_episode(
            parent,
            prefix,
            StagedModelSettings(maximum_requests=3),
            "sequential_restart",
            branch_delta=True,
        )
    result, resumed = run(
        parent, [raw, corrected], "sequential_restart", branch_delta=True
    )
    assert prefix.calls == resumed.calls
    assert result["eligible_candidate_ids"] == ["call1-option0"]


def test_common_zero_flag_reaches_request_citation_and_child_gates(parent, monkeypatch):
    original = base._assessment
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs.get("common_zero_forcing"))
        return original(*args, **kwargs)

    monkeypatch.setattr(base, "_assessment", record)
    run(parent, [patch(), patch(), patch()], "independent", common_zero_forcing=True)
    assert seen and all(seen)


def test_options_branch_delta_never_merges_sibling_declarations(parent):
    raws = [
        patch("a*mem+" + "*".join([name] * (i + 1)), name=name)
        for i, name in enumerate(("X", "Y", "Z"))
    ]
    candidates = [
        engine.assess_candidate(
            parent, raw, candidate_id=f"call0-option{i}", option_index=i
        )
        for i, raw in enumerate(raws)
    ]
    selected = engine._failed_branches(candidates, seed=3, maximum_repairs=2)
    responses = [{"options": [option(i, raw) for i, raw in enumerate(raws)]}]
    for candidate in selected:
        name = candidate["raw_reply"]["new_states"][0]["name"]
        responses.append(
            branch_reply(
                candidate,
                {
                    "edits": [
                        {
                            "component": "q",
                            "kind": "algebraic",
                            "expression": f"a*mem*Gp+{name}",
                        }
                    ]
                },
            )
        )
    result, client = run(
        parent, responses, "options_shallow", seed=3, branch_delta=True
    )
    assert len(result["eligible_candidate_ids"]) == 2
    for expected, actual, request in zip(
        selected, result["candidates"][3:], client.calls[1:], strict=True
    ):
        assert actual["branch_id"] == expected["id"]
        assert (
            actual["composed_patch"]["new_states"]
            == expected["raw_reply"]["new_states"]
        )
        packet = json.loads(request["user"])
        assert packet["repair_feedback"]["branch_reference"] == branch.reference(
            expected
        )
        assert packet["response_format"]["schema"] == "BranchRepairReply"
    assert client.calls[0]["schema"] == engine.OptionsReply.model_json_schema()


def test_branch_composed_conflicting_existing_parameter_role_rejected(parent):
    raw = patch("a*mem+X", name="X")
    initial = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    change = branch_reply(
        initial, {"new_parameters": [{"name": "a", "role": "coefficient"}]}
    )
    result = engine._assess_branch_reply(
        parent, change, initial, common_zero_forcing=False, candidate_id="call1-option0"
    )
    assert result["status"] == "rejected"
    assert result["diagnostic"]["code"] == "DECLARATION_IDENTITY"
    assert parent["problem"]["base_candidate"]["parameters"][0]["role"] == "rate"


@pytest.mark.parametrize("failure", ["truncated", "unreadable", "provider"])
def test_branch_unavailable_reply_preserves_pending_for_cached_next_delta(
    parent, failure
):
    raw = patch(
        "a*mem+b*X", name="X", new_parameters=[{"name": "b", "role": "coefficient"}]
    )
    preserved = {"id": "call1-option0", "composed_patch": raw}
    correction = branch_reply(
        preserved,
        {
            "edits": [
                {"component": "q", "kind": "algebraic", "expression": "a*mem*Gp+b*X"}
            ]
        },
    )

    class FailedSecond(Client):
        def call(self, **kwargs):
            record = super().call(**kwargs)
            if kwargs["attempt"] == 1:
                if failure == "truncated":
                    record["raw_response"]["choices"][0]["finish_reason"] = "length"
                elif failure == "unreadable":
                    record["raw_response"]["choices"][0]["message"]["content"] = (
                        "not JSON"
                    )
                else:
                    record["status"] = "failed"
                    record["raw_response"] = {"error": "provider unavailable"}
            return record

    def episode():
        client = FailedSecond([raw, {}, correction])
        result = engine.run_episode(
            parent,
            client,
            StagedModelSettings(maximum_requests=3),
            "sequential_restart",
            branch_delta=True,
        )
        return result, client

    result, client = episode()
    failed = result["candidates"][1]
    assert failed["status"] == "unavailable"
    assert failed["composed_patch"] == raw
    assert (
        failed["branch_composition"]["status"]
        == "response_unavailable_pending_preserved"
    )
    assert (
        failed["branch_composition"]["previous_pending_assessment"]
        == "trial_has_qualified_conflict"
    )
    assert result["eligible_candidate_ids"] == ["call2-option0"]
    packet = json.loads(client.calls[2]["user"])
    assert packet["repair_feedback"]["complete_intended_patch"] == raw
    assert packet["repair_feedback"]["branch_reference"] == branch.reference(preserved)
    assert (
        packet["repair_feedback"]["scientific_assessment"]
        == "no_new_trial_composed_previous_pending_patch_kept"
    )
    assert (
        result["candidates"][2]["composed_patch"]["new_parameters"]
        == raw["new_parameters"]
    )
    replay, replay_client = episode()
    assert replay == result and replay_client.calls == client.calls


@pytest.mark.parametrize("citation_only", [True, False])
def test_branch_packet_has_one_authoritative_repair_scope(parent, citation_only):
    raw = patch(
        "a*mem+X",
        name="X",
        evidence_ids=["invalid_label"] if citation_only else ["public_task"],
    )
    candidate = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    if not citation_only:
        candidate["diagnostic"]["details"]["assessment"]["repair_contract"] = (
            "Return a complete patch relative to current_model; "
            "include every intended correction again."
        )
    before = deepcopy(candidate)
    packet = json.loads(
        engine.request(
            parent,
            "sequential_restart",
            intended=candidate,
            history=[candidate],
            branch_delta=True,
        )
    )
    scopes = []

    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"repair_scope", "repair_contract"}:
                    scopes.append(child)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            assert "include every intended correction again" not in value
            assert "Return a complete patch relative" not in value

    inspect(packet)
    assert scopes and all("repair_feedback.instruction" in scope for scope in scopes)
    assert packet["repair_feedback"]["complete_intended_patch"] == raw
    assert candidate == before
    historical = engine.request(parent, "sequential_restart", intended=candidate)
    assert "Return a complete patch relative" in historical


def test_branch_scope_rewrite_preserves_same_named_scientific_symbols(parent):
    candidate = engine.assess_candidate(parent, patch("a*mem"))
    relations = {"repair_scope": "k*x", "repair_contract": "-k*x"}
    candidate["diagnostic"]["details"]["assessment"]["model_relations"] = relations
    before = deepcopy(candidate)
    packet = json.loads(
        engine.request(
            parent, "sequential_restart", intended=candidate, branch_delta=True
        )
    )
    shown = packet["repair_feedback"]["diagnostic"]["details"]["assessment"]
    assert shown["model_relations"] == relations
    assert "BranchRepairReply" in shown["repair_scope"]
    assert candidate == before
