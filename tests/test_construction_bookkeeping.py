"""Current-only bookkeeping, immutable evidence and explicit representation edits."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def shared_draft():
    return apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )


def normalize(raw, draft, improved=True):
    return ledger.normalize_reply(
        brief(), raw, draft=draft, ignore_definition_description=improved
    )


def test_only_description_difference_normalizes_preserving_both_texts_and_science():
    draft = shared_draft()
    repeated = equation("p", "x")  # Different description, same structured term.
    raw = patch(equations=[repeated], stage_complete=True).model_dump(mode="json")
    original = json.dumps(raw, sort_keys=True)
    assert normalize(raw, draft, False) == (raw, [])  # Frozen strict comparator.
    fixed, log = normalize(raw, draft)
    result = ledger.apply_patch(brief(), draft, ledger.DraftPatch.model_validate(fixed))
    assert result == draft
    assert fixed["equations"] == []
    assert log[0]["equation"] == repeated
    assert log[0]["canonical_definition"]["terms"][0]["scientific_role"] == (
        draft.processes[0].scientific_meaning
    )
    assert not log[0]["scientific_equivalence_asserted"]
    assert original == json.dumps(raw, sort_keys=True)
    assert len(ledger.assembled_equations(brief(), result)) == 3


@pytest.mark.parametrize("fault", ["drivers", "sign", "term_count", "remove", "type"])
def test_repetition_never_hides_computational_conflicts_or_partial_edits(fault):
    draft = shared_draft()
    repeated = equation("p", "x")
    edits = {"equations": [repeated], "variables": [variable("new_state")]}
    if fault == "drivers":
        repeated["terms"][0]["sources"] = ["u"]
    elif fault == "sign":
        repeated["terms"][0]["outer_weight_sign"] = "negative"
    elif fault == "term_count":
        repeated["terms"] *= 2
    elif fault == "remove":
        edits["remove_equations"] = ["p"]
    else:
        edits["variables"].append(variable("p"))
    raw = patch(**edits).model_dump(mode="json")
    fixed, _ = normalize(raw, draft)
    with pytest.raises(ValueError):
        ledger.apply_patch(brief(), draft, ledger.DraftPatch.model_validate(fixed))
    assert draft == shared_draft()
    if fault in {"drivers", "sign", "term_count"}:
        conflicts = bookkeeping.definition_conflicts(draft, ledger.DraftPatch(**fixed))
        assert conflicts[0]["process"] == "p"
        assert conflicts[0]["ordinary_definition"] == repeated
        assert "processes" in conflicts[0]["instruction"]


def test_same_reply_process_revision_owns_definition_signs_and_conversions():
    draft = shared_draft()
    declaration = process(depends_on=["u"])
    raw = patch(
        processes=[declaration], equations=[equation("p", "u")], stage_complete=True
    ).model_dump(mode="json")
    fixed, _ = normalize(raw, draft)
    updated = ledger.apply_patch(brief(), draft, ledger.DraftPatch(**fixed))
    assert updated.processes[0].model_dump(mode="json") == declaration
    assert updated.processes[0].uses == draft.processes[0].uses
    assert updated.equations == draft.equations


def test_read_only_preview_and_atomic_type_change_follow_structured_declarations():
    draft = shared_draft()
    readout = equation("y", "x", "area")
    readout["terms"][0]["scientific_role"] = "algebraic"
    prose_only = apply(draft, equations=[readout])
    assert bookkeeping.balance_view(prose_only, "y")["lhs"] == "d(y)/dt"
    assert next(v for v in prose_only.variables if v.name == "y").definition == (
        "differential"
    )
    revised = patch(variables=[variable("y", "algebraic")], equations=[readout])
    updated = ledger.apply_patch(brief(), draft, revised)
    snapshot = bookkeeping.snapshot(brief(), updated)
    assert "assembled_equations" not in snapshot
    assert snapshot["declarations"] == updated.model_dump(mode="json")
    generated = snapshot["read_only_generated_process_definitions"][0]
    assert generated["name"] == "p" and generated["edit_via"] == "processes"
    view = next(v for v in snapshot["read_only_balances"] if v["name"] == "y")
    assert view["lhs"] == "y" and view["balance_topology"].startswith("y = ")
    assert view["ordinary_terms"] == readout["terms"]
    assert view["runtime_inserted_process_uses"][0]["conversion"] == "1/area"
    # No new cross-stage exception; separate type changes wait for overall repair.
    with pytest.raises(ValueError, match="fixes the variable inventory"):
        schedules.validate_scope("separate", "equations", "y", revised, draft)
    schedules.validate_scope("separate", "repair", None, revised, draft)
    schedules.validate_scope("joint_adaptive", "equations", None, revised, draft)
    # Invalid companion edits roll back the whole transaction, including the type.
    bad = revised.model_copy(update={"remove_equations": ("y",)})
    with pytest.raises(ValueError, match="ambiguous"):
        ledger.apply_patch(brief(), draft, bad)
    assert bookkeeping.balance_view(draft, "y")["lhs"] == "d(y)/dt"


def test_composite_question_displays_balance_without_splitting_or_removing_effects():
    draft = apply(shared_draft(), equations=[equation("x", "u", "p")])
    original = draft.model_dump(mode="json")
    check = ledger.assess(brief(), context(), {}, draft)
    enhanced = bookkeeping.assessment_context(check, draft)
    assert not enhanced["eligible"] and enhanced["eligible"] == check["eligible"]
    issue = next(
        e for e in enhanced["errors"] if e["code"] == "process_consumer_decision"
    )
    assert "already includes" in issue["question"]
    balance = issue["current_balance"]
    assert "phi0(u, p)" in balance["balance_topology"]
    assert "- gP0 * (1) * p" in balance["balance_topology"]
    assert len(balance["ordinary_terms"]) == 1
    assert issue["clarification_actions"][1]["if"] == "a distinct joint interaction"
    assert draft.model_dump(mode="json") == original
    assert "current_balance" not in check["errors"][0]


def execute(root, send, **kwargs):
    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=root / "calls",
        namespace="bookkeeping",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        **kwargs,
    )
    return schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        root / "construction",
        "joint_adaptive",
        bookkeeping_policy=bookkeeping.POLICY,
    )


def test_live_feedback_normalization_rejection_and_cached_resume(tmp_path):
    calls = []

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        assert payload["bookkeeping_policy"] == bookkeeping.POLICY
        draft = payload["current_draft"]
        assert "read_only_balances" in draft and "assembled_equations" not in draft
        if payload["stage"] == "relationships":
            reply = patch(
                variables=[variable("x"), variable("y")],
                processes=[process()],
                stage_complete=True,
            )
        elif payload["stage"] == "equations":
            reply = patch(
                equations=[
                    equation("p", "x"),
                    equation("x", "u", "p"),
                    {"name": "y", "terms": []},
                ],
                stage_complete=True,
            )
        elif len(calls) == 3:
            failures = payload["runtime_diagnostics"]["remaining_topology_failures"]
            issue = next(
                e for e in failures if e["code"] == "process_consumer_decision"
            )
            assert issue["current_balance"]["lhs"] == "d(x)/dt"
            # Conflicting repair: nothing in this response should survive.
            reply = patch(
                variables=[variable("bad")],
                equations=[equation("p", "u")],
                stage_complete=True,
            )
        else:
            receipt = payload["last_edit_result"]
            assert not receipt["transaction_applied"]
            assert receipt["definition_conflicts"][0]["process"] == "p"
            assert all(v["name"] != "bad" for v in draft["declarations"]["variables"])
            reply = patch(equations=[equation("x", "u")], stage_complete=True)
        return response(reply.model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        execute(tmp_path, send, can_start=lambda: len(calls) < 2)
    result = execute(tmp_path, send)
    assert result["status"] == "topology_complete" and len(calls) == 4
    event = sealed_read(tmp_path / "construction/events/001.json")
    assert event["normalizations"][0]["code"] == "repeated_process_definition"
    assert not result["before_repair"]["assessment"]["eligible"]
    assert execute(tmp_path, send) == result and len(calls) == 4
    assert len(result["draft"]["processes"]) == 1


def test_frozen_policy_is_explicit_and_not_enabled_in_minimal_comparison(tmp_path):
    source_fixture(tmp_path)
    source = tmp_path / "new"
    with pytest.raises(ValueError, match="separate from the prompt comparison"):
        campaign.freeze(
            source,
            tmp_path / "disallowed",
            study="prompt_comparison",
            bookkeeping_policy=bookkeeping.POLICY,
        )
    plan = campaign.freeze(
        source,
        tmp_path / "current",
        study="shared_law_comparison",
        bookkeeping_policy=bookkeeping.POLICY,
    )
    assert campaign.verify(tmp_path / "current") == plan
    assert plan["config"]["bookkeeping_policy"] == bookkeeping.POLICY
    assert len(plan["tasks"]) == 12
    baseline = campaign.freeze(source, tmp_path / "minimal", study="prompt_comparison")
    assert baseline["config"]["bookkeeping_policy"] == "legacy"
    for family in ("minimal", "current"):
        with pytest.raises(ValueError, match="separate from the prompt comparison"):
            bookkeeping.validate_policy(bookkeeping.POLICY, family)


def test_saved_replies_keep_original_predecessors_and_never_call_a_provider(tmp_path):
    from scripts.audit_construction_handoff import audit

    source_fixture(tmp_path)
    source = tmp_path / "campaign"
    campaign.freeze(tmp_path / "new", source, study="basin_confirmation")
    result = audit(
        source, tmp_path / "audit.json", bookkeeping_policy=bookkeeping.POLICY
    )
    assert result["llm_calls"] == result["optimizer_calls"] == 0
    assert result["bookkeeping_policy"] == bookkeeping.POLICY
    assert result["saved_attempts"] == 0
    assert result["planned_tasks"] == 6


def test_no_use_question_does_not_claim_a_contribution_has_been_inserted():
    draft = shared_draft()
    draft = apply(draft, variables=[variable("z")], equations=[equation("z", "p", "u")])
    check = bookkeeping.assessment_context(
        ledger.assess(brief(), context(), {}, draft), draft
    )
    issue = next(e for e in check["errors"] if e["code"] == "process_consumer_decision")
    assert "No signed use" in issue["question"]
    assert "already includes" not in issue["question"]
    assert not issue["current_balance"]["runtime_inserted_process_uses"]
    assert handoff.process_reference_issues(draft)  # Existing gate, no new gate.


def test_preview_does_not_double_count_an_already_assembled_identity_reference():
    repeated = equation("y", "p")
    repeated["terms"][0]["outer_weight_sign"] = "positive"
    # Forward-reference delivery may retain an ordinary record until closure.
    draft = apply(shared_draft(), equations=[repeated])
    assembled = next(
        e for e in ledger.assembled_equations(brief(), draft) if e.name == "y"
    )
    assert len(assembled.terms) == 1
    view = bookkeeping.balance_view(draft, "y")
    assert view["ordinary_indices_already_represented_by_process_uses"] == [0]
    assert "phi" not in view["balance_topology"]
    assert view["balance_topology"] == "d(y)/dt = gP0 * (1/area) * p"
    assert view["ordinary_terms"] == repeated["terms"]  # Original evidence retained.
