"""Retained states cannot inherit process-owned equation deletion advice."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_compact as compact
from autoformalism.search import construction_edit_feedback as editing
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_ownership_feedback as ownership
from tests.test_construction_edit_feedback import issue_rows
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)


def packet(draft, *, last=None, stage="repair", policy=ownership):
    """Use the complete payload projection and its unchanged deterministic checks."""
    b = brief()
    return policy.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, None, last
    )


def state_draft(*, definition="differential", with_equation=True):
    """Keep a state and a separately owned process to distinguish deletion keys."""
    return apply(
        variables=[variable("x", definition), variable("y")],
        equations=[equation("y", "u")]
        + ([equation("x", "u")] if with_equation else []),
        processes=[
            process(kind="influence", uses=[{"target": "y", "sign": "positive"}])
        ],
    )


def collision_last(draft, *, receiver="x", **updates):
    """Reproduce the diagnostics emitted for an actual rejected state/process edit."""
    proposal = patch(
        processes=[
            process(
                name="x",
                depends_on=["u"],
                kind="influence",
                uses=[{"target": receiver, "sign": "positive"}],
            )
        ],
        **updates,
    )
    return {
        "error": "x: shared instantaneous law conflicts with state",
        "rejected_reply": proposal.model_dump(mode="json"),
        "name_conflicts": compact.naming_conflicts(brief(), draft, proposal),
        "definition_conflicts": bookkeeping.definition_conflicts(draft, proposal),
    }


@pytest.mark.parametrize("stage", ["shared_laws", "equations", "repair"])
def test_retained_state_overrides_process_equation_deletion_advice(stage):
    draft = state_draft()
    last = collision_last(draft)
    before = deepcopy(last)
    old = packet(draft, last=last, stage=stage, policy=editing)
    new = packet(draft, last=last, stage=stage)
    row = issue_rows(new, "state_process_ownership")[0]
    assert row["name"] == "x"
    assert row["retained_owner"] == "differential_state"
    assert row["ordinary_equation_retained"] and not row["process_retained"]
    assert row["attempted_process_addition"]
    assert row["if_retaining_state"]["field_to_edit"] == "equations[x]"
    assert "Do not delete the state equation" in row["if_retaining_state"]["action"]
    assert (
        "distinct unused process name"
        in row["if_retaining_state"]["if_distinct_instantaneous_law_is_intended"]
    )
    conflict = issue_rows(new, "conflicting_process_definition")[0]
    old_conflict = issue_rows(old, "conflicting_process_definition")[0]
    assert conflict["ordinary_definition"] == old_conflict["ordinary_definition"]
    assert "state_process_ownership" in conflict["instruction"]
    assert "omit this name from equations" not in conflict["instruction"]
    name_issue = issue_rows(new, "process_name_collision")[0]
    assert name_issue["proposed_drivers"] == ["u"]
    assert len(name_issue["repair_options"]) == 1
    assert "state_process_ownership" in name_issue["repair_options"][0]
    assert last == before
    assert packet(draft, last=last, stage=stage, policy=editing) == old
    assert "omit this name from equations" in old_conflict["instruction"]
    for key in new.keys() - {"feedback"}:
        assert new[key] == old[key]


def test_unknown_process_removal_identifies_state_without_a_new_process_proposal():
    draft = state_draft()
    last = {
        "error": "cannot remove unknown processes: ['x']",
        "rejected_reply": {"remove_processes": ["x", "unknown"]},
    }
    row = issue_rows(packet(draft, last=last), "state_process_ownership")[0]
    assert row["attempted_process_removal"]
    assert not row["attempted_process_addition"]
    assert "no process with this name" in row["action"]
    assert "never installed" in row["action"]
    assert apply(draft, stage_complete=True) == draft
    with pytest.raises(ValueError, match="unknown processes"):
        apply(draft, remove_processes=["x"])


def test_missing_state_equation_is_completed_not_treated_as_a_process_definition():
    draft = state_draft(with_equation=False)
    new = packet(draft, last=collision_last(draft))
    row = issue_rows(new, "state_process_ownership")[0]
    assert not row["ordinary_equation_retained"]
    assert "otherwise add it" in row["if_retaining_state"]["action"]
    assert not issue_rows(new, "conflicting_process_definition")


@pytest.mark.parametrize("also_retained", [False, True])
def test_simultaneous_replace_remove_guidance_is_scoped_to_actual_ownership(
    also_retained,
):
    draft = state_draft()
    raw = {"processes": [{"name": "x"}], "remove_processes": ["x"]}
    if also_retained:
        raw["processes"].append(process())
        raw["remove_processes"].append("p")
    new = packet(draft, last={"error": "ambiguous operations", "rejected_reply": raw})
    row = issue_rows(new, "process_replace_remove_conflict")[0]
    assert row["state_owned_names"] == ["x"]
    assert row["uncommitted_process_names"] == ["x"]
    assert row["retained_process_names"] == (["p"] if also_retained else [])
    assert "state_process_ownership" in row["action"]
    if also_retained:
        assert row["other_process_names"] == ["p"]
        assert "For other_process_names only: Choose one operation" in row["action"]
    else:
        assert not row["other_process_names"]
        assert "return its complete processes entry" not in row["action"]


def test_scientific_type_change_remains_explicit_and_is_not_applied_by_feedback():
    draft = state_draft()
    last = collision_last(draft, receiver="y", variables=[variable("x", "algebraic")])
    before = draft.model_dump()
    row = issue_rows(packet(draft, last=last), "state_process_ownership")[0]
    assert row["retained_owner"] == "differential_state"
    assert "explicit variables edit" in row["if_changing_scientific_type"]
    assert "If abandoning the uncommitted addition, omit" in row["action"]
    assert "abandon it by" not in row["action"]
    assert draft.model_dump() == before
    # A scientific redesign still uses the existing transaction schema/checks.
    redesigned = apply(
        draft,
        variables=[variable("x", "algebraic")],
        processes=last["rejected_reply"]["processes"],
        remove_equations=["x"],
    )
    assert next(v for v in redesigned.variables if v.name == "x").definition == (
        "algebraic"
    )
    assert ledger.assess(brief(), context(), {}, redesigned)["eligible"]


def test_corrected_local_rhs_or_distinct_process_edits_use_unchanged_acceptance():
    draft = state_draft()
    before = draft.model_dump()
    check = ledger.assess(brief(), context(), {}, draft)
    last = collision_last(draft)
    packet(draft, last=last)
    assert ledger.assess(brief(), context(), {}, draft) == check
    assert draft.model_dump() == before
    with pytest.raises(ValueError, match="conflicts with state"):
        apply(draft, processes=last["rejected_reply"]["processes"])
    local = apply(draft, equations=[equation("x", "u", "x")])
    assert ledger.assess(brief(), context(), {}, local)["eligible"]
    distinct = apply(
        draft,
        processes=[{**last["rejected_reply"]["processes"][0], "name": "q"}],
    )
    assert ledger.assess(brief(), context(), {}, distinct)["eligible"]
    assert distinct.variables == draft.variables
    assert distinct.equations == draft.equations


@pytest.mark.parametrize("raw", [None, "x", 1, [], [None, {}]])
def test_nonobject_rejected_reply_does_not_create_ownership_claim(raw):
    draft = state_draft()
    last = {"error": "bad reply", "rejected_reply": raw}
    assert packet(draft, last=last) == packet(draft, last=last, policy=editing)


@pytest.mark.parametrize("value", [None, "x", {"name": "x"}, [None, {}, []]])
def test_malformed_rejected_process_fields_do_not_create_names(value):
    draft = state_draft()
    last = {
        "error": "bad reply",
        "rejected_reply": {"processes": value, "remove_processes": value},
    }
    assert packet(draft, last=last) == packet(draft, last=last, policy=editing)


def test_malformed_entries_are_ignored_without_losing_valid_exact_names():
    draft = state_draft()
    raw = {
        "processes": [None, [], {"name": []}, {"name": "x"}],
        "remove_processes": [None, {}, [], "x"],
    }
    rows = issue_rows(
        packet(draft, last={"error": "bad reply", "rejected_reply": raw}),
        "state_process_ownership",
    )
    assert [row["name"] for row in rows] == ["x"]


def test_rejected_diagnostics_can_qualify_advice_without_a_raw_reply():
    draft = state_draft()
    last = collision_last(draft)
    last.pop("rejected_reply")
    assert issue_rows(packet(draft, last=last), "state_process_ownership")


@pytest.mark.parametrize("definition", ["differential", "algebraic"])
def test_unaffected_packets_and_variable_stage_preserve_v1(definition):
    draft = state_draft(definition=definition)
    assert packet(draft) == packet(draft, policy=editing)
    last = collision_last(draft)
    assert packet(draft, last=last, stage="variables") == packet(
        draft, last=last, stage="variables", policy=editing
    )
    if definition == "algebraic":
        assert packet(draft, last=last) == packet(draft, last=last, policy=editing)
    last = {"error": None, "rejected_reply": last["rejected_reply"]}
    assert packet(draft, last=last) == packet(draft, last=last, policy=editing)


def test_existing_process_removal_advice_and_policy_exports_are_unchanged():
    draft = state_draft()
    last = {
        "error": "unknown equations",
        "rejected_reply": {"remove_equations": ["p"]},
    }
    assert packet(draft, last=last) == packet(draft, last=last, policy=editing)
    assert ownership.POLICY == "minimal-edit-feedback-2"
    for name in ("SYSTEM", "EDITING", "model_view", "stage_text", "repair_budget"):
        assert getattr(ownership, name) is getattr(editing, name)
