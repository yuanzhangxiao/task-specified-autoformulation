"""Editing feedback preserves scientific choices and historical acceptance."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_decisions as decisions
from autoformalism.search import construction_edit_feedback as editing
from autoformalism.search import construction_ledger as ledger
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    process,
    variable,
)


def packet(draft, *, stage="repair", last=None, public=None, types=None):
    """Exercise the live payload builder, including the existing checks."""
    b = public or brief()
    return editing.payload(
        stage, b, context(), types or {}, b.model_dump(mode="json"), draft, None, last
    )


def issue_rows(packet, code):
    return [r for r in packet["feedback"]["issues"] if r["code"] == code]


def duplicate_draft(kind="influence", signs=("positive", "negative")):
    return apply(
        variables=[variable("y"), variable("x")],
        equations=[equation("y", "u"), equation("x", "u")],
        processes=[
            process(kind=kind, uses=[{"target": "x", "sign": s} for s in signs])
        ],
    )


@pytest.mark.parametrize("kind", ["influence", "transfer"])
@pytest.mark.parametrize("signs", [("positive", "negative"), ("positive", "positive")])
@pytest.mark.parametrize("stage", ["shared_laws", "repair"])
def test_duplicate_receiver_feedback_never_treats_kind_change_as_a_fix(
    kind, signs, stage
):
    d = duplicate_draft(kind, signs)
    before = d.model_dump()
    check = ledger.assess(brief(), context(), {}, d)
    p = packet(d, stage=stage)
    issue = issue_rows(p, "duplicate_receiver")[0]
    assert issue["field_to_edit"] == "processes[p].uses"
    assert [u["sign"] for u in issue["retained_duplicate_uses"]] == list(signs)
    assert "also applies to influence" in issue["warning"]
    assert "not automatically cancelled" in issue["warning"]
    assert "distinct" in issue["repair_options"][1]
    for r in issue_rows(p, "process_uses"):
        assert "repair_options" not in r
        assert "duplicate_receiver" in r["action"]
    assert d.model_dump() == before
    assert ledger.assess(brief(), context(), {}, d) == check
    assert not check["eligible"]


def test_nonduplicate_transfer_diagnostic_and_unchanged_draft_are_preserved():
    d = apply(
        variables=[variable("x"), variable("y")],
        equations=[equation("x", "u"), equation("y", "u")],
        processes=[process(uses=[{"target": "x", "sign": "positive"}])],
    )
    args = ("repair", brief(), context(), {}, brief().model_dump(mode="json"), d, None)
    assert editing.payload(*args) == decisions.payload(*args)
    assert issue_rows(packet(d), "transfer_signs")


def test_process_removal_feedback_uses_retained_ownership_and_is_only_conditional():
    d = duplicate_draft()
    last = {
        "error": "cannot remove unknown equations",
        "rejected_reply": {"remove_equations": ["p", "y", "rejected_only"]},
    }
    before = deepcopy(last)
    p = packet(d, last=last)
    rows = issue_rows(p, "process_edit_destination")
    assert [r["name"] for r in rows] == ["p"]
    assert rows[0]["choices"]["if_removal_is_intended"] == {"remove_processes": ["p"]}
    assert rows[0]["affected_consumers"] == ["x"]
    assert "committed nothing" in rows[0]["action"]
    assert "Omit this name from remove_equations" in rows[0]["action"]
    assert last == before and d.processes
    corrected = apply(d, **rows[0]["choices"]["if_removal_is_intended"])
    assert not corrected.processes
    assert corrected.variables == d.variables
    assert corrected.equations == d.equations
    with pytest.raises(ValueError, match="unknown equations"):
        apply(d, remove_equations=["p"])


@pytest.mark.parametrize("value", [None, "p", [None, {}, []], {"p": True}])
def test_malformed_rejected_fields_are_not_coerced_into_names(value):
    p = packet(
        duplicate_draft(),
        last={
            "error": "bad reply",
            "rejected_reply": {
                "remove_equations": value,
                "remove_processes": value,
                "processes": value,
            },
        },
    )
    assert not issue_rows(p, "process_edit_destination")
    assert not issue_rows(p, "process_replace_remove_conflict")


def test_mutually_exclusive_operations_are_explained_without_choosing_one():
    d = duplicate_draft()
    raw = {"processes": [process()], "remove_processes": ["p"]}
    p = packet(d, last={"error": "ambiguous operations", "rejected_reply": raw})
    issue = issue_rows(p, "process_replace_remove_conflict")[0]
    assert issue["names"] == ["p"]
    assert "Choose one operation" in issue["action"]
    assert issue["retained_process_names"] == ["p"]
    assert not issue["uncommitted_process_names"]
    assert d.processes[0].kind == "influence"
    with pytest.raises(ValueError, match="ambiguous replace/remove"):
        apply(d, **raw)


def test_rejected_only_process_is_omitted_not_removed_from_the_retained_draft():
    d = duplicate_draft()
    raw = {"processes": [process(name="new")], "remove_processes": ["new"]}
    p = packet(d, last={"error": "ambiguous operations", "rejected_reply": raw})
    issue = issue_rows(p, "process_replace_remove_conflict")[0]
    assert issue["uncommitted_process_names"] == ["new"]
    assert not issue["retained_process_names"]
    assert "omit both its proposed entry and its removal key" in issue["action"]
    assert not issue_rows(p, "process_edit_destination")
    with pytest.raises(ValueError, match="unknown processes"):
        apply(d, remove_processes=["new"])
    assert apply(d, stage_complete=True) == d


def all_lhs_draft():
    b = brief().model_copy(
        update={
            "public_variables": (
                *brief().public_variables,
                brief()
                .public_variables[1]
                .model_copy(update={"name": "t", "data_role": "time"}),
            )
        }
    )
    d = apply(
        public=b,
        equations=[equation(n, "u") for n in ("a", "u", "area", "t", "y", "z")],
    )
    return b, d


def test_all_invalid_lhs_are_reported_with_distinct_public_role_options():
    b, d = all_lhs_draft()
    check = ledger.assess(b, context(), {"y": "differential"}, d)
    p = packet(d, public=b, types={"y": "differential"})
    issue = issue_rows(p, "undeclared_lhs_entries")[0]
    rows = {r["exact_name"]: r for r in issue["entries"]}
    assert set(rows) == {"a", "u", "area", "t", "y", "z"}
    assert len(rows["a"]["choices"]) == 2
    assert rows["a"]["choices"][0]["edit"] == {"remove_equations": ["a"]}
    assert rows["a"]["choices"][1]["allowed_types"] == ["differential", "algebraic"]
    assert len(rows["y"]["choices"]) == 1
    assert rows["y"]["choices"][0]["allowed_types"] == ["differential"]
    for name in ("u", "area", "t"):
        assert len(rows[name]["choices"]) == 1
        assert rows[name]["choices"][0]["edit"] == {"remove_equations": [name]}
    assert len(rows["z"]["choices"]) == 2
    assert issue_rows(p, "compiler")[0]["action"].startswith(
        "See undeclared_lhs_entries"
    )
    assert ledger.assess(b, context(), {"y": "differential"}, d) == check
    assert not check["eligible"]


@pytest.mark.parametrize("model_observed", [False, True])
def test_supplied_observation_may_be_kept_supplied_or_explicitly_generated(
    model_observed,
):
    d = apply(
        variables=[variable("y")], equations=[equation("y", "a"), equation("a", "u")]
    )
    assert not ledger.assess(brief(), context(), {}, d)["eligible"]
    if model_observed:
        repaired = apply(d, variables=[variable("a", "algebraic")])
    else:
        choice = issue_rows(packet(d), "undeclared_lhs_entries")[0]["entries"][0][
            "choices"
        ][0]
        repaired = apply(d, **choice["edit"])
    assert ledger.assess(brief(), context(), {}, repaired)["eligible"]
    assert repaired.equations[0] == d.equations[0]
    assert not issue_rows(packet(repaired), "undeclared_lhs_entries")


@pytest.mark.parametrize("name", ["u", "area", "t"])
def test_external_input_covariate_and_time_still_cannot_be_generated(name):
    b, d = all_lhs_draft()
    with pytest.raises(ValueError, match="already available public"):
        apply(d, public=b, variables=[variable(name)])


def test_process_owned_algebraic_definition_is_not_missing_ordinary_declaration():
    d = apply(
        variables=[variable("y"), variable("x")],
        processes=[process()],
        equations=[equation("y", "u"), equation("x", "u")],
    )
    assert not issue_rows(packet(d), "undeclared_lhs_entries")
    assert ledger.assess(brief(), context(), {}, d)["eligible"]


def test_variable_stage_and_old_policy_packet_are_unchanged_and_not_mutated():
    b, d = all_lhs_draft()
    args = ("repair", b, context(), {}, b.model_dump(mode="json"), d, None)
    old = decisions.payload(*args)
    saved = deepcopy(old)
    new = editing.payload(*args)
    assert old == saved == decisions.payload(*args)
    for name in (
        "model_view",
        "requirements_checklist",
        "response_template",
        "stage_instructions",
        "public_brief",
    ):
        assert new[name] == old[name]
    args = ("variables", b, context(), {}, b.model_dump(mode="json"), d, None)
    assert editing.payload(*args) == decisions.payload(*args)
