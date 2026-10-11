"""Minimal notation, conditional repairs and unchanged scientific acceptance."""

import json
from copy import deepcopy

import pytest

from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_readable as readable
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    process,
    variable,
)


def packet(draft, stage="repair", public=None, last=None, definitions=None):
    b = public or brief()
    return readable.payload(
        stage,
        b,
        context(),
        definitions or {},
        b.model_dump(mode="json"),
        draft,
        None,
        last,
    )


def test_view_has_one_name_per_shared_process_and_no_fitted_labels():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u", "x"), {"name": "y", "terms": []}],
    )
    v = readable.model_view(d)
    assert v["processes"][0]["topology"] == "p depends on (x)"
    assert v["balances"][0]["topology"] == "d(x)/dt = ± f(u, x) - (1) * p"
    assert v["balances"][1]["topology"] == "d(y)/dt = (1/area) * p"
    assert v["balances"][0]["local_process_meanings_in_order"] == ["joint effect"]
    text = json.dumps(v)
    assert all(s not in text for s in ("Phi_", "g_p_use", "x_local_0", "c_x_local"))
    assert "not fixed to one" in v["notation"]


def test_local_processes_keep_joint_arguments_multiplicity_and_constant_terms():
    e = equation("y", "u", "a")
    e["terms"] += [deepcopy(e["terms"][0]), {**e["terms"][0], "sources": []}]
    d = apply(variables=[variable("y", "algebraic")], equations=[e])
    assert readable.model_view(d)["balances"][0]["topology"] == (
        "y = ± f(u, a) ± f(u, a) ± f()"
    )
    assert "separate local process" in readable.model_view(d)["notation"]


def test_parameter_question_does_not_reinterpret_or_delete_any_name():
    d = apply(
        variables=[
            variable("y"),
            {**variable("k", "algebraic"), "scientific_role": "parameter"},
        ],
        equations=[equation("y", "k", "z")],
    )
    before = deepcopy(d)
    feedback = packet(d)["feedback"]
    roles = feedback["symbol_role_questions"]
    assert {r["name"] for r in roles} == {"k", "z"}
    assert all("Confirm through your edits" in r["if_coefficient"] for r in roles)
    assert "sources" in roles[0]["locations"][0]
    assert d == before
    assert not deferred.assessment(brief(), context(), {}, d, None)["eligible"]
    # A k-prefixed, fully defined physical quantity is not diagnosed as a parameter.
    valid = apply(
        variables=[variable("y"), variable("k_state")],
        equations=[equation("k_state", "u"), equation("y", "k_state")],
    )
    assert not packet(valid)["feedback"]["symbol_role_questions"]


def test_cycle_feedback_can_change_type_without_touching_process_or_uses():
    d = apply(
        variables=[variable("x", "algebraic"), variable("y")],
        processes=[process()],
        equations=[{"name": "x", "terms": []}, {"name": "y", "terms": []}],
    )
    cycle = next(
        i
        for i in packet(d)["feedback"]["issues"]
        if i["code"] == "algebraic_dependency_cycle"
    )
    edit = cycle["conditional_type_edits"][0]["edit"]
    assert set(edit) == {"variables"}
    changed = apply(d, **edit)
    assert changed.processes == d.processes
    assert not any(
        i["code"] == "algebraic_dependency_cycle"
        for i in packet(changed)["feedback"]["issues"]
    )
    forbidden = next(
        i
        for i in packet(d, definitions={"x": "algebraic"})["feedback"]["issues"]
        if i["code"] == "algebraic_dependency_cycle"
    )
    assert not forbidden["conditional_type_edits"]


def test_conversion_clarification_preserves_rejection_and_real_dependencies():
    d = apply(variables=[variable("y")])
    last = {
        "status": "rejected",
        "transaction_applied": False,
        "error": "p -> y: invalid fixed conversion 'k'",
        "rejected_reply": {
            "processes": [
                process(uses=[{"target": "y", "sign": "positive", "conversion": "k"}])
            ]
        },
    }
    before = deepcopy(d)
    p = packet(d, last=last)
    q = next(
        i for i in p["feedback"]["issues"] if i["code"] == "conversion_role_decision"
    )
    assert q["attempted"][0]["uses"][0]["conversion"] == "k"
    assert "null" in q["action"] and "confirm" in q["action"]
    assert p["previous_edit"]["applied"] is False and before == d


def test_required_memory_path_is_visible_with_and_without_compilation():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("y"), variable("m")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
        equations=[equation("m", "u"), equation("y", "u")],
    )
    f = packet(d, public=b)["feedback"]
    issue = next(i for i in f["issues"] if i["code"] == "memory_target_path")
    assert issue["binding_context"][0]["memory_states"] == ["m"]
    assert "mechanism_topology:memory" in f["repair_agenda"]["failed_requirements"]
    broken = apply(d, public=b, equations=[equation("y", "unknown")])
    f = packet(broken, public=b)["feedback"]
    assert any(
        r["source"] == "m" and r["target"] == "y"
        for r in f["repair_agenda"]["required_paths"]
    )


def test_process_reference_conflict_asks_additive_or_composite_without_splitting():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[{"name": "x", "terms": []}, equation("y", "p", "u")],
    )
    before = deepcopy(d)
    row = next(
        i
        for i in packet(d)["feedback"]["issues"]
        if i["code"] == "process_consumer_decision"
    )
    assert "composite" in row["decision"] and "cannot split" in row["decision"]
    assert row["existing_contribution"][0]["process"] == "p" and before == d


def test_missing_topology_does_not_ask_to_redeclare_automatic_contributions():
    d = apply(variables=[variable("x"), variable("y")], processes=[process()])
    p = packet(d, stage="equations")
    issue = next(i for i in p["feedback"]["issues"] if i["code"] == "compiler")
    assert issue["pending_topology_entries"] == ["x", "y"]
    assert "terms=[]" in issue["action"]
    assert not deferred.assessment(brief(), context(), {}, d, None)["eligible"]


def test_changed_process_receipt_shows_lost_effect_without_restoring_it():
    d = apply(variables=[variable("x"), variable("y")], processes=[process()])
    after = apply(
        d,
        processes=[
            process(
                kind="influence",
                depends_on=["u"],
                uses=[{"target": "y", "sign": "positive"}],
            )
        ],
    )
    last = {
        "status": "accepted",
        "transaction_applied": True,
        "edit_effects": handoff.edit_effects(d, after),
    }
    row = packet(after, last=last)["feedback"]["recently_changed_processes"][0]
    assert row["dependencies_before"] == ["x"] and row["dependencies_after"] == ["u"]
    assert len(row["uses_before"]) == 2 and len(row["uses_after"]) == 1
    assert len(after.processes[0].uses) == 1


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_public_context_remains_verbatim_and_projection_does_not_mutate(stage):
    d = apply(variables=[variable("y")])
    before = deepcopy(d)
    p = packet(d, stage)
    assert p["public_brief"] == brief().model_dump(mode="json")
    assert d == before and p["stage_instructions"] == readable.stage_text(stage)


def test_rejected_supplied_declarations_list_all_and_keep_valid_retry_content():
    raw = {
        "variables": [
            variable("y"),
            variable("u", "algebraic"),
            variable("area", "algebraic"),
        ]
    }
    last = {
        "status": "rejected",
        "transaction_applied": False,
        "error": "u is already supplied",
        "rejected_reply": raw,
    }
    d = readable.ledger.Draft()
    row = next(
        i
        for i in packet(d, "variables", last=last)["feedback"]["issues"]
        if i["code"] == "supplied_declarations_in_rejected_reply"
    )
    assert row["names"] == ["area", "u"]
    assert row["conditional_variables_retry"]["variables"] == [variable("y")]
    assert not d.variables and len(raw["variables"]) == 3
