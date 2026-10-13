"""Coverage guidance preserves science authority, acceptance and historical prompts."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_completion as completion
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_deferred as deferred
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)


def packet(draft, stage="repair", public=None, last=None):
    b = public or brief()
    return coverage.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, None, last
    )


def test_missing_public_path_points_to_target_without_assigning_a_scientific_edge():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("y"), variable("m")],
        equations=[equation("m", "u"), equation("y", "y")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
    )
    before = deepcopy(d)
    issues = packet(d, public=b)["feedback"]["issues"]
    public = next(i for i in issues if i["code"] == "public_path")
    memory = next(i for i in issues if i["code"] == "memory_target_path")
    assert public["missing_path"] == {"from": "u", "to": "y"}
    assert memory["missing_path"] == {"from_any": ["m"], "to": "y"}
    assert "at least one" in memory["repair_options"][0]
    assert "no edge or sign" in public["repair_options"][0]
    assert d == before
    assert not deferred.assessment(b, context(), {}, d, None)["eligible"]
    # A mediator is allowed; there is no forced direct u -> y edge.
    changed = apply(d, public=b, equations=[equation("y", "m")])
    assert deferred.assessment(b, context(), {}, changed, None)["eligible"]
    assert not any(
        "missing_path" in i for i in packet(changed, public=b)["feedback"]["issues"]
    )


def test_memory_driver_and_alternative_memory_states_keep_original_quantifiers():
    row = coverage._path_destination(
        {"code": "memory_driver_path", "driver": "u", "memory": "m"}
    )
    assert row["missing_path"] == {"from": "u", "to": "m"}
    row = coverage._path_destination(
        {"code": "memory_target_path", "memories": ["m1", "m2"], "target": "y"}
    )
    assert row["missing_path"] == {"from_any": ["m1", "m2"], "to": "y"}
    for issue in (
        {"code": "forbidden_path", "source": "u", "target": "y"},
        {"code": "public_path", "kind": "composition_path"},
        {"code": "public_path", "kind": "memory_path"},
    ):
        assert coverage._path_destination(issue) == issue


def test_flow_feedback_offers_scientific_sign_correction_not_classification_escape():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[
            process(
                uses=[
                    {"target": "x", "sign": "negative"},
                    {"target": "y", "sign": "negative"},
                ]
            )
        ],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )
    before = deepcopy(d)
    p = packet(d, "shared_laws")
    decision = next(
        i
        for i in p["feedback"]["issues"]
        if i["code"] == "flow_classification_decision"
    )
    assert (
        "correct it instead" in decision["if_boundary_flow_or_other_nontransfer_effect"]
    )
    assert decision["retained_entry"] == d.processes[0].model_dump(mode="json")
    assert d == before
    assert not deferred.assessment(brief(), context(), {}, d, None)["eligible"]
    changed = apply(d, processes=[process()])
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]
    # Historical policy retains its old conditional example verbatim.
    old = completion.payload(
        "shared_laws", brief(), context(), {}, brief().model_dump(mode="json"), d, None
    )
    old_decision = next(
        i
        for i in old["feedback"]["issues"]
        if i["code"] == "flow_classification_decision"
    )
    assert "edit" in old_decision["if_boundary_flow_or_other_nontransfer_effect"]


def test_duplicate_equations_need_one_complete_list_but_never_merge_automatically():
    d = apply(variables=[variable("y")], equations=[equation("y", "u")])
    rejected = {"equations": [equation("y", "u"), equation("y", "a")]}
    with pytest.raises(ValueError, match="duplicate keys in equations"):
        patch(**rejected)
    before = deepcopy(d)
    p = packet(
        d, last={"error": "duplicate keys in equations", "rejected_reply": rejected}
    )
    q = next(
        i for i in p["feedback"]["issues"] if i["code"] == "duplicate_equation_entries"
    )
    assert q["names"] == ["y"] and "entire previous transaction" in q["action"]
    assert d == before and rejected["equations"][1]["terms"][0]["sources"] == ["a"]
    assert not any(
        i["code"] == "duplicate_equation_entries"
        for i in packet(d)["feedback"]["issues"]
    )
    # Distinct terms, including equal dependencies, remain a proposer decision.
    two_terms = equation("y", "u")
    two_terms["terms"] *= 2
    changed = apply(d, equations=[two_terms])
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]


def test_supplied_observation_allows_either_explicit_modeling_or_boundary_effect():
    p = process(
        depends_on=["y", "a"],
        uses=[
            {"target": "y", "sign": "negative"},
            {"target": "a", "sign": "positive"},
        ],
    )
    d = apply(
        variables=[variable("y")], processes=[p], equations=[{"name": "y", "terms": []}]
    )
    before = deepcopy(d)
    q = next(
        i
        for i in packet(d, "shared_laws")["feedback"]["issues"]
        if i["code"] == "supplied_receiver"
    )["preserve_valid_effects"]
    assert "conditional_process_edit" not in q
    assert q["currently_valid_uses"] == [
        {"target": "y", "sign": "negative", "conversion": None}
    ]
    assert d == before
    modeled = apply(
        d, variables=[variable("a")], equations=[{"name": "a", "terms": []}]
    )
    assert deferred.assessment(brief(), context(), {}, modeled, None)["eligible"]
    boundary = apply(d, processes=[{**p, "kind": "influence", "uses": p["uses"][:1]}])
    assert deferred.assessment(brief(), context(), {}, boundary, None)["eligible"]
    with pytest.raises(ValueError):
        apply(d, variables=[variable("u")])


@pytest.mark.parametrize(
    "raw", [None, [], {"equations": None}, {"equations": [None, {"name": []}]}]
)
def test_untrusted_rejected_shapes_do_not_break_feedback(raw):
    assert coverage._duplicate_equations({"rejected_reply": raw}) == []


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_public_evidence_schema_and_checks_preserved_without_more_prompt_copies(stage):
    b = brief(True)
    d = apply(public=b, variables=[variable("y")])
    old = completion.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), d, None
    )
    new = packet(d, stage, b)
    for key in (
        "public_brief",
        "requirements_checklist",
        "model_view",
        "response_template",
        "editing_rules",
        "receiver_choices",
    ):
        assert new[key] == old[key]
    if stage == "variables":
        assert new == old
    else:
        assert new["stage_instructions"] != old["stage_instructions"]
