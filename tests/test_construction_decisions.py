"""Generic proposer decisions preserve scientific checks and prior prompt policy."""

import json
from copy import deepcopy

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_decisions as decisions
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_prompts as prompts
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    process,
    variable,
)


def memory_brief(required=False):
    """Provide a generic pathway with either mandatory or optional memory."""
    b = brief(True)
    return b.model_copy(
        update={
            "requirements": (
                b.requirements[0].model_copy(
                    update={"requires_dynamic_memory": required}
                ),
            )
        }
    )


def packet(draft, stage="repair", public=None, validation_context=None):
    """Build the new policy with the same public evidence as the old policy."""
    b = public or memory_brief()
    return decisions.payload(
        stage,
        b,
        validation_context or context(),
        {},
        b.model_dump(mode="json"),
        draft,
        None,
    )


def bound_draft(public, valid=False):
    """Keep the public pathway valid while varying the optional memory claim."""
    return apply(
        public=public,
        variables=[variable("y"), variable("m")],
        equations=[
            equation("y", "m" if valid else "u"),
            equation("m", "u" if valid else "m"),
        ],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
    )


@pytest.mark.parametrize("required", [False, True])
def test_unassigned_optional_mechanisms_are_not_binding_tasks(required):
    b = memory_brief(required)
    d = apply(public=b, variables=[variable("y"), variable("m")])
    p = packet(d, "variables", b)
    options = p["memory_binding_options"]
    assert options == (
        [{"requirement_id": "memory", "memory_assignment_required": True}]
        if required
        else []
    )
    requested = [
        row
        for row in p["feedback"]["issues"]
        if row["code"] == "confirm_memory_assignment"
    ]
    assert bool(requested) is required
    assert not d.mechanism_bindings
    assert "scientific_review" not in p["feedback"]
    assert "memory_assignment_decisions" not in p["feedback"]
    assert "mechanism_bindings" in p["memory_assignment_rules"]
    assert "remove_bindings" in p["memory_assignment_rules"]


def test_valid_optional_memory_claim_stays_visible_without_withdrawal_request():
    b = memory_brief()
    d = bound_draft(b, valid=True)
    p = packet(d, public=b)
    assert p["memory_binding_options"] == [
        {"requirement_id": "memory", "memory_assignment_required": False}
    ]
    assert p["model_view"]["memory_assignments"] == [
        {"requirement_id": "memory", "memory_states": ["m"]}
    ]
    assert not p["feedback"]["memory_assignment_decisions"]
    assert deferred.assessment(b, context(), {}, d, None)["eligible"]


def test_optional_claim_decision_is_deduplicated_and_withdrawal_preserves_science():
    b = memory_brief()
    d = bound_draft(b)
    before = deepcopy(d)
    p = packet(d, public=b)
    choices = p["feedback"]["memory_assignment_decisions"]
    assert len(choices) == 1
    choice = choices[0]
    assert choice["retained_entry"] == {
        "requirement_id": "memory",
        "memory_states": ["m"],
    }
    assert choice["memory_assignment_required"] is False
    assert choice["if_history_is_intended"]
    failures = [
        row
        for row in p["feedback"]["issues"]
        if row["code"] in {"memory_driver_path", "memory_target_path"}
    ]
    assert {row["code"] for row in failures} == {
        "memory_driver_path",
        "memory_target_path",
    }
    assert all("assignment_decision" not in row for row in failures)
    assert all("assignment_choice" in row for row in failures)
    assert all("repair_options" not in row for row in failures)
    assert all(row["if_assignment_retained"] for row in failures)
    assert d == before
    assert not deferred.assessment(b, context(), {}, d, None)["eligible"]

    edit = choice["if_only_coverage_was_intended"]["edit"]
    assert edit == {"remove_bindings": ["memory"]}
    changed = apply(d, public=b, **edit)
    assert changed.variables == d.variables
    assert changed.equations == d.equations
    assert changed.processes == d.processes
    assert not changed.mechanism_bindings
    assert deferred.assessment(b, context(), {}, changed, None)["eligible"]
    disconnected = apply(changed, public=b, equations=[equation("y", "m")])
    assert not deferred.assessment(b, context(), {}, disconnected, None)["eligible"]


def test_optional_choice_does_not_conditionalize_the_public_path_requirement():
    b = memory_brief()
    d = apply(bound_draft(b), public=b, equations=[equation("y", "y")])
    args = ("repair", b, context(), {}, b.model_dump(mode="json"), d, None)
    old = coverage.payload(*args)
    new = decisions.payload(*args)
    public_before = [r for r in old["feedback"]["issues"] if r["code"] == "public_path"]
    public_after = [r for r in new["feedback"]["issues"] if r["code"] == "public_path"]
    # Remove the duplicate binding choice, never the public path's obligation.
    for row in public_before:
        row.pop("assignment_decision", None)
    assert public_before and public_after == public_before
    assert all("if_assignment_retained" not in row for row in public_after)


def test_required_assignment_failures_never_offer_optional_withdrawal():
    b = memory_brief(True)
    d = bound_draft(b)
    p = packet(d, public=b)
    assert not p["feedback"]["memory_assignment_decisions"]
    assignments = [
        row["assignment_decision"]
        for row in p["feedback"]["issues"]
        if "assignment_decision" in row
    ]
    assert assignments
    assert all(row["memory_assignment_required"] for row in assignments)
    assert all("if_only_coverage_was_intended" not in row for row in assignments)
    assert not deferred.assessment(b, context(), {}, d, None)["eligible"]


def test_optional_invalid_memory_type_uses_the_same_explicit_assignment_decision():
    b = memory_brief()
    d = apply(
        public=b,
        variables=[variable("y"), variable("m", "algebraic")],
        equations=[equation("y", "m"), equation("m", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
    )
    p = packet(d, public=b)
    choices = p["feedback"]["memory_assignment_decisions"]
    assert len(choices) == 1
    assert choices[0]["retained_entry"]["memory_states"] == ["m"]
    failure = next(
        row for row in p["feedback"]["issues"] if row["code"] == "memory_type"
    )
    assert "assignment_decision" not in failure
    assert "assignment_choice" in failure
    edit = choices[0]["if_only_coverage_was_intended"]["edit"]
    corrected = apply(d, public=b, **edit)
    assert corrected.equations == d.equations and corrected.variables == d.variables
    assert deferred.assessment(b, context(), {}, corrected, None)["eligible"]


def test_provisional_memory_gap_keeps_optional_decision_when_compiler_is_blocked():
    b = memory_brief()
    d = apply(bound_draft(b), public=b, variables=[variable("a", "algebraic")])
    p = packet(d, public=b)
    feedback = p["feedback"]
    assert any(row["code"] == "compiler" for row in feedback["issues"])
    assert any(
        row["requirement"] == "memory" and row["status"] == "missing_in_draft"
        for row in feedback["provisional_paths"]
    )
    choices = feedback["memory_assignment_decisions"]
    assert len(choices) == 1
    assert choices[0]["if_only_coverage_was_intended"]["edit"] == {
        "remove_bindings": ["memory"]
    }
    assert not deferred.assessment(b, context(), {}, d, None)["eligible"]


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_all_stages_preserve_public_evidence_schema_views_checks_and_v1(stage):
    b = memory_brief()
    d = bound_draft(b)
    enriched = {**b.model_dump(mode="json"), "training_observations": {"count": 7}}
    last = {"status": "accepted", "transaction_applied": True}
    continuation = {"reason": "finish outstanding declarations"}
    args = (stage, b, context(), {}, enriched, d, None, last, continuation)
    before = deepcopy((b, enriched, d, last, continuation))
    old = coverage.payload(*args)
    old_before = deepcopy(old)
    check_before = deferred.assessment(b, context(), {}, d, None)
    schema_before = prompts.response_model(stage).model_json_schema()
    new = decisions.payload(*args)
    assert (b, enriched, d, last, continuation) == before
    assert old == old_before == coverage.payload(*args)
    assert deferred.assessment(b, context(), {}, d, None) == check_before
    assert prompts.response_model(stage).model_json_schema() == schema_before
    for key in (
        "public_brief",
        "requirements_checklist",
        "model_view",
        "response_template",
        "editing_rules",
        "receiver_choices",
        "previous_edit",
    ):
        assert new[key] == old[key]
    assert decisions.SYSTEM == coverage.SYSTEM
    assert decisions.model_view is coverage.model_view
    assert decisions.repair_budget is coverage.repair_budget
    assert "memory_assignment_rules" not in old
    assert "scientific_review" not in old["feedback"]
    assert "memory_assignment_decisions" not in old["feedback"]
    assert ("scientific_review" in new["feedback"]) is (stage != "variables")


def test_scientific_questions_reference_actual_model_without_classifying_prose():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )
    p = packet(d, public=brief())
    review = p["feedback"]["scientific_review"]
    rates = next(q for q in review["questions"] if q["topic"] == "rate_directions")
    assert rates["processes"] == ["p"]
    assert rates["local_rhs"] == ["x"]
    assert set(rates["processes"]) <= {r["name"] for r in p["model_view"]["processes"]}
    assert set(rates["local_rhs"]) <= {
        r["variable"] for r in p["model_view"]["balances"]
    }
    assert "score" not in review and "passed" not in review
    assert all("edit" not in q for q in review["questions"])
    changed = apply(
        d,
        processes=[
            process(scientific_meaning="Ambiguous signed net effect, not a flow")
        ],
    )
    assert packet(changed, public=brief())["feedback"]["scientific_review"] == review
    assert deferred.assessment(brief(), context(), {}, d, None)["eligible"]
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]


def test_unrelated_targets_and_no_named_flows_do_not_create_required_connections():
    raw = brief().model_dump(mode="json")
    raw["public_variables"].append({"name": "z", "data_role": "target"})
    raw["scientific_context"] = "Independent outputs y and z have unrelated drivers."
    b = PublicScientificBrief.model_validate(raw)
    c = ValidationContext(
        targets=("y", "z"),
        auxiliaries=("a",),
        external_inputs=("u",),
        fixed_covariates=("area",),
    )
    d = apply(
        public=b,
        variables=[variable("y"), variable("z")],
        equations=[equation("y", "u"), equation("z", "a")],
    )
    before = deepcopy(d)
    p = packet(d, public=b, validation_context=c)
    questions = {q["topic"]: q for q in p["feedback"]["scientific_review"]["questions"]}
    assert questions["rate_directions"]["processes"] == []
    assert questions["rate_directions"]["local_rhs"] == ["y", "z"]
    assert questions["related_quantities"]["prediction_targets"] == ["y", "z"]
    assert not p["feedback"]["issues"]
    assert not p["memory_binding_options"]
    assert not d.processes and not d.mechanism_bindings
    assert d == before
    assert deferred.assessment(b, c, {}, d, None)["eligible"]


def test_empty_draft_has_no_rate_inventory_or_invented_process():
    p = packet(ledger.Draft(), "shared_laws", brief())
    questions = p["feedback"]["scientific_review"]["questions"]
    assert [q["topic"] for q in questions] == ["related_quantities"]
    assert p["model_view"]["processes"] == []


def test_optional_decisions_follow_identifiers_under_generic_renaming():
    b = memory_brief()
    d = bound_draft(b)
    names = {"memory": "history_claim", "m": "reservoir", "y": "readout", "u": "drive"}

    def rename(value):
        if isinstance(value, str):
            return names.get(value, value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    renamed_b = PublicScientificBrief.model_validate(rename(b.model_dump(mode="json")))
    renamed_d = ledger.Draft.model_validate(rename(d.model_dump(mode="json")))
    c = ValidationContext(
        targets=("readout",),
        auxiliaries=("a",),
        external_inputs=("drive",),
        fixed_covariates=("area",),
    )
    old = packet(d, public=b)["feedback"]["memory_assignment_decisions"]
    new = packet(renamed_d, public=renamed_b, validation_context=c)["feedback"][
        "memory_assignment_decisions"
    ]
    assert new == rename(old)
    assert "meal" not in json.dumps(new) and "insulin" not in json.dumps(new)
