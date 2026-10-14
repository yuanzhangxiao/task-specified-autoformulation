"""General mechanism realization prompts retain evidence and explicit edit ownership."""

import json

import pytest
from pydantic import ValidationError

from autoformalism.search import construction_protocol as protocol
from autoformalism.search import construction_realizations as prompts
from autoformalism.search import mechanism_realizations as core
from tests.test_construction_ledger import brief, context, equation, process, variable


def public(memory=False):
    """One public mechanism can have an algebraic or explicit memory contract."""
    b = brief(True)
    return b.model_copy(
        update={
            "requirements": tuple(
                r.model_copy(
                    update={
                        "id": "effect",
                        "requires_dynamic_memory": memory,
                        "public_requirement": "Represent the input effect on y.",
                    }
                )
                for r in b.requirements
            )
        }
    )


def apply(draft=None, *, b=None, **changes):
    """Apply a typed opt-in patch using the real transactional implementation."""
    patch = core.DraftPatch.model_validate({"stage_complete": False, **changes})
    return core.apply_patch(b or public(), draft or core.Draft(), patch)


def packet(draft, stage="repair", *, b=None, last=None):
    """Use the exact live payload entry point with unchanged public evidence."""
    b = b or public()
    enriched = b.model_dump(mode="json")
    enriched["training_summary"] = {"source": "training", "observed_count": 3}
    return prompts.payload(stage, b, context(), {}, enriched, draft, None, last)


def record(*components, memory_states=()):
    """Represent a mechanism explicitly without requiring a memory annotation."""
    return {
        "requirement_id": "effect",
        "components": list(components),
        "memory_states": list(memory_states),
    }


def algebraic_draft():
    """Use algebraic components within the runtime's required dynamic model."""
    return apply(
        variables=[
            variable("y", "algebraic"),
            variable("m", "algebraic"),
            variable("x"),
        ],
        equations=[equation("m", "u"), equation("y", "m"), equation("x", "u")],
        mechanism_realizations=[record("m", "y")],
    )


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_new_requests_show_algebraic_realization_and_preserve_public_brief(stage):
    d = algebraic_draft()
    p = packet(d, stage)
    expected = public().model_dump(mode="json")
    expected["training_summary"] = {"source": "training", "observed_count": 3}
    assert p["public_brief"] == expected
    assert p["model_view"]["mechanism_realizations"] == [record("m", "y")]
    assert p["response_template"]["mechanism_realizations"] == []
    assert p["response_template"]["remove_realizations"] == []
    text = json.dumps(p)
    for obsolete in (
        "mechanism_bindings",
        "remove_bindings",
        "memory_binding_options",
        "memory_assignment_rules",
        "memory_assignments",
        "memory_assignment_decisions",
    ):
        assert obsolete not in text
    row = next(
        r for r in p["requirements_checklist"] if r.get("requirement") == "effect"
    )
    assert row["declaration_status"] == "satisfied"
    assert row["inferred_type"] == "algebraic"
    assert row["behavior_status"] == "unverified"
    assert row["structural_status"] == (
        "passed" if stage in {"equations", "repair"} else "deferred"
    )
    assert "variables, processes" in prompts.SYSTEM
    assert "scientific coverage" in prompts.SYSTEM


def test_missing_algebraic_mechanism_record_has_actionable_variable_feedback():
    d = apply(variables=[variable("y", "algebraic")])
    p = packet(d, "variables")
    issue = next(
        i for i in p["feedback"]["issues"] if i["code"] == "mechanism_realization"
    )
    assert issue["name"] == "effect"
    assert issue["status"] == "missing"
    assert issue["field_to_edit"] == "mechanism_realizations"
    assert issue["retained_entry"] is None
    assert "Do not add a memory state" in issue["action"]
    assert "For every public mechanism" in p["stage_instructions"]
    assert "No topology is needed yet" in p["stage_instructions"]
    assert not any(i["code"] == "equation_destination" for i in p["feedback"]["issues"])
    assert "mechanism_bindings" not in json.dumps(p)


def test_missing_required_memory_uses_realization_feedback_without_legacy_options():
    b = public(True)
    p = packet(apply(b=b, variables=[variable("y"), variable("m")]), b=b)
    issue = next(
        i for i in p["feedback"]["issues"] if i["code"] == "mechanism_realization"
    )
    assert "needs differential components" in issue["memory_rule"]
    assert "distinct from its listed drivers and targets" in issue["memory_rule"]
    assert "memory_binding" not in json.dumps(p)
    assert "if_optional_history_was_unintended" not in json.dumps(p)


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_response_schema_exposes_only_new_records_and_retains_stage_boundaries(stage):
    model = prompts.response_model(stage)
    schema = model.model_json_schema()
    assert "mechanism_bindings" not in json.dumps(schema)
    assert "remove_bindings" not in json.dumps(schema)
    reply = model.model_validate(
        {"stage_complete": True, "mechanism_realizations": [record("y")]}
    )
    assert reply.mechanism_realizations[0].components == ("y",)
    with pytest.raises(ValidationError):
        model.model_validate({"stage_complete": True, "mechanism_bindings": []})
    assert ("equations" in model.model_fields) == (stage in {"equations", "repair"})
    assert ("processes" in model.model_fields) == (stage != "variables")


def test_process_realization_has_derived_type_and_no_second_memory_mapping():
    d = apply(
        variables=[variable("y")],
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": None}],
            )
        ],
        equations=[{"name": "y", "terms": []}],
        mechanism_realizations=[record("p", "y")],
    )
    p = packet(d)
    row = next(
        r for r in p["requirements_checklist"] if r.get("requirement") == "effect"
    )
    assert row["inferred_type"] == "mixed"
    assert row["structural_status"] == "passed"
    assert p["model_view"]["mechanism_realizations"] == [record("p", "y")]
    assert "memory_assignments" not in p["model_view"]


@pytest.mark.parametrize("memory", [False, True])
def test_failed_memory_path_targets_complete_record_and_only_optional_history_can_clear(
    memory,
):
    b = public(memory)
    d = apply(
        b=b,
        variables=[variable("y"), variable("m")],
        equations=[equation("m", "u"), equation("y", "u")],
        mechanism_realizations=[record("m", memory_states=["m"])],
    )
    p = packet(d, b=b)
    issue = next(
        i for i in p["feedback"]["issues"] if i["code"] == "memory_target_path"
    )
    assert issue["field_to_edit"] == "mechanism_realizations"
    assert issue["retained_entry"] == record("m", memory_states=["m"])
    if memory:
        assert "if_optional_history_was_unintended" not in issue
    else:
        assert issue["if_optional_history_was_unintended"]["edit"] == {
            "mechanism_realizations": [record("m")]
        }
    text = json.dumps(p)
    assert "mechanism_bindings" not in text
    assert "remove_bindings" not in text
    assert "memory_assignment_decisions" not in text


def test_wrong_deletion_category_identifies_realization_or_quantity_owner():
    d = algebraic_draft()
    last = {
        "error": "unknown entry",
        "rejected_reply": {
            "remove_variables": ["effect"],
            "remove_realizations": ["m"],
        },
    }
    p = packet(d, last=last)
    issues = {
        r["name"]: r
        for r in p["feedback"]["issues"]
        if r["code"] == "removal_namespace_mismatch"
    }
    assert issues["effect"]["retained_entries"] == [
        {
            "field": "mechanism_realizations",
            "key": "requirement_id",
            "removal_if_intended": {"remove_realizations": ["effect"]},
        }
    ]
    assert {r["field"] for r in issues["m"]["retained_entries"]} == {
        "variables",
        "equations",
    }
    assert "do not remove every owner automatically" in issues["m"]["action"]
    assert "remove_bindings" not in json.dumps(p)


@pytest.mark.parametrize("field", ["mechanism_bindings", "remove_bindings"])
def test_obsolete_reply_field_gets_explicit_migration_feedback_without_inferred_edit(
    field,
):
    p = packet(
        algebraic_draft(),
        last={
            "error": "extra field rejected",
            "rejected_reply": {field: []},
        },
    )
    issue = next(
        i for i in p["feedback"]["issues"] if i["code"] == "obsolete_mechanism_fields"
    )
    assert issue["rejected_fields"] == [field]
    assert "components and memory_states" in issue["action"]
    assert "No fields were migrated automatically" in issue["action"]
    assert field not in p["response_template"]


def test_previous_edit_receipt_keeps_complete_realization_edit_category():
    before = algebraic_draft()
    after = apply(before, mechanism_realizations=[record("m")])
    p = packet(
        after,
        last={
            "status": "accepted",
            "transaction_applied": True,
            "edit_effects": core.edit_effects(before, after),
        },
    )
    assert p["previous_edit"]["changed"]["mechanism_realizations"] == ["effect"]
    assert "mechanism_bindings" not in json.dumps(p["previous_edit"])


def test_reply_controls_are_not_offered_as_scientific_quantity_choices():
    d = core.Draft.model_validate(
        {
            "variables": [variable("y")],
            "equations": [equation("remove_realizations", "u")],
        }
    )
    p = packet(
        d,
        last={
            "error": "unknown variables",
            "rejected_reply": {"remove_variables": ["remove_realizations"]},
        },
    )
    issues = p["feedback"]["issues"]
    assert any(i["code"] == "reply_control_is_not_quantity" for i in issues)
    destination = next(i for i in issues if i["code"] == "removal_namespace_mismatch")
    assert destination["retained_entries"][0]["removal_if_intended"] == {
        "remove_equations": ["remove_realizations"]
    }
    assert all(
        q["name"] != "remove_realizations"
        for q in p["feedback"]["symbol_role_questions"]
    )


def test_repair_budget_and_signature_keep_protocol_policy_boundaries():
    d = algebraic_draft()
    p = packet(d)
    prompts.repair_budget(p, 10, 2, 262144, 1000, 262144)
    assert p["feedback"]["repair_budget"]["remaining_after_this_call"] == 7
    assert "three consecutive" in p["feedback"]["repair_budget"]["early_stop"]
    assert prompts.REPAIR_REQUESTS == protocol.REPAIR_REQUESTS
    assert prompts.REPAIR_TOKENS == protocol.REPAIR_TOKENS
    assert prompts.repair_signature(d, None, []) != prompts.repair_signature(
        apply(d, mechanism_realizations=[record("m")]), None, []
    )
    assert protocol.POLICY == "minimal-protocol-clarity-1"
    assert "mechanism_bindings" in protocol.MEMORY_RULE


@pytest.mark.parametrize("definition", ["differential", "algebraic"])
def test_repair_signature_ignores_realization_member_order_in_actual_errors(definition):
    d = apply(
        variables=[variable("y"), variable("m", definition), variable("n", definition)],
        equations=[equation(name, "u") for name in ("y", "m", "n")],
        mechanism_realizations=[record("m", "n", memory_states=["m", "n"])],
    )
    reordered = apply(
        d, mechanism_realizations=[record("n", "m", memory_states=["n", "m"])]
    )
    errors = core.assess(public(), context(), {}, d)["errors"]
    reordered_errors = core.assess(public(), context(), {}, reordered)["errors"]
    assert errors != reordered_errors
    original = json.dumps(errors)
    assert prompts.repair_signature(d, None, errors) == prompts.repair_signature(
        reordered, None, reordered_errors
    )
    assert json.dumps(errors) == original


def test_repair_signature_preserves_ordered_paths_and_scientific_error_text():
    d = algebraic_draft()
    error = {"code": "path", "witness": ["u", "m", "y"], "meaning": "positive inflow"}
    signature = prompts.repair_signature(d, None, [error])
    assert signature != prompts.repair_signature(
        d, None, [{**error, "witness": ["y", "m", "u"]}]
    )
    assert signature != prompts.repair_signature(
        d, None, [{**error, "meaning": "positive outflow"}]
    )
    assert signature["errors"] == [error]
