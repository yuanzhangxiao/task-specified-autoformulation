"""Reply controls never become science under the opt-in construction policy."""

from copy import deepcopy

import pytest

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_control_fields as controls
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_ownership_feedback as ownership
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)


def checked_apply(draft=None, **kwargs):
    """Exercise the prospective call order without modifying historical ledger code."""
    value = patch(**kwargs)
    controls.validate_patch(value, brief=brief())
    return ledger.apply_patch(brief(), draft or ledger.Draft(), value)


def packet(draft, *, last=None, stage="repair"):
    b = brief()
    old = ownership.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, None, last
    )
    return controls.augment_payload(old, draft, last)


def issues(value, code):
    return [row for row in value["feedback"]["issues"] if row["code"] == code]


@pytest.mark.parametrize("name", sorted(controls.CONTROL_NAMES))
@pytest.mark.parametrize("field", ["variables", "equations", "processes"])
def test_exact_reply_controls_rejected_in_all_scientific_declaration_categories(
    name, field
):
    entry = {
        "variables": variable(name),
        "equations": equation(name, "u"),
        "processes": process(
            name=name,
            depends_on=["u"],
            kind="influence",
            uses=[{"target": "y", "sign": "positive"}],
        ),
    }[field]
    draft = apply(variables=[variable("y")], equations=[equation("y", "u")])
    before = draft.model_dump()
    with pytest.raises(ValueError, match="reply-control fields") as caught:
        checked_apply(draft, **{field: [entry]}, stage_complete=True)
    assert f"{field}[{name}]" in str(caught.value)
    assert "No part of this reply was committed" in str(caught.value)
    assert draft.model_dump() == before


def test_control_reservation_matches_only_top_level_edit_operations():
    assert (
        ledger.DraftPatch.model_fields.keys() - ledger.Draft.model_fields.keys()
    ) == controls.CONTROL_NAMES


@pytest.mark.parametrize(
    "name",
    [
        "stage_complete_rate",
        "my_stage_complete",
        "Stage_complete",
        "remove_parameters",
        "name",
        "terms",
        "variables",
        "processes",
    ],
)
def test_prefixes_and_generic_schema_words_remain_valid_scientific_names(name):
    draft = checked_apply(
        variables=[variable("y"), variable(name)],
        equations=[equation("y", name), equation(name, "u")],
    )
    assert ledger.assess(brief(), context(), {}, draft)["eligible"]


def test_public_name_collision_is_reported_as_schema_ambiguity_not_scientific_error():
    b = PublicScientificBrief.model_validate(
        {
            **brief().model_dump(mode="json"),
            "public_variables": [
                *brief().model_dump(mode="json")["public_variables"],
                {"name": "stage_complete", "data_role": "target"},
            ],
        }
    )
    with pytest.raises(ValueError, match="ambiguous public/schema namespace") as error:
        controls.validate_patch(patch(variables=[variable("stage_complete")]), brief=b)
    assert "Do not automatically rename or delete a public quantity" in str(error.value)
    value = controls.augment_payload(
        {"feedback": {"issues": []}},
        ledger.Draft(),
        {
            "error": str(error.value),
            "rejected_reply": {"variables": [variable("stage_complete")]},
        },
        brief=b,
    )
    assert issues(value, "public_control_namespace_conflict")
    assert not issues(value, "reply_control_is_not_quantity")


def test_actual_accepted_completion_process_pattern_is_blocked_only_by_new_policy():
    draft = apply(variables=[variable("y")])
    scientific_process = process(
        name="input_to_y",
        depends_on=["u"],
        kind="influence",
        uses=[{"target": "y", "sign": "positive"}],
    )
    metadata_process = {
        **scientific_process,
        "name": "stage_complete",
        "depends_on": ["input_to_y"],
        "scientific_meaning": "This is a placeholder to mark the stage as complete.",
    }
    historical = apply(
        draft,
        processes=[scientific_process, metadata_process],
        equations=[{"name": "y", "terms": []}],
        stage_complete=True,
    )
    assert ledger.assess(brief(), context(), {}, historical)["eligible"]
    with pytest.raises(ValueError, match=r"processes\[stage_complete\]"):
        checked_apply(
            draft,
            processes=[scientific_process, metadata_process],
            equations=[{"name": "y", "terms": []}],
            stage_complete=True,
        )
    # Valid edits in the same rejected reply stay uncommitted.
    assert not draft.processes


def test_actual_alien_device_pattern_gets_exact_equation_removal_without_type_choices():
    draft = apply(
        variables=[variable("y"), variable("memory")],
        equations=[
            equation("y", "memory"),
            equation("memory", "u"),
            {"name": "stage_complete", "terms": []},
        ],
    )
    before = draft.model_dump()
    last = {
        "error": "cannot remove unknown variables: ['stage_complete']",
        "rejected_reply": {
            "remove_variables": ["stage_complete"],
            "remove_equations": ["stage_complete"],
            "stage_complete": True,
        },
    }
    value = packet(draft, last=last)
    row = issues(value, "removal_namespace_mismatch")[0]
    assert row["attempted_field"] == "remove_variables"
    assert row["retained_entries"] == [
        {
            "field": "equations",
            "key": "name",
            "removal_if_intended": {"remove_equations": ["stage_complete"]},
        }
    ]
    assert "Omit 'stage_complete' from remove_variables" in row["action"]
    assert not any(
        r["name"] == "stage_complete"
        for r in value["feedback"]["symbol_role_questions"]
    )
    lhs = issues(value, "undeclared_lhs_entries")[0]["entries"][0]
    assert lhs["choices"] == [
        {"action": "Follow reply_control_is_not_quantity for this name."}
    ]
    with pytest.raises(ValueError, match="unknown variables"):
        apply(draft, **last["rejected_reply"])
    repaired = checked_apply(draft, remove_equations=["stage_complete"])
    assert ledger.assess(brief(), context(), {}, repaired)["eligible"]
    assert draft.model_dump() == before


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_rejected_control_addition_was_never_installed_in_any_stage(stage):
    draft = apply(variables=[variable("y")], equations=[equation("y", "u")])
    last = {
        "error": "reply-control fields are not scientific declarations",
        "rejected_reply": {"equations": [{"name": "stage_complete", "terms": []}]},
    }
    row = issues(packet(draft, last=last, stage=stage), "reply_control_is_not_quantity")
    row = row[0]
    assert not row["retained_entries"]
    assert row["attempted_declaration_rejected"]
    assert "do not request its removal" in row["action"]


def test_wrong_removal_lists_all_retained_owners_without_deleting_either():
    draft = apply(variables=[variable("y")], equations=[equation("y", "u")])
    last = {
        "error": "unknown processes",
        "rejected_reply": {"remove_processes": ["y", "never_retained"]},
    }
    rows = issues(packet(draft, last=last), "removal_namespace_mismatch")
    row = next(r for r in rows if r["name"] == "y")
    assert {r["field"] for r in row["retained_entries"]} == {"variables", "equations"}
    assert "do not delete every listed owner automatically" in row["action"]
    absent = next(r for r in rows if r["name"] == "never_retained")
    assert not absent["retained_entries"]
    assert "absent from every retained edit category" in absent["action"]


@pytest.mark.parametrize("raw", [None, "bad", 1, [], {"remove_variables": [None, {}]}])
def test_malformed_reply_does_not_create_names_or_modify_existing_packet(raw):
    draft = apply(variables=[variable("y")], equations=[equation("y", "u")])
    base = {"feedback": {"issues": [{"code": "unchanged"}]}}
    before = deepcopy(base)
    result = controls.augment_payload(
        base, draft, {"error": "bad", "rejected_reply": raw}
    )
    assert result == base
    assert before == base


def test_successful_reply_or_correct_removal_does_not_add_wrong_namespace_feedback():
    draft = apply(variables=[variable("y")], equations=[equation("y", "u")])
    for last in [
        {"error": None, "rejected_reply": {"remove_processes": ["y"]}},
        {"error": "another error", "rejected_reply": {"remove_equations": ["y"]}},
    ]:
        assert not issues(packet(draft, last=last), "removal_namespace_mismatch")
