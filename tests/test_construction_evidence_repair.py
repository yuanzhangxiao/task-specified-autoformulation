"""Repair-only guidance preserves initial requests and scientific ownership."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_evidence as v1
from autoformalism.search import construction_evidence_repair as v2
from tests.test_construction_evidence import apply, assess, boundary, choice, draft
from tests.test_construction_ledger import context, equation, process, variable
from tests.test_construction_realizations import public, record


def packet(d, stage="repair", *, api=v2, **kwargs):
    """Build the exact live prompt facade without making model or network calls."""
    return api.payload(
        stage,
        public(),
        context(),
        {},
        public().model_dump(mode="json"),
        d,
        None,
        **kwargs,
    )


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations"])
def test_every_initial_stage_payload_and_schema_is_identical_to_v1(stage):
    d = draft("u")
    kwargs = {
        "boundary_contract": boundary(),
        "training_highlights": "TRAIN: y varies.",
    }
    assert v2.SYSTEM == v1.SYSTEM
    assert v2.stage_text(stage) == v1.stage_text(stage)
    assert v2.response_model(stage).model_json_schema() == (
        v1.response_model(stage).model_json_schema()
    )
    assert packet(d, stage, api=v1, **kwargs) == packet(d, stage, **kwargs)


def test_all_schemas_and_validation_are_preserved_including_repair():
    assert v2.Draft is v1.Draft
    assert v2.DraftPatch is v1.DraftPatch
    assert v2.apply_patch is v1.apply_patch
    assert v2.assess is v1.assess
    assert v2.response_model("repair").model_json_schema() == (
        v1.response_model("repair").model_json_schema()
    )
    assert v2.POLICIES == (v1.POLICY, "minimal-evidence-feedback-2")


def test_source_only_record_feedback_does_not_request_equation_deletion():
    d = draft("u")
    p = packet(d)
    work = p["feedback"]["repair_work"]
    assert work["structural_eligibility"] is False
    assert not work["model_or_structural_corrections"]
    assert work["record_corrections"][0]["code"] == "mechanism_realization_source_only"
    assert work["record_corrections"][0]["public_path_status"] == "passed"
    assert "repair only those records" in work["instruction"]
    assert "no equation deletion" in work["instruction"]
    assert "repair_agenda" not in p["feedback"]
    assert not assess(d)["eligible"]


def test_missing_path_is_not_hidden_by_record_only_correction():
    d = apply(draft("u"), equations=[equation("y", "y")])
    work = packet(d)["feedback"]["repair_work"]
    assert work["model_or_structural_corrections"]
    assert work["record_corrections"][0]["public_path_status"] == "failed"
    assert "currently failed checks concern records" not in work["instruction"]


def test_unknown_requirement_record_cleanup_keeps_valid_model_components():
    d = apply(
        draft(),
        mechanism_realizations=[{**record("y"), "requirement_id": "extra"}],
    )
    p = packet(d)
    work = p["feedback"]["repair_work"]
    assert not work["model_or_structural_corrections"]
    assert work["record_corrections"][0]["requirement"] == "extra"
    issue = next(
        row for row in p["feedback"]["issues"] if row.get("requirement") == "extra"
    )
    assert issue["record_only_removal"] == {"remove_realizations": ["extra"]}
    assert "does not request removing its variables" in issue["action"]
    cleaned = apply(d, **issue["record_only_removal"])
    assert cleaned.variables == d.variables
    assert cleaned.equations == d.equations
    assert assess(cleaned)["eligible"]
    assert "completion_example" not in p["feedback"]


def test_boundary_choices_distinguish_inserted_one_use_process_from_local_index():
    d = apply(
        draft(),
        processes=[
            process(
                name="outlet",
                kind="influence",
                depends_on=["y"],
                uses=[{"target": "y", "sign": "negative", "conversion": None}],
            )
        ],
    )
    choices = v2.external_reference_choices(public(), d, boundary())[0]
    selectors = [row["selector"] for row in choices["reference_choices"]]
    assert selectors == [
        {"obligation_id": "loss", "receiver": "y", "local_term_index": 0},
        {"obligation_id": "loss", "receiver": "y", "local_term_index": 1},
        {"obligation_id": "loss", "receiver": "y", "shared_process": "outlet"},
    ]
    p = packet(d, boundary_contract=boundary())
    assert (
        "even when the model view calls it a one-receiver" in p["external_effect_rules"]
    )
    assert "scientifically represents the loss" in p["external_effect_rules"]
    updated = apply(
        d, external_exchanges=[choice(local_term_index=None, shared_process="outlet")]
    )
    assert assess(updated, boundary())["eligible"]
    # A displayed inserted use is never offered as the nonexistent local index 2.
    invalid = apply(d, external_exchanges=[choice(local_term_index=2)])
    assert not assess(invalid, boundary())["eligible"]


def test_boundary_choices_exclude_internal_transfer_and_unrelated_receiver():
    d = apply(
        draft(),
        variables=[variable("x")],
        equations=[equation("x", "x")],
        processes=[
            process(
                depends_on=["y"],
                uses=[
                    {"target": "x", "sign": "positive", "conversion": None},
                    {"target": "y", "sign": "negative", "conversion": None},
                ],
            )
        ],
    )
    options = v2.external_reference_choices(public(), d, boundary())[0][
        "reference_choices"
    ]
    assert all(row["selector"]["receiver"] == "y" for row in options)
    assert all("shared_process" not in row["selector"] for row in options)
    assert len(options) == 2


def test_boundary_choices_allow_equivalent_balance_coordinate_without_inference():
    d = apply(
        draft(),
        variables=[variable("y", "algebraic"), variable("storage")],
        equations=[equation("y", "storage"), equation("storage", "storage", "u")],
    )
    options = v2.external_reference_choices(public(), d, boundary())[0][
        "reference_choices"
    ]
    assert [row["selector"] for row in options] == [
        {"obligation_id": "loss", "receiver": "storage", "local_term_index": 0}
    ]


def test_question_only_ready_draft_still_invites_scientific_revision():
    questions = [{"question_id": "q1", "question": "Conditional witness"}]
    p = packet(draft(), compatibility_questions=questions)
    work = p["feedback"]["repair_work"]
    assert work["mode"] == "advisory_review"
    assert work["structural_eligibility"]
    assert work["pending_advisory_questions"] == ["q1"]
    assert "structurally accepted but inadequate" in p["stage_instructions"]
    choices = p["feedback"]["advisory_reply_choices"]
    assert {row["disposition"] for row in choices["choices"]} == {
        "revise",
        "retain_approximation",
        "dispute_assumption",
    }
    assert choices["completion_control"] == {"stage_complete": True}
    assert "no choice is preselected" in choices["instruction"]
    assert "completion_example" not in p["feedback"]


def test_revise_answer_stays_visible_if_question_remains_pending():
    question = {"question_id": "q1", "question": "Conditional witness"}
    decision = {
        "question_id": "q1",
        "disposition": "revise",
        "explanation": "Add a dependence.",
    }
    p = packet(
        draft(),
        compatibility_questions=[question],
        compatibility_state={"decisions": [decision]},
    )
    work = p["feedback"]["repair_work"]
    assert work["pending_questions_previously_answered_revise"] == ["q1"]
    assert "does not change topology or resolve" in p["stage_instructions"]


def test_completion_preserves_acknowledged_conflict_and_scientific_reason():
    question = {
        "question_id": "q1",
        "target": "y",
        "property": "scalar_autonomous",
        "question": "Conditional reversal witness",
    }
    decision = {
        "question_id": "q1",
        "disposition": "retain_approximation",
        "explanation": "Retain the coarse model for later fitted-error assessment.",
    }
    state = {"advisory_questions": [question], "decisions": [decision]}
    p = packet(draft(), compatibility_state=state)
    work = p["feedback"]["repair_work"]
    assert work["mode"] == "completion_confirmation"
    assert not work["pending_advisory_questions"]
    assert work["accepted_advisory_dispositions"] == [
        {
            **decision,
            "target": "y",
            "property": "scalar_autonomous",
            "question_status": "answered",
            "scientific_status": "unverified",
        }
    ]
    assert "still has the acknowledged" in work["advisory_disposition_scope"]
    assert "not scientific approval" in work["advisory_disposition_scope"]
    assert "witnesses" not in work["accepted_advisory_dispositions"][0]
    assert state["decisions"] == [decision]


def test_structural_summaries_distinguish_driver_and_memory_paths():
    b = public(memory=True)
    b = b.model_copy(
        update={
            "requirements": tuple(
                r.model_copy(
                    update={"positive_requirements": ("Delayed input pathway",)}
                )
                for r in b.requirements
            )
        }
    )
    edit = v1.DraftPatch(
        variables=[variable("y"), variable("m")],
        equations=[equation("y", "m"), equation("m", "m")],
        mechanism_realizations=[record("m", "y", memory_states=["m"])],
        stage_complete=True,
    )
    d = v1.apply_patch(b, v1.Draft(), edit)
    p = v2.payload("repair", b, context(), {}, b.model_dump(mode="json"), d, None)
    summaries = p["feedback"]["repair_work"]["model_or_structural_corrections"]
    paths = [row for row in summaries if row["code"] == "public_path"]
    assert {row["kind"] for row in paths} == {"driver_path", "memory_path"}
    assert all(row["driver"] == "u" for row in paths)
    assert len({tuple(row["feedback_issue_indices"]) for row in paths}) == 2
    memory = next(row for row in summaries if row["code"] == "memory_driver_path")
    assert memory["driver"] == "u"
    assert memory["memory"] == "m"


def test_completion_only_state_preserves_explicit_false_without_demanding_more_edits():
    d = draft()
    last = {
        "status": "accepted",
        "transaction_applied": True,
        "stage_complete": False,
        "edit_effects": {},
    }
    continuation = {"no_progress": True, "instruction": "Change topology again."}
    p = packet(d, last=last, continuation=continuation)
    work = p["feedback"]["repair_work"]
    assert work["mode"] == "completion_confirmation"
    assert work["previous_accepted_stage_complete"] is False
    assert "never silently overridden" in work["instruction"]
    assert "Change topology again" not in p["feedback"]["continuation"]["instruction"]
    assert p["feedback"]["continuation"]["no_progress"] is True
    example = p["feedback"]["completion_example"]["reply"]
    edit = v2.DraftPatch.model_validate(example)
    assert v2.apply_patch(public(), d, edit) == d
    assert edit.stage_complete is True
    assert last["stage_complete"] is False
    assert continuation["instruction"] == "Change topology again."


def test_required_check_loss_is_factual_not_rollback_or_scientific_regression():
    before = draft()
    after = apply(before, equations=[equation("y", "y")])
    losses = v2.lost_required_predicates(assess(before), assess(after))
    assert any(
        row.get("driver") == "u" and row["after_status"] == "failed" for row in losses
    )
    p = packet(
        after, last={"transaction_applied": True, "required_predicates_lost": losses}
    )
    warning = p["feedback"]["previously_passed_checks_changed"]
    assert warning["automatic_rollback"] is False
    assert warning["additional_acceptance_gate"] is False
    assert (
        "not evidence that the previous model was scientifically correct"
        in warning["action"]
    )
    assert after.equations != before.equations


def test_unavailable_graph_is_not_reported_as_failed_path_or_scientific_error():
    before = assess(draft())
    after = deepcopy(before)
    after["public_structure_checks"][0]["passed"] = None
    losses = v2.lost_required_predicates(before, after)
    assert losses[0]["after_status"] == "unavailable"
    assert losses[0]["before_status"] == "passed"
    assert v2.lost_required_predicates(before, before) == []


def test_rejected_transaction_cannot_claim_required_checks_were_lost():
    p = packet(
        draft(),
        last={
            "transaction_applied": False,
            "required_predicates_lost": [{"predicate": "fabricated"}],
        },
    )
    assert "previously_passed_checks_changed" not in p["feedback"]
