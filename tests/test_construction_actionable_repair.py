"""Actionable repair keeps uncertainty and maps only explicitly chosen effects."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.search import construction_actionable_repair as prompts
from autoformalism.search import construction_evidence as core
from autoformalism.search import construction_evidence_repair as previous
from tests.test_construction_evidence import apply, assess, boundary, choice, draft
from tests.test_construction_ledger import context, equation, process, variable
from tests.test_construction_realizations import public


def packet(d, *, stage="repair", **kwargs):
    return prompts.payload(
        stage,
        public(),
        context(),
        {},
        public().model_dump(mode="json"),
        d,
        None,
        **kwargs,
    )


def raw_reply(**updates):
    value = prompts.response_model("repair")(stage_complete=True).model_dump(
        mode="json"
    )
    value.update(updates)
    return value


def with_outlet():
    return apply(
        draft(),
        equations=[equation("y", "u")],
        processes=[
            process(
                name="outlet",
                kind="influence",
                depends_on=["y"],
                uses=[{"target": "y", "sign": "negative", "conversion": None}],
            )
        ],
    )


def select(d, *, shared="outlet"):
    options = prompts.external_handle_options(public(), d, boundary())[0]
    option = next(
        row
        for row in options["contributions"]
        if row["stored_reference"].get("shared_process") == shared
    )
    return {
        "obligation_id": "loss",
        "contribution_handle": option["contribution_handle"],
        "explanation": "This is my intended external contribution.",
    }


def normalize(d, raw, *, contract=None):
    value, changes = prompts.normalize_reply(
        public(), raw, draft=d, boundary_contract=contract or boundary()
    )
    return prompts.DraftPatch.model_validate(value), changes


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations"])
def test_initial_prompts_schemas_and_public_evidence_remain_exact(stage):
    d = draft()
    kwargs = {"boundary_contract": boundary(), "training_highlights": "Train only."}
    old = previous.payload(
        stage,
        public(),
        context(),
        {},
        public().model_dump(mode="json"),
        d,
        None,
        **kwargs,
    )
    assert packet(d, stage=stage, **kwargs) == old
    assert prompts.response_model(stage).model_json_schema() == (
        previous.response_model(stage).model_json_schema()
    )
    assert prompts.SYSTEM == previous.SYSTEM
    assert prompts.assess is core.assess


def test_reply_has_no_approximation_completion_disposition_or_legacy_selector():
    schema = prompts.response_model("repair").model_json_schema()
    decision = schema["$defs"]["CompatibilityDecision"]
    assert decision["properties"]["disposition"]["enum"] == [
        "revise",
        "dispute_assumption",
    ]
    selection = schema["$defs"]["ExternalExchangeSelection"]["properties"]
    assert set(selection) == {"obligation_id", "contribution_handle", "explanation"}
    with pytest.raises(ValidationError):
        prompts.response_model("repair").model_validate(
            raw_reply(external_exchanges=[choice()])
        )
    with pytest.raises(ValidationError):
        prompts.response_model("repair").model_validate(
            raw_reply(
                compatibility_decisions=[
                    {
                        "question_id": "q",
                        "disposition": "retain_approximation",
                        "explanation": "I accept it.",
                    }
                ]
            )
        )


def test_exact_handle_maps_one_use_process_without_changing_stored_draft_schema():
    d = with_outlet()
    selected = select(d)
    patch, changes = normalize(d, raw_reply(external_exchanges=[selected]))
    assert patch.external_exchanges[0].shared_process == "outlet"
    assert patch.external_exchanges[0].local_term_index is None
    assert changes[-1]["code"] == "external_contribution_handle_resolved"
    assert changes[-1]["scientific_role_inferred"] is False
    updated = prompts.apply_patch(public(), d, patch)
    updated, _ = prompts.normalize_consumers(updated, patch, d)
    assert assess(updated, boundary())["eligible"]
    assert updated.external_exchanges[0].shared_process == "outlet"
    assert "contribution_handle" not in updated.external_exchanges[0].model_dump()
    assert updated.equations == d.equations
    assert updated.processes == d.processes


def test_runtime_does_not_select_outlet_from_process_meaning():
    d = with_outlet()
    options = prompts.external_handle_options(public(), d, boundary())[0]
    assert len(options["contributions"]) == 2
    selected = select(
        d, shared=None
    )  # User explicitly chooses the ordinary input term.
    patch, _ = normalize(d, raw_reply(external_exchanges=[selected]))
    assert patch.external_exchanges[0].local_term_index == 0
    assert patch.external_exchanges[0].shared_process is None
    assert "Valid reference syntax" in options["scope"]


def test_stale_handle_cannot_silently_retarget_changed_contribution():
    d = with_outlet()
    selected = select(d)
    changed = d.processes[0].model_dump(mode="json")
    changed["scientific_meaning"] = "Changed physical interpretation."
    after = apply(d, processes=[changed])
    with pytest.raises(ValueError, match="absent or stale"):
        normalize(after, raw_reply(external_exchanges=[selected]))
    with pytest.raises(ValueError, match="selected contribution changed"):
        normalize(d, raw_reply(external_exchanges=[selected], processes=[changed]))
    assert not d.external_exchanges
    assert d.processes[0].scientific_meaning != changed["scientific_meaning"]


def test_unrelated_same_reply_edit_keeps_selected_handle_valid():
    d = with_outlet()
    patch, _ = normalize(
        d,
        raw_reply(
            external_exchanges=[select(d)],
            variables=[variable("z")],
            equations=[equation("z", "u")],
        ),
    )
    after = prompts.apply_patch(public(), d, patch)
    assert after.external_exchanges[0].shared_process == "outlet"
    assert "z" in {row.name for row in after.variables}


def test_handles_are_obligation_specific_and_undeclared_effects_have_no_handle():
    d = with_outlet()
    selected = select(d)
    with pytest.raises(ValueError, match="absent or stale"):
        normalize(
            d, raw_reply(external_exchanges=[{**selected, "obligation_id": "other"}])
        )
    with pytest.raises(ValueError, match="absent or stale"):
        normalize(
            d,
            raw_reply(
                external_exchanges=[
                    {
                        **selected,
                        "contribution_handle": "contribution-" + "0" * 64,
                    }
                ]
            ),
        )


def test_supported_challenge_is_logged_without_model_or_assumption_change():
    d = draft()
    with pytest.raises(ValidationError, match="requires supporting_evidence"):
        prompts.CompatibilityDecision(
            question_id="q1", disposition="dispute_assumption", explanation="No."
        )
    decision = {
        "question_id": "q1",
        "disposition": "dispute_assumption",
        "explanation": "The intended initialization differs by trajectory.",
        "supporting_evidence": ["The displayed initial-data contract permits this."],
    }
    patch, _ = normalize(
        d,
        raw_reply(
            compatibility_decisions=[decision],
            stage_complete=False,
        ),
    )
    assert patch.compatibility_decisions[0].supporting_evidence == tuple(
        decision["supporting_evidence"]
    )
    assert prompts.apply_patch(public(), d, patch) == d
    assert "supporting_evidence" not in d.model_dump()


def test_actionable_conflict_requests_structured_edits_not_disposition_completion():
    question = {
        "question_id": "q1",
        "property": "closed_subsystem",
        "target": "y",
        "repair_status": "repair_required",
        "assumptions": ["Common initial state."],
        "witnesses": [{"error_intervals_disjoint": True}],
        "question": "Retain an intended approximation?",
    }
    p = packet(
        draft(),
        compatibility_questions=[question],
        compatibility_state={
            "pending_questions": [question],
            "completion_allowed": False,
            "repair_status": "repair_required",
        },
    )
    work = p["feedback"]["repair_work"]
    assert work["actionable_conflicts"] == ["q1"]
    assert work["runtime_revalidation"]["completion_allowed"] is False
    assert "advisory_reply_choices" not in p["feedback"]
    assert "completion_example" not in p["feedback"]
    revised = p["compatibility_questions"][0]
    assert "approximation" not in revised["question"]
    assert revised["assumptions"] == question["assumptions"]
    assert revised["witnesses"] == question["witnesses"]
    assert "A statement that you will revise is not an edit" in p["stage_instructions"]


def test_uncertain_evidence_is_unresolved_without_becoming_an_error():
    question = {
        "question_id": "q1",
        "target": "y",
        "property": "scalar_autonomous",
        "repair_status": "uncertainty_limited",
        "witnesses": [{"error_intervals_disjoint": None}],
    }
    p = packet(draft(), compatibility_questions=[question])
    work = p["feedback"]["repair_work"]
    assert not work["actionable_conflicts"]
    assert work["mode"] == "assessment_unresolved"
    assert (
        "not an established model error"
        in p["compatibility_questions"][0]["interpretation"]
    )
    assert work["structural_eligibility"] is True


def test_latent_deferral_without_current_question_is_not_completion():
    state = {
        "completion_allowed": False,
        "repair_status": "review_deferred",
        "baseline_outcomes": [
            {
                "target": "y",
                "property": "closed_subsystem",
                "resolved": False,
                "status": "deferred",
                "reason": "Unknown latent initialization.",
            }
        ],
    }
    p = packet(draft(), compatibility_state=state)
    work = p["feedback"]["repair_work"]
    assert work["mode"] == "assessment_unresolved"
    assert work["unresolved_previous_diagnostics"][0]["reason"] == (
        "Unknown latent initialization."
    )
    assert "completion_example" not in p["feedback"]


def test_repetition_feedback_and_invalid_external_index_are_specific_and_nonmutating():
    d = apply(with_outlet(), external_exchanges=[choice(local_term_index=2)])
    repetitions = {"same_error_count": 3, "code": "contribution_reference_unresolved"}
    saved = deepcopy(repetitions)
    p = packet(d, boundary_contract=boundary(), repetition_feedback=repetitions)
    feedback = p["feedback"]
    issue = next(
        row for row in feedback["issues"] if row["code"] == "external_exchange_coverage"
    )
    facts = issue["reference_namespace_facts"]
    assert facts["ordinary_term_indices"] == [0]
    assert facts["separate_process_uses"] == ["outlet"]
    assert "contribution_handle" in issue["action"]
    assert "external_reference_choices" not in p
    assert p["external_effect_options"][0]["contributions"]
    reminder = feedback["repeated_repair_feedback"]
    assert reminder["observations"]["same_error_count"] == 3
    assert "does not change dependencies" in reminder["instruction"]
    assert repetitions == saved
    assert d.external_exchanges[0].local_term_index == 2


def test_repetition_receipt_does_not_reintroduce_historical_completion_advice():
    legacy_text = "Answer pending scientific questions or explicitly finish when ready."
    legacy_instruction = "An answer alone resolves these advisory questions."
    repetition = {
        "consecutive_unchanged": 2,
        "continuation": {
            "no_progress": True,
            "repeated_semantic_attempts": 2,
            "instruction": legacy_text,
        },
        "previous_reply": {"stage_complete": True, "equations": []},
        "previous_error": "contribution_reference_unresolved",
        "remaining_error_codes": ["external_exchange_coverage"],
        "instruction": legacy_instruction,
    }
    saved = deepcopy(repetition)
    p = packet(
        draft(),
        continuation=repetition["continuation"],
        repetition_feedback=repetition,
    )
    observed = p["feedback"]["repeated_repair_feedback"]["observations"]
    assert observed == {
        key: value
        for key, value in repetition.items()
        if key not in {"continuation", "instruction"}
    }
    assert observed["previous_reply"] == repetition["previous_reply"]
    text = json.dumps(p)
    assert legacy_text not in text
    assert legacy_instruction not in text
    assert "Neither stage_complete=true nor an explanatory response overrides" in text
    assert "retain_approximation" not in p["editing_rules"]
    assert "completion_example" not in p["feedback"]
    assert p["feedback"]["continuation"]["repeated_semantic_attempts"] == 2
    assert repetition == saved
