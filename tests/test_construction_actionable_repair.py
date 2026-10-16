"""Actionable repair keeps uncertainty and maps only explicitly chosen effects."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.search import construction_actionable_repair as prompts
from autoformalism.search import construction_evidence as core
from autoformalism.search import construction_evidence_repair as previous
from autoformalism.search import mechanism_realizations
from autoformalism.staged_topology import content_hash
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


@pytest.mark.parametrize(
    ("stage", "historical_hash"),
    [
        (
            "variables",
            "911dbb44b9f8ad7cf20f04e6e34d3bf088861a11fe5ded2ca3c5cbd07f88eb12",
        ),
        (
            "shared_laws",
            "582b63c071ba4778a595edea231c45541a7a42a94bcc2be58a367f313e934420",
        ),
        (
            "equations",
            "2005f9c388ca6219a3b436947a04da3a656a402327e701e371905289a20ddba7",
        ),
        ("repair", "9cf5e4f0c99c4ba79cba0edde59d713ea05c3aed4447f81c924b04f91da4b108"),
    ],
)
def test_v2_is_opt_in_and_historical_payloads_are_unchanged(stage, historical_hash):
    value = packet(draft(), stage=stage)
    assert content_hash(value) == historical_hash
    assert packet(draft(), stage=stage, policy=prompts.POLICY) == value
    # No v2 repair-specific condition occurs in this complete fixture.
    assert packet(draft(), stage=stage, policy=prompts.POLICY_V2) == value
    with pytest.raises(ValueError, match="unknown saved actionable repair policy"):
        packet(draft(), stage=stage, policy="unregistered")


def rejected_memory_record():
    """An intended memory variable is omitted from its own realization record."""
    return {
        "status": "rejected",
        "transaction_applied": False,
        "error": "memory_states must be a subset of components",
        "rejected_reply": raw_reply(
            mechanism_realizations=[
                {
                    "requirement_id": "effect",
                    "components": ["u"],
                    "memory_states": ["y"],
                }
            ]
        ),
    }


def test_v2_subset_feedback_identifies_exact_conditional_record_without_edit():
    d, last = draft(), rejected_memory_record()
    saved_last, saved_draft = deepcopy(last), d.model_dump(mode="json")
    old = packet(d, last=last)
    new = packet(d, last=last, policy=prompts.POLICY_V2)
    matches = [
        row
        for row in new["feedback"]["issues"]
        if row["code"] == "memory_state_not_in_components"
    ]
    assert len(matches) == 1
    issue = matches[0]
    assert issue["requirement_id"] == "effect"
    assert issue["rejected_record_index"] == 0
    assert issue["missing_from_components"] == ["y"]
    assert issue["record_if_assignment_is_intended"] == {
        "requirement_id": "effect",
        "components": ["u", "y"],
        "memory_states": ["y"],
    }
    assert "Otherwise explicitly correct the memory assignment" in issue["action"]
    assert "no edit has been applied" in issue["scope"]
    assert "Omitted realization records stay unchanged" in issue["scope"]
    assert new["model_view"] == old["model_view"]
    assert new["requirements_checklist"] == old["requirements_checklist"]
    assert new["public_brief"] == old["public_brief"]
    assert not any(
        row["code"] == "memory_state_not_in_components"
        for row in old["feedback"]["issues"]
    )
    assert d.model_dump(mode="json") == saved_draft and last == saved_last
    # The schema continues rejecting the original; feedback does not normalize it.
    with pytest.raises(ValidationError, match="must be a subset"):
        prompts.response_model("repair").model_validate(last["rejected_reply"])


@pytest.mark.parametrize("raw", [None, [], {"mechanism_realizations": "invalid"}])
def test_v2_subset_feedback_handles_missing_or_wrong_shape_rejected_text(raw):
    last = rejected_memory_record()
    last["rejected_reply"] = raw
    p = packet(draft(), last=last, policy=prompts.POLICY_V2)
    assert not any(
        row["code"] == "memory_state_not_in_components"
        for row in p["feedback"]["issues"]
    )


def test_v2_does_not_suggest_record_repair_for_an_accepted_transaction():
    last = {
        **rejected_memory_record(),
        "transaction_applied": True,
        "status": "accepted",
    }
    assert prompts._memory_record_issues(last) == []


@pytest.mark.parametrize("status", ["repair_required", "review_deferred"])
def test_v2_noop_error_cannot_suggest_finishing_unresolved_work(status):
    error = "no draft change; make an explicit edit or finish this stage"
    last = {
        "status": "rejected",
        "transaction_applied": False,
        "error": error,
        "rejected_reply": raw_reply(stage_complete=False),
    }
    repetitions = {"previous_error": error, "previous_reply": last["rejected_reply"]}
    state = {"completion_allowed": False, "repair_status": status}
    kwargs = {
        "last": last,
        "repetition_feedback": repetitions,
        "compatibility_state": state,
    }
    old = packet(draft(), **kwargs)
    new = packet(draft(), policy=prompts.POLICY_V2, **kwargs)
    assert "or finish this stage" in json.dumps(old)
    assert "or finish this stage" not in json.dumps(new)
    assert "Finishing does not resolve" in new["previous_edit"]["error"]
    assert (
        "Uncertain evidence alone does not require a model edit"
        in (
            new["feedback"]["repeated_repair_feedback"]["observations"][
                "previous_error"
            ]
        )
    )
    assert last["error"] == repetitions["previous_error"] == error


def test_v2_noop_keeps_completion_available_when_checks_are_clear():
    error = "no draft change; make an explicit edit or finish this stage"
    last = {"status": "rejected", "transaction_applied": False, "error": error}
    p = packet(
        draft(),
        last=last,
        policy=prompts.POLICY_V2,
        compatibility_state={
            "completion_allowed": True,
            "repair_status": "checks_clear",
        },
    )
    assert p["previous_edit"]["error"] == error


def closed_question():
    """A generic matched-conditioning witness, without benchmark-specific names."""
    return {
        "question_id": "closed",
        "target": "y",
        "property": "closed_subsystem",
        "repair_status": "repair_required",
        "assumptions": ["Shared deterministic laws have unique solutions."],
        "conditioning": {
            "matched_initial_information": {"a": {"y": 0}, "b": {"y": 0}},
            "matched_complete_forcing_channels": ["u"],
        },
        "witnesses": [
            {
                "trajectories": ["a", "b"],
                "times": [3.0, 3.0],
                "observations": [10.0, 20.0],
                "error_intervals_disjoint": True,
            }
        ],
    }


def test_v2_closed_subsystem_feedback_preserves_witness_and_scientific_ownership():
    question = closed_question()
    saved = deepcopy(question)
    old = packet(draft(), compatibility_questions=[question])
    new = packet(draft(), compatibility_questions=[question], policy=prompts.POLICY_V2)
    displayed = new["compatibility_questions"][0]
    reason = displayed.pop("why_this_still_conflicts")
    assert displayed == old["compatibility_questions"][0]
    assert reason["witness_reference"] == "witnesses[0]"
    assert "a: y(3.0)=10.0; b: y(3.0)=20.0" in reason["observed_difference"]
    assert "self-only decay" in reason["invariant"]
    assert "shared constant production term" in reason["invariant"]
    assert "matched permitted initial information" in reason["invariant"]
    assert "does not select a new dependence" in reason["repair_choice"]
    assert "fit quality remain unverified" in reason["repair_choice"]
    assert question == saved


@pytest.mark.parametrize(
    "variation", ["uncertain", "challenged", "no_bounds", "no_initials"]
)
def test_v2_does_not_promote_unqualified_closed_comparisons(variation):
    question = closed_question()
    if variation == "uncertain":
        question["repair_status"] = "uncertainty_limited"
    elif variation == "challenged":
        question["repair_status"] = "assumption_challenged"
    elif variation == "no_bounds":
        question["witnesses"][0]["error_intervals_disjoint"] = None
    else:
        del question["conditioning"]["matched_initial_information"]
    p = packet(draft(), compatibility_questions=[question], policy=prompts.POLICY_V2)
    assert "why_this_still_conflicts" not in p["compatibility_questions"][0]


def qualified_packet(*, corrected=False, policy=prompts.POLICY_V2, stage="repair"):
    """Exercise an actual qualified assessment, not a fabricated eligibility flag."""
    from tests.test_required_memory_path_qualification import draft as memory_draft
    from tests.test_required_memory_path_qualification import public as memory_public

    b = memory_public()
    d = core.Draft.model_validate(memory_draft().model_dump(mode="json"))
    if corrected:
        d = core.apply_patch(
            b,
            d,
            core.DraftPatch(equations=[equation("y", "I", "X")], stage_complete=True),
        )
    c = ValidationContext(
        targets=("y", "I"), auxiliaries=("a", "b"), external_inputs=("u",)
    )
    original = core.assess(b, c, {}, d)
    assert original["eligible"]
    assessment = mechanism_realizations.qualify_required_memory_paths(b, d, original)
    saved = deepcopy(assessment)
    args = (stage, b, c, {}, b.model_dump(mode="json"), d, None)
    packet = prompts.payload(*args, policy=policy, qualified_assessment=assessment)
    assert assessment == saved
    return packet, assessment, prompts.payload(*args, policy=policy)


def test_v2_qualified_cycle_failure_has_exact_witness_and_consistent_work_status():
    p, assessment, unqualified = qualified_packet()
    feedback = p["feedback"]
    assert (
        feedback["required_memory_path_qualification"]
        == (assessment["required_memory_path_qualification"])
    )
    work = feedback["repair_work"]
    assert work["structural_eligibility"] is False
    assert work["mode"] == "repair_required"
    issue = next(
        row
        for row in feedback["issues"]
        if row["code"] == "required_memory_path_revisits_driver"
    )
    assert issue["memory_witnesses"][0]["ordinary_path"] == ["X", "I", "y"]
    assert issue["memory_witnesses"][0]["qualified_path"] is None
    correction = next(
        row
        for row in work["model_or_structural_corrections"]
        if row["code"] == issue["code"]
    )
    assert correction["memories"] == ["X"]
    assert feedback["issues"][correction["feedback_issue_indices"][0]] == issue
    checklist = {row["id"]: row for row in p["requirements_checklist"]}
    assert checklist["structural_validity"]["status"] == "failed"
    mechanism = checklist["mechanism_realization:history_effect"]
    assert mechanism["status"] == mechanism["structural_status"] == "failed"
    assert mechanism["required_memory_path_qualification_status"] == "failed"
    assert p["model_view"] == unqualified["model_view"]
    assert p["public_brief"] == unqualified["public_brief"]


def test_v2_qualified_history_path_can_pass_without_extra_repair():
    p, assessment, _ = qualified_packet(corrected=True)
    assert assessment["eligible"]
    work = p["feedback"]["repair_work"]
    assert work["structural_eligibility"] is True
    assert work["mode"] == "recheck_completion"
    assert not any(
        row["code"] == "required_memory_path_revisits_driver"
        for row in p["feedback"]["issues"]
    )
    record = next(
        row
        for row in p["requirements_checklist"]
        if row["id"] == "mechanism_realization:history_effect"
    )
    assert record["status"] == "passed"
    assert record["required_memory_path_qualification_status"] == "passed"


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_qualified_assessment_cannot_change_historical_requests(stage):
    p, _, unchanged = qualified_packet(policy=prompts.POLICY, stage=stage)
    assert p == unchanged


def test_v2_runtime_structural_failure_cannot_appear_as_completion_only():
    p = packet(
        draft(),
        policy=prompts.POLICY_V2,
        compatibility_state={
            "repair_status": "structurally_blocked",
            "completion_allowed": False,
        },
    )
    assert p["feedback"]["repair_work"]["mode"] == "repair_required"
    assert p["feedback"]["repair_work"]["structural_eligibility"] is False
    assert (
        next(
            row
            for row in p["requirements_checklist"]
            if row["id"] == "structural_validity"
        )["status"]
        == "failed"
    )
