"""Focused prompts preserve evidence and make previous-reply repetition explicit."""

import json
from copy import deepcopy

import pytest

from autoformalism.search import construction_initialization_focus as focus


def packet(*, error=None, reply=None, questions=None):
    value = {
        "public_brief": {
            "science": "All targets must be generated.",
            "observations": [{"t": 0, "x": 2}],
        },
        "training_highlights": {"summary": "Same input, different x."},
        "stage": "repair",
        "stage_instructions": "old universal instructions",
        "editing_rules": "old universal rules",
        "mechanism_realization_rules": "old mapping rules",
        "response_template": {"stage_complete": False, "initialization": None},
        "response_schema": {"type": "object"},
        "previous_edit": {"error": error, "applied": error is None},
        "model_view": {"processes": [{"name": "P"}], "balances": ["x'=P"]},
        "initialization": {"fixed_runtime_rules": {"x": "x0"}},
        "compatibility_questions": questions or [],
        "feedback": {
            "issues": [],
            "repair_work": {"mode": "repair_required"},
            "repair_budget": {"reservation_rule": "request UTF-8 bytes"},
        },
        "requirements_checklist": [{"id": "a", "status": "unverified"}],
    }
    if reply is not None:
        value["feedback"]["repeated_repair_feedback"] = {
            "observations": {"previous_reply": reply, "previous_error": error},
            "instruction": "Correct the recurring problem.",
        }
    return value


def process(name="P", **kwargs):
    return {
        "name": name,
        "depends_on": ["x", "y"],
        "kind": "influence",
        "scientific_meaning": "An intended scientific effect. " * 8,
        "uses": [{"target": "z", "sign": "positive"}],
        **kwargs,
    }


def test_system_terms_include_public_targets_and_distinguish_mapping_from_coverage():
    assert "algebraic equations" in focus.SYSTEM
    assert "instantaneous equations" not in focus.SYSTEM
    assert "including a declared public target output" in focus.SYSTEM
    assert "mechanism_realizations records" in focus.SYSTEM
    assert "not a record of satisfied" in focus.SYSTEM
    assert "supplied-only input or" in focus.SYSTEM


def test_projection_changes_presentation_but_preserves_all_scientific_context():
    original = packet()
    saved = deepcopy(original)
    result = focus.project(original)
    assert original == saved
    assert list(result)[:3] == ["stage", "stage_instructions", "feedback"]
    assert list(result)[-2:] == ["training_highlights", "public_brief"]
    assert next(iter(result["feedback"])) == "repair_work"
    for key in (
        "public_brief",
        "training_highlights",
        "model_view",
        "initialization",
        "response_schema",
        "response_template",
        "requirements_checklist",
        "compatibility_questions",
    ):
        assert result[key] == original[key]
    assert result["feedback"]["repair_guidance"] == []
    assert len(result["stage_instructions"].split()) < 100
    assert (
        "serving input tokens"
        in result["feedback"]["repair_budget"]["reservation_rule"]
    )
    assert "UTF-8 bytes" not in json.dumps(result)


def test_first_request_can_have_no_previous_edit():
    original = packet()
    original["previous_edit"] = None
    assert focus.project(original)["previous_edit"] is None


def test_all_33_repetitions_keep_exact_records_multiplicity_and_order():
    original = packet(reply={"processes": [process()] * 33, "variables": []})
    compact = focus.compact_repeated_replies(original)
    receipt = compact["feedback"]["repeated_repair_feedback"]["observations"]
    repeated = receipt["previous_reply"]["processes"]
    assert repeated["original_count"] == 33
    assert repeated["ordered_runs"] == [{"record": process(), "count": 33}]
    assert receipt["previous_reply_presentation"]["record_lists"] == ["processes"]
    assert len(json.dumps(compact)) < len(json.dumps(original))
    assert focus.expand_repeated_reply(compact) == original
    assert (
        original["feedback"]["repeated_repair_feedback"]["observations"][
            "previous_reply"
        ]["processes"]
        == [process()] * 33
    )


def test_nonconsecutive_records_and_changed_parameters_are_not_merged():
    a, b = process(), process(coefficient=2)
    original = packet(reply={"processes": [a] * 20 + [b] * 3 + [a] * 10})
    compact = focus.compact_repeated_replies(original)
    runs = compact["feedback"]["repeated_repair_feedback"]["observations"][
        "previous_reply"
    ]["processes"]["ordered_runs"]
    assert [row["count"] for row in runs] == [20, 3, 10]
    assert [row["record"] for row in runs] == [a, b, a]
    assert focus.expand_repeated_reply(compact) == original


def test_only_previous_reply_lists_are_encoded_and_scalar_types_stay_distinct():
    original = packet(reply={"processes": [process(value=True), process(value=1)]})
    original["model_view"]["processes"] = [process()] * 33
    original["feedback"]["issues"] = [{"code": "x"}] * 33
    assert focus.compact_repeated_replies(original) == original
    assert focus.expand_repeated_reply(original) == original


def test_small_repetitions_do_not_expand_the_prompt():
    original = packet(reply={"processes": [{"name": "P"}] * 2})
    assert focus.compact_repeated_replies(original) == original


@pytest.mark.parametrize("count", [0, -1, True, "33"])
def test_bad_expansion_counts_fail_closed(count):
    compact = focus.compact_repeated_replies(
        packet(reply={"processes": [process()] * 33})
    )
    repeated = compact["feedback"]["repeated_repair_feedback"]["observations"][
        "previous_reply"
    ]["processes"]
    repeated["ordered_runs"][0]["count"] = count
    with pytest.raises(ValueError, match="invalid previous reply run"):
        focus.expand_repeated_reply(compact)


def test_double_projection_of_repetition_is_rejected():
    compact = focus.compact_repeated_replies(
        packet(reply={"processes": [process()] * 33})
    )
    with pytest.raises(ValueError, match="already applied"):
        focus.compact_repeated_replies(compact)


def test_specific_collision_feedback_names_exact_conflicting_equations_entry():
    original = packet(
        error="a shared process already has one runtime-generated definition",
        reply={"processes": [process()], "equations": [{"name": "P", "terms": []}]},
    )
    guidance = focus.project(original)["feedback"]["repair_guidance"]
    collision = next(
        row for row in guidance if row["for"] == "shared-process definition collision"
    )
    assert collision["conflicting_equations_entries"] == ["P"]
    assert "omit the conflicting equations entry" in collision["instruction"]
    assert "Do not delete a retained process" in collision["instruction"]


def test_compatibility_guidance_is_conditional_and_keeps_witnesses_and_assumptions():
    question = {
        "target": "x",
        "repair_status": "actionable_conflict",
        "witnesses": [{"times": [0, 0], "observations": [1.2, 1.9]}],
        "assumptions": ["same shared initialization"],
    }
    original = packet(questions=[question])
    result = focus.project(original)
    assert result["compatibility_questions"] == [question]
    assert len(result["feedback"]["repair_guidance"]) == 1
    assert (
        "supporting_evidence" in result["feedback"]["repair_guidance"][0]["instruction"]
    )
    assert "meal" not in json.dumps(result)
    assert "insulin" not in json.dumps(result)


def test_initializer_stage_keeps_its_exclusive_schema_and_instructions():
    original = packet()
    original["stage"] = "initialization"
    original["stage_instructions"] = "Return initial dependencies only."
    original["editing_rules"] = "Return only stage_complete and initialization."
    original["response_template"] = {
        "stage_complete": False,
        "initialization": {"records": []},
    }
    result = focus.project(original)
    for key in ("stage_instructions", "editing_rules", "response_template"):
        assert result[key] == original[key]
    assert "repair_guidance" not in result["feedback"]


def test_unsupported_stage_is_not_silently_rewritten():
    value = packet()
    value["stage"] = "variables"
    with pytest.raises(ValueError, match="initialization or repair"):
        focus.project(value)

def compatibility_question(target="x", *, assumption="common assumptions", initial=1):
    return {
        "target": target,
        "question_id": "question_" + target,
        "repair_status": "actionable_conflict",
        "witnesses": [{"observations": [initial, 2]}],
        "conditioning": {"complete_evidence": "Recorded input knots. " * 50},
        "initialization_qualification": {"shared_initial_map": "Initial values. " * 50},
        "assumptions": [assumption] * 30,
        "why_this_still_conflicts": {
            "invariant": "The same inputs and initial state give the same solution. "
            * 20,
            "observed_difference": "Target " + target + " differs.",
        },
    }


def test_compatibility_factoring_preserves_all_questions_and_nested_differences():
    original = packet(
        questions=[compatibility_question("x"), compatibility_question("y")]
    )
    before = deepcopy(original)
    compact = focus.factor_compatibility_details(original)
    assert "compatibility_details" in compact
    assert focus.expand_compatibility_details(compact) == original == before
    assert len(json.dumps(compact)) < len(json.dumps(original))
    for old, new in zip(
        original["compatibility_questions"],
        compact["compatibility_questions"],
        strict=True,
    ):
        for key in ("target", "question_id", "repair_status", "witnesses"):
            assert new[key] == old[key]
        assert (
            new["why_this_still_conflicts"]["observed_difference"]
            == old["why_this_still_conflicts"]["observed_difference"]
        )
        assert "shared_details" in new["why_this_still_conflicts"]
    assert compact["public_brief"] == original["public_brief"]
    assert compact["feedback"] == original["feedback"]


def test_distinct_assumptions_and_argument_order_cannot_be_shared():
    a = compatibility_question("x", assumption="Assumption A")
    b = compatibility_question("y", assumption="Assumption B", initial=3)
    a["scope"] = ["u", "z"]
    b["scope"] = ["z", "u"]
    original = packet(questions=[a, b])
    compact = focus.factor_compatibility_details(original)
    assert compact["compatibility_questions"][0]["assumptions"] == a["assumptions"]
    assert compact["compatibility_questions"][1]["assumptions"] == b["assumptions"]
    assert compact["compatibility_questions"][0]["scope"] == ["u", "z"]
    assert compact["compatibility_questions"][1]["scope"] == ["z", "u"]
    assert focus.expand_compatibility_details(compact) == original


def test_compatibility_factoring_does_not_expand_a_single_or_small_question():
    for questions in (
        [],
        [compatibility_question()],
        [{"target": "x"}, {"target": "y"}],
    ):
        original = packet(questions=questions)
        assert focus.factor_compatibility_details(original) == original
        assert focus.expand_compatibility_details(original) == original


def test_compatibility_projection_markers_fail_closed():
    original = packet(
        questions=[compatibility_question("x"), compatibility_question("y")]
    )
    compact = focus.factor_compatibility_details(original)
    with pytest.raises(ValueError, match="already applied"):
        focus.factor_compatibility_details(compact)
    invalid = deepcopy(compact)
    invalid["compatibility_projection"]["policy"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported compatibility"):
        focus.expand_compatibility_details(invalid)
    invalid = deepcopy(compact)
    del invalid["compatibility_projection"]
    with pytest.raises(ValueError, match="require a projection marker"):
        focus.expand_compatibility_details(invalid)
