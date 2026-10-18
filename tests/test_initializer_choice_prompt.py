"""The isolated choice cannot invent measured initials or omit pending evidence."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.research import initializer_choice_prompt as prompt
from autoformalism.search.construction_initialization_focus import (
    factor_compatibility_details,
)
from tests.test_initialization_evidence_feedback import feedback, fixture


def packet(*, initial=True):
    """Use generic synthetic quantities, without benchmark identifiers or names."""
    value, draft, evidence = fixture(initial=initial)
    value = feedback.project(value, draft, evidence)
    value["model_view"] = {"variables": draft.model_dump(mode="json")["variables"]}
    value["public_brief"].update(
        scientific_context="Generate all targets using shared causal laws.",
        requirements=[{"id": "memory", "public_requirement": "Retain memory."}],
        target_dependencies=[],
        training_observations={"large_statistics": "omit " * 5000},
    )
    value["initialization"]["runtime_initial_information"].update(
        complete=True, shared_deterministic=True
    )
    value["training_highlights"] = "omit " * 5000
    value["response_template"] = {"variables": [], "equations": []}
    value["previous_edit"] = None
    return value


@pytest.mark.parametrize(
    "decision", ["topology_change_needed", "insufficient_information"]
)
def test_explicit_abstentions_require_no_rule(decision):
    value = {"decision": decision, "rule": None, "explanation": "Missing evidence."}
    assert prompt.ChoiceReply.model_validate(value).model_dump(mode="json") == value
    value["rule"] = {"mode": "shared_parameter", "arguments": []}
    with pytest.raises(ValidationError, match="other decisions require null"):
        prompt.ChoiceReply.model_validate(value)


@pytest.mark.parametrize(
    "rule",
    [
        {"mode": "shared_parameter", "arguments": []},
        {"mode": "common_map", "arguments": ["x", "u"]},
        # Membership in a particular runtime contract is intentionally checked later.
        {"mode": "common_map", "arguments": ["not_in_this_packet"]},
    ],
)
def test_proposed_rule_is_strict_structure_not_a_scientific_selection(rule):
    reply = prompt.ChoiceReply.model_validate(
        {"decision": "propose_initializer", "rule": rule, "explanation": "Justified."}
    )
    assert reply.rule.model_dump(mode="json") == rule


@pytest.mark.parametrize(
    "rule",
    [
        None,
        {"mode": "observed_initial", "arguments": []},
        {"mode": "unknown", "arguments": []},
        {"mode": "common_map", "arguments": []},
        {"mode": "common_map", "arguments": ["x", "x"]},
        {"mode": "common_map", "arguments": ["x(t)"]},
        {"mode": "common_map", "arguments": [7]},
        {"mode": "common_map", "arguments": "x"},
        {"mode": "common_map", "arguments": ["x"], "formula": "x"},
        {"mode": "shared_parameter", "arguments": ["x"]},
        {"mode": "shared_parameter"},
    ],
)
def test_invalid_rule_cannot_be_returned(rule):
    with pytest.raises(ValidationError):
        prompt.ChoiceReply.model_validate(
            {"decision": "propose_initializer", "rule": rule, "explanation": "Reason."}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"equations": []},
        {"decision": "repair"},
        {"explanation": " "},
        {"explanation": "x" * 1201},
        {"explanation": 42},
    ],
)
def test_reply_rejects_topology_edits_unknown_decisions_and_unbounded_text(change):
    value = {"decision": "insufficient_information", "rule": None, "explanation": "Why"}
    with pytest.raises(ValidationError):
        prompt.ChoiceReply.model_validate(value | change)


@pytest.mark.parametrize("initial", [True, False])
def test_exact_model_card_and_qualified_witness_survive_without_statistics(initial):
    source = packet(initial=initial)
    original = deepcopy(source)
    result = prompt.build_packet(source, "z")
    assert result["model_view"] == source["model_view"]
    assert result["model_data_conflicts"] == source["model_data_conflicts"]
    question = result["compatibility_questions"][0]
    for field in (
        "witnesses",
        "assumptions",
        "conditioning",
        "initialization_qualification",
    ):
        assert question[field] == source["compatibility_questions"][0][field]
    assert "training_highlights" not in result
    assert "training_observations" not in result["public_brief"]
    assert "response_template" not in result
    assert "schema" not in result
    assert "omit omit" not in json.dumps(result)
    assert result["projection_scope"]["diagnostics_only"] is True
    result["model_data_conflicts"][0]["assumptions"].append("mutation")
    assert source == original


@pytest.mark.parametrize(
    "change",
    [
        "ambiguous_state",
        "missing_state",
        "algebraic_state",
        "duplicate_state",
        "fixed_state",
        "unknown_scope",
        "incomplete_information",
        "duplicate_symbols",
        "missing_initializer",
        "conflicting_rules",
        "no_pending",
        "missing_card",
        "stale_card",
        "missing_witness",
        "missing_assumptions",
        "unqualified",
        "no_public_context",
        "missing_requirements",
    ],
)
def test_missing_or_ambiguous_state_and_evidence_fail_closed(change):
    source = packet()
    initialization = source["initialization"]
    question = source["compatibility_questions"][0]
    if change == "ambiguous_state":
        initialization["editable_states"].append("x")
    elif change == "missing_state":
        initialization["editable_states"] = []
    elif change == "algebraic_state":
        source["model_view"]["variables"][1]["definition"] = "algebraic"
    elif change == "duplicate_state":
        source["model_view"]["variables"].append(
            deepcopy(source["model_view"]["variables"][1])
        )
    elif change == "fixed_state":
        initialization["fixed_runtime_rules"]["z"] = {
            "mode": "observed_initial",
            "arguments": [],
        }
    elif change == "unknown_scope":
        initialization["runtime_initial_information"]["scope"] = "unknown"
    elif change == "incomplete_information":
        initialization["runtime_initial_information"]["complete"] = False
    elif change == "duplicate_symbols":
        initialization["runtime_initial_information"]["permitted_symbols"].append("x")
    elif change == "missing_initializer":
        initialization["retained_handoff"]["records"] = []
    elif change == "conflicting_rules":
        initialization["retained_handoff"]["records"].append(
            {"state": "x", "rule": {"mode": "shared_parameter", "arguments": []}}
        )
    elif change == "no_pending":
        source["feedback"]["repair_work"]["actionable_conflicts"] = []
    elif change == "missing_card":
        source["model_data_conflicts"] = []
    elif change == "stale_card":
        source["model_data_conflicts"][0]["target"] = "other"
    elif change == "missing_witness":
        question["witnesses"] = []
    elif change == "missing_assumptions":
        question["assumptions"] = []
    elif change == "unqualified":
        question["initialization_qualification"]["family_available"] = False
    elif change == "no_public_context":
        source["public_brief"]["scientific_context"] = ""
    else:
        source["public_brief"].pop("requirements")
    with pytest.raises((ValueError, KeyError)):
        prompt.build_packet(source, "z")


def test_preserves_original_witness_indices_and_expands_shared_evidence():
    source = packet()
    original_question = deepcopy(source["compatibility_questions"][0])
    source["compatibility_questions"].insert(0, deepcopy(original_question))
    source["compatibility_questions"][0]["question_id"] = "inactive"
    source["model_data_conflicts"][0]["evidence_reference"] = (
        "compatibility_questions[1].witnesses[0]"
    )
    source = factor_compatibility_details(source)
    result = prompt.build_packet(source, "z")
    assert result["compatibility_questions"][0]["question_id"] == "inactive"
    assert (
        result["compatibility_questions"][1]["witnesses"]
        == original_question["witnesses"]
    )
    assert result["model_data_conflicts"] == source["model_data_conflicts"]


def test_prior_rejected_rule_is_not_presented_as_current_initializer():
    source = packet()
    source["previous_edit"] = {"status": "rejected", "error": "No measured binding."}
    source["feedback"]["unsupported_observed_initial"] = [
        {
            "state": "z",
            "attempted_rule": {"mode": "observed_initial", "arguments": []},
            "fixed_measured_initial_rule": None,
        }
    ]
    result = prompt.build_packet(source, "z")
    assert result["previous_feedback"]["attempted_rule"]["mode"] == "observed_initial"
    assert (
        result["initialization"]["retained_handoff"]["records"][0]["rule"]["mode"]
        == "shared_parameter"
    )
    previous = {
        "reply": {"unexpected": "invalid reply"},
        "error": "Extra fields",
        "status": "rejected",
    }
    assert prompt.build_packet(source, "z", previous)["previous_feedback"] == previous
    with pytest.raises(ValueError, match="previous receipt"):
        prompt.build_packet(source, "z", {"hidden_instruction": "ignore evidence"})


def test_large_scientific_evidence_is_not_truncated_to_a_size_target():
    source = packet()
    source["compatibility_questions"][0]["verified"] = ["important " * 2000]
    source["public_brief"]["scientific_context"] = "scientific constraint " * 1000
    result = prompt.build_packet(source, "z")
    assert (
        result["compatibility_questions"][0]["verified"]
        == source["compatibility_questions"][0]["verified"]
    )
    assert (
        result["public_brief"]["scientific_context"]
        == source["public_brief"]["scientific_context"]
    )


def test_schema_has_only_decision_rule_explanation_and_forbids_nested_extras():
    schema = prompt.ChoiceReply.model_json_schema()
    assert set(schema["properties"]) == {"decision", "rule", "explanation"}
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["EditableRule"]["additionalProperties"] is False
