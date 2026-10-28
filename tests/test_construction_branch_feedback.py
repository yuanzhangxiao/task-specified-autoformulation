"""Typed branch advice preserves patches, evidence, and scientific choice."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_branch_feedback as feedback


def parent():
    return {
        "problem": {
            "base_candidate": {
                "parameters": [{"name": "a", "role": "rate", "scope": "global"}],
                "state_equations": [{"state": "I", "rhs": "-a*I"}],
                "processes": [{"name": "production", "expression": "a*I"}],
            }
        }
    }


def candidate(code, patch, **details):
    return {
        "id": "call0-option1",
        "composed_patch": patch,
        "diagnostic": {"code": code, "details": details},
    }


def test_parameter_equation_advice_is_conditional_and_preserves_other_actions():
    patch = {
        "edits": [
            {"component": "production", "kind": "algebraic", "expression": "a*I+k_I0"},
            {"component": "k_I0", "kind": "dynamic", "expression": "0"},
        ],
        "new_parameters": [{"name": "k_I0", "role": "nonnegative_coefficient"}],
    }
    selected = candidate("COMPONENT_IDENTITY", patch, component="k_I0")
    before = deepcopy(selected)
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["category"] == "parameter_used_as_equation_target"
    option = card["choices"][0]
    assert "trainable parameter" in option["if"]
    assert "not a fixed-value condition" in option["if"]
    assert option["reply_fragment"] == {
        "drop_actions": [{"field": "edits", "key": "k_I0"}]
    }
    assert card["existing_parameter_declaration"] == patch["new_parameters"][0]
    assert card["other_existing_equation_edits_with_matching_kind"] == ["production"]
    assert "does not prohibit editing" in card["explanation"]
    assert selected == before


def test_parameter_identity_never_inferred_from_a_coefficient_like_name():
    selected = candidate(
        "COMPONENT_IDENTITY",
        {"edits": [{"component": "k_I0", "kind": "dynamic", "expression": "0"}]},
        component="k_I0",
    )
    assert feedback.view(parent(), selected)["action_choices"] == []


def test_new_process_in_edits_distinguished_from_an_undeclared_symbol():
    patch = {
        "edits": [
            {"component": "I", "kind": "dynamic", "expression": "basal_insulin-a*I"},
            {
                "component": "basal_insulin",
                "kind": "algebraic",
                "expression": "k_basal",
            },
        ],
        "new_parameters": [{"name": "k_basal", "role": "nonnegative_coefficient"}],
    }
    selected = candidate("UNKNOWN_FORMULA_SYMBOL", patch, unknown=["basal_insulin"])
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["category"] == "new_definition_placed_in_existing_edits"
    assert "new_processes" in card["instruction"]
    assert card["required_explicit_cleanup"] == {
        "drop_actions": [{"field": "edits", "key": "basal_insulin"}]
    }
    assert "intended to introduce a new quantity" in card["if"]


@pytest.mark.parametrize("name", ["k_absorption", "new_unknown"])
def test_unresolved_symbol_does_not_choose_a_parameter_role(name):
    selected = candidate(
        "UNKNOWN_FORMULA_SYMBOL",
        {"new_states": [{"name": "memory", "rhs": f"{name}*(I-memory)"}]},
        unknown=[name],
    )
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["category"] == "unresolved_formula_symbol"
    assert "spelling does not determine" in card["instruction"]
    assert any("chosen admissible role" in choice for choice in card["choices"])
    assert any("new_processes" in choice for choice in card["choices"])
    assert "reply_fragment" not in card


def test_ambiguous_new_definition_has_no_suggested_automatic_move():
    selected = candidate(
        "UNKNOWN_FORMULA_SYMBOL",
        {
            "edits": [
                {"component": "extra", "kind": "dynamic", "expression": "I-extra"},
                {"component": "extra", "kind": "algebraic", "expression": "I"},
            ]
        },
        unknown=["extra"],
    )
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["category"] == "unresolved_formula_symbol"
    assert "required_explicit_cleanup" not in card


def test_invalid_existing_initializer_names_the_permitted_locations():
    patch = {
        "initialization_edits": [{"state": "I", "initial": {"scope": "trajectory"}}]
    }
    selected = candidate(
        "INITIALIZATION_IDENTITY",
        patch,
        unavailable=["I"],
        editable_initial_states=["memory"],
    )
    before = deepcopy(selected)
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["eligible_existing_initializer_targets"] == ["memory"]
    assert (
        card["locations"]["existing_eligible_latent_state"]
        == "patch.initialization_edits"
    )
    assert "patch.new_states[].initial" in card["locations"]["new_differential_state"]
    assert "Recorded/fixed initial identities remain fixed" in card["preserve"]
    assert "if_introducing_this_new_state" not in card
    assert selected == before


def test_new_state_initialization_in_wrong_locations_requires_both_explicit_drops():
    selected = candidate(
        "INITIALIZATION_IDENTITY",
        {
            "edits": [
                {"component": "X_mem", "kind": "dynamic", "expression": "I-X_mem"}
            ],
            "initialization_edits": [
                {"state": "X_mem", "initial": {"scope": "trajectory"}}
            ],
        },
        unavailable=["X_mem"],
        editable_initial_states=["memory"],
    )
    card = feedback.view(parent(), selected)["action_choices"][0]
    assert card["if_introducing_this_new_state"]["drop_actions"] == [
        {"field": "edits", "key": "X_mem"},
        {"field": "initialization_edits", "key": "X_mem"},
    ]


def test_qualified_failure_explains_that_the_mathematical_patch_was_already_tested():
    patch = {"edits": [{"component": "I", "kind": "dynamic", "expression": "-a*I"}]}
    selected = candidate("QUALIFIED_MODEL_DATA_CONFLICT", patch)
    selected["repeats_candidate_ids"] = ["call0-option0"]
    report = feedback.view(parent(), selected)
    assert "passed normalization and static validation" in report["repair_goal"]
    assert "Reapplying these actions" in report["repair_goal"]
    assert "actual mathematical premise" in report["repair_goal"]
    assert report["already_tested_scientific_actions"] == patch
    assert report["repeated_candidate_ids"] == ["call0-option0"]


def test_citation_failure_does_not_claim_the_trial_has_been_scientifically_checked():
    report = feedback.view(parent(), candidate("UNKNOWN_EVIDENCE", {}))
    assert "has not reached scientific assessment" in report["repair_goal"]
    assert "first complete check" in report["repair_goal"]


def test_apply_replaces_only_the_known_generic_instruction_field():
    quoted = "Correct the listed locations in one complete patch."
    packet = {
        "repair_feedback": {
            "guidance": {
                "specific_feedback": quoted,
                "model_relations": {"repair_scope": quoted},
            },
            "diagnostic": {"details": {"quoted_model": quoted}},
            "complete_intended_patch": {"hypothesis": quoted},
        },
        "training_consistency": {"model_relations": {"repair_scope": quoted}},
        "search_instruction": {"instruction": "old"},
    }
    before = deepcopy(packet)
    feedback.apply(packet, parent(), candidate("OTHER_INTERFACE_FAILURE", {}))
    assert packet["repair_feedback"]["guidance"]["specific_feedback"] != quoted
    assert (
        packet["repair_feedback"]["guidance"]["model_relations"]
        == before["repair_feedback"]["guidance"]["model_relations"]
    )
    assert (
        packet["repair_feedback"]["diagnostic"]
        == before["repair_feedback"]["diagnostic"]
    )
    assert (
        packet["repair_feedback"]["complete_intended_patch"]
        == before["repair_feedback"]["complete_intended_patch"]
    )
    assert packet["training_consistency"] == before["training_consistency"]


def test_malformed_pending_fields_do_not_generate_invented_action_advice():
    selected = candidate("UNKNOWN_FORMULA_SYMBOL", {"edits": "invalid"}, unknown=None)
    report = feedback.view(parent(), selected)
    assert report["action_choices"] == []
    assert not report["patch_changed_by_runtime"]
    assert not report["scientific_role_selected_by_runtime"]
