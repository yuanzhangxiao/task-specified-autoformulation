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


def test_v2_existing_process_move_is_explicit_and_preserves_initializer():
    from autoformalism.research.construction_branch_patch import (
        BranchPatchFields,
        DropAction,
    )

    patch = {
        "new_processes": [
            {"name": "production", "expression": "a*I+1", "sources": ["I"]}
        ],
        "initialization_edits": [
            {"state": "memory", "initial": {"mode": "value", "scope": "trajectory"}}
        ],
        "remove_processes": ["production"],
    }
    selected = candidate(
        "DECLARATION_IDENTITY",
        patch,
        declaration_issues=[{"name": "production", "category": "existing_process"}],
    )
    before = deepcopy(selected)
    assert feedback.view(parent(), selected)["action_choices"] == []
    report = feedback.view(parent(), selected, repair_contract=True)
    assert report["policy"] == feedback.REPAIR_POLICY
    card = report["action_choices"][0]
    example = card["if_this_supplied_definition_is_your_intended_law"]
    BranchPatchFields.model_validate(example["patch"])
    for action in example["drop_actions"]:
        DropAction.model_validate(action)
    assert example["patch"]["edits"] == [
        {
            "component": "production",
            "kind": "algebraic",
            "expression": "a*I+1",
            "sources": ["I"],
        }
    ]
    assert example["drop_actions"] == [{"field": "new_processes", "key": "production"}]
    assert card["if_retaining_this_process_also_drop_its_removal"]["drop_actions"] == [
        {"field": "remove_processes", "key": "production"}
    ]
    assert selected == before


def test_v2_ambiguous_process_definitions_do_not_choose_one_law():
    selected = candidate(
        "DECLARATION_IDENTITY",
        {
            "new_processes": [
                {"name": "production", "expression": "I"},
                {"name": "production", "expression": "2*I"},
            ]
        },
        declaration_issues=[{"name": "production", "category": "existing_process"}],
    )
    card = feedback.view(parent(), selected, repair_contract=True)["action_choices"][0]
    assert "if_this_supplied_definition_is_your_intended_law" not in card
    assert card["required_explicit_cleanup"] == {
        "drop_actions": [{"field": "new_processes", "key": "production"}]
    }


def test_v2_parameter_component_collision_offers_both_meanings():
    selected = candidate(
        "DECLARATION_IDENTITY",
        {
            "new_parameters": [{"name": "k_basal", "role": "coefficient"}],
            "new_processes": [{"name": "k_basal", "expression": "1"}],
        },
        declaration_issues=[
            {"name": "k_basal", "category": "parameter_component_collision"}
        ],
    )
    choices = feedback.view(parent(), selected, repair_contract=True)["action_choices"][
        0
    ]["choices"]
    assert choices[0]["reply_fragment"] == {
        "drop_actions": [{"field": "new_processes", "key": "k_basal"}]
    }
    assert choices[1]["reply_fragment"] == {
        "drop_actions": [{"field": "new_parameters", "key": "k_basal"}]
    }
    assert selected["composed_patch"]["new_parameters"][0]["role"] == "coefficient"


def test_v2_initializer_feedback_uses_its_actual_local_alias_not_another_maps_name():
    from autoformalism.research.construction_structural_revision import InitialMapChoice
    from tests.test_construction_revision_edits import problem

    selected = candidate(
        "REVISION_VALIDATION",
        {
            "initialization_edits": [
                {
                    "state": "memory",
                    "initial": {
                        "mode": "map",
                        "expression": "public_1",
                        "parameters": [],
                        "public_initials": [{"alias": "I0", "channel": "I"}],
                        "state_initials": [],
                    },
                }
            ]
        },
    )
    before = deepcopy(selected)
    card = feedback.view(
        {"problem": problem().model_dump(mode="json")}, selected, repair_contract=True
    )["action_choices"][0]
    assert card["undeclared_symbols"] == ["public_1"]
    assert card["unused_local_names"] == ["I0"]
    example = card["syntax_only_example_using_one_of_your_bindings"]
    InitialMapChoice.model_validate(example["initial"])
    assert example["initial"]["expression"] == "I0"
    assert "only if" in example["condition"]
    assert "protected initial" in card["authority"]
    assert selected == before


def test_v2_state_initial_alias_example_keeps_state_reference_and_parameter_scope():
    from tests.test_construction_revision_edits import problem

    selected = candidate(
        "REVISION_VALIDATION",
        {
            "new_states": [
                {
                    "name": "memory",
                    "initial": {
                        "mode": "map",
                        "expression": "gain*wrong",
                        "parameters": [{"name": "gain", "role": "coefficient"}],
                        "state_initials": [{"alias": "root", "state": "I"}],
                        "public_initials": [],
                    },
                }
            ]
        },
    )
    card = feedback.view(
        {"problem": problem().model_dump(mode="json")}, selected, repair_contract=True
    )["action_choices"][0]
    assert card["local_parameters"] == [{"name": "gain", "role": "coefficient"}]
    assert card["state_initials"] == [{"alias": "root", "state": "I"}]
    assert card["syntax_only_example_using_one_of_your_bindings"]["initial"][
        "state_initials"
    ] == [{"alias": "root", "state": "I"}]


def test_v2_decision_contract_error_is_not_presented_as_a_scientific_failure():
    selected = candidate(
        "OPTION_DECISION_CONTRACT",
        {"decision": "requires_wider_revision", "new_states": [{"name": "extra"}]},
    )
    before = deepcopy(selected)
    report = feedback.view(parent(), selected, repair_contract=True)
    assert "before scientific assessment" in report["repair_goal"]
    assert "already supports adding latent states" in report["repair_goal"]
    assert report["action_choices"][0]["reply_fragment"] == {
        "patch": {"decision": "revise"}
    }
    assert "no-actions" in report["repair_goal"]
    assert selected == before


def test_default_feedback_does_not_call_the_new_dependency_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("historical policy must not inspect new dependencies")

    monkeypatch.setattr(feedback.influence, "view", forbidden)
    assert (
        feedback.view(parent(), candidate("QUALIFIED_MODEL_DATA_CONFLICT", {}))[
            "policy"
        ]
        == feedback.POLICY
    )


def test_v2_malformed_alias_and_declaration_fields_do_not_crash_or_invent_choice():
    selected = candidate(
        "REVISION_VALIDATION",
        {
            "initialization_edits": [
                {
                    "state": "I",
                    "initial": {
                        "mode": "map",
                        "expression": None,
                        "public_initials": "wrong",
                    },
                }
            ]
        },
    )
    assert (
        feedback.view(parent(), selected, repair_contract=True)["action_choices"] == []
    )
    selected = candidate("DECLARATION_IDENTITY", {}, declaration_issues="wrong")
    assert (
        feedback.view(parent(), selected, repair_contract=True)["action_choices"] == []
    )


@pytest.mark.parametrize("name,can_edit_initial", [("mem", True), ("I", False)])
def test_v2_existing_state_keeps_law_and_initialization_as_separate_intentions(
    name, can_edit_initial
):
    from autoformalism.research.construction_branch_patch import BranchPatchFields
    from tests.test_construction_revision_edits import problem

    source = {"problem": problem().model_dump(mode="json")}
    selected = candidate(
        "DECLARATION_IDENTITY",
        {
            "new_states": [
                {
                    "name": name,
                    "description": "intended changed memory",
                    "rhs": f"I-{name}",
                    "sources": ["I", name],
                    "initial": {"mode": "value", "scope": "trajectory"},
                }
            ]
        },
        declaration_issues=[{"name": name, "category": "existing_state"}],
    )
    before = deepcopy(selected)
    card = feedback.view(source, selected, repair_contract=True)["action_choices"][0]
    assert card["if_retaining_the_parent_initial_rule"]["drop_actions"] == [
        {"field": "new_states", "key": name}
    ]
    assert "do not silently discard" in card["instruction"]
    assert (
        "if_preserving_the_supplied_law_and_initializer" in card
    ) == can_edit_initial
    if can_edit_initial:
        fragment = card["if_preserving_the_supplied_law_and_initializer"]
        BranchPatchFields.model_validate(fragment["patch"])
        assert fragment["patch"]["initialization_edits"] == [
            {"state": name, "initial": {"mode": "value", "scope": "trajectory"}}
        ]
    assert selected == before


def test_v2_known_latent_initializer_remains_protected():
    from tests.test_construction_revision_edits import problem

    payload = problem().model_dump(mode="json")
    payload["graph"]["rules"]["mem"]["initial"] = {
        "mode": "known",
        "value": 1.0,
        "justification": "known from the task",
    }
    selected = candidate(
        "DECLARATION_IDENTITY",
        {
            "new_states": [
                {
                    "name": "mem",
                    "rhs": "I-mem",
                    "initial": {"mode": "value", "scope": "trajectory"},
                }
            ]
        },
        declaration_issues=[{"name": "mem", "category": "existing_state"}],
    )
    card = feedback.view({"problem": payload}, selected, repair_contract=True)[
        "action_choices"
    ][0]
    assert "mem" not in card["eligible_existing_initializer_targets"]
    assert "if_preserving_the_supplied_law_and_initializer" not in card


def test_v2_unavailable_initializer_binding_is_not_shown_as_a_valid_example():
    from tests.test_construction_revision_edits import problem

    selected = candidate(
        "REVISION_VALIDATION",
        {
            "initialization_edits": [
                {
                    "state": "mem",
                    "initial": {
                        "mode": "map",
                        "expression": "wrong",
                        "parameters": [],
                        "public_initials": [
                            {"alias": "readout", "channel": "unavailable"}
                        ],
                        "state_initials": [],
                    },
                }
            ]
        },
    )
    card = feedback.view(
        {"problem": problem().model_dump(mode="json")}, selected, repair_contract=True
    )["action_choices"][0]
    assert "syntax_only_example_using_one_of_your_bindings" not in card
    assert card["public_initials"] == [{"alias": "readout", "channel": "unavailable"}]


def test_v2_retain_parent_initializer_explicitly_drops_existing_pending_override():
    from autoformalism.research import construction_branch_patch as branch
    from autoformalism.research.construction_structural_revision import (
        apply_structural_revision,
    )
    from tests.test_construction_revision_edits import problem

    source = problem()
    parent_payload = {"problem": source.model_dump(mode="json")}
    pending = {
        "decision": "revise",
        "hypothesis": "Revise memory dynamics and its initialization.",
        "evidence_ids": ["training:I"],
        "expected_observable_effect": "A different memory response.",
        "new_states": [
            {
                "name": "mem",
                "description": "Existing memory",
                "rhs": "I-mem+1",
                "sources": ["I", "mem"],
                "initial": {"mode": "value", "scope": "shared_training"},
            }
        ],
        "initialization_edits": [
            {
                "state": "mem",
                "initial": {"mode": "value", "scope": "shared_training"},
            }
        ],
    }
    selected = candidate(
        "DECLARATION_IDENTITY",
        pending,
        declaration_issues=[
            {"name": "mem", "category": "existing_state"},
        ],
    )
    selected["branch_id"] = selected["id"]
    card = feedback.view(parent_payload, selected, repair_contract=True)[
        "action_choices"
    ][0]
    fragment = card["if_retaining_the_parent_initial_rule"]
    assert fragment["drop_actions"] == [
        {"field": "new_states", "key": "mem"},
        {"field": "initialization_edits", "key": "mem"},
    ]
    composed, _ = branch.compose(
        {
            "mode": "delta",
            **branch.reference(selected),
            **fragment,
        },
        selected,
        repair_contract=True,
    )
    assert composed["initialization_edits"] == []
    applied = apply_structural_revision(source, composed, evidence_ids=["training:I"])
    assert applied["problem"].graph.rules["mem"] == source.graph.rules["mem"]
    assert applied["problem"].graph.rules["mem"].initial.scope == "trajectory"

    keep_supplied = card["if_preserving_the_supplied_law_and_initializer"]
    assert keep_supplied["drop_actions"] == [{"field": "new_states", "key": "mem"}]
    supplied, _ = branch.compose(
        {
            "mode": "delta",
            **branch.reference(selected),
            "patch": keep_supplied["patch"],
            "drop_actions": keep_supplied["drop_actions"],
        },
        selected,
        repair_contract=True,
    )
    assert supplied["initialization_edits"][0]["initial"]["scope"] == "shared_training"


def failed_removal_delta(name="mem", field="remove_states"):
    """A rejected reply both retains and drops one stored removal intention."""
    from autoformalism.research import construction_branch_patch as branch

    selected = candidate("BRANCH_PATCH_COMPOSITION", {field: [name]})
    selected["branch_id"] = selected["id"]
    selected["submitted_reply"] = {
        "mode": "delta",
        **branch.reference(selected),
        "patch": {field: [name]},
        "drop_actions": [{"field": field, "key": name}],
    }
    return selected


@pytest.mark.parametrize(
    "field,name", [("remove_states", "mem"), ("remove_processes", "q")]
)
def test_v3_removal_and_intention_deletion_are_mutually_exclusive(field, name):
    from autoformalism.research import construction_branch_patch as branch
    from tests.test_construction_revision_edits import problem

    selected = failed_removal_delta(name, field)
    before = deepcopy(selected)
    source = {"problem": problem().model_dump(mode="json")}
    old = feedback.view(source, selected, repair_contract=True)
    assert old["action_choices"] == []
    report = feedback.view(source, selected, repair_contract=True, admission_guard=True)
    assert report["policy"] == feedback.ADMISSION_POLICY
    card = report["action_choices"][0]
    assert card["category"] == "remove_component_or_retract_removal"
    assert card["removal_authority"] == "in_scope_subject_to_full_validation"
    keep, remove = card["choices"]
    for choice, expected in ((keep, []), (remove, [name])):
        composed, _ = branch.compose(
            {"mode": "delta", **branch.reference(selected), **choice["reply_fragment"]},
            selected,
            repair_contract=True,
        )
        assert composed[field] == expected
    assert "none of that reply was applied" in remove["instruction"]
    assert "every remaining equation and initializer reference" in remove["instruction"]
    assert selected == before


def test_v3_protected_or_unavailable_removal_is_not_offered_as_executable_choice():
    from tests.test_construction_revision_edits import problem

    source = {"problem": problem().model_dump(mode="json")}
    for payload, name, expected in (
        (source, "I", "protected"),
        ({}, "mem", "unavailable"),
    ):
        selected = failed_removal_delta(name)
        card = feedback.view(
            payload, selected, repair_contract=True, admission_guard=True
        )["action_choices"][0]
        assert card["removal_authority"] == expected
        assert len(card["choices"]) == 1
        assert card["choices"][0]["reply_fragment"]["patch"] == {}


def test_v3_wrong_or_malformed_failed_reply_produces_no_invented_removal_advice():
    selected = failed_removal_delta()
    selected["submitted_reply"]["drop_actions"] = [
        {"field": "new_states", "key": "mem"}
    ]
    assert not feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )["action_choices"]
    selected["submitted_reply"] = "not structured"
    assert not feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )["action_choices"]


def test_v3_drop_for_absent_intention_is_not_presented_as_a_valid_fragment():
    selected = failed_removal_delta()
    selected["composed_patch"]["remove_states"] = []
    card = feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )["action_choices"][0]
    assert card["stored_removal_intention"] is False
    assert card["choices"][0]["reply_fragment"] == {"patch": {}, "drop_actions": []}
    assert "no stored" in card["choices"][0]["instruction"]


def test_v3_conflicting_new_state_initializers_have_one_owner_and_explicit_choices():
    from autoformalism.research import construction_branch_patch as branch

    initial_a = {"mode": "value", "scope": "trajectory"}
    initial_b = {
        "mode": "map",
        "expression": "0",
        "parameters": [],
        "public_initials": [],
        "state_initials": [],
    }
    declaration = {
        "name": "lag",
        "description": "a proposed lag",
        "rhs": "I-lag",
        "sources": ["I", "lag"],
        "initial": initial_a,
    }
    selected = candidate(
        "DECLARATION_CONFLICT",
        {
            "new_states": [declaration],
            "initialization_edits": [{"state": "lag", "initial": initial_b}],
        },
        name="lag",
        fields=["new_states", "initialization_edits"],
    )
    selected["branch_id"] = selected["id"]
    before = deepcopy(selected)
    card = feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )["action_choices"][0]
    assert card["category"] == "new_state_initializer_has_one_owner"
    assert len(card["choices"]) == 2
    for choice, expected in zip(card["choices"], (initial_a, initial_b), strict=True):
        composed, _ = branch.compose(
            {"mode": "delta", **branch.reference(selected), **choice["reply_fragment"]},
            selected,
            repair_contract=True,
        )
        assert composed["new_states"][0]["initial"] == expected
        assert composed["new_states"][0]["rhs"] == "I-lag"
        assert composed["initialization_edits"] == []
    assert selected == before


def test_v3_ambiguous_initialization_declarations_do_not_choose_one_rule():
    selected = candidate(
        "DECLARATION_CONFLICT",
        {
            "new_states": [{"name": "lag"}, {"name": "lag"}],
            "initialization_edits": [{"state": "lag", "initial": {}}],
        },
        name="lag",
        fields=["new_states", "initialization_edits"],
    )
    assert not feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )["action_choices"]


def test_v3_typed_initial_scope_is_displayed_without_interpreting_prose(monkeypatch):
    monkeypatch.setattr(feedback.influence, "view", lambda *args: {})
    patch = {
        "hypothesis": "I intend to set the memory initial to zero.",
        "initialization_edits": [
            {
                "state": "mem",
                "initial": {
                    "mode": "value",
                    "scope": "trajectory",
                    "role": "nonnegative_coefficient",
                },
            }
        ],
    }
    selected = candidate("QUALIFIED_MODEL_DATA_CONFLICT", patch)
    selected["repeats_candidate_ids"] = ["earlier"]
    before = deepcopy(selected)
    report = feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )
    view = report["typed_initializer_meanings"][0]
    assert view["typed_initial_rule"] == patch["initialization_edits"][0]["initial"]
    assert "does not enforce zero" in view["meaning"]
    assert (
        "does not assert that your prose disagrees" in report["initializer_instruction"]
    )
    patch["hypothesis"] = "An unrelated explanation with no initial-value claim."
    changed = feedback.view(
        parent(), selected, repair_contract=True, admission_guard=True
    )
    assert changed["typed_initializer_meanings"] == report["typed_initializer_meanings"]
    assert selected["diagnostic"] == before["diagnostic"]


def test_v3_requires_the_existing_repair_contract_and_preserves_known_packet_fields():
    selected = failed_removal_delta()
    with pytest.raises(ValueError, match="requires repair_contract"):
        feedback.view(parent(), selected, admission_guard=True)
    packet = {
        "repair_feedback": {
            "guidance": {"specific_feedback": "old"},
            "diagnostic": {"quoted": "unchanged"},
        },
        "search_instruction": {"instruction": "old"},
    }
    feedback.apply(
        packet, parent(), selected, repair_contract=True, admission_guard=True
    )
    assert (
        packet["repair_feedback"]["action_feedback"]["policy"]
        == feedback.ADMISSION_POLICY
    )
    assert packet["repair_feedback"]["diagnostic"] == {"quoted": "unchanged"}
