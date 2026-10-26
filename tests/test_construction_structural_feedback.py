"""Structural feedback keeps every target witness and does not change acceptance."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_structural_feedback as feedback
from autoformalism.research.construction_affine_family_consistency import (
    assess as affine_assess,
)
from autoformalism.research.construction_function_consistency import (
    assess as function_assess,
)
from autoformalism.research.construction_initial_family_consistency import (
    assess as initial_assess,
)
from tests.test_construction_affine_family_consistency import (
    evidence as affine_evidence,
)
from tests.test_construction_affine_family_consistency import problem as affine_problem
from tests.test_construction_function_consistency import evidence, problem
from tests.test_construction_initial_family_consistency import (
    evidence as initial_evidence,
)
from tests.test_construction_initial_family_consistency import (
    problem as initial_problem,
)


def parent():
    return {"problem": problem().model_dump(mode="json")}


def report(**changes):
    return {
        "status": "conflict",
        "conflicts": [],
        "checks": [],
        "assumptions": [],
        "child_problem_sha256": "model",
        "evidence_sha256": "evidence",
        **changes,
    }


def patch(**changes):
    return {
        "decision": "revise",
        "hypothesis": "Add justified coupling",
        "evidence_ids": ["training_consistency"],
        "expected_observable_effect": "Different trajectories",
        "edits": [
            {"component": "x", "kind": "dynamic", "sources": [], "expression": "-k*x+m"}
        ],
        **changes,
    }


def test_function_witness_keeps_direction_zero_forcing_and_widest_available_gap():
    source = problem(rhs="-k*x")
    checked = function_assess(source, evidence(source, y=(0, 2, 4), u=(0, 0, 0)))
    conflict = next(c for c in checked["conflicts"] if c["target"] == "y")
    weak = deepcopy(conflict)
    weak["witnesses"][0]["contradiction_margin_lower_bound"] = 0.01
    weak["witnesses"][0]["trajectory"] = "weak"
    checked["conflicts"].insert(0, weak)
    full = report(function_consistency=checked)
    before = deepcopy(full)
    shown = feedback.consistency_view(full, parent())
    check = next(
        c for c in shown["function_consistency"]["conflicts"] if c["target"] == "y"
    )
    assert check["witness"]["trajectory"] != "weak"
    assert check["witness"]["model_implied_change_bound"]["comparison"] == "at_most"
    assert "upper bound is too low" in check["witness"]["directional_explanation"]
    assert check["zero_forcing"]["channels"] == ["u"]
    assert (
        "does not assert those states are zero" in check["zero_forcing"]["explanation"]
    )
    assert "initializers remain fixed" not in str(shown)
    assert "new latent states" in shown["repair_scope"]
    assert shown["editable_components"]["initialization_edits"] == ["m"]
    assert full == before


def test_every_affine_target_retains_its_own_witness_and_assumptions():
    source = affine_problem(pin_root=True)
    checked = affine_assess(
        source,
        affine_evidence(source, [[2, 3, 4], [1, 1, 2], [3, 4, 5]], z=[[2, 2, 2]] * 3),
    )
    first = next(
        c
        for c in checked["conflicts"]
        if c["property"] == "affine_initial_superposition"
    )
    second = deepcopy(first)
    second["target"] = "different_target"
    second["witnesses"][0]["observations"] = [9, 1, 2]
    checked["conflicts"].append(second)
    before = deepcopy(checked)
    view = feedback.consistency_view(report(affine_family_consistency=checked))[
        "affine_family_consistency"
    ]
    rows = [
        c for c in view["conflicts"] if c["property"] == "affine_initial_superposition"
    ]
    assert {c["target"] for c in rows} == {"y", "different_target"}
    assert next(c for c in rows if c["target"] == "different_target")["witnesses"][0][
        "observations"
    ] == [9, 1, 2]
    assert view["assumptions"] == checked["assumptions"]
    assert "shared linear coefficients" in rows[0]["repair_feedback"]
    assert "basal term" in rows[0]["repair_feedback"]
    assert checked == before


def test_stronger_superposition_uses_same_identity_and_preserves_original_witness():
    source = affine_problem(pin_root=True)
    data = affine_evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 100]], z=[[2, 2, 2]] * 3
    )
    checked = affine_assess(source, data)
    before = deepcopy(checked)
    # The displayed certificate may belong to a rejected trial. The current
    # parent can have a different equation family and is only the edit anchor.
    starting_parent = affine_problem(pin_root=True, nonlinear=True)
    shown = feedback.consistency_view(
        report(affine_family_consistency=checked),
        {
            "problem": starting_parent.model_dump(mode="json"),
            "_consistency_evidence": data,
        },
    )["affine_family_consistency"]
    check = next(
        c for c in shown["conflicts"] if c["property"] == "affine_initial_superposition"
    )
    assert check["original_witness"]["time"] == 1
    assert check["witnesses"][0]["time"] == 2
    assert (
        check["witnesses"][0]["weights_exact"]
        == check["original_witness"]["weights_exact"]
    )
    assert check["witness_refinement"]["qualification_same_family"] is True
    assert not check["witness_refinement"]["acceptance_changed"]
    assert checked == before


def test_refinement_rejects_different_evidence_and_cannot_clear_conflict():
    source = affine_problem(pin_root=True)
    data = affine_evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 100]], z=[[2, 2, 2]] * 3
    )
    checked = affine_assess(source, data)
    different = affine_evidence(
        source, [[2, 3, 4], [1, 1, 2], [3, 4, 90]], z=[[2, 2, 2]] * 3
    )
    with pytest.raises(ValueError, match="evidence identity differs"):
        feedback.consistency_view(
            report(affine_family_consistency=checked),
            {
                "problem": source.model_dump(mode="json"),
                "_consistency_evidence": different,
            },
        )


def test_rank_refinement_keeps_single_free_initial_direction_and_guard():
    source = affine_problem()
    data = affine_evidence(source, [[0, 1, 1], [0, 2, 3], [0, 3, 10]])
    checked = affine_assess(source, data)
    shown = feedback.consistency_view(
        report(affine_family_consistency=checked),
        {
            "problem": source.model_dump(mode="json"),
            "_consistency_evidence": data,
        },
    )["affine_family_consistency"]
    check = next(
        c
        for c in shown["conflicts"]
        if c["property"] == "linear_initial_difference_rank_one"
    )
    assert check["witnesses"][0]["possibly_different_initial_states"] == ["m"]
    assert check["witness_refinement"]["comparisons"] == 3
    assert check["witness_refinement"]["qualification_same_family"] is True


def test_initial_witness_offers_supported_initial_edit_without_freezing_all_initials():
    source = initial_problem()
    checked = initial_assess(source, initial_evidence(source))
    shown = feedback.consistency_view(report(initial_family_consistency=checked))
    check = shown["initial_family_consistency"]["conflicts"][0]
    assert "eligible latent initializer" in check["repair_direction"]
    assert "changing only the ode" in check["repair_direction"].lower()
    assert check["witnesses"][0]["maximum_initial_absolute_error_lower_bound"] > 0


def test_equality_feedback_explains_basal_source_cannot_distinguish_trajectories():
    check = {
        "property": "closed_subsystem_equality",
        "target": "seen",
        "status": "conditional_conflict",
        "witnesses": [{"observation_gap": 5}],
    }
    shown = feedback.consistency_view(report(conflicts=[check]))
    assert "does not distinguish" in shown["conflicts"][0]["repair_feedback"]
    assert shown["status"] == "conflict"


def test_rank_one_and_deferred_checks_keep_distinct_interpretations():
    checked = {
        "protocol": "affine",
        "status": "conflict",
        "assumptions": [],
        "coverage": {},
        "child_problem_sha256": "model",
        "evidence_sha256": "evidence",
        "checks": [],
        "conflicts": [
            {
                "property": "linear_initial_difference_rank_one",
                "target": "y",
                "witnesses": [{"determinant": 3}],
            }
        ],
    }
    shown = feedback.consistency_view(
        report(
            affine_family_consistency=checked,
            checks=[
                {
                    "property": "other",
                    "target": "z",
                    "status": "deferred",
                    "unavailable": ["nonlinear"],
                }
            ],
        )
    )
    assert (
        "new latent state/initializer"
        in shown["affine_family_consistency"]["conflicts"][0]["repair_feedback"]
    )
    assert shown["other_checks"][0]["status"] == "deferred"
    assert "Deferred checks do not certify" in shown["interpretation"]


def test_duplicate_declarations_get_specific_advice_without_editing_the_reply():
    reply = patch()
    reply["edits"] *= 33
    before = deepcopy(reply)
    result = feedback.feedback_guidance("REVISION_SCHEMA", {}, parent(), reply)
    assert result["duplicate_declarations"] == [
        {"field": "edits", "name": "x", "declarations": 33}
    ]
    assert "one complete declaration" in result["duplicate_feedback"]
    assert "do not delete unrelated" in result["duplicate_feedback"]
    assert reply == before


def test_feedback_identifies_new_state_repeated_as_existing_component_and_initializer():
    initial = {"mode": "value", "scope": "trajectory"}
    reply = patch(
        new_states=[
            {
                "name": "n",
                "description": "Memory",
                "rhs": "x-n",
                "initial": initial,
            }
        ],
        edits=[
            {"component": "n", "kind": "dynamic", "sources": [], "expression": "x-n"}
        ],
        initialization_edits=[{"state": "n", "initial": initial}],
    )
    before = deepcopy(reply)
    result = feedback.feedback_guidance("DECLARATION_CONFLICT", {}, parent(), reply)
    assert result["duplicate_declarations"] == [
        {"field": "edits", "name": "n", "also_declared_in": "new_states"},
        {
            "field": "initialization_edits",
            "name": "n",
            "also_declared_in": "new_states",
        },
    ]
    assert "different actions" in result["specific_feedback"]
    assert "original list limits" in result["duplicate_feedback"]
    assert reply == before


@pytest.mark.parametrize(
    "code, expected",
    [
        ("COMPONENT_IDENTITY", "initialization_graph and decision are control fields"),
        ("INITIALIZATION_IDENTITY", "Public recorded time-zero values"),
        ("REPEATED_EQUATION_EDIT", "changing the explanation"),
        ("CONFLICTING_OPERATIONS", "Do not both edit and remove"),
        ("DECLARATION_CONFLICT", "alias bindings are not formatting differences"),
        ("UNKNOWN_EVIDENCE", "allowed_evidence_ids exactly"),
        ("UNUSED_NEW_PARAMETER", "used only in an initializer"),
        ("REVISION_VALIDATION", "does not bind its initial value"),
    ],
)
def test_failure_guidance_is_scope_specific(code, expected):
    details = {"location": "x"}
    result = feedback.feedback_guidance(code, details, parent())
    assert expected in result["specific_feedback"]
    assert result["details"] == details
    assert "does not accept the model" in result["repair_scope"]


def test_feedback_distinguishes_coefficients_controls_and_new_state_initials():
    components = feedback.feedback_guidance("COMPONENT_IDENTITY", {}, parent())
    assert "parameter belongs in new_parameters" in components["specific_feedback"]
    assert "map's parameters list" in components["specific_feedback"]
    assert "kind requires a wider revision" in components["specific_feedback"]
    initial = feedback.feedback_guidance("INITIALIZATION_IDENTITY", {}, parent())
    assert "new_states[].initial" in initial["specific_feedback"]
    assert (
        "Do not list a new state in initialization_edits"
        in initial["specific_feedback"]
    )


def test_trial_assessment_gets_compacted_without_mutating_saved_failure():
    details = {"assessment": report(status="deferred")}
    before = deepcopy(details)
    result = feedback.feedback_guidance("TRAINING_CONFLICT", details, parent())
    assert result["details"]["assessment"]["status"] == "deferred"
    assert details == before


def test_compact_trial_feedback_is_idempotent_and_independent_of_starting_model():
    source = problem(rhs="-k*x")
    checked = function_assess(source, evidence(source, y=(0, 2, 4)))
    compact = feedback.consistency_view(report(function_consistency=checked))
    assert feedback.consistency_view(compact, parent()) == compact
    before = deepcopy(compact)
    result = feedback.feedback_guidance(
        "TRAINING_CONFLICT", {"assessment": compact}, parent()
    )
    assert result["details"]["assessment"] == compact
    assert compact == before


def test_signature_ignores_prose_and_advisory_sources_but_tracks_structural_changes():
    original = patch(
        new_states=[
            {
                "name": "n",
                "description": "new memory",
                "rhs": "x-n",
                "sources": ["x", "n"],
                "initial": {"mode": "value", "scope": "shared_training"},
            }
        ]
    )
    changed = deepcopy(original)
    changed["hypothesis"] = "Another description"
    changed["edits"][0].update(expression="d(x)/dt = (-k * x) + m", sources=["x", "m"])
    changed["new_states"][0]["description"] = "Renamed description"
    assert feedback.patch_signature(original) is not None
    assert feedback.patch_signature(original) == feedback.patch_signature(changed)
    changed["new_states"][0]["initial"]["scope"] = "trajectory"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)
    changed = deepcopy(original)
    changed["new_states"][0]["rhs"] = "x-2*n"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)


def test_signature_tracks_initializer_maps_removals_and_parameter_roles():
    original = patch(
        initialization_edits=[
            {
                "state": "m",
                "initial": {
                    "mode": "map",
                    "expression": "a*reading",
                    "public_initials": [{"alias": "reading", "channel": "u"}],
                    "parameters": [{"name": "a", "role": "coefficient"}],
                },
            }
        ]
    )
    for updated in (
        {"remove_processes": ["flow"]},
        {"new_parameters": [{"name": "b", "role": "coefficient"}]},
    ):
        assert feedback.patch_signature(original) != feedback.patch_signature(
            original | updated
        )
    changed = deepcopy(original)
    changed["initialization_edits"][0]["initial"]["public_initials"][0]["channel"] = "y"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)
    changed = deepcopy(original)
    changed["initialization_edits"][0]["initial"]["expression"] = "m = a * reading"
    assert feedback.patch_signature(original) == feedback.patch_signature(changed)


def test_signature_uses_same_safe_declaration_normalization_as_application():
    original = patch(
        new_states=[
            {
                "name": "n",
                "description": "Memory",
                "rhs": "x-n",
                "initial": {"mode": "value", "scope": "trajectory"},
            }
        ]
    )
    redundant = deepcopy(original)
    redundant["new_states"] *= 2
    redundant["edits"] += [
        {
            "component": "n",
            "kind": "dynamic",
            "sources": [],
            "expression": "d(n)/dt = x - (n)",
        }
    ]
    redundant["initialization_edits"] = [
        {"state": "n", "initial": {"mode": "value", "scope": "trajectory"}}
    ]
    assert feedback.patch_signature(original) is not None
    assert feedback.patch_signature(original) == feedback.patch_signature(redundant)
    redundant["edits"][-1]["kind"] = "algebraic"
    assert feedback.patch_signature(redundant) is None


def test_signature_ignores_initializer_reference_list_order_but_keeps_alias_bindings():
    original = patch(
        initialization_edits=[
            {
                "state": "m",
                "initial": {
                    "mode": "map",
                    "expression": "a+b+c+d",
                    "state_initials": [
                        {"alias": "a", "state": "x"},
                        {"alias": "b", "state": "i"},
                    ],
                    "public_initials": [
                        {"alias": "c", "channel": "u"},
                        {"alias": "d", "channel": "y"},
                    ],
                },
            }
        ]
    )
    changed = deepcopy(original)
    initial = changed["initialization_edits"][0]["initial"]
    initial["state_initials"].reverse()
    initial["public_initials"].reverse()
    assert feedback.patch_signature(original) is not None
    assert feedback.patch_signature(original) == feedback.patch_signature(changed)
    initial["state_initials"][0]["state"] = "x"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        patch(edits="bad"),
        patch(edits=[{}]),
        patch(unrecognized=True),
        patch(remove_states=[1]),
        patch(
            edits=[
                {
                    "component": "x",
                    "kind": "dynamic",
                    "sources": [],
                    "expression": "__import__('os')",
                }
            ]
            * 2
        ),
        patch(
            edits=[
                {"component": "x", "kind": "dynamic", "sources": [], "expression": "x"}
            ]
            * 33
        ),
    ],
)
def test_signature_invalid_or_oversized_is_unavailable(invalid):
    assert feedback.patch_signature(invalid) is None


def test_wider_request_is_explicit_and_not_a_scientific_edit():
    wider = patch(decision="requires_wider_revision", edits=[])
    assert feedback.patch_signature(wider) is not None
    assert feedback.patch_signature(wider | {"edits": patch()["edits"]}) is None


def test_parent_identity_feedback_exposes_per_name_repair_without_reclassification():
    details = {
        "names": ["init_k"],
        "declaration_issues": [
            {
                "name": "init_k",
                "category": "reserved_namespace",
                "repair": "Declare this initializer coefficient in its local map.",
            }
        ],
    }
    for code in (
        "DECLARATION_IDENTITY",
        "RESERVED_DECLARATION_NAME",
        "INITIALIZER_PARAMETER_SCOPE",
    ):
        result = feedback.feedback_guidance(code, details, parent())
        assert result["details"] == details
        assert (
            "reserved prefix and an existing identity are different"
            in result["specific_feedback"]
        )
        assert "without redeclaring unchanged" in result["specific_feedback"]
    assert details["declaration_issues"][0]["category"] == "reserved_namespace"
