"""Requirement coverage, safe routing and preservation of complete saved models."""

import copy

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.search.requirement_feedback import (
    apply_requirement_repair,
    diagnose_requirements,
    model_review_facts,
)
from scripts.smoke_prefit_requirements import synthetic_plan


@pytest.fixture
def case(tmp_path):
    return next(c for c in synthetic_plan(tmp_path)["cases"] if c["cohort"] == "repair")


def patch_for(case, expression=None):
    slot = next(s for s in case["bundle"]["slots"] if s["selected_term"]["lhs"] == "m")
    reply = copy.deepcopy(slot["accepted_reply"])
    reply["expression"] = expression or reply["expression"].replace("v01", "v01**2")
    return {**reply, "interaction_id": slot["interaction_id"]}


def test_global_obligation_does_not_depend_on_local_keyword(case):
    assert all(
        not s["selected_term"]["functional_obligation"][
            "requires_nonlinear_source_dependence"
        ]
        for s in case["bundle"]["slots"]
    )
    diagnosis = diagnose_requirements(case["bundle"], case["bindings"])
    assert diagnosis["requirement_gap"] and diagnosis["route"] == "function_repair"
    assert diagnosis["findings"][0]["code"] == "REQUIRED_NONLINEAR_FEEDBACK_MISSING"


def test_repair_preserves_topology_siblings_and_initializers(case):
    before = copy.deepcopy(case)
    result = apply_requirement_repair(case["bundle"], case["bindings"], patch_for(case))
    assert not result["diagnosis"]["requirement_gap"]
    assert result["protected_slots_preserved"] == len(case["bundle"]["slots"]) - 1
    assert result["initialization"]["plan"] == case["bundle"]["initialization"]["plan"]
    assert (
        result["candidate"]["initial_conditions"]
        == case["bundle"]["candidate"]["initial_conditions"]
    )
    assert case == before


@pytest.mark.parametrize(
    "expression",
    [
        "-a*m+b*u01+c*v01",
        "-a*m+b*u01+c*v01+0*v01**2",
        "-a*m+b*u01+c*v01+(v01**2-v01**2)",
        "-a*m+b*u01+c*v01+0/v01",
        "-a*m+b*u01+c*v01+(-0)*v01**2",
        "-a*m+b*u01+c*v01**0",
        "m = -a*m+b*u01+c*v01**2",
        "-a*m+b*unknown+c*v01**2",
        "__import__('os').system('echo nope')",
    ],
)
def test_invalid_or_ineffective_revision_rolls_back(case, expression):
    before = copy.deepcopy(case)
    with pytest.raises((ValueError, TypeError, ModelValidationError)):
        apply_requirement_repair(
            case["bundle"], case["bindings"], patch_for(case, expression)
        )
    assert case == before


def test_frozen_requirement_text_cannot_be_reworded(case):
    bindings = copy.deepcopy(case["bindings"])
    bindings[0]["public_requirement"] = "A different requirement"
    with pytest.raises(ValueError, match="frozen public requirement"):
        diagnose_requirements(case["bundle"], bindings)


def test_no_binding_does_not_invent_nonlinearity_requirement(case):
    result = diagnose_requirements(case["bundle"], [])
    assert not result["requirement_gap"] and result["route"] == "preserve"


def test_feedforward_input_nonlinearity_is_not_feedback_evidence(case):
    changed = copy.deepcopy(case["bundle"])
    for slot in changed["slots"]:
        slot["canonical_function"]["expression"] = slot["canonical_function"][
            "expression"
        ].replace("u01", "u01**2")
    assert diagnose_requirements(changed, case["bindings"])["requirement_gap"]


def test_unrelated_or_nonexistent_patch_is_rejected(case):
    patch = {**patch_for(case), "interaction_id": "term_99_0"}
    with pytest.raises(ValueError, match="protected or unrelated"):
        apply_requirement_repair(case["bundle"], case["bindings"], patch)


def test_extra_topology_or_sibling_edits_are_schema_errors(case):
    with pytest.raises(ValueError):
        apply_requirement_repair(
            case["bundle"], case["bindings"], {**patch_for(case), "state_equations": []}
        )


def test_absent_feedback_scaffold_routes_to_topology_without_guessing(case):
    bundle = copy.deepcopy(case["bundle"])
    bundle["slots"] = [
        {
            "interaction_id": "term_0_0",
            "selected_term": {"lhs": "v01", "sources": ["u01"]},
            "canonical_function": {"expression": "u01"},
        }
    ]
    diagnosis = diagnose_requirements(bundle, case["bindings"])
    assert diagnosis["route"] == "topology_revision_required"
    assert diagnosis["eligible_slots"] == []


def test_identity_terms_are_advisory_not_new_rejections(case):
    assert model_review_facts(case["bundle"])["status"] == "advisory_only"


def test_nonlinear_algebraic_feedback_is_supported(case):
    bundle = copy.deepcopy(case["bundle"])
    bundle["slots"] = [
        {
            "interaction_id": "term_0_0",
            "selected_term": {"lhs": "v01", "sources": ["u01", "a"]},
            "canonical_function": {"expression": "u01-a"},
        },
        {
            "interaction_id": "term_1_0",
            "selected_term": {"lhs": "a", "sources": ["v01"]},
            "canonical_function": {"expression": "v01**2"},
        },
    ]
    assert not diagnose_requirements(bundle, case["bindings"])["requirement_gap"]


def test_public_binding_cannot_move_to_another_cell(case):
    bindings = copy.deepcopy(case["bindings"])
    bindings[0]["cell"] = "another_cell"
    with pytest.raises(ValueError, match="different source cell"):
        diagnose_requirements(case["bundle"], bindings)


def test_disconnected_nonlinearity_does_not_satisfy_target_path(case):
    bundle = copy.deepcopy(case["bundle"])
    bundle["slots"].append(
        {
            "interaction_id": "term_99_0",
            "selected_term": {"lhs": "unrelated", "sources": ["unrelated", "u01"]},
            "canonical_function": {"expression": "unrelated**2 + u01"},
        }
    )
    assert diagnose_requirements(bundle, case["bindings"])["requirement_gap"]
