"""Explicit whole-model RHS revisions retain the parent's initialization contract."""

from copy import deepcopy

import pytest

from autoformalism.research.trajectory_function_revision import (
    assess_fixed,
    assess_revision,
    permitted_sources,
)
from tests.test_preproposal_topology import case_and_choice


def example():
    _, _, frozen, _ = case_and_choice()
    frozen["required_histories"] = []
    names = ("level", "upstream")
    parent = {
        "variable_types": dict.fromkeys(names, "differential"),
        "sources": {"level": ["rain"], "upstream": ["spill"]},
        "initializers": {n: {"mode": "observed_initial", "channel": n} for n in names},
    }
    formula = {
        "equations": {"level": "rain + k*spill", "upstream": "spill"},
        "parameters": {"k": 2},
        "initializers": {},
    }
    return parent, frozen, formula


def test_full_child_passes_with_exact_patch_and_retains_parent():
    parent, frozen, formula = example()
    before = deepcopy(parent)
    outcome = assess_revision(parent, frozen, formula, "Add the second water supply.")
    assert outcome["status"] == "eligible"
    assert outcome["patch"] == {"level": {"added": ["spill"], "removed": []}}
    assert outcome["graph"]["initializers"] == before["initializers"]
    assert outcome["graph"]["variable_types"] == before["variable_types"]
    assert parent == before
    assert set(outcome["assessment"]["training_checks"]) == {"level", "upstream"}
    assert outcome["balance_check"]["status"] == "not_applicable"


def test_admission_uses_all_training_even_if_previous_full_check_flag_is_false():
    parent, frozen, formula = example()
    parent["sources"]["level"] = ["rain", "spill"]
    frozen["check_all_measured"] = False
    formula["equations"]["level"] = "k*rain"
    outcome = assess_revision(parent, frozen, formula, "Remove the second supply.")
    assert outcome["status"] == "training_conflict"
    assert outcome["assessment"]["public_graph_check"] == "passed"
    assert outcome["assessment"]["training_checks"]["level"]["conflicting_pairs"] > 0
    assert "a at t=2 has level=0" in outcome["reason"]
    assert "c at t=2 has level=4" in outcome["reason"]
    assert "graph" not in outcome


def test_public_path_blocks_revised_equations_and_feedback_explains_it():
    parent, frozen, formula = example()
    frozen["public_brief"]["requirements"] = [
        {
            "id": "storage",
            "drivers": ["upstream"],
            "targets": ["level"],
            "public_requirement": "Upstream conditions must affect level.",
        }
    ]
    outcome = assess_revision(parent, frozen, formula, "Consider direct inflows.")
    assert outcome["status"] == "invalid_topology"
    assert "Missing required path" in outcome["reason"]


@pytest.mark.parametrize(
    ("expression", "extra", "message"),
    [
        ("rain+unknown", {}, "unavailable"),
        ("rain + spill", {"upstream": 1}, "shadow"),
        ("rain+0*spill", {}, "cancels"),
        ("__import__('x')", {}, "equations.level"),
        ("rain + state0_level", {"state0_level": 1}, "shadow"),
    ],
)
def test_reject_unknown_shadows_disallowed_names_and_cancelled_sources(
    expression, extra, message
):
    parent, frozen, formula = example()
    formula["parameters"] = extra
    formula["equations"]["level"] = expression
    result = assess_revision(parent, frozen, formula, "Propose source edit.")
    assert result["status"] == "invalid_formula" and message in result["reason"]


def test_unchanged_reason_and_initializer_errors_do_not_advance():
    parent, frozen, formula = example()
    assert (
        assess_revision(parent, frozen, formula, "   ")["status"] == "invalid_topology"
    )
    formula["initializers"] = {"level": "1"}
    assert assess_revision(parent, frozen, formula, "Source edit.")["status"] == (
        "invalid_formula"
    )
    formula["initializers"] = {}
    parent["sources"]["level"] = ["spill", "rain"]
    assert assess_revision(parent, frozen, formula, "Source edit.")["status"] == (
        "no_topology_change"
    )


def test_adding_initial_only_channel_differs_from_keeping_an_admitted_source():
    parent, frozen, formula = example()
    context = frozen["evidence"]["context"]
    context["fixed_covariates"] = ["survey", "area"]
    frozen["measured_initial_bindings"] = {"upstream": "survey"}
    assert {"area", "t"} <= set(permitted_sources(parent, frozen))
    assert "survey" not in permitted_sources(parent, frozen)
    formula["equations"]["level"] = "rain + survey"
    formula["parameters"] = {}
    result = assess_revision(parent, frozen, formula, "Use initial survey throughout.")
    assert "unavailable RHS sources: ['survey']" in result["reason"]
    # Validating an unchanged source set does not spend a graph-check budget.
    parent["sources"]["level"] = ["rain", "survey"]
    result = assess_revision(parent, frozen, formula, "Retain the existing source.")
    assert result["status"] == "no_topology_change"


def test_fixed_path_preserves_sources_and_reserves_all_public_symbols():
    parent, frozen, formula = example()
    assert assess_fixed(parent, frozen, formula)["status"] == "invalid_formula"
    formula["equations"]["level"] = "k * rain"
    assert assess_fixed(parent, frozen, formula)["graph"] == parent
    # A surveyed coefficient outside the original graph is still a public name.
    frozen["evidence"]["context"]["fixed_covariates"] = ["area"]
    formula["parameters"] = {"area": 1}
    formula["equations"]["level"] = "area * rain"
    blocked = assess_fixed(parent, frozen, formula)
    assert blocked["status"] == "invalid_formula" and "shadow" in blocked["reason"]


def test_fixed_and_revised_routes_honor_same_proven_balance_veto(monkeypatch):
    from autoformalism.research import trajectory_function_revision as revisions

    monkeypatch.setattr(
        revisions,
        "check_transfer_balance",
        lambda *args: {"status": "fail", "reason": "Different transfer magnitudes."},
    )
    parent, frozen, formula = example()
    assert assess_revision(parent, frozen, formula, "Add spill.")["status"] == (
        "balance_conflict"
    )
    parent["sources"]["level"].append("spill")
    blocked = assess_fixed(parent, frozen, formula)
    assert blocked["status"] == "balance_conflict"
    assert "formula" not in blocked and "graph" not in blocked
