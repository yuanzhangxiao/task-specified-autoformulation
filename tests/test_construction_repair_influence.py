"""Dependency views explain failed trials without selecting or applying repairs."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_repair_influence as influence
from autoformalism.staged_topology import content_hash
from tests.test_construction_revision_edits import problem


def add_state(payload, name="extra", rhs="I-extra"):
    base = payload["base_candidate"]
    base["states"].append({"name": name, "kind": "latent"})
    base["state_equations"].append({"state": name, "rhs": rhs})
    base["initial_conditions"].append(
        {"state": name, "fixed_value": 0, "scope": "global"}
    )
    payload["graph"]["rules"][name] = {
        "initial": {"mode": "value", "scope": "trajectory", "guess": 0.1}
    }


def case(change=None):
    source = problem().model_dump(mode="json")
    trial = deepcopy(source)
    if change:
        change(trial)
    selected = {
        "problem": trial,
        "audit": {
            "static_eligibility": "eligible",
            "child_problem_sha256": content_hash(trial),
            "parent_problem_sha256": content_hash(source),
        },
        "diagnostic": {
            "code": "QUALIFIED_MODEL_DATA_CONFLICT",
            "details": {"assessment": {"conflicts": [{"target": "Gp"}]}},
        },
    }
    return {"problem": source}, selected


def connection(report, component="extra", target="Gp"):
    item = next(c for c in report["components"] if c["name"] == component)
    return next(c for c in item["target_connections"] if c["target"] == target)


def test_no_new_components_is_an_empty_report_not_a_scientific_pass():
    report = influence.view(*case())
    assert report["status"] == "reported"
    assert report["components"] == []
    assert not report["acceptance_changed"]
    assert "not proof" in report["interpretation"]


def test_new_state_driven_by_existing_model_is_not_itself_a_path_back_to_target():
    parent, selected = case(add_state)
    before = deepcopy((parent, selected))
    report = influence.view(parent, selected)
    assert connection(report)["status"] == "disconnected"
    assert "How is extra intended to enter Gp?" in report["components"][0]["question"]
    assert (parent, selected) == before


def test_additive_term_and_nested_algebraic_use_reach_target():
    def change(trial):
        add_state(trial)
        base = trial["base_candidate"]
        base["processes"].extend(
            [
                {"name": "relay", "expression": "extra"},
                {"name": "addition", "expression": "a*relay"},
            ]
        )
        next(row for row in base["state_equations"] if row["state"] == "Gp")["rhs"] = (
            "q+addition"
        )

    report = influence.view(*case(change))
    row = connection(report)
    assert row["status"] == "equation_path"
    assert row["path"] == ["extra", "relay", "addition", "Gp", "output:Gp"]
    assert "question" not in next(
        c for c in report["components"] if c["name"] == "extra"
    )


def test_initialization_only_connection_does_not_claim_a_later_state_value_path():
    def change(trial):
        add_state(trial)
        trial["graph"]["rules"]["mem"] = {
            "initial": {
                "mode": "map",
                "expression": "root",
                "parameters": [],
                "state_initials": {"root": "extra"},
                "public_initials": {},
            }
        }

    report = influence.view(*case(change))
    row = connection(report)
    assert row["status"] == "initialization_only_path"
    assert row["path"] == [
        "initial:extra",
        "initial:mem",
        "mem",
        "q",
        "Gp",
        "output:Gp",
    ]
    assert "not later state values" in report["interpretation"]


def test_zero_rhs_does_not_create_a_reference_and_zero_multiplier_is_not_overproved():
    def change(trial):
        add_state(trial, rhs="0")
        trial["base_candidate"]["processes"].append(
            {"name": "addition", "expression": "0*extra"}
        )
        next(
            row
            for row in trial["base_candidate"]["state_equations"]
            if row["state"] == "Gp"
        )["rhs"] = "q+addition"

    report = influence.view(*case(change))
    assert connection(report)["status"] == "equation_path"
    assert "not proof of nonzero influence" in report["interpretation"]


@pytest.mark.parametrize(
    "bad", ["unknown_symbol", "__import__('os')", "I[0]", "I." + "x"]
)
def test_unknown_or_unsupported_expression_defers_all_disconnection_claims(bad):
    parent, selected = case(lambda trial: add_state(trial, rhs=bad))
    report = influence.view(parent, selected)
    assert report["status"] == "unknown"
    assert report["components"] == []


def test_unresolved_readout_and_stale_static_receipt_defer():
    parent, selected = case(add_state)
    selected["problem"]["base_candidate"]["observation_mappings"][0]["expression"] = (
        "absent"
    )
    assert influence.view(parent, selected)["status"] == "unknown"
    selected["audit"]["child_problem_sha256"] = content_hash(selected["problem"])
    assert influence.view(parent, selected)["status"] == "unknown"


def test_unsupported_initializer_reference_or_cycle_defers():
    def change(trial):
        add_state(trial)
        trial["graph"]["rules"]["extra"] = {
            "initial": {
                "mode": "map",
                "expression": "root",
                "parameters": [],
                "state_initials": {"root": "extra"},
                "public_initials": {},
            }
        }

    report = influence.view(*case(change))
    assert report["status"] == "unknown"
    assert "cyclic" in report["reason"]


def test_parent_identity_failure_is_not_hidden_by_a_well_formed_trial():
    parent, selected = case(add_state)
    parent["problem"]["base_candidate"]["state_equations"][0]["state"] = "missing"
    selected["audit"]["parent_problem_sha256"] = content_hash(parent["problem"])
    assert influence.view(parent, selected)["status"] == "unknown"


def test_nonmathematical_failure_never_gets_dependency_advice():
    parent, selected = case(add_state)
    selected["diagnostic"]["code"] = "DECLARATION_IDENTITY"
    assert influence.view(parent, selected)["status"] == "unknown"
    assert influence.view(parent, selected)["components"] == []


def test_display_limits_preserve_omission_counts_and_one_question_per_component():
    def change(trial):
        for number in range(12):
            trial["base_candidate"]["processes"].append(
                {"name": f"extra{number}", "expression": "1"}
            )

    parent, selected = case(change)
    selected["diagnostic"]["details"]["assessment"]["conflicts"].append({"target": "I"})
    report = influence.view(parent, selected)
    assert len(report["components"]) == influence.MAX_DISPLAY_COMPONENTS
    assert report["omitted_components"] == 4
    for component in report["components"]:
        assert "Gp, I" in component["question"]
        assert all("question" not in row for row in component["target_connections"])
