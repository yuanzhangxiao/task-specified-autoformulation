"""Selection and data checks use realized graph roots and forecast-time ownership."""

import json

import pytest

from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.research.preproposal_topology import (
    assess_topology,
    topology_messages,
    topology_schema,
)
from autoformalism.search.preproposal_screen import screen_target
from tests.test_preproposal_screen import complementary
from tests.test_topology_repair_generator import evidence


def case_and_choice():
    packet = evidence(complementary())
    screen = screen_target(
        packet, "level", histories=("rain", "spill"), initials=("upstream",)
    )
    brief = {
        "scientific_context": "Explain reservoir levels under supplied inputs.",
        "public_variables": [
            {"name": n, "data_role": "target"} for n in packet.context.targets
        ]
        + [
            {"name": n, "data_role": "external_input"}
            for n in packet.context.external_inputs
        ],
        "requirements": [],
    }
    case = SelectionCase.model_validate(
        {
            "case_id": "paired",
            "target": "level",
            "inventory": [
                {"name": n, "kind": "differential"}
                for n in ("level", "upstream", "store")
            ],
            "public_task": brief["scientific_context"],
            "evidence_sha256": packet.sha256,
            "required_options": screen.required_options,
            "evidence_summary": "Three divergent records.",
            "candidates": [
                {"candidate_id": c.id, "sources": c.options} for c in screen.candidates
            ],
        }
    )
    chosen = next(c for c in case.candidates if "initial:upstream" in c.sources)
    choice = {
        "candidate_id": chosen.candidate_id,
        "rationale": "Store carries prior level.",
    }
    frozen = {
        "case_id": case.case_id,
        "target": case.target,
        "evidence": packet.model_dump(mode="json"),
        "public_brief": brief,
        "initials": ["upstream"],
        "histories": ["rain", "spill"],
    }
    value = {
        "hypothesis": "A latent store carries the initial upstream reading.",
        "sources": {
            "level": ["store", "rain"],
            "upstream": ["spill"],
            "store": ["store"],
        },
        "latent_initializers": [
            {
                "state": "store",
                "rule": {"mode": "map", "public_initials": ["upstream"]},
                "justification": "At baseline the store depends on upstream level.",
            }
        ],
    }
    return case, choice, frozen, value


def test_conditional_pass_depends_on_both_realized_routes():
    case, choice, frozen, value = case_and_choice()
    result = assess_topology(case, choice, value, frozen)
    assert result["status"] == "topology_screen_passed"
    assert result["training_checks"]["level"]["actual_options"] == [
        "history:rain",
        "initial:level",
        "initial:upstream",
    ]
    assert set(result["training_checks"]) == {"level", "upstream"}
    value["latent_initializers"][0]["rule"] = {
        "mode": "value",
        "scope": "shared_training",
    }
    result = assess_topology(case, choice, value, frozen)
    assert result["status"] == "training_conflict"
    assert result["training_checks"]["level"]["conflicting_pairs"] == 1
    assert result["missing_selected_or_required"] == ["initial:upstream"]
    value["sources"]["level"].append("spill")
    result = assess_topology(case, choice, value, frozen)
    assert result["status"] == "selected_route_missing"
    assert result["training_checks"]["level"]["conflicting_pairs"] == 0


@pytest.mark.parametrize(
    "field", ["forbidden_source", "missing_state", "free_initial", "algebraic_cycle"]
)
def test_inadmissible_draft_is_rejected(field):
    case, choice, frozen, value = case_and_choice()
    if field == "forbidden_source":
        frozen["histories"].remove("spill")
    elif field == "missing_state":
        value["sources"].pop("level")
    elif field == "free_initial":
        value["latent_initializers"][0]["rule"] = {
            "mode": "value",
            "scope": "trajectory",
        }
    else:
        inv = [v.model_dump() for v in case.inventory]
        inv[-1]["kind"] = "algebraic"
        case = SelectionCase.model_validate(case.model_dump() | {"inventory": inv})
        value["latent_initializers"] = []
    assert assess_topology(case, choice, value, frozen)["status"] == "invalid_topology"


def test_model_receives_inventory_and_public_evidence_summary_only():
    case, choice, frozen, value = case_and_choice()
    schema = topology_schema(case)
    assert (
        schema.model_validate(value).model_dump(mode="json", by_alias=True)["sources"]
        == value["sources"]
    )
    _, user = topology_messages(case, choice, frozen)
    content = json.loads(user)
    assert content["target"] == "level"
    assert set(content["selected_information"]) == {"history:rain", "initial:upstream"}
    assert "trajectories" not in user and "source_path" not in user


def test_selected_target_margin_preserves_exact_comparison():
    case, choice, frozen, value = case_and_choice()
    frozen["minimum_target_gap"] = 5.0
    value["latent_initializers"][0]["rule"] = {
        "mode": "value",
        "scope": "shared_training",
    }
    checked = assess_topology(case, choice, value, frozen)
    assert checked["training_checks"]["level"]["conflicting_pairs"] == 0
    assert checked["exact_selected_target_check"]["conflicting_pairs"] == 1
    assert checked["training_checks"]["upstream"]["minimum_target_gap"] == 0


def test_permitted_initials_and_explicit_record_dependency():
    case, choice, frozen, value = case_and_choice()
    frozen["initials"] = []
    # Training observes upstream. This forecast cannot use its baseline.
    assert assess_topology(case, choice, value, frozen)["status"] == "invalid_topology"
    value["latent_initializers"] = [
        {
            "state": n,
            "rule": {"mode": "value", "scope": "shared_training"},
            "justification": "A common initial quantity.",
        }
        for n in ("store", "upstream")
    ]
    checks = assess_topology(case, choice, value, frozen)["training_checks"]
    assert "initial:upstream" not in checks["level"]["actual_options"]
    assert checks["upstream"]["status"] == "unresolved"


def test_structure_identity_ignores_source_order_and_rationale():
    case, choice, frozen, value = case_and_choice()
    first = assess_topology(case, choice, value, frozen)
    value["hypothesis"] = "Another explanation."
    value["sources"]["level"].reverse()
    value["latent_initializers"][0]["justification"] = "Restated."
    second = assess_topology(case, choice, value, frozen)
    assert first["structure_sha256"] == second["structure_sha256"]
    value["sources"]["level"].append("spill")
    assert (
        first["structure_sha256"]
        != assess_topology(case, choice, value, frozen)["structure_sha256"]
    )
