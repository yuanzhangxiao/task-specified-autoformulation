"""At topology time a proposer can introduce a fully specified latent variable."""

import json
from copy import deepcopy

import pytest

from autoformalism.research import preproposal_graph_repair as repair
from autoformalism.research import preproposal_topology as topology
from tests.test_preproposal_topology import case_and_choice


def extended_graph(kind):
    case, choice, frozen, value = case_and_choice()
    frozen["allow_topology_latents"] = True
    value["sources"]["level"] = ["store", "capture"]
    value["new_latents"] = [{"name": "capture", "kind": kind, "sources": ["rain"]}]
    if kind == "differential":
        value["latent_initializers"].append(
            {
                "state": "capture",
                "rule": {"mode": "value", "scope": "shared_training"},
                "justification": "Common stored quantity at the start.",
            }
        )
    return case, choice, frozen, value


@pytest.mark.parametrize("kind", ["differential", "algebraic"])
def test_new_latent_completes_selected_or_open_graph(kind):
    case, choice, frozen, value = extended_graph(kind)
    for selected in (choice, None):
        checked = topology.assess_topology(case, selected, value, frozen)
        assert checked["status"] == repair.PASS
        assert set(checked["sources"]["capture"]) == {"rain"}
        assert "history:rain" in checked["training_checks"]["level"]["actual_options"]
    assert (
        repair.assess_and_repair(case, value, frozen)["repair_status"]
        == "already_passed"
    )
    frozen.pop("allow_topology_latents")
    assert (
        topology.assess_topology(case, None, value, frozen)["status"]
        == "invalid_topology"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "undeclared_rhs",
        "no_sources",
        "no_type",
        "missing_initializer",
        "duplicate",
        "owned_channel",
        "observation_label",
        "algebraic_cycle",
        "trajectory_initial",
        "initializer_cycle",
        "common_preparation",
        "unused",
    ],
)
def test_incomplete_or_ambiguous_extension_rejected(failure):
    case, _, frozen, value = extended_graph("differential")
    extra = value["new_latents"][0]
    if failure == "undeclared_rhs":
        value["sources"]["level"].append("hypothetical")
    elif failure == "no_sources":
        extra.pop("sources")
    elif failure == "no_type":
        extra.pop("kind")
    elif failure == "missing_initializer":
        value["latent_initializers"].pop()
    elif failure == "duplicate":
        value["new_latents"].append(extra.copy())
    elif failure == "owned_channel":
        extra["name"] = "rain"
    elif failure == "observation_label":
        extra["name"] = "history_rain"
    elif failure == "algebraic_cycle":
        extra.update(kind="algebraic", sources=["capture"])
        value["latent_initializers"].pop()
    elif failure == "trajectory_initial":
        value["latent_initializers"][-1]["rule"]["scope"] = "trajectory"
    elif failure == "initializer_cycle":
        value["latent_initializers"][-1]["rule"] = {
            "mode": "map",
            "state_initials": ["capture"],
        }
    elif failure == "common_preparation":
        frozen["shared_hidden_initials"] = True
    elif failure == "unused":
        value["sources"]["level"] = ["store", "rain"]
    assert (
        topology.assess_topology(case, None, value, frozen)["status"]
        == "invalid_topology"
    )


def test_repair_enumerates_wiring_and_initials_for_new_states():
    case, _, frozen, value = extended_graph("differential")
    value["new_latents"][0]["sources"] = []
    catalogue = repair._catalogue(case, frozen, value)
    edits = (
        repair.Edit("add_source", "capture", "rain"),
        repair.Edit("add_public_initial", "capture", "upstream"),
    )
    assert set(edits) <= set(catalogue)
    assert repair.Edit("add_source", "store", "capture") in catalogue
    child = repair._apply(value, edits)
    assert value["new_latents"][0]["sources"] == []
    assert child["new_latents"][0]["sources"] == ["rain"]
    assert topology.assess_topology(case, None, child, frozen)["status"] == repair.PASS
    assert any(
        row["graph"]["new_latents"][0]["sources"] == ["rain"]
        for row in repair.assess_and_repair(case, value, frozen)["repairs"]
    )


def test_new_graph_instruction_names_observed_and_owned_fields():
    case, choice, frozen, _ = extended_graph("differential")
    frozen["required_histories"] = []
    frozen["inventory_roles"] = {"store": "holds earlier inflow"}
    frozen["mechanism_bindings"] = [
        {"memory_states": ["store"], "requirement_id": "upstream_memory"}
    ]
    for system, user in [
        topology.topology_messages(case, choice, frozen),
        repair.graph_first_messages(case, frozen),
    ]:
        packet = json.loads(user)
        assert packet["inventory_roles"] == frozen["inventory_roles"]
        assert packet["assigned_memory_states"] == frozen["mechanism_bindings"]
        assert set(packet["available_rhs_names"]) == {
            "level",
            "upstream",
            "store",
            "rain",
            "spill",
            "t",
        }
        assert packet["initializer_owners"] == ["store"]
        assert "new_latents" in system
    assert (
        "new_latents"
        in topology.reply_schema(case, frozen).model_json_schema()["properties"]
    )


def test_merge_only_equivalent_inventory_rows_and_keep_genuine_new_state():
    case, choice, frozen, graph = extended_graph("algebraic")
    expected = topology.assess_topology(case, choice, graph, frozen)
    graph["new_latents"].append(
        {"name": "level", "kind": "differential", "sources": ["capture", "store"]}
    )
    original = deepcopy(graph)
    checked = topology.assess_topology(case, choice, graph, frozen)
    assert checked["status"] == repair.PASS
    assert checked["merged_inventory_redeclarations"] == ["level"]
    assert checked["structure_sha256"] == expected["structure_sha256"]
    assert "capture" in checked["sources"]
    assert graph == original


@pytest.mark.parametrize(
    ("kind", "rhs"),
    [
        ("algebraic", ["store", "capture"]),
        ("differential", ["rain"]),
        ("differential", ["store", "store", "capture"]),
    ],
)
def test_conflicting_inventory_row_reports_both_definitions_without_patch_search(
    kind, rhs
):
    case, choice, frozen, graph = extended_graph("algebraic")
    graph["new_latents"].append({"name": "level", "kind": kind, "sources": rhs})
    checked = topology.assess_topology(case, choice, graph, frozen)
    assert checked["status"] == "invalid_topology"
    assert (
        "level: inventory differential with sources ['store', 'capture']"
        in checked["reason"]
    )
    assert f"new_latents {kind} with sources {rhs}" in checked["reason"]
    result = repair.assess_and_repair(case, graph, frozen)
    assert (result["repair_status"], result["checked"]) == (
        "outside_additive_repair_scope",
        0,
    )
