"""Graph-first edits are algorithmic, bounded, and verified on whole graphs."""

from copy import deepcopy

from autoformalism.research import preproposal_graph_repair as repair
from autoformalism.research.preproposal_topology import assess_topology
from tests.test_preproposal_topology import case_and_choice


def test_recheck_every_minimal_edit_and_preserve_parent():
    case, _, frozen, graph = case_and_choice()
    graph["sources"]["level"] = ["rain"]
    graph["latent_initializers"][0]["rule"] = {
        "mode": "value",
        "scope": "shared_training",
    }
    original = deepcopy(graph)
    result = repair.assess_and_repair(case, graph, frozen)
    assert graph == original
    assert result["public_graph_check"] == "failed"  # store has no path to an output
    assert result["repair_status"] == "repaired"
    assert not result["budget_exhausted"]
    patches = []
    for row in result["repairs"]:
        child = assess_topology(case, None, row["graph"], frozen)
        assert (child["status"], child["structure_sha256"]) == (
            repair.PASS,
            row["structure_sha256"],
        )
        patches.append(frozenset(repair.Edit(**edit) for edit in row["edits"]))
    assert not any(p < other for p in patches for other in patches)
    assert (
        frozenset(
            (
                repair.Edit("add_source", "level", "store"),
                repair.Edit("add_public_initial", "store", "upstream"),
            )
        )
        in patches
    )


def test_common_hidden_preparation_and_check_budget():
    case, _, frozen, graph = case_and_choice()
    frozen["shared_hidden_initials"] = True
    graph["latent_initializers"][0]["rule"] = {
        "mode": "value",
        "scope": "shared_training",
    }
    result = repair.assess_and_repair(case, graph, frozen, max_checks=1)
    assert result["checked"] == 1 and result["budget_exhausted"]
    assert all(e.kind == "add_source" for e in repair._catalogue(case, frozen, graph))


def test_already_good_and_unparsable_graphs():
    case, _, frozen, graph = case_and_choice()
    accepted = repair.assess_and_repair(case, graph, frozen)
    assert accepted["status"] == repair.PASS and accepted["checked"] == 0
    invalid = repair.assess_and_repair(case, {}, frozen)
    assert (invalid["repair_status"], invalid["checked"]) == (
        "invalid_parent_schema",
        0,
    )


def test_apply_repairs_to_canonical_graph_while_preserving_original_parent():
    case, _, frozen, graph = case_and_choice()
    frozen["allow_topology_latents"] = True
    graph["sources"]["level"] = ["rain"]  # store has no path to a target
    original_run = repair.assess_and_repair(case, graph, frozen)
    graph["new_latents"] = [
        {"name": "level", "kind": "differential", "sources": ["rain"]}
    ]
    saved = deepcopy(graph)
    repeated = repair.assess_and_repair(case, graph, frozen)
    assert repeated["parent"] == graph == saved
    assert repeated["merged_inventory_redeclarations"] == ["level"]
    assert repeated["repair_status"] == "repaired"
    assert repeated["repairs"] == original_run["repairs"]
    assert all(c["graph"]["new_latents"] == [] for c in repeated["repairs"])
    assert any("store" in c["graph"]["sources"]["level"] for c in repeated["repairs"])


def test_permanently_undeclared_source_skips_additive_search():
    case, _, frozen, graph = case_and_choice()
    graph["sources"]["level"].append("missing_channel")
    result = repair.assess_and_repair(case, graph, frozen)
    assert "missing_channel" in result["reason"]
    assert result["repair_status"] == "outside_additive_repair_scope"
    assert result["checked"] == 0 and result["repairs"] == []
