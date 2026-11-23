"""Complete graph selections reference the same preverified pool."""

import json
import random
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.research import preproposal_complete_graphs as graphs
from autoformalism.research import preproposal_topology as topology
from autoformalism.research.preproposal_observed_gate import check_bound_memories
from autoformalism.research.preproposal_topology import assess_topology
from tests.test_preproposal_topology import case_and_choice


def inputs():
    case, _, frozen, _ = case_and_choice()
    frozen.update(
        inventory=[v.model_dump() for v in case.inventory],
        check_all_measured=True,
        required_histories=[],
        inventory_roles={"store": "Temporary upstream storage."},
    )
    return case, frozen


def test_whole_graphs_verified_and_reproducible_with_matching_selections():
    case, frozen = inputs()
    before = deepcopy(frozen)
    pool = graphs.generate_pool(case, frozen, max_candidates=3, max_checks=64)
    assert frozen == before
    assert pool == graphs.generate_pool(case, frozen, max_candidates=3, max_checks=64)
    assert len(pool["candidates"]) == 3
    assert pool["stop_reason"] == "pool_filled"
    assert sum(pool["status_counts"].values()) == pool["checked"] <= 64
    assert sum(pool["skipped_counts"].values()) + pool["checked"] == pool["draws"]
    assert len({r["structure_sha256"] for r in pool["candidates"]}) == 3
    for row in pool["candidates"]:
        result = assess_topology(case, None, row["graph"], frozen)
        assert result["status"] == "topology_screen_passed"
        assert row["structure_sha256"] == result["structure_sha256"]
        assert set(row["graph"]["sources"]) == {v.name for v in case.inventory}

    random_rows = graphs.random_selection(pool, 2, 3)
    assert random_rows == graphs.random_selection(pool, 2, 3)
    ids = [row["candidate_id"] for row in random_rows]
    schema = graphs.graph_selection_schema(pool, 2)
    assert list(schema(graph_ids=ids).graph_ids) == ids
    assert graphs.resolve_selection(pool, ids, 2) == random_rows
    with pytest.raises(ValidationError):
        schema(graph_ids=["unknown", ids[0]])
    with pytest.raises(ValueError, match="distinct"):
        graphs.resolve_selection(pool, [ids[0], ids[0]], 2)
    _, message = graphs.graph_selection_messages(case, pool, 2)
    packet = json.loads(message)
    assert {row["graph_id"] for row in packet["graphs"]} == {
        row["candidate_id"] for row in pool["candidates"]
    }
    assert packet["inventory_roles"] == frozen["inventory_roles"]
    assert "trajectories" not in packet


def test_budget_empty_pool_and_identity_checks():
    case, frozen = inputs()
    with pytest.raises(ValueError, match="full measured"):
        graphs.generate_pool(case, dict(frozen, check_all_measured=False))
    with pytest.raises(ValueError, match="limits"):
        graphs.generate_pool(case, frozen, max_checks=0)
    # No generated equation owns "spill": this public obligation yields no pool.
    frozen["public_brief"]["requirements"] = [
        {
            "id": "out",
            "drivers": ["rain"],
            "targets": ["spill"],
            "public_requirement": "Rain must affect spill.",
        }
    ]
    pool = graphs.generate_pool(case, frozen, max_checks=2)
    assert pool["stop_reason"] == "check_limit"
    assert pool["checked"] == 2 and not pool["candidates"]
    assert graphs.random_selection(pool, 0, 5) == []
    with pytest.raises(ValueError, match="empty"):
        graphs.graph_selection_schema(pool, 0)
    pool["checked"] += 1
    with pytest.raises(ValueError, match="identity"):
        graphs.resolve_selection(pool, [], 0)


def test_common_hidden_preparation_does_not_create_initial_maps():
    case, frozen = inputs()
    frozen["shared_hidden_initials"] = True
    pool = graphs.generate_pool(case, frozen, max_candidates=3)
    assert pool["candidates"]
    for row in pool["candidates"]:
        assert all(
            r["rule"] == {"mode": "value", "scope": "shared_training"}
            for r in row["graph"]["latent_initializers"]
        )


def test_constructed_route_carries_the_driver_through_the_assigned_state():
    case, frozen = inputs()
    frozen["public_brief"]["requirements"] = [
        {
            "id": "storage",
            "drivers": ["upstream"],
            "targets": ["level"],
            "public_requirement": "Upstream conditions affect level via stored water.",
            "requires_dynamic_memory": True,
        }
    ]
    bindings = [{"requirement_id": "storage", "memory_states": ["store"]}]
    frozen["mechanism_bindings"] = bindings
    evidence, brief = topology._inputs(case, frozen)
    states = topology._state_names(case)
    observed, permitted = topology._initial_channels(states, evidence, frozen)
    for seed in range(24):
        graph = graphs._graph_proposal(
            case,
            frozen,
            {"variables": {}},
            states,
            observed,
            permitted,
            brief,
            0,
            random.Random(seed),
            2,
        )
        check_bound_memories(graph["sources"], states, brief, bindings)
