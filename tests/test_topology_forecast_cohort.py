"""The topology-to-forecast comparison matches inventories before advancing graphs."""

import pytest

from scripts.topology_forecast_cohort import ARMS, match_inventory


def entry(digest, variable_types, origin):
    return {
        "graph_id": digest,
        "graph": {"variable_types": variable_types, "sources": {}, "initializers": {}},
        "origin": origin,
    }


def test_equal_sampling_requires_identical_generated_names_and_types():
    kinds = {"x": "differential"}
    pools = {
        ARMS[0]: [entry(str(i), kinds, "algorithm") for i in range(10)],
        ARMS[1]: [
            entry("a", kinds, "seed2"),
            entry("a", kinds, "seed1"),
            entry("b", kinds, "seed3"),
            entry("extra", {**kinds, "z": "algebraic"}, "seed4"),
            entry("changed", {"x": "algebraic"}, "seed5"),
        ],
    }
    selected, receipt = match_inventory("inventory", kinds, pools, 8)
    assert receipt["selected_per_arm"] == 2
    assert receipt["arms"][ARMS[1]] == {
        "admitted_occurrences": 5,
        "distinct_admitted": 4,
        "matching_inventory": 2,
        "different_inventory": ["changed", "extra"],
    }
    assert next(r for r in selected if r["graph_id"] == "a")["origins"] == [
        "seed1",
        "seed2",
    ]
    assert all(r["graph"]["variable_types"] == kinds for r in selected)
    assert (selected, receipt) == match_inventory(
        "inventory", kinds, {k: list(reversed(v)) for k, v in pools.items()}, 8
    )


def test_no_matched_controls_never_fills_from_another_inventory():
    kinds = {"x": "algebraic"}
    pools = {ARMS[0]: [entry("x", kinds, "algorithm")], ARMS[1]: []}
    selected, receipt = match_inventory("other_inventory", kinds, pools, 8)
    assert selected == []
    assert receipt["selected_per_arm"] == 0
    assert receipt["arms"][ARMS[0]]["matching_inventory"] == 1


def test_conflicting_digests_are_rejected():
    pools = {
        ARMS[0]: [entry("same", {"x": "differential"}, "a"), entry("same", {}, "b")],
        ARMS[1]: [],
    }
    with pytest.raises(ValueError, match="conflicting structure"):
        match_inventory("inventory", {}, pools, 1)
