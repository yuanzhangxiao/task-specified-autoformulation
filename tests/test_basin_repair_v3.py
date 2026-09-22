"""Parameter equality is not a dynamical algebraic loop."""

from copy import deepcopy

import pytest

from autoformalism.search import basin_model_repair_v3 as repair
from tests.test_basin_model_repair import bundle
from tests.test_basin_repair_v2 import patch


def test_cycle_retains_one_parameter_and_propagates_all_uses():
    parent = bundle()
    raw = patch(
        parameter_bindings=[
            {"parameter": "k", "same_as": "d"},
            {"parameter": "d", "same_as": "k"},
        ]
    )
    tx = repair.apply(parent, raw, "coupled")
    assert [p["name"] for p in tx["bundle"]["candidate"]["parameters"]] == ["d"]
    assert tx["parameter_replacements"] == {"k": "d"}
    assert tx["binding_normalizations"]["equality_cycles"] == [
        {"members": ["d", "k"], "representative": "d"}
    ]
    assert repair.apply(tx["bundle"], raw, "coupled")["changed"] is False
    assert tx["reply"] == raw


def test_empty_binding_does_not_create_fix_or_free_parameter():
    parent = bundle()
    tx = repair.apply(parent, patch(parameter_bindings=[{"parameter": "k"}]), "coupled")
    assert not tx["changed"]
    assert tx["binding_normalizations"]["empty_bindings_removed"] == ["k"]
    with pytest.raises(ValueError, match="unavailable"):
        repair.apply(
            parent, patch(parameter_bindings=[{"parameter": "missing"}]), "coupled"
        )


def test_new_declaration_with_empty_entry_and_self_identity():
    raw = patch(
        new_parameters=[{"name": "new_rate", "role": "rate"}],
        equations=[
            {"component": "o", "expression": "new_rate*max(0,h_down-crest_down)"}
        ],
        parameter_bindings=[
            {"parameter": "new_rate", "value": None, "same_as": None},
            {"parameter": "k", "same_as": "k"},
        ],
    )
    tx = repair.apply(bundle(), raw, "coupled")
    assert "new_rate" in {p["name"] for p in tx["bundle"]["candidate"]["parameters"]}
    assert tx["binding_normalizations"]["empty_bindings_removed"] == ["new_rate"]


def test_incompatible_cycle_and_conflicting_operations_remain_atomic():
    parent = bundle()
    before = deepcopy(parent)
    for raw in [
        patch(
            new_parameters=[{"name": "positive", "role": "rate"}],
            parameter_bindings=[
                {"parameter": "k", "same_as": "positive"},
                {"parameter": "positive", "same_as": "k"},
            ],
        ),
        patch(parameter_bindings=[{"parameter": "k", "same_as": "d", "value": 1}]),
        patch(parameter_bindings=[{"parameter": "k", "same_as": "area_up"}]),
        patch(parameter_bindings=[{"parameter": "k"}, {"parameter": "k", "value": 1}]),
    ]:
        with pytest.raises(ValueError):
            repair.apply(parent, raw, "coupled")
        assert parent == before


def test_acyclic_binding_keeps_declared_direction():
    tx = repair.apply(
        bundle(),
        patch(parameter_bindings=[{"parameter": "d", "same_as": "k"}]),
        "coupled",
    )
    assert tx["parameter_replacements"] == {"d": "k"}
    assert tx["binding_normalizations"]["equality_cycles"] == []
