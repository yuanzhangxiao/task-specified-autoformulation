"""Exact affine lower bounds and constructive original-readout replay."""

from copy import deepcopy

import pytest

from scripts import t2_initial_disposal_family as family


@pytest.fixture
def example(monkeypatch):
    graph = {
        "variable_types": {
            "I": "differential",
            "Gp": "differential",
            "I_act": "differential",
            "U": "algebraic",
        },
        "initializers": {
            "I": {"mode": "observed_initial", "channel": "I"},
            "Gp": {"mode": "observed_initial", "channel": "Gp"},
            "I_act": {"mode": "map", "state_initials": ["I"]},
        },
        "sources": {
            "I": [],
            "Gp": [],
            "I_act": ["I"],
            "U": ["I_act", "Gt", "Gp", "Uii"],
        },
    }
    formula = {
        "equations": {
            "I": "0",
            "Gp": "0",
            "I_act": "afcoef10*I",
            "U": "afcoef11+afcoef12*Gp+afcoef13*I_act+afcoef14*(Uii-1)"
            "+disp_tissue*((Gt-100)/100)",
        },
        "initializers": {"I_act": "afcoef15+afcoef16*state0_I"},
        "parameters": {
            **{f"afcoef{i}": 1.0 for i in range(10, 17)},
            "disp_tissue": 1.0,
        },
    }
    monkeypatch.setattr(
        family,
        "GRAPHS",
        {family.content_hash([graph, formula["equations"], formula["initializers"]])},
    )
    rows = [
        {
            "trajectory_id": f"train_{i:03}",
            "time": [0, 1],
            "covariates": {},
            "inputs": {},
            "targets": {"Gp": [101 - i] * 2, "I": [10] * 2, "U": [u, 5]},
            "auxiliaries": {"Gt": [99 + i] * 2, "Uii": [1] * 2},
        }
        for i, u in enumerate([2, 2.5, 2])
    ]
    return {"formula": formula, "graph": graph}, rows


def test_convex_combination_excludes_and_constructive_replay_attains(example):
    model, rows = example
    original = deepcopy(model)
    measured = family.assess_family(model, rows)
    assert measured["status"] == "no_absolute_tolerance_declared"
    assert measured["lower_bound"] == measured["upper_bound"] == 0.25
    assert measured["lower_bound_witness"]["middle_event"] == "train_001"
    assert measured["lower_bound_witness"]["weight_on_second_endpoint"] == "1/2"
    assert list(measured["witness_predictions"].values()) == [2.25] * 3
    assert family.assess_family(model, rows, tolerance=0.25)["status"] == (
        "initial_tolerance_attainable"
    )
    assert family.assess_family(model, rows, tolerance=0.24)["status"] == (
        "family_conflict_at_initial_time"
    )
    assert model == original


def test_lower_bound_requires_all_affine_inputs_to_interpolate(example):
    model, rows = example
    rows[1]["targets"]["I"][0] += 1
    outcome = family.assess_family(model, rows, tolerance=0.24)
    assert outcome["status"] == "unresolved"
    assert outcome["lower_bound"] == 0


def test_heldout_changed_models_and_invalid_tolerances_refused(example):
    model, rows = example
    with pytest.raises(ValueError, match="tolerance"):
        family.assess_family(model, rows, tolerance=float("nan"))
    with pytest.raises(ValueError, match="unique train"):
        family.assess_family(model, [rows[0] | {"trajectory_id": "val_000"}])
    model["formula"]["equations"]["U"] += "+1"
    with pytest.raises(ValueError, match="qualified"):
        family.assess_family(model, rows)


def test_equal_initial_features_and_initial_physical_domain(example):
    model, rows = example
    repeated = deepcopy(rows[0]) | {"trajectory_id": "train_003"}
    repeated["targets"]["U"][0] = 3
    result = family.assess_family(model, [rows[0], repeated])
    assert result["lower_bound"] == result["upper_bound"] == 0.5
    assert result["lower_bound_witness"]["same_initial_features"] == [
        "train_000",
        "train_003",
    ]
    repeated["targets"]["U"][0] = 0.25
    result = family.assess_family(model, [repeated], tolerance=0.1)
    assert result["witness_max_initial_error"] == 0
    assert result["upper_bound"] is None
    assert result["status"] == "unresolved"
