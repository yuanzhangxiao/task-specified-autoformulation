"""A single observed initial Gt augments a shared latent initializer."""

from copy import deepcopy

import pytest

from autoformalism.research.trajectory_function_fitting import initial_target_events
from scripts import probe_t2_initial_tissue_map as probe


def inputs():
    graph = {
        "sources": {"I": ["meal_rate_g_per_min"], "I_act": ["I"], "U": ["I_act"]},
        "variable_types": {
            "I": "differential",
            "I_act": "differential",
            "U": "algebraic",
        },
        "initializers": {
            "I": {"mode": "observed_initial", "channel": "I"},
            "I_act": {"mode": "map", "state_initials": ["I"], "public_initials": []},
        },
    }
    model = {
        "graph": graph,
        "formula": {
            "equations": {"I": "meal_rate_g_per_min", "I_act": "I", "U": "I_act"},
            "initializers": {"I_act": "afcoef15+afcoef16*state0_I"},
            "parameters": {"afcoef15": 1.0, "afcoef16": 2.0},
        },
    }
    rows = [
        {
            "trajectory_id": f"train_{i:03}",
            "time": [0, 1],
            "inputs": {"meal_rate_g_per_min": [0, 0]},
            "auxiliaries": {"Gt": [100.0 + i, 400.0]},
            "targets": {"I": [2.0, 2.0], "U": [5.0, 7.0], "Gp": [200.0 - i, 222.0]},
            "covariates": {},
        }
        for i in (0, 1)
    ]
    return model, rows


def test_only_observed_time_zero_changes_action_and_zero_recovers_parent():
    model, rows = inputs()
    original = deepcopy(model)
    child, evidence = probe.tissue_map(model, rows)
    assert model == original
    assert evidence["centered_Gp_equals_negative_centered_Gt"]
    assert evidence["reference_initial_Gt"] == 100.0

    def predictions(m):
        return [
            r["predicted"]["U"][0]
            for r in initial_target_events(m["graph"], m["formula"], rows, ["U"])
        ]

    assert predictions(child) == predictions(model) == [5.0, 5.0]
    child["formula"]["parameters"][probe.GAIN] = 50.0
    assert predictions(child) == [5.0, 5.5]
    rows[1]["auxiliaries"]["Gt"][1] = 0.0
    assert predictions(child) == [5.0, 5.5]
    with pytest.raises(ValueError, match="Only the registered"):
        probe.TissueTraining(model, child, rows)


def test_check_all_baselines_and_source_initializer():
    model, rows = inputs()
    rows[1]["targets"]["Gp"][0] += 1.0
    assert not probe.tissue_map(model, rows)[1][
        "centered_Gp_equals_negative_centered_Gt"
    ]
    model["formula"]["initializers"]["I_act"] = "afcoef15"
    with pytest.raises(ValueError, match="original"):
        probe.tissue_map(model, rows)
    with pytest.raises(ValueError, match="fasting"):
        probe.tissue_map(model, rows[1:])
    rows[1]["trajectory_id"] = "val_001"
    with pytest.raises(ValueError, match="training"):
        probe.tissue_map(model, rows)
