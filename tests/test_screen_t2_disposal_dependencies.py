"""Regression checks for exact matched readouts and unobserved-state histories."""

from copy import deepcopy

import pytest

from scripts.screen_t2_disposal_dependencies import LINEAR, screen_dependencies


def graph():
    return {
        "variable_types": {"I_act": "differential", "U": "algebraic"},
        "initializers": {"I_act": {"mode": "value", "scope": "shared_training"}},
        "sources": {"I_act": ["dose"], "U": ["Uii", "I_act"]},
    }


def event(index, dose, diff, gp=(1, 2, 3)):
    uii = [1, 2, 3]
    return {
        "trajectory_id": f"train_{index}",
        "time": [0, 1, 2],
        "inputs": {"dose": dose},
        "targets": {
            "I": [10, 11, 12],
            "U": [a + b for a, b in zip(uii, diff, strict=True)],
        },
        "auxiliaries": {"Gp": list(gp), "Gt": [4, 5, 6], "Uii": uii},
        "covariates": {},
    }


def test_shared_hidden_start_and_distinguishing_measurement():
    rows = [
        event(0, [0, 1, 0], [2, 2, 2]),
        event(1, [0, 4, 0], [4, 4, 4], gp=(7, 8, 9)),
    ]
    result = screen_dependencies(graph(), rows, interpolation={"dose": LINEAR})
    h = result["hidden_candidates"]
    assert not h["I_act"]["eligible"] and h["I_act"]["minimum_uniform_error"] == 1
    assert h["I_act+Gp"]["eligible"] and not h["I_act+Gt"]["eligible"]
    observed = result["measured_candidates"]
    assert not observed["I"]["eligible"] and observed["Gp"]["eligible"]


def test_identical_forcing_prefixes_only_with_known_interpolation():
    rows = [
        event(0, [0, 1, 0], [2, 2, 2]),
        event(1, [0, 1, 0], [2, 6, 6]),
    ]
    checked = screen_dependencies(graph(), rows, interpolation={"dose": LINEAR})
    h = checked["hidden_candidates"]["I_act"]
    assert h["initial"]["eligible"] and h["minimum_uniform_error"] == 2
    assert h["repeated_argument_groups"] == 3
    unknown = screen_dependencies(graph(), rows)["hidden_candidates"]["I_act"]
    assert unknown["eligible"] and unknown["repeated_argument_groups"] == 1
    rows[1]["inputs"]["dose"][1] = 2
    divergent = screen_dependencies(graph(), rows, interpolation={"dose": LINEAR})
    assert divergent["hidden_candidates"]["I_act"]["eligible"]


def test_observed_target_histories_are_not_treated_as_continuous_forcing():
    g = graph()
    g["sources"]["I_act"] = ["I"]
    g["sources"]["I"] = ["dose"]
    g["variable_types"]["I"] = "differential"
    g["initializers"]["I"] = {"mode": "observed_initial", "channel": "I"}
    rows = [
        event(0, [0, 1, 0], [2, 2, 2]),
        event(1, [0, 2, 0], [2, 4, 2]),
    ]
    report = screen_dependencies(g, rows, interpolation={"dose": LINEAR})
    assert not report["measured_candidates"]["I"]["eligible"]
    assert report["hidden_candidates"]["I_act"]["eligible"]


def test_independent_initials_and_split_contract():
    g = graph()
    g["initializers"]["I_act"]["scope"] = "trajectory"
    rows = [event(0, [0, 1, 0], [2, 2, 2]), event(1, [0, 1, 0], [5, 5, 5])]
    r = screen_dependencies(g, rows, interpolation={"dose": LINEAR})
    assert r["hidden_candidates"]["I_act"]["repeated_argument_groups"] == 0
    bad = deepcopy(rows)
    bad[0]["trajectory_id"] = "val_000"
    with pytest.raises(ValueError, match="train"):
        screen_dependencies(g, bad)


def test_adding_action_readout_resolves_newly_referenced_initial_state():
    g = graph()
    g["sources"]["U"] = ["Uii"]
    rows = [event(0, [0, 1, 0], [2, 2, 2]), event(1, [0, 1, 0], [4, 4, 4])]
    report = screen_dependencies(g, rows, interpolation={"dose": LINEAR})
    assert report["hidden_candidates"]["I_act"]["minimum_uniform_error"] == 1
