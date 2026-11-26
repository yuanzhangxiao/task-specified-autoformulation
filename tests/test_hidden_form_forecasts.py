"""Matched form construction, held-event prediction and fit accounting."""

import numpy as np
import pytest

from scripts import run_hidden_form_forecasts as forecast

GRAPH = {
    "variable_types": {"y": "differential", "z": "differential"},
    "sources": {"y": ["u", "z"], "z": ["u"]},
    "initializers": {
        "y": {"mode": "observed_initial", "channel": "y"},
        "z": {"mode": "value", "scope": "shared_training"},
    },
}
FORM = {
    "equations": {"y": "b+k*((u-(2))/0.5)+h*z", "z": "q*u"},
    "initializers": {"z": "s"},
    "parameters": {"b": 1.0, "k": 1.0, "q": 0.0, "s": 2.0, "h": 0.0},
}


def event(i, u):
    return {
        "time": [0.0, 1.0, 2.0],
        "trajectory_id": f"train_{i:03}",
        "targets": {"y": [2.0, 3.0, 4.0]},
        "inputs": {"u": [u] * 3},
        "auxiliaries": {},
        "covariates": {},
    }


def test_immutable_graph_starts_and_source_basis():
    forms = forecast.competing_forms(GRAPH, FORM, "y", "u")
    affine, square = forms.values()
    assert affine == FORM
    assert square["equations"]["z"] == affine["equations"]["z"]
    assert square["initializers"] == affine["initializers"]
    starts = forecast.starting_vectors(GRAPH, affine, ["y"])
    assert starts == forecast.starting_vectors(GRAPH, square, ["y"])
    assert starts["constant_observed"] == {
        "b": 0.0,
        "k": 0.0,
        "q": 0.0,
        "s": 2.0,
        "h": 0.0,
    }
    for u in (2.0, 3.0):
        fit = {"parameters": FORM["parameters"], "scales": {"y": 1.0}}
        a = forecast._forecast(GRAPH, affine, fit, event(10, u), ["y"])
        b = forecast._forecast(GRAPH, square, fit, event(10, u), ["y"])
        np.testing.assert_allclose(
            a["predicted"]["y"], 2 + np.arange(3) * (1 + (u - 2) / 0.5)
        )
        np.testing.assert_allclose(
            b["predicted"]["y"], 2 + np.arange(3) * (1 + (u**2 - 4) / 2.0)
        )
        if u == 2.0:
            assert a["nmse"] == pytest.approx(0) and b["nmse"] == pytest.approx(0)
        else:
            # Post-initial error, on the two forecast samples only.
            assert a["nmse"] == pytest.approx((2.0**2 + 4.0**2) / 2)
    with pytest.raises(ValueError, match="single normalized"):
        forecast.competing_forms(GRAPH, FORM, "y", "z")


def test_rotations_and_private_split_rejection():
    ids = [f"train_{i:03}" for i in range(16)]
    train, other = ids[::2], ids[1::2]
    a, b = forecast.rotations(train, other)
    assert a["selection_ids"] == b["held_ids"]
    assert a["held_ids"] == b["selection_ids"]
    assert not set(a["selection_ids"]) & set(a["held_ids"] + train)
    with pytest.raises(ValueError, match="distinct"):
        forecast.rotations(other, other)
    with pytest.raises(ValueError, match="training"):
        forecast.rotations(train, [*other[:-1], "val_000"])


def test_failed_held_bound_keeps_diagnostic_and_no_primary_score():
    fit = {"parameters": FORM["parameters"], "scales": {"y": 1.0}}
    curve = forecast._forecast(GRAPH, FORM, fit, event(10, 0.0), ["y"])
    assert curve["nmse"] is None and curve["status"] == "negative_forecast"
    assert curve["diagnostic_nmse"] > 0 and curve["negative_samples"]["y"] > 0


def test_forecast_is_independent_of_held_observations_after_initial_time():
    fit = {"parameters": FORM["parameters"], "scales": {"y": 1.0}}
    row = event(8, 2.0)
    before = forecast._forecast(GRAPH, FORM, fit, row, ["y"])
    row["targets"]["y"][1:] = [10.0, 20.0]
    after = forecast._forecast(GRAPH, FORM, fit, row, ["y"])
    assert before["predicted"] == after["predicted"]
    assert after["nmse"] > before["nmse"]


def test_resume_does_not_repeat_fit(tmp_path, monkeypatch):
    calls = []

    def fit(graph, form, rows, *, targets, **budget):
        calls.append(budget)
        return {"status": "fitted", "evaluations": 1, "budget": budget}

    monkeypatch.setattr(forecast.bounded, "fit_nonnegative", fit)
    args = tmp_path, "plan", "fold_local", GRAPH, FORM, [event(0, 1.0)], ["y"]
    assert forecast._train_start(*args) == forecast._train_start(*args)
    assert calls == [forecast.BUDGET]
    with pytest.raises(ValueError, match="Different fit"):
        forecast._train_start(tmp_path, "other", *args[2:])
