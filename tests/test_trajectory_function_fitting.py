"""Exercise complete public equations, causal initials and shared fits."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
    validate_formula,
)


def problem():
    graph = {
        "variable_types": {
            "x": "differential",
            "y": "differential",
            "output": "algebraic",
        },
        "sources": {"x": ["u"], "y": ["x"], "output": ["y", "u"]},
        "initializers": {
            "x": {"mode": "observed_initial", "channel": "survey"},
            "y": {"mode": "map", "public_initials": [], "state_initials": ["x"]},
        },
    }
    formulas = {
        "equations": {"x": "k*u", "y": "x", "output": "y+sin(u)"},
        "initializers": {"y": "state0_x"},
        "parameters": {"k": 0.3},
    }

    def event(split, j):
        t = np.array([0, 0.1, 0.25, 0.5, 0.75, 1]) * (1 + 0.25 * j)
        survey = 1 + j
        # u(t)=t, x(t)=survey+k*t^2/2, y(t)=survey+survey*t+k*t^3/6.
        output = survey * (1 + t) + 1.5 * t**3 / 6 + np.sin(t)
        return {
            "trajectory_id": f"{split}_{j:03d}",
            "time": t.tolist(),
            "targets": {"output": output.tolist()},
            "inputs": {"u": t.tolist()},
            "covariates": {"survey": survey},
            "auxiliaries": {},
        }

    return graph, formulas, [event("train", 0), event("train", 1)], [event("val", 2)]


def test_analytic_fit_on_distinct_grids_and_genuine_free_validation():
    g, f, t, v = problem()
    fit = fit_formula(g, f, t, max_evaluations=60, seed=0)
    assert fit["status"] == "fitted"
    assert set(fit["runtime_versions"]) == {"python", "numpy", "scipy", "casadi"}
    assert all(fit["runtime_versions"].values())
    assert fit["parameters"]["k"] == pytest.approx(1.5, abs=1e-6)
    assert fit["training_nmse"] < 1e-16
    score = score_formula(g, f, fit, v)
    assert score["status"] == "scored" and score["pooled_nmse"] < 1e-16
    assert score["negative_samples"] == {"output": 0}
    assert score["curves"][0]["time"] == v[0]["time"]
    assert "curves" not in fit
    assert fit["effective_sources"] == g["sources"]
    altered = deepcopy(v)
    altered[0]["targets"]["output"][2] += 10
    changed = score_formula(g, f, fit, altered)
    assert changed["pooled_nmse"] == pytest.approx(
        100 / (6 * fit["scales"]["output"] ** 2)
    )
    assert changed["minimum_prediction"] == score["minimum_prediction"]
    assert fit["evaluations"] <= 60


def test_no_validation_fit_and_frozen_formula():
    g, f, t, v = problem()
    with pytest.raises(ValueError, match="train trajectory"):
        fit_formula(g, f, v)
    result = fit_formula(g, f, t, max_evaluations=1)
    with pytest.raises(ValueError, match="val trajectory"):
        score_formula(g, f, result, t)
    altered = deepcopy(f)
    altered["parameters"]["k"] = 100
    with pytest.raises(ValueError, match="identity"):
        score_formula(g, altered, result, v)
    assert result["parameters"] == {"k": 0.3} and result["evaluations"] == 1


@pytest.mark.parametrize(
    "equation", ["u+output", "u+0*x", "0*u", "k*(u-u)", "__import__('os')"]
)
def test_reject_graph_drift_invalid_text_or_cancelled_dependence(equation):
    g, f, _, _ = problem()
    f["equations"]["x"] = equation
    with pytest.raises(ValueError):
        validate_formula(g, f)


def test_reject_extra_observed_initial_or_dynamic_input_in_map():
    g, f, _, _ = problem()
    f["initializers"]["x"] = "u"
    with pytest.raises(ValueError, match="initializers only"):
        validate_formula(g, f)
    f["initializers"] = {"y": "state0_x+u"}
    with pytest.raises(ValueError, match="initial y needs"):
        validate_formula(g, f)


def test_discretization_failure_does_not_supply_finite_ranking_score():
    g = {
        "variable_types": {"x": "differential"},
        "sources": {"x": ["x"]},
        "initializers": {"x": {"mode": "observed_initial", "channel": "x"}},
    }
    f = {"equations": {"x": "-5*x"}, "initializers": {}, "parameters": {}}
    t = [
        {
            "trajectory_id": "train_000",
            "time": [0, 1, 2],
            "targets": {"x": [1, 0.1, 0]},
            "inputs": {},
            "auxiliaries": {},
            "covariates": {},
        }
    ]
    result = fit_formula(g, f, t, max_evaluations=2)
    assert (
        result["status"] == "resolution_unverified" and result["training_nmse"] is None
    )
    assert result["diagnostic_fine_grid"]["pooled_nmse"] is not None


@pytest.mark.parametrize("expr", ["k*log(u-5)", "k*(exp(1000*u))"])
def test_numerical_failure_cannot_be_reported_as_nmse(expr):
    g, f, t, _ = problem()
    f["equations"]["x"] = expr
    result = fit_formula(g, f, t, max_evaluations=2)
    assert result["status"] == "fit_failed" and result["training_nmse"] is None
    assert result["failed_rollouts"] == result["evaluations"] == 2
    assert result["runtime_versions"]["casadi"]


def test_validation_numerical_score_overflow_is_never_rankable():
    g, f, t, v = problem()
    fit = fit_formula(g, f, t, max_evaluations=1)
    v[0]["targets"]["output"][-1] = 1e300
    score = score_formula(g, f, fit, v)
    assert score["status"] == "score_failed" and score["pooled_nmse"] is None


def test_fitted_zero_eliminates_a_source_without_changing_the_frozen_graph():
    g, f, t, _ = problem()
    f["parameters"]["k"] = 0.0
    result = fit_formula(g, f, t, max_evaluations=1)
    assert result["status"] == "fitted"
    assert result["effective_sources"]["x"] == [] and g["sources"]["x"] == ["u"]
