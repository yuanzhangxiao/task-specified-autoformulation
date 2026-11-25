from copy import deepcopy

import pytest

from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
)
from autoformalism.research.trajectory_nonnegative_fitting import fit_nonnegative


def slope_problem():
    """Positive measurements; unconstrained least squares crosses below zero."""
    graph = {
        "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
        "variable_types": {"y": "differential"},
        "sources": {"y": []},
    }
    formula = {
        "equations": {"y": "rate"},
        "initializers": {},
        "parameters": {"rate": -1.0},
    }
    train = [
        {
            "trajectory_id": "train_00",
            "time": [0, 1, 2, 3, 4, 5],
            "targets": {"y": [1, 0, 0, 0, 0, 0]},
            "covariates": {},
            "inputs": {},
            "auxiliaries": {},
        }
    ]
    return graph, formula, train


def test_hard_constraint_recovers_boundary_optimum_from_infeasible_start():
    graph, formula, train = slope_problem()
    old = fit_formula(graph, formula, train, max_evaluations=100)
    assert old["minimum_prediction"]["y"] < -0.3
    actual = fit_nonnegative(graph, formula, train, max_evaluations=100, starts=2)
    assert actual["status"] == "fitted"
    assert actual["parameters"]["rate"] == pytest.approx(-0.2, abs=1e-6)
    assert actual["negative_samples"] == {"y": 0}
    assert actual["minimum_coarse_prediction"]["y"] >= -1e-10
    assert (
        actual["evaluations"]
        == actual["coarse_evaluations"] + actual["fine_evaluations"]
    )
    assert 0 < actual["evaluations"] <= 100
    assert actual["starts_attempted"] == 2
    validation = [train[0] | {"trajectory_id": "val_00"}]
    assert score_formula(graph, formula, actual, validation)["status"] == "scored"


def test_infeasible_fixed_model_keeps_failed_slot():
    graph, formula, train = slope_problem()
    formula = {**formula, "parameters": {}, "equations": {"y": "-1"}}
    result = fit_nonnegative(graph, formula, train, starts=2, max_evaluations=5)
    assert result["status"] == "fit_failed"
    assert result["parameters"] is None and result["training_nmse"] is None
    assert result["coarse_evaluations"] == 2 and result["fine_evaluations"] == 0


def test_resolution_check_and_budget_before_retaining_vector():
    graph, formula, train = slope_problem()
    formula = {**formula, "parameters": {}, "equations": {"y": "-5*y"}}
    graph = {**graph, "sources": {"y": ["y"]}}
    result = fit_nonnegative(graph, formula, train, max_evaluations=2)
    assert result["status"] == "fit_failed" and result["resolution_rejections"] == 1
    result = fit_nonnegative(graph, formula, train, max_evaluations=1)
    assert result["evaluations"] == 1 and result["parameters"] is None


def test_verify_target_names_and_training_identity():
    graph, formula, train = slope_problem()
    for targets in (["z"], [], ["y", "y"]):
        with pytest.raises(ValueError, match="distinct observed"):
            fit_nonnegative(graph, formula, train, targets=targets)
    with pytest.raises(ValueError, match="budget"):
        fit_nonnegative(graph, formula, train, wall_seconds=-1)
    validation = deepcopy(train)
    validation[0]["trajectory_id"] = "val_00"
    with pytest.raises(ValueError, match="unique train"):
        fit_nonnegative(graph, formula, validation)
