"""Affine order compilation, projection, and full-flow accounting."""

import numpy as np
import pytest

from scripts import t2_order_constraint_solvers as solvers
from tests.test_fit_t2_initial_disposal_order import candidate


def test_wedge_projection_is_nearest_and_preserves_other_bounds():
    a = np.array([[1.0, 1.0, 0.0], [2.0, -2.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    b = np.array([2.0, 4.0, 3.0, 2.0])
    chart = solvers.wedge(a, b, np.array([-np.inf, -np.inf, 1.0]))
    x = chart.project(np.array([0.0, 0.0, 4.0]))
    np.testing.assert_allclose(x, [2, 0, 4], atol=1e-12)
    x = chart.project(np.array([2.0, 1.0, 3.0]))
    np.testing.assert_allclose(x, [2.5, 0.5, 3], atol=1e-12)
    # The third coordinate retains its original finite parameter bound.
    np.testing.assert_allclose(chart.lower, [0, 0, 1])
    for y in ([0.0, 0.0, 1.0], [2.1, 1.2, 5.0]):
        x = chart.physical(np.array(y))
        assert min(a @ x - b) >= -1e-12
        assert x[2] >= 1
        np.testing.assert_allclose(chart.coordinates(x), y, atol=1e-12)
        delta = np.array([0.6, 0.11, -0.5])
        g = np.array([3.0, -2.0, 1.2])
        assert chart.gradients(g) @ delta == pytest.approx(
            g @ (chart.physical(np.array(y) + delta) - x)
        )
    with pytest.raises(ValueError, match="bounded coefficients"):
        solvers.wedge(a, b, np.array([0.0, -np.inf, 1.0]))


def test_non_wedge_is_rejected():
    a = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="independent bound facets"):
        solvers.wedge(a, np.array([-1.0, -1.0]), np.full(2, -np.inf))


def test_same_reference_and_feasible_solution_no_infeasible_solver_evaluations(
    monkeypatch,
):
    model, rows = candidate(monkeypatch)
    existing = solvers.reference.fit(model, rows, "half_gap", 40, "warm")
    reference, feasible = [
        solvers.fit(model, rows, arm, "warm", budget=40)
        for arm in ["reference", "feasible"]
    ]
    assert reference["parameters"] == existing["parameters"]
    assert reference["forward_evaluations"] == existing["forward_evaluations"]
    assert reference["cost"]["infeasible_residual_batches"] > 0
    assert feasible["solver_success"]
    assert feasible["training_nmse"] == pytest.approx(
        reference["training_nmse"], rel=1e-8
    )
    assert feasible["parameters"]["gain"] == pytest.approx(1.0, abs=1e-8)
    assert feasible["projection"]["min_projected_slack"] >= -1e-12
    assert feasible["matrix_check"]["max_absolute_U_error"] < 1e-13
    assert feasible["cost"]["infeasible_residual_batches"] == 0
    assert feasible["cost"]["infeasible_jacobian_batches"] == 1
    assert feasible["cost"]["infeasible_setup_jacobians"] == 1
    for result in (reference, feasible):
        c = result["cost"]
        assert c["full_residual_trajectories"] == 2 * c["full_residual_batches"]
        assert c["full_jacobian_trajectories"] == 2 * c["full_jacobian_batches"]
        assert c["setup_seconds"] + c["solver_seconds"] + c[
            "projection_seconds"
        ] == pytest.approx(c["elapsed_seconds"])


def test_compiler_checks_parameter_products_before_treating_them_as_affine(monkeypatch):
    model, rows = candidate(monkeypatch)
    f = model["formula"]
    f["equations"]["U"] = "offset*gain+gain*Gt+afcoef13*I"
    identity = solvers.reference.content_hash(
        [model["graph"], f["equations"], f["initializers"]]
    )
    monkeypatch.setattr(solvers.reference.base, "GRAPHS", {identity})
    with pytest.raises(ValueError, match="not affine"):
        solvers.fit(model, rows, "feasible", "warm", budget=4)
