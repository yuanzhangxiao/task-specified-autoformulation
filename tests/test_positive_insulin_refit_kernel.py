"""Fit-stage boundaries, trajectory derivatives and handoff replay agree."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import transfer_positive_insulin_law as portable
from scripts.positive_insulin_refit_kernel import FitProblem
from tests.test_validate_positive_insulin_transfer import toy


def example():
    row, model = toy("train_001")
    model["formula"]["equations"]["Gp"] = "gain*I-Gp"
    model["formula"]["parameters"]["gain"] = 0.2
    return vars(row), model


def test_rk4_knot_grid_and_sensitivities_match_original():
    row, model = example()
    problem = FitProblem(model, [row])
    p = model["formula"]["parameters"]
    vec = np.array([p[n] for n in problem.names])
    predicted = portable._predict(model, row, model["knots"], 4)
    target = np.array([row["targets"][k] for k in portable.TARGETS])
    np.testing.assert_allclose(problem.residual(vec), (predicted - target).T.ravel())
    jac = problem.jacobian(vec)
    for k in range(len(vec)):
        delta = np.zeros_like(vec)
        delta[k] = 1e-5
        central = (problem.residual(vec + delta) - problem.residual(vec - delta)) / 2e-5
        np.testing.assert_allclose(jac[:, k], central, atol=1e-8, rtol=1e-6)


def test_fixed_then_joint_keeps_bounds_and_incumbents():
    row, model = example()
    original = deepcopy(model)
    problem = FitProblem(model, [row])
    p = model["formula"]["parameters"]
    fixed = problem.fit_stage("fixed_insulin", p, max_nfev=25)
    assert fixed["training_nmse"] < problem.score(p)["pooled_nmse"]
    assert fixed["diagnostics"]["active_parameters"] == ["gain"]
    assert all(p[k] == fixed["parameters"][k] for k in problem.insulin_parameters)
    joint = problem.fit_stage("joint", fixed["parameters"], max_nfev=25)
    assert joint["training_nmse"] <= fixed["training_nmse"] + 1e-12
    assert all(joint["parameters"][k] >= 0 for k in problem.insulin_parameters)
    assert joint["parameters"]["pos_tau"] > 1e-12
    assert joint["diagnostics"]["nfev"] <= 25
    assert model == original


def test_rejects_held_out_input_and_out_of_domain_parameters():
    row, model = example()
    with pytest.raises(ValueError, match="train"):
        FitProblem(model, [row | {"trajectory_id": "val_001"}])
    problem = FitProblem(model, [row])
    p = model["formula"]["parameters"]
    for fail in ({"pos_tau": 1e-12}, {"pos_Gt_1": -1}):
        with pytest.raises(ValueError, match="admissible"):
            problem.fit_stage("joint", p | fail)
