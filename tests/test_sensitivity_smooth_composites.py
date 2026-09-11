"""Classical first sensitivities through certified composites and zero crossings."""

import ast
import json
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.sensitivity_probe import (
    SensitivityContractError,
    SymbolicODE,
    symbolic_rollout,
)
from autoformalism.schemas import CandidateModel


def model(*, rhs="a", observation="f/(1+abs(f))", processes=()):
    return compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "smooth_composite",
                "parent_candidate_id": None,
                "states": [{"name": "f", "kind": "latent"}],
                "state_equations": [{"state": "f", "rhs": rhs}],
                "processes": list(processes),
                "observation_mappings": [
                    {"channel": "v01", "expression": observation}
                ],
                "parameters": [{"name": "a", "role": "rate", "scope": "global"}],
                "initial_conditions": [
                    {"state": "f", "scope": "global", "fixed_value": -1.0}
                ],
            }
        ),
        ValidationContext(targets=("v01",)),
    )


def trajectory(end=4.0, name="train"):
    time = np.linspace(0, end, 41)
    state = -1 + 0.5 * time
    return Trajectory(name, time, {"v01": state / (1 + abs(state))}, {}, {}, {}, {})


def settings(method="Radau"):
    return FitConfig(
        integration_method=method, relative_tolerance=1e-10, absolute_tolerance=1e-12
    )


def test_exact_candidate_output_crosses_zero_with_analytic_sensitivity():
    system = SymbolicODE(model())
    data = trajectory()
    values, jacobian, states, _ = symbolic_rollout(
        system, data, np.array([0.5]), settings(), None, sensitivities=True
    )
    expected_state = -1 + 0.5 * data.time
    np.testing.assert_allclose(states[:, 0], expected_state, atol=1e-10)
    np.testing.assert_allclose(values[:, 0], data.targets["v01"], atol=1e-10)
    np.testing.assert_allclose(
        jacobian[:, 0, 0], data.time / (1 + abs(expected_state)) ** 2, atol=1e-9
    )
    assert float(system.local(2, [0], [0.5], [])[2]) == 1.0
    assert system.requires_first_order_solver
    # A C1 observation alone needs no second derivatives in the augmented ODE.
    assert system.augmented_jacobian is not None
    assert system.audit["collocation_hessian"] == "limited-memory"


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_c1_rhs_crossing_matches_production_and_parameter_differences(method):
    compiled = model(rhs="a - f/(1+abs(f))", observation="f")
    system = SymbolicODE(compiled)
    assert system.augmented_jacobian is None
    assert system.audit["augmented_solver_jacobian"] == "numerical_newton_approximation"
    data = trajectory()
    values, jacobian, states, _ = symbolic_rollout(
        system, data, np.array([0.5]), settings(method), None, sensitivities=True
    )
    assert states[0, 0] < 0 < states[-1, 0]
    production = simulate_trajectory(
        compiled, data, {"a": 0.5}, {}, settings(method), reset_observed_states=False
    )
    assert production.success
    np.testing.assert_allclose(values[:, 0], production.predictions["v01"], atol=2e-8)
    for step in (1e-4, 1e-5):
        replays = []
        for parameter in (0.5 + step, 0.5 - step):
            replay = simulate_trajectory(
                compiled, data, {"a": parameter}, {}, settings(method),
                reset_observed_states=False,
            )
            assert replay.success
            replays.append(replay.predictions["v01"])
        numerical = (replays[0] - replays[1]) / (2 * step)
        np.testing.assert_allclose(jacobian[:, 0, 0], numerical, atol=3e-5)


@pytest.mark.parametrize(
    "expression,scale,offset",
    [
        ("a*f/(1+abs(f))", 0.7, 1.0),
        ("-(a*f)/(abs(f)+1)", -0.7, 1.0),
        ("f/(2+abs(f))", 1.0, 2.0),
    ],
)
def test_factored_composites_have_correct_first_partials(expression, scale, offset):
    system = SymbolicODE(model(observation=expression))
    for value in (-1.0, -1e-8, 0, 1e-8, 1.0):
        _, _, hx, _ = system.local(0, [value], [0.7], [])
        assert float(hx) == pytest.approx(scale * offset / (offset + abs(value)) ** 2)


def test_composite_is_certified_across_process_aliases_without_mutating_source():
    compiled = model(
        observation="q/(1+magnitude)",
        processes=[
            {"name": "q", "expression": "f+a"},
            {"name": "magnitude", "expression": "abs(q)"},
        ],
    )
    before = deepcopy(compiled.validated.candidate.model_dump(mode="json"))
    trees_before = {
        name: ast.dump(expression.tree)
        for name, expression in compiled.validated.process_expressions.items()
    }
    system = SymbolicODE(compiled)
    _, _, hx, hp = system.local(0, [-0.7], [0.7], [])
    assert float(hx) == 1.0 and float(hp) == 1.0
    assert compiled.validated.candidate.model_dump(mode="json") == before
    assert trees_before == {
        name: ast.dump(expression.tree)
        for name, expression in compiled.validated.process_expressions.items()
    }


def test_current_candidate_saturation_forms_work_together():
    system = SymbolicODE(model(
        observation="f/(1+abs(f)) + m**2/(1+m**2) + f**2/(1+f**2)",
        processes=[{"name": "m", "expression": "f+a"}],
    ))
    _, _, hx, hp = system.local(0, [0], [0.5], [])
    assert float(hx) == pytest.approx(1.64)
    assert float(hp) == pytest.approx(0.64)


def test_expanded_process_budget_fails_closed():
    processes = []
    previous = "f"
    for index in range(16):
        name = f"p{index}"
        processes.append({"name": name, "expression": f"{previous}+{previous}"})
        previous = name
    with pytest.raises(
        SensitivityContractError, match="expanded structural audit budget"
    ):
        SymbolicODE(model(processes=processes, observation=previous))


@pytest.mark.parametrize("expression", ["max(f,0)**2", "min(0,f)**2"])
def test_squared_rectifier_crossing_has_continuous_first_derivative(expression):
    system = SymbolicODE(model(observation=expression))
    for state in (-1e-4, 0, 1e-4):
        _, _, hx, _ = system.local(0, [state], [0.5], [])
        expected = 2 * (
            max(state, 0) if expression.startswith("max") else min(state, 0)
        )
        assert float(hx) == pytest.approx(expected)
    assert system.requires_first_order_solver
    values, jacobian, states, _ = symbolic_rollout(
        system, trajectory(), np.array([0.5]), settings(), None, sensitivities=True
    )
    assert np.isfinite(values).all() and np.isfinite(jacobian).all()
    assert states[0, 0] < 0 < states[-1, 0]


def test_even_abs_power_has_true_second_derivative_at_zero():
    system = SymbolicODE(model(rhs="a+abs(f)**2", observation="f"))
    assert not system.requires_first_order_solver
    matrix = np.asarray(system.augmented_jacobian(0, [0, 1], [0.5], []))
    assert matrix[1, 0] == 2.0


def test_stable_softplus_translation_does_not_claim_exact_second_ad_at_zero():
    system = SymbolicODE(model(rhs="a+softplus(f)", observation="f"))
    for state in (-20, 0, 20):
        assert float(system.local(0, [state], [0.5], [])[0]) == pytest.approx(
            1 / (1 + np.exp(-state))
        )
    assert system.augmented_jacobian is None
    assert system.audit["collocation_hessian"] == "limited-memory"


@pytest.mark.parametrize("expression", ["max(f,f)", "min(f,f)", "abs(-2)*f"])
def test_identical_branches_and_constants_are_not_rejected_by_function_name(expression):
    system = SymbolicODE(model(observation=expression))
    assert not system.requires_first_order_solver
    assert float(system.local(0, [0], [0.5], [])[2]) == (
        2.0 if expression.startswith("abs") else 1.0
    )


@pytest.mark.parametrize(
    "expression",
    ["abs(f)", "max(f,0)", "min(f,0)", "1/(1+abs(f))", "f/(a+abs(f))"],
)
def test_uncertified_crossings_fail_before_optimization(expression):
    with pytest.raises(SensitivityContractError, match="uncertified nonsmooth"):
        SymbolicODE(model(observation=expression))


@pytest.mark.parametrize("expression", ["log(1+f**2)", "sqrt(1+f**2)"])
def test_domain_restricted_functions_need_a_separate_certificate(expression):
    with pytest.raises(SensitivityContractError, match="domain not certified"):
        SymbolicODE(model(observation=expression))


def test_collocation_and_sensitivity_fit_through_output_zero_crossing(tmp_path):
    train = DatasetSplit(SplitName.TRAIN, (trajectory(),), "smooth-training")
    validation = DatasetSplit(
        SplitName.VALIDATION, (trajectory(5, "validation"),), "smooth-validation"
    )
    result = fit_collocation_forward_sensitivity(
        model(), train, validation,
        CollocationSensitivityConfig(
            initializer_seconds=20, refinement_seconds=15,
            maximum_function_evaluations=30,
            collocation_node_start="rollout_or_observed",
        ),
        tmp_path, initial_parameters={"a": 0.3},
    )
    assert result["status"] == "complete", result
    assert result["initializer"]["success"], result["initializer"]
    assert result["initializer"]["hessian_approximation"] == "limited-memory"
    assert result["sensitivity_audit"]["collocation_hessian"] == "limited-memory"
    assert json.loads((tmp_path / "sensitivity_audit.json").read_text()) == (
        result["sensitivity_audit"]
        | {"parameter_order": list(result["sensitivity_audit"]["parameter_order"])}
    )
    assert result["parameters"]["a"] == pytest.approx(0.5, abs=1e-6)
    assert result["validation"]["normalized_mse"] < 1e-9
