"""Physical initial-state estimation, causal maps, and independent rollout checks."""

from dataclasses import replace
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.fitting.simulation import (
    simulate_trajectory,
    trajectory_initial_state,
)
from autoformalism.schemas import CandidateModel


def problem():
    """Hidden reservoir excites an initially zero observed state without input."""
    model = compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "hidden_initial",
                "parent_candidate_id": None,
                "states": [
                    {"name": "m", "kind": "latent"},
                    {"name": "y", "kind": "latent"},
                ],
                "state_equations": [
                    {"state": "m", "rhs": "-rate*m"},
                    {"state": "y", "rhs": "m-0.7*y"},
                ],
                "parameters": [{"name": "rate", "scope": "global", "role": "rate"}],
                "observation_mappings": [{"channel": "v01", "expression": "y"}],
                "initial_conditions": [
                    {"state": "m", "scope": "global", "fixed_value": 0},
                    {"state": "y", "scope": "global", "fixed_value": 0},
                ],
            }
        ),
        ValidationContext(targets=("v01",), fixed_covariates=("dose",)),
    )
    splits = []
    for name, doses in ((SplitName.TRAIN, (1.0, 2.0)), (SplitName.VALIDATION, (3.0,))):
        rows = []
        for index, dose in enumerate(doses):
            time = np.linspace(0, 4, 31)
            y = (1 + dose) * (np.exp(-0.3 * time) - np.exp(-0.7 * time)) / 0.4
            rows.append(
                Trajectory(
                    f"{name.value}_{index}",
                    time,
                    {"v01": y},
                    {},
                    {},
                    {"dose": dose},
                    {},
                )
            )
        splits.append(DatasetSplit(name, tuple(rows), name.value))
    return model, *splits


def map_plan():
    return LatentInitializationPlan.model_validate(
        {
            "rules": {
                "m": {
                    "initial": {
                        "mode": "map",
                        "expression": "a+b*dose+v01",
                        "parameters": [
                            {"name": "a", "guess": 0},
                            {"name": "b", "guess": 0},
                        ],
                    }
                }
            }
        }
    )


def test_boundary_derivatives_match_finite_differences_and_production():
    original, train, _ = problem()
    model, guesses, _ = apply_initialization_plan(original, map_plan())
    assert guesses == {"init_m_a": 0, "init_m_b": 0}
    system = SymbolicODE(model)
    theta = np.array([0.3, 1.0, 1.0])
    row = train.trajectories[1]
    np.testing.assert_allclose(
        system.initial_sensitivity(row, theta), [[0, 1, 2], [0, 0, 0]]
    )
    settings = CollocationSensitivityConfig(
        relative_tolerance=1e-10, absolute_tolerance=1e-12
    ).fit_config()
    values, jac, _, _ = symbolic_rollout(
        system, row, theta, settings, monotonic() + 10, sensitivities=True
    )
    for column in range(len(theta)):
        shift = np.zeros(3)
        shift[column] = 1e-4
        plus = symbolic_rollout(system, row, theta + shift, settings, monotonic() + 10)[
            0
        ]
        minus = symbolic_rollout(
            system, row, theta - shift, settings, monotonic() + 10
        )[0]
        np.testing.assert_allclose(
            jac[:, :, column], (plus - minus) / 2e-4, atol=2e-7, rtol=2e-6
        )
    result = simulate_trajectory(
        model, row, dict(zip(system.names, theta, strict=True)), {}, settings
    )
    assert result.success
    np.testing.assert_allclose(result.predictions["v01"], values[:, 0], atol=1e-8)


def test_future_targets_cannot_change_initials_and_observed_boundary_is_exact():
    original, train, _ = problem()
    model, _, _ = apply_initialization_plan(original, map_plan())
    row = train.trajectories[0]
    altered = row.targets["v01"].copy()
    altered[1:] = 1e5
    changed = replace(row, targets={"v01": altered})
    parameters = {"rate": 0.3, "init_m_a": 1.0, "init_m_b": 1.0}
    np.testing.assert_array_equal(
        trajectory_initial_state(model, row, {}, parameters=parameters),
        trajectory_initial_state(model, changed, {}, parameters=parameters),
    )
    altered = altered.copy()
    altered[0] = 4
    changed = replace(row, targets={"v01": altered})
    np.testing.assert_array_equal(
        trajectory_initial_state(model, changed, {}, parameters=parameters), [6, 4]
    )


@pytest.mark.parametrize("expression", ["future", "m", "v01[1]", "__import__('os')"])
def test_unsafe_or_unavailable_map_rejected(expression):
    model, _, _ = problem()
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "m": {
                    "initial": {
                        "mode": "map",
                        "expression": expression,
                    }
                }
            }
        }
    )
    with pytest.raises((ValueError, ModelValidationError)):
        apply_initialization_plan(model, plan)


def test_cannot_fit_directly_observed_initial_or_change_legacy_default():
    model, train, _ = problem()
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "y": {
                    "initial": {
                        "mode": "value",
                        "guess": 2,
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="latent states"):
        apply_initialization_plan(model, plan)
    np.testing.assert_array_equal(
        SymbolicODE(model).initial_for(train.trajectories[0]), [0, 0]
    )


def test_piecewise_initial_map_is_included_in_solver_audit():
    model, _, _ = problem()
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "m": {
                    "initial": {
                        "mode": "map",
                        "expression": "max(a+v01,0)",
                        "parameters": [{"name": "a"}],
                    }
                }
            }
        }
    )
    compiled, _, _ = apply_initialization_plan(model, plan)
    system = SymbolicODE(compiled, allow_piecewise=True)
    assert system.has_piecewise
    assert any(
        p["location"] == "initial_condition:m"
        for p in system.audit["piecewise_expressions"]
    )


def test_known_initial_is_fixed_and_observed_initial_uses_measurement():
    model, train, _ = problem()
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "m": {
                    "initial": {
                        "mode": "known",
                        "value": 2.0,
                        "justification": "Known prepared amount",
                    }
                }
            }
        }
    )
    compiled, guesses, audit = apply_initialization_plan(model, plan)
    assert guesses == {}
    row = train.trajectories[0]
    altered = row.targets["v01"].copy()
    altered[0] = 3.0
    changed = replace(row, targets={"v01": altered})
    np.testing.assert_array_equal(
        SymbolicODE(compiled).initial_for(changed, np.array([0.3])), [2, 3]
    )
    assert audit["bindings"]["m"]["mode"] == "known"


def test_initializer_parameter_names_cannot_collide():
    model, _, _ = problem()
    payload = model.validated.candidate.model_dump(mode="json")
    payload["parameters"][0]["name"] = "init_m_value"
    payload["state_equations"][0]["rhs"] = "-init_m_value*m"
    model = compile_candidate(
        CandidateModel.model_validate(payload), model.validated.context
    )
    plan = LatentInitializationPlan.model_validate(
        {"rules": {"m": {"initial": {"mode": "value", "guess": 0}}}}
    )
    with pytest.raises(ValueError, match="collision"):
        apply_initialization_plan(model, plan)


def test_invalid_validation_map_retains_boundary_failure_report():
    from autoformalism.fitting.collocation_sensitivity import _physical_initial_report

    model, _, val = problem()
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "m": {
                    "initial": {
                        "mode": "map",
                        "expression": "exp(a*v01)",
                        "parameters": [{"name": "a"}],
                    }
                }
            }
        }
    )
    compiled, _, _ = apply_initialization_plan(model, plan)
    row = val.trajectories[0]
    target = row.targets["v01"].copy()
    target[0] = 1000.0
    row = replace(row, targets={"v01": target})
    val = DatasetSplit(SplitName.VALIDATION, (row,), "large_initial_observation")
    initials, errors = _physical_initial_report(
        compiled, val, {"rate": 0.3, "init_m_a": 1.0}
    )
    assert initials == {}
    assert row.trajectory_id in errors
    with pytest.raises(ValueError, match="physical initialization"):
        SymbolicODE(compiled).initial_for(row, np.array([0.3, 1.0]))


def test_collocation_and_refinement_recover_causal_map_from_zero_guesses(tmp_path):
    model, train, val = problem()
    result = fit_collocation_forward_sensitivity(
        model,
        train,
        val,
        CollocationSensitivityConfig(
            initializer_seconds=25,
            refinement_seconds=20,
            maximum_function_evaluations=50,
            collocation_node_start="rollout_or_observed",
        ),
        tmp_path,
        initial_parameters={"rate": 1.0},
        initialization_plan=map_plan(),
    )
    assert result["status"] == "complete", result
    assert result["initializer"]["success"], result["initializer"]
    assert result["initializer"]["initial_conditions_optimized"]
    assert result["validation"]["normalized_mse"] < 1e-9
    assert result["parameters"]["init_m_a"] == pytest.approx(1, abs=1e-4)
    assert result["parameters"]["init_m_b"] == pytest.approx(1, abs=1e-4)
    assert result["validation"]["trajectory_initial_conditions"]["val_0"][
        "m"
    ] == pytest.approx(4, abs=1e-4)
