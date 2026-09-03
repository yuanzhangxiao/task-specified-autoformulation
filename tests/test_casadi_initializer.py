"""Optional CasADi nonlinear initialization tests."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.schemas import CandidateModel, ParameterDomain, ParameterSpec
from autoformalism.search.controller import _fit_from_dict, _fit_to_dict


def _split(name: SplitName, identifier: str, rate: float) -> DatasetSplit:
    time = np.linspace(0.0, 4.0, 41)
    target = 2.0 * np.exp(-rate * time)
    trajectory = Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"target": target},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    return DatasetSplit(name, (trajectory,), f"{name.value}-fingerprint")


def _model() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "casadi_decay",
            "parent_candidate_id": None,
            "change_summary": "Test one positive decay rate.",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed decaying state.",
                }
            ],
            "state_equations": [{"state": "x", "rhs": "-rate * x"}],
            "observation_mappings": [
                {"channel": "target", "expression": "x", "unit": "unit"}
            ],
            "parameters": [
                {
                    "name": "rate",
                    "scope": "global",
                    "role": "rate",
                    "unit": "1/time",
                    "description": "Positive decay rate.",
                }
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "target"}
            ],
        }
    )


def test_parameter_role_derives_qualitative_domain() -> None:
    parameter = ParameterSpec.model_validate(
        {"name": "tau", "scope": "global", "role": "time_constant"}
    )

    assert parameter.domain is ParameterDomain.POSITIVE
    with pytest.raises(ValueError, match="requires domain positive"):
        ParameterSpec.model_validate(
            {
                "name": "tau",
                "scope": "global",
                "role": "time_constant",
                "domain": "real",
            }
        )


def test_casadi_multiple_shooting_initializes_then_uses_existing_fitter() -> None:
    true_rate = 0.55
    model = compile_candidate(_model(), ValidationContext(targets=("target",)))
    result = fit_candidate(
        model,
        _split(SplitName.TRAIN, "train", true_rate),
        _split(SplitName.VALIDATION, "validation", true_rate),
        FitConfig(
            nonlinear_initializer="casadi_multiple_shooting",
            nonlinear_initializer_failure_policy="raise",
            integration_backend="fixed_rk4",
            maximum_function_evaluations=20,
            casadi_maximum_iterations=100,
            casadi_maximum_wall_time_seconds=30.0,
        ),
    )

    assert result.success
    assert result.global_parameters["rate"] == pytest.approx(true_rate, abs=1e-5)
    assert result.validation_metrics.normalized_mse < 1e-10
    assert len(result.initialization_diagnostics) == 1
    diagnostic = result.initialization_diagnostics[0]
    assert diagnostic.backend == "casadi_multiple_shooting"
    assert diagnostic.success
    assert diagnostic.iterations is not None
    restored = _fit_from_dict(_fit_to_dict(result))
    assert restored.initialization_diagnostics == result.initialization_diagnostics


def test_casadi_initializer_optimizes_latent_states_without_latent_labels() -> None:
    true_rate = 0.65
    time = np.linspace(0.0, 4.0, 41)
    target = 0.5 * np.exp(-0.2 * time) + (
        np.exp(-0.2 * time) - np.exp(-true_rate * time)
    ) / (true_rate - 0.2)
    trajectory = Trajectory(
        trajectory_id="latent-train",
        time=time,
        targets={"target": target},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "casadi_latent_decay",
            "parent_candidate_id": None,
            "change_summary": "Test an unobserved transient state.",
            "states": [
                {"name": "x", "kind": "observed"},
                {"name": "z", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "x", "rhs": "z - 0.2 * x"},
                {"state": "z", "rhs": "-rate * z"},
            ],
            "observation_mappings": [
                {"channel": "target", "expression": "x"}
            ],
            "parameters": [
                {"name": "rate", "scope": "global", "role": "rate"}
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "target"},
                {"state": "z", "scope": "global", "fixed_value": 1.0},
            ],
        }
    )
    model = compile_candidate(candidate, ValidationContext(targets=("target",)))
    split = DatasetSplit(SplitName.TRAIN, (trajectory,), "latent-fingerprint")
    result = fit_candidate(
        model,
        split,
        DatasetSplit(
            SplitName.VALIDATION,
            (trajectory,),
            "latent-validation-fingerprint",
        ),
        FitConfig(
            nonlinear_initializer="casadi_multiple_shooting",
            nonlinear_initializer_failure_policy="raise",
            integration_backend="fixed_rk4",
            maximum_function_evaluations=20,
            casadi_maximum_wall_time_seconds=30.0,
        ),
    )

    assert result.success
    assert result.global_parameters["rate"] == pytest.approx(true_rate, abs=1e-5)
    assert result.validation_metrics.normalized_mse < 1e-10
    assert result.initialization_diagnostics[0].success
