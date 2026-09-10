"""Transfer-adapter checks for collocation plus forward sensitivities."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.schemas import CandidateModel


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "collocation_transfer_smoke",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": "-rate*x + gain*u01"}],
            "observation_mappings": [{"channel": "v01", "expression": "x"}],
            "parameters": [
                {"name": "rate", "scope": "global", "role": "rate"},
                {
                    "name": "gain",
                    "scope": "global",
                    "role": "nonnegative_coefficient",
                },
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "fixed_value": 0.0}
            ],
        }
    )


def _split(
    name: SplitName,
    amplitudes: tuple[float, ...],
    initial_values: tuple[float, ...] | None = None,
) -> DatasetSplit:
    rate, gain = 0.7, 1.3
    time = np.linspace(0.0, 2.0, 21)
    rows = []
    starts = initial_values or tuple(0.0 for _ in amplitudes)
    for index, (amplitude, initial) in enumerate(zip(amplitudes, starts, strict=True)):
        output = initial * np.exp(-rate * time) + amplitude * gain / rate * (
            1.0 - np.exp(-rate * time)
        )
        rows.append(
            Trajectory(
                trajectory_id=f"{name.value}-{index}",
                time=time,
                targets={"v01": output},
                auxiliaries={},
                external_inputs={"u01": np.full_like(time, amplitude)},
                fixed_covariates={},
                derivatives={},
            )
        )
    return DatasetSplit(name, tuple(rows), f"{name.value}-fingerprint")


def test_adapter_uses_train_only_initialization_and_causal_replay(tmp_path) -> None:
    model = compile_candidate(
        _candidate(), ValidationContext(targets=("v01",), external_inputs=("u01",))
    )
    result = fit_collocation_forward_sensitivity(
        model,
        _split(SplitName.TRAIN, (1.0, 1.7)),
        _split(SplitName.VALIDATION, (0.6,)),
        CollocationSensitivityConfig(
            initializer_seconds=15,
            refinement_seconds=15,
            maximum_function_evaluations=20,
        ),
        tmp_path,
        initial_parameters={"rate": 0.7, "gain": 1.3},
    )

    assert result["status"] == "complete", result
    assert result["initializer"]["training_only"]
    assert result["training_only_parameter_estimation"]
    assert not result["validation_used_for_fitting"]
    assert result["forward_sensitivity_jacobian_used"]
    assert not result["collocation_states_used_for_final_score"]
    assert result["training"]["normalized_mse"] < 1e-12
    assert result["validation"]["normalized_mse"] < 1e-12


def test_adapter_uses_each_trajectory_direct_observation_initial_value(
    tmp_path,
) -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"][0]["name"] = "v01"
    payload["state_equations"][0].update(
        state="v01", rhs="-rate*v01 + gain*u01"
    )
    payload["observation_mappings"][0]["expression"] = "v01"
    payload["initial_conditions"][0].update(
        state="v01", fixed_value=None, expression="v01"
    )
    model = compile_candidate(
        CandidateModel.model_validate(payload),
        ValidationContext(targets=("v01",), external_inputs=("u01",)),
    )
    result = fit_collocation_forward_sensitivity(
        model,
        _split(SplitName.TRAIN, (1.0, 1.7), (2.0, -0.4)),
        _split(SplitName.VALIDATION, (0.6,), (3.1,)),
        CollocationSensitivityConfig(
            initializer_seconds=15,
            refinement_seconds=15,
            maximum_function_evaluations=20,
        ),
        tmp_path,
        initial_parameters={"rate": 0.7, "gain": 1.3},
    )

    assert result["status"] == "complete", result
    assert result["training"]["normalized_mse"] < 1e-12
    assert result["validation"]["normalized_mse"] < 1e-12
