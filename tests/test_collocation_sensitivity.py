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
    assert result["refinement"]["selected_training_rollout_verified"]
    assert result["refinement"]["production_training_rollout_verified"]


@pytest.mark.parametrize("failing_split", ["train", "validation"])
def test_fresh_replay_failure_and_training_gate(tmp_path, monkeypatch, failing_split):
    from types import SimpleNamespace

    from autoformalism.fitting import collocation_sensitivity as adapter

    model = compile_candidate(
        _candidate(), ValidationContext(targets=("v01",), external_inputs=("u01",))
    )
    monkeypatch.setattr(
        adapter, "bounded_latent_start", lambda *a, **kw: {"success": False}
    )

    def evaluate(model, data, **kwargs):
        failed = data.name is (
            SplitName.TRAIN if failing_split == "train" else SplitName.VALIDATION
        )
        return {}, SimpleNamespace(
            normalized_mse=1e6 if failed else 0.0,
            per_target_normalized_mse={"v01": 1e6 if failed else 0.0},
            failed_trajectories=[data.trajectories[0].trajectory_id] if failed else [],
        )

    monkeypatch.setattr(adapter, "evaluate_fitted_candidate", evaluate)
    result = fit_collocation_forward_sensitivity(
        model,
        _split(SplitName.TRAIN, (1.0,)),
        _split(SplitName.VALIDATION, (0.6,)),
        CollocationSensitivityConfig(
            initializer_seconds=5,
            refinement_seconds=15,
            maximum_function_evaluations=10,
        ),
        tmp_path,
        initial_parameters={"rate": 0.7, "gain": 1.3},
    )
    assert result["status"] == "rollout_failed"
    assert result["refinement"]["optimizer_native_success"]
    assert result["refinement"]["optimizer_success"] == (failing_split != "train")
    assert result["refinement"]["production_training_rollout_verified"] == (
        failing_split != "train"
    )


def test_all_failure_adapter_does_not_score_validation(tmp_path, monkeypatch):
    from autoformalism.fitting import collocation_sensitivity as adapter

    payload = _candidate().model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "rate*x**2 + gain*u01"
    payload["initial_conditions"][0]["fixed_value"] = 2.0
    model = compile_candidate(
        CandidateModel.model_validate(payload),
        ValidationContext(targets=("v01",), external_inputs=("u01",)),
    )
    monkeypatch.setattr(
        adapter, "bounded_latent_start", lambda *a, **kw: {"success": False}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no parameter fit exists to score")

    monkeypatch.setattr(adapter, "evaluate_fitted_candidate", forbidden)
    result = fit_collocation_forward_sensitivity(
        model,
        _split(SplitName.TRAIN, (1.0,)),
        _split(SplitName.VALIDATION, (0.6,)),
        CollocationSensitivityConfig(initializer_seconds=5, refinement_seconds=15),
        tmp_path,
        initial_parameters={"rate": 0.7, "gain": 1.3},
    )
    assert result["status"] == "fit_failed"
    assert result["parameters"] is None and result["validation"] is None
    assert result["refinement"]["optimizer_native_success"]
    assert not result["refinement"]["optimizer_success"]
    assert result["refinement"]["numerical_status"] == "no_feasible_rollout"


def test_adapter_uses_each_trajectory_direct_observation_initial_value(
    tmp_path,
) -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"][0]["name"] = "v01"
    payload["state_equations"][0].update(state="v01", rhs="-rate*v01 + gain*u01")
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
