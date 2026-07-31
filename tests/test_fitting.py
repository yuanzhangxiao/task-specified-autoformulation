"""Synthetic recovery and numerical-failure tests for Milestone 5."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import (
    FitConfig,
    evaluate_fitted_candidate,
    fit_candidate,
    simulate_trajectory,
)
from autoformalism.schemas import CandidateModel


def _candidate(
    equations: Mapping[str, str],
    observation_state: str,
    parameters: tuple[tuple[str, float, float], ...],
    *,
    latent_local_state: str | None = None,
) -> CandidateModel:
    states = tuple(equations)
    return CandidateModel.model_validate(
        {
            "candidate_id": "synthetic_recovery",
            "parent_candidate_id": None,
            "change_summary": "Synthetic recovery model.",
            "states": [
                {
                    "name": state,
                    "kind": (
                        "latent" if state == latent_local_state else "observed"
                    ),
                    "unit": "unit",
                    "description": f"State {state}.",
                }
                for state in states
            ],
            "state_equations": [
                {"state": state, "rhs": rhs} for state, rhs in equations.items()
            ],
            "observation_mappings": [
                {
                    "channel": "target",
                    "expression": observation_state,
                    "unit": "unit",
                }
            ],
            "parameters": [
                {
                    "name": name,
                    "scope": "global",
                    "bounds": {"lower": lower, "upper": upper},
                    "initialization_range": {"lower": lower, "upper": upper},
                    "unit": "unit",
                    "description": f"Parameter {name}.",
                }
                for name, lower, upper in parameters
            ],
            "initial_conditions": [
                {
                    "state": state,
                    "scope": (
                        "trajectory_specific"
                        if state == latent_local_state
                        else "global"
                    ),
                    "initialization_range": {
                        "lower": 0.05,
                        "upper": 3.0,
                    },
                }
                for state in states
            ],
        }
    )


def _trajectory(identifier: str, time: np.ndarray, target: np.ndarray) -> Trajectory:
    return Trajectory(
        trajectory_id=identifier,
        time=time.copy(),
        targets={"target": target.copy()},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )


def _split(
    name: SplitName,
    trajectories: tuple[Trajectory, ...],
) -> DatasetSplit:
    return DatasetSplit(name, trajectories, f"{name.value}-fingerprint")


def _config(seed: int = 13) -> FitConfig:
    return FitConfig(
        number_of_starts=5,
        random_seed=seed,
        maximum_function_evaluations=1_500,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
    )


def test_recovers_linear_one_state_ode_and_is_reproducible() -> None:
    time = np.linspace(0.0, 3.0, 31)
    decay = 0.72
    initial = 1.8
    target = initial * np.exp(-decay * time)
    model = compile_candidate(
        _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.5),)),
        ValidationContext(targets=("target",)),
    )
    train = _split(
        SplitName.TRAIN, (_trajectory("train-1", time, target),)
    )
    validation = _split(
        SplitName.VALIDATION, (_trajectory("val-1", time, target),)
    )

    first = fit_candidate(model, train, validation, _config())
    second = fit_candidate(model, train, validation, _config())

    assert first.success
    assert first.global_parameters["decay"] == pytest.approx(decay, abs=2e-4)
    assert first.global_initial_conditions["x"] == pytest.approx(initial, abs=2e-4)
    assert first.training_metrics.normalized_mse < 1e-10
    assert first.validation_metrics.normalized_mse < 1e-10
    assert first.target_scales["target"] == pytest.approx(float(np.std(target)))
    assert first.global_parameters == second.global_parameters
    assert first.best_start_index == second.best_start_index


def test_recovers_nonlinear_exponential_internal_parameter() -> None:
    time = np.linspace(0.0, 2.0, 41)
    rate, exponent, initial = 0.35, 0.65, 1.4
    target = -np.log(
        np.exp(-exponent * initial) + exponent * rate * time
    ) / exponent
    model = compile_candidate(
        _candidate(
            {"x": "-rate * exp(exponent * x)"},
            "x",
            (("rate", 0.1, 0.8), ("exponent", 0.2, 1.2)),
        ),
        ValidationContext(targets=("target",)),
    )
    train = _split(
        SplitName.TRAIN, (_trajectory("train-1", time, target),)
    )
    validation = _split(
        SplitName.VALIDATION, (_trajectory("val-1", time, target),)
    )

    result = fit_candidate(model, train, validation, _config(8))

    assert result.success
    assert result.global_parameters["rate"] == pytest.approx(rate, abs=3e-3)
    assert result.global_parameters["exponent"] == pytest.approx(
        exponent, abs=8e-3
    )
    assert result.global_initial_conditions["x"] == pytest.approx(
        initial, abs=2e-3
    )
    assert result.validation_metrics.normalized_mse < 1e-7


def test_recovers_latent_system_and_trajectory_specific_initial_states() -> None:
    time = np.linspace(0.0, 4.0, 41)
    latent_decay, observed_decay, observed_initial = 0.45, 0.9, 0.4

    def target(latent_initial: float) -> np.ndarray:
        return (
            observed_initial * np.exp(-observed_decay * time)
            + latent_initial
            * (
                np.exp(-latent_decay * time)
                - np.exp(-observed_decay * time)
            )
            / (observed_decay - latent_decay)
        )

    model = compile_candidate(
        _candidate(
            {"z": "-latent_decay * z", "y": "z - observed_decay * y"},
            "y",
            (
                ("latent_decay", 0.2, 0.7),
                ("observed_decay", 0.7, 1.2),
            ),
            latent_local_state="z",
        ),
        ValidationContext(targets=("target",)),
    )
    train_initials = {"train-1": 0.8, "train-2": 1.5, "train-3": 2.2}
    validation_initials = {"val-1": 1.1, "val-2": 1.9}
    train = _split(
        SplitName.TRAIN,
        tuple(
            _trajectory(identifier, time, target(value))
            for identifier, value in train_initials.items()
        ),
    )
    validation = _split(
        SplitName.VALIDATION,
        tuple(
            _trajectory(identifier, time, target(value))
            for identifier, value in validation_initials.items()
        ),
    )

    result = fit_candidate(model, train, validation, _config(21))

    assert result.success
    assert result.global_parameters["latent_decay"] == pytest.approx(
        latent_decay, abs=3e-3
    )
    assert result.global_parameters["observed_decay"] == pytest.approx(
        observed_decay, abs=3e-3
    )
    assert result.global_initial_conditions["y"] == pytest.approx(
        observed_initial, abs=2e-3
    )
    for identifier, expected in train_initials.items():
        assert result.training_trajectory_initial_conditions[identifier][
            "z"
        ] == pytest.approx(expected, abs=4e-3)
    for identifier, expected in validation_initials.items():
        assert result.validation_trajectory_initial_conditions[identifier][
            "z"
        ] == pytest.approx(expected, abs=4e-3)
    assert result.validation_metrics.normalized_mse < 1e-7


def test_simulation_returns_failure_for_missing_forcing() -> None:
    candidate = _candidate({"x": "-decay * x + u"}, "x", (("decay", 0.1, 1.0),))
    model = compile_candidate(
        candidate,
        ValidationContext(targets=("target",), external_inputs=("u",)),
    )
    trajectory = _trajectory(
        "missing-input", np.asarray([0.0, 1.0]), np.asarray([1.0, 0.5])
    )

    result = simulate_trajectory(
        model,
        trajectory,
        {"decay": 0.5},
        {"x": 1.0},
        FitConfig(),
    )

    assert not result.success
    assert "missing forcing channel u" in (result.message or "")


def test_rejects_trajectory_specific_model_parameter() -> None:
    payload = _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),))
    changed = payload.model_dump(mode="json")
    changed["parameters"][0]["scope"] = "trajectory_specific"
    model = compile_candidate(
        CandidateModel.model_validate(changed),
        ValidationContext(targets=("target",)),
    )
    time = np.asarray([0.0, 1.0])
    train = _split(
        SplitName.TRAIN, (_trajectory("train", time, np.asarray([1.0, 0.5])),)
    )
    validation = _split(
        SplitName.VALIDATION,
        (_trajectory("validation", time, np.asarray([1.0, 0.5])),),
    )

    with pytest.raises(ValueError, match="trajectory-specific model parameters"):
        fit_candidate(model, train, validation)


def test_target_free_evaluation_uses_midpoint_latent_initial() -> None:
    model = compile_candidate(
        _candidate(
            {"z": "0", "y": "z"},
            "y",
            (),
            latent_local_state="z",
        ),
        ValidationContext(targets=("target",)),
    )
    time = np.asarray([0.0, 1.0])
    first = _split(
        SplitName.TEST,
        (_trajectory("test", time, np.asarray([0.0, 0.0])),),
    )
    second = _split(
        SplitName.TEST,
        (_trajectory("test", time, np.asarray([1000.0, -1000.0])),),
    )

    first_initials, _ = evaluate_fitted_candidate(
        model,
        first,
        global_parameters={},
        global_initial_conditions={"y": 1.0},
        target_scales={"target": 1.0},
        fit_trajectory_initial_conditions=False,
    )
    second_initials, _ = evaluate_fitted_candidate(
        model,
        second,
        global_parameters={},
        global_initial_conditions={"y": 1.0},
        target_scales={"target": 1.0},
        fit_trajectory_initial_conditions=False,
    )

    assert first_initials == second_initials
    assert first_initials["test"]["z"] == pytest.approx(1.525)
