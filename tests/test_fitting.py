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


def test_fixed_initial_condition_is_not_sent_to_optimizer() -> None:
    time = np.linspace(0.0, 3.0, 31)
    decay = 0.72
    initial = 1.8
    target = initial * np.exp(-decay * time)
    payload = _candidate(
        {"x": "-decay * x"}, "x", (("decay", 0.1, 1.5),)
    ).model_dump(mode="json")
    payload["initial_conditions"][0]["initialization_range"] = {
        "lower": initial,
        "upper": initial,
    }
    model = compile_candidate(
        CandidateModel.model_validate(payload),
        ValidationContext(targets=("target",)),
    )
    train = _split(SplitName.TRAIN, (_trajectory("train-1", time, target),))
    validation = _split(
        SplitName.VALIDATION,
        (_trajectory("val-1", time, target),),
    )

    result = fit_candidate(model, train, validation, _config())

    assert result.success
    assert result.global_initial_conditions["x"] == initial
    assert all(
        "initial:x" not in diagnostic.parameters_at_lower_bound
        for diagnostic in result.diagnostics
    )


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

    train = _split(SplitName.TRAIN, (trajectory,))
    validation = _split(
        SplitName.VALIDATION,
        (
            _trajectory(
                "missing-input-val",
                trajectory.time,
                trajectory.targets["target"],
            ),
        ),
    )
    fit = fit_candidate(
        model,
        train,
        validation,
        FitConfig(number_of_starts=1, maximum_function_evaluations=2),
    )

    assert not fit.success
    assert fit.diagnostics[0].integration_failure_messages == (
        "trajectory is missing forcing channel u",
    )


def test_finite_fit_at_evaluation_budget_remains_eligible() -> None:
    candidate = _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),))
    model = compile_candidate(candidate, ValidationContext(targets=("target",)))
    time = np.linspace(0.0, 1.0, 11)
    trajectory = _trajectory("finite-budget", time, np.exp(-0.4 * time))

    fit = fit_candidate(
        model,
        _split(SplitName.TRAIN, (trajectory,)),
        _split(SplitName.VALIDATION, (trajectory,)),
        FitConfig(number_of_starts=1, maximum_function_evaluations=1),
    )

    assert fit.success
    assert fit.diagnostics[0].status == 0
    assert "budget reached" in (fit.message or "")


def test_parameter_warm_start_is_used_for_first_optimizer_start() -> None:
    candidate = _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),))
    model = compile_candidate(candidate, ValidationContext(targets=("target",)))
    time = np.linspace(0.0, 1.0, 11)
    trajectory = _trajectory("warm-start", time, np.exp(-0.4 * time))

    fit = fit_candidate(
        model,
        _split(SplitName.TRAIN, (trajectory,)),
        _split(SplitName.VALIDATION, (trajectory,)),
        FitConfig(number_of_starts=1, maximum_function_evaluations=1),
        initial_global_parameters={"decay": 0.4},
    )

    assert fit.success
    assert fit.global_parameters["decay"] == pytest.approx(0.4)


def test_parameter_warm_start_rejects_unknown_names() -> None:
    candidate = _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),))
    model = compile_candidate(candidate, ValidationContext(targets=("target",)))
    time = np.linspace(0.0, 1.0, 5)
    trajectory = _trajectory("unknown-warm-start", time, np.exp(-0.4 * time))

    with pytest.raises(ValueError, match="unknown parameters"):
        fit_candidate(
            model,
            _split(SplitName.TRAIN, (trajectory,)),
            _split(SplitName.VALIDATION, (trajectory,)),
            initial_global_parameters={"invented": 1.0},
        )


def test_lagged_target_forcing_is_strictly_one_step_causal() -> None:
    candidate = _candidate({"x": "target"}, "x", ())
    context = ValidationContext(
        targets=("target",), lagged_targets=("target",)
    )
    model = compile_candidate(candidate, context)
    time = np.asarray([0.0, 1.0, 2.0])
    first = _trajectory("first", time, np.asarray([1.0, 3.0, 10.0]))
    changed_future = _trajectory(
        "changed", time, np.asarray([1.0, 3.0, 4000.0])
    )

    first_result = simulate_trajectory(
        model, first, {}, {"x": 0.0}, FitConfig()
    )
    changed_result = simulate_trajectory(
        model, changed_future, {}, {"x": 0.0}, FitConfig()
    )

    assert first_result.success
    assert changed_result.success
    assert first_result.states is not None
    assert changed_result.states is not None
    assert first_result.states[0] == pytest.approx([1.0, 2.0, 6.0])
    assert changed_result.states[0] == pytest.approx(first_result.states[0])


def test_one_step_reset_observed_state_but_propagates_latent_state() -> None:
    candidate = _candidate(
        {"z": "1", "x": "z"},
        "x",
        (),
        latent_local_state="z",
    )
    payload = candidate.model_dump(mode="json")
    payload["initial_conditions"][0].pop("initialization_range")
    payload["initial_conditions"][0]["expression"] = "0.1 * target"
    candidate = CandidateModel.model_validate(payload)
    model = compile_candidate(
        candidate,
        ValidationContext(targets=("target",), lagged_targets=("target",)),
    )
    trajectory = _trajectory(
        "rolling", np.asarray([0.0, 1.0, 2.0]), np.asarray([10.0, 20.0, 30.0])
    )

    result = simulate_trajectory(model, trajectory, {}, {}, FitConfig())

    assert result.success
    assert result.states is not None
    assert result.states[0] == pytest.approx([1.0, 2.0, 3.0])
    assert result.states[1] == pytest.approx([10.0, 11.5, 22.5])


def test_fixed_rk4_advances_one_step_without_adaptive_solver() -> None:
    candidate = _candidate({"x": "1"}, "x", ())
    model = compile_candidate(
        candidate,
        ValidationContext(targets=("target",), lagged_targets=("target",)),
    )
    trajectory = _trajectory(
        "fixed-rk4",
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0, 2.5, 3.0]),
    )

    result = simulate_trajectory(
        model,
        trajectory,
        {},
        {},
        FitConfig(integration_backend="fixed_rk4"),
    )

    assert result.success
    assert result.states is not None
    assert result.states[0] == pytest.approx([2.0, 2.5, 3.0])


def test_fit_timeout_is_a_recoverable_failed_result() -> None:
    time = np.linspace(0.0, 1.0, 11)
    target = np.exp(-0.4 * time)
    model = compile_candidate(
        _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),)),
        ValidationContext(targets=("target",), lagged_targets=("target",)),
    )
    trajectory = _trajectory("timeout", time, target)

    fit = fit_candidate(
        model,
        _split(SplitName.TRAIN, (trajectory,)),
        _split(SplitName.VALIDATION, (trajectory,)),
        FitConfig(
            integration_backend="fixed_rk4",
            maximum_wall_time_seconds=1e-12,
        ),
    )

    assert not fit.success
    assert fit.diagnostics[0].status == -2
    assert "wall-clock limit" in (fit.message or "")
    assert np.isfinite(fit.training_metrics.normalized_mse)


def test_one_step_causally_initializes_observed_channel_rate_state() -> None:
    candidate = _candidate(
        {"y": "y_rate", "y_rate": "0"},
        "y",
        (),
        latent_local_state="y_rate",
    )
    payload = candidate.model_dump(mode="json")
    payload["initial_conditions"] = [
        item
        for item in payload["initial_conditions"]
        if item["state"] != "y_rate"
    ]
    model = compile_candidate(
        CandidateModel.model_validate(payload),
        ValidationContext(targets=("target",), lagged_targets=("target",)),
    )
    trajectory = _trajectory(
        "rate", np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 3.0, 6.0])
    )

    result = simulate_trajectory(model, trajectory, {}, {}, FitConfig())

    assert result.success
    assert result.states is not None
    assert model.validated.causal_derivative_initials == {"y_rate": "y"}
    assert result.states[0] == pytest.approx([1.0, 1.0, 5.0])
    assert result.states[1] == pytest.approx([0.0, 0.0, 2.0])


def test_one_step_fit_does_not_optimize_observed_state_initial_value() -> None:
    time = np.linspace(0.0, 2.0, 21)
    target = 1.7 * np.exp(-0.6 * time)
    model = compile_candidate(
        _candidate({"x": "-decay * x"}, "x", (("decay", 0.1, 1.0),)),
        ValidationContext(targets=("target",), lagged_targets=("target",)),
    )
    def with_derivative(identifier: str) -> Trajectory:
        return Trajectory(
            identifier,
            time.copy(),
            {"target": target.copy()},
            {},
            {},
            {},
            {"target": -0.6 * target.copy()},
        )

    train = _split(SplitName.TRAIN, (with_derivative("train"),))
    validation = _split(
        SplitName.VALIDATION, (with_derivative("validation"),)
    )

    fit = fit_candidate(model, train, validation, _config())

    assert fit.success
    assert fit.global_initial_conditions == {}
    assert fit.global_parameters["decay"] == pytest.approx(0.6, abs=2e-4)
    assert {item.backend for item in fit.diagnostics} == {
        "derivative_regression"
    }
    assert all(
        "initial:x" not in diagnostic.parameters_at_lower_bound
        and "initial:x" not in diagnostic.parameters_at_upper_bound
        for diagnostic in fit.diagnostics
    )


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
