"""Whole-term contribution pruning and support-selection tests."""

from __future__ import annotations

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig
from autoformalism.pruning import PruningConfig, prune_candidate
from autoformalism.schemas import CandidateModel


def _candidate(
    rhs: str,
    parameters: tuple[tuple[str, float, float], ...],
) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "pruning_model",
            "parent_candidate_id": None,
            "change_summary": "Synthetic pruning candidate.",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed state.",
                }
            ],
            "state_equations": [{"state": "x", "rhs": rhs}],
            "observation_mappings": [
                {"channel": "target", "expression": "x", "unit": "unit"}
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
                    "state": "x",
                    "scope": "global",
                    "initialization_range": {"lower": -0.1, "upper": 1.5},
                }
            ],
        }
    )


def _trajectory(
    identifier: str,
    time: np.ndarray,
    target: np.ndarray,
    inputs: dict[str, np.ndarray] | None = None,
) -> Trajectory:
    return Trajectory(
        trajectory_id=identifier,
        time=time.copy(),
        targets={"target": target.copy()},
        auxiliaries={},
        external_inputs={
            name: value.copy() for name, value in (inputs or {}).items()
        },
        fixed_covariates={},
        derivatives={},
    )


def _split(name: SplitName, trajectory: Trajectory) -> DatasetSplit:
    return DatasetSplit(name, (trajectory,), f"{name.value}-fingerprint")


def _fit_config(seed: int = 5) -> FitConfig:
    return FitConfig(
        number_of_starts=4,
        random_seed=seed,
        maximum_function_evaluations=1_000,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
    )


def test_prunes_complete_negligible_term_and_its_exclusive_parameter() -> None:
    time = np.linspace(0.0, 3.0, 31)
    target = np.exp(-0.7 * time)
    model = compile_candidate(
        _candidate(
            "-decay * x + offset",
            (("decay", 0.3, 1.1), ("offset", -0.2, 0.2)),
        ),
        ValidationContext(targets=("target",)),
    )
    train = _split(SplitName.TRAIN, _trajectory("train", time, target))
    validation = _split(
        SplitName.VALIDATION, _trajectory("validation", time, target)
    )

    result = prune_candidate(
        model,
        train,
        validation,
        fit_config=_fit_config(),
        pruning_config=PruningConfig(validation_mse_tolerance=1e-8),
    )

    assert result.selected_removed_terms == ("equation:x:term:1",)
    assert result.selected_removed_parameters == ("offset",)
    assert [item.name for item in result.selected_candidate.parameters] == ["decay"]
    assert result.selected_candidate.state_equations[0].rhs == "-decay * x"
    assert result.selected_fit.validation_metrics.normalized_mse < 1e-8
    assert all(candidate.retained_term_ids for candidate in result.candidates)
    assert result.persistence_validation_mse >= 0.0
    assert len(result.candidates) <= 2


def test_raw_coefficient_magnitude_does_not_measure_term_contribution() -> None:
    time = np.linspace(0.0, 2.0, 41)
    large_input = 100.0 * np.cos(time)
    small_input = 0.02 * time
    small_coefficient = 0.01
    large_coefficient = 1.0
    derivative = (
        small_coefficient * large_input + large_coefficient * small_input
    )
    target = np.concatenate(
        (
            np.asarray([0.0]),
            np.cumsum(
                np.diff(time) * (derivative[:-1] + derivative[1:]) / 2.0
            ),
        )
    )
    inputs = {"large_input": large_input, "small_input": small_input}
    model = compile_candidate(
        _candidate(
            "small_coefficient * large_input + large_coefficient * small_input",
            (
                ("small_coefficient", 0.005, 0.02),
                ("large_coefficient", 0.5, 1.5),
            ),
        ),
        ValidationContext(
            targets=("target",),
            external_inputs=("large_input", "small_input"),
        ),
    )
    train = _split(
        SplitName.TRAIN, _trajectory("train", time, target, inputs)
    )
    validation = _split(
        SplitName.VALIDATION,
        _trajectory("validation", time, target, inputs),
    )

    result = prune_candidate(
        model,
        train,
        validation,
        fit_config=_fit_config(9),
        pruning_config=PruningConfig(validation_mse_tolerance=1e-10),
    )
    contributions = {
        item.expression: item.normalized_rms for item in result.contributions
    }

    fitted = result.unpruned_fit.global_parameters
    assert fitted["large_coefficient"] > 20.0 * fitted["small_coefficient"]
    assert (
        contributions["small_coefficient * large_input"]
        > 20.0 * contributions["large_coefficient * small_input"]
    )
    assert not result.selected_removed_terms


def test_parameters_inside_nonlinear_term_are_pruned_as_one_unit() -> None:
    time = np.linspace(0.0, 1.0, 21)
    rate, shape, initial = 0.2, 0.5, 1.0
    target = -np.log(
        np.exp(-shape * initial) + shape * rate * time
    ) / shape
    model = compile_candidate(
        _candidate(
            "-rate * exp(shape * x)",
            (("rate", 0.1, 0.4), ("shape", 0.2, 0.8)),
        ),
        ValidationContext(targets=("target",)),
    )
    train = _split(SplitName.TRAIN, _trajectory("train", time, target))
    validation = _split(
        SplitName.VALIDATION, _trajectory("validation", time, target)
    )

    result = prune_candidate(
        model,
        train,
        validation,
        fit_config=_fit_config(17),
        pruning_config=PruningConfig(validation_mse_tolerance=1e-10),
    )

    assert len(result.contributions) == 1
    assert result.contributions[0].parameters == ("rate", "shape")
    removed_parameter_sets = {
        candidate.removed_parameters
        for candidate in result.candidates
        if candidate.removed_term_ids
    }
    assert removed_parameter_sets <= {("rate", "shape")}


def test_pruning_never_removes_external_input_or_all_target_dynamics() -> None:
    time = np.linspace(0.0, 2.0, 21)
    pulse = np.zeros_like(time)
    target = np.ones_like(time)
    model = compile_candidate(
        _candidate(
            "gain * pulse + offset",
            (("gain", 0.1, 2.0), ("offset", -0.1, 0.1)),
        ),
        ValidationContext(targets=("target",), external_inputs=("pulse",)),
    )
    data = _trajectory("trajectory", time, target, {"pulse": pulse})

    result = prune_candidate(
        model,
        _split(SplitName.TRAIN, data),
        _split(SplitName.VALIDATION, data),
        fit_config=_fit_config(21),
    )

    assert "equation:x:term:0" not in result.selected_removed_terms
    assert result.selected_candidate.state_equations[0].rhs != "0"
    assert all(candidate.retained_term_ids for candidate in result.candidates)


def test_threshold_generation_is_capped() -> None:
    time = np.linspace(0.0, 1.0, 21)
    target = np.exp(-0.5 * time)
    model = compile_candidate(
        _candidate("-decay * x", (("decay", 0.1, 1.0),)),
        ValidationContext(targets=("target",)),
    )
    data = _trajectory("trajectory", time, target)
    result = prune_candidate(
        model,
        _split(SplitName.TRAIN, data),
        _split(SplitName.VALIDATION, data),
        fit_config=_fit_config(22),
        pruning_config=PruningConfig(maximum_normalized_contribution=0.01),
    )

    assert all(threshold <= 0.01 for threshold in result.thresholds)
    assert not result.selected_removed_terms
