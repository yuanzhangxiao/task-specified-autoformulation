"""Shared derivative-regression tables and causal rollout evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from autoformalism.data import (
    BenchmarkSpec,
    DatasetSplit,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.execution import (
    _development_forcing_bounds,
    _numeric_declared_channels,
)
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import EvaluationMetrics, evaluate_fitted_candidate
from autoformalism.schemas import CandidateModel


def baseline_validation_context(
    dataset: DevelopmentDataset,
    spec: BenchmarkSpec,
) -> ValidationContext:
    """Build the leakage-safe numeric context shared by baseline stages."""
    forcing_bounds = _development_forcing_bounds(
        dataset,
        include_targets=spec.one_step_target_history,
    )
    return ValidationContext(
        targets=dataset.roles.targets,
        auxiliaries=dataset.roles.auxiliaries,
        external_inputs=_numeric_declared_channels(
            spec.external_inputs,
            forcing_bounds,
        ),
        fixed_covariates=_numeric_declared_channels(
            spec.fixed_covariates,
            forcing_bounds,
        ),
        lagged_targets=(
            dataset.roles.targets if spec.one_step_target_history else ()
        ),
        forcing_bounds=forcing_bounds,
    )


def combine_development_splits(
    train: DatasetSplit,
    validation: DatasetSplit,
) -> DatasetSplit:
    """Combine public train and validation after hyperparameter selection."""
    if train.name is not SplitName.TRAIN:
        raise ValueError("combined development data requires the train split")
    if validation.name is not SplitName.VALIDATION:
        raise ValueError("combined development data requires the validation split")
    return DatasetSplit(
        SplitName.TRAIN,
        (*train.trajectories, *validation.trajectories),
        f"{train.fingerprint}+{validation.fingerprint}",
    )


def feature_names(context: ValidationContext) -> tuple[str, ...]:
    """Return current, causally available regression variables without lags."""
    return tuple(
        dict.fromkeys(
            (*context.targets, *context.auxiliaries, *context.external_inputs,
             *context.fixed_covariates)
        )
    )


def numeric_feature_names(
    split: DatasetSplit, context: ValidationContext
) -> tuple[str, ...]:
    """Return regression features that are numeric on every trajectory.

    Manifests may expose structured fixed covariates, such as a JSON meal
    schedule, for task context.  Derivative-regression baselines cannot use
    those objects as scalar columns, so they are omitted rather than coerced.
    """
    names = feature_names(context)
    numeric: list[str] = []
    for name in names:
        try:
            values = [
                _channel_value(trajectory, name, 0)
                for trajectory in split.trajectories
            ]
        except (TypeError, ValueError):
            continue
        if np.isfinite(np.asarray(values, dtype=float)).all():
            numeric.append(name)
    return tuple(numeric)


def regression_table(
    split: DatasetSplit,
    names: Sequence[str],
    targets: Sequence[str],
    extra_features: Mapping[str, Callable[[Trajectory, int], float]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Build rows within trajectories using supplied derivative labels."""
    extras = extra_features or {}
    rows: list[list[float]] = []
    labels: list[list[float]] = []
    for trajectory in split.trajectories:
        for index in range(trajectory.number_of_rows):
            rows.append(
                [_channel_value(trajectory, name, index) for name in names]
                + [float(function(trajectory, index)) for function in extras.values()]
            )
            labels.append(
                [float(trajectory.derivatives[target][index]) for target in targets]
            )
    x = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("baseline regression table contains nonfinite values")
    return x, y, (*names, *extras)


def _channel_value(trajectory: Trajectory, name: str, index: int) -> float:
    for mapping in (
        trajectory.targets,
        trajectory.auxiliaries,
        trajectory.external_inputs,
    ):
        if name in mapping:
            return float(mapping[name][index])
    if name in trajectory.fixed_covariates:
        return float(trajectory.fixed_covariates[name])
    raise ValueError(f"trajectory is missing baseline feature {name}")


def candidate_from_equations(
    equations: Mapping[str, str], context: ValidationContext, *, identifier: str
) -> CandidateModel:
    """Construct a parameter-free observed-state candidate for safe rollout."""
    return CandidateModel.model_validate(
        {
            "candidate_id": re.sub(r"[^A-Za-z0-9_]", "_", identifier),
            "parent_candidate_id": None,
            "change_summary": "Derivative-regression baseline.",
            "states": [
                {"name": target, "kind": "observed"} for target in context.targets
            ],
            "state_equations": [
                {"state": target, "rhs": equations[target]}
                for target in context.targets
            ],
            "observation_mappings": [
                {"channel": target, "expression": target}
                for target in context.targets
            ],
            "parameters": [],
            "initial_conditions": [
                {"state": target, "scope": "global", "expression": target}
                for target in context.targets
            ],
        }
    )


def target_scales(training: DatasetSplit, targets: Sequence[str]) -> dict[str, float]:
    """Fit target standard deviations on training data only."""
    return {
        target: max(
            float(
                np.std(
                    np.concatenate(
                        [
                            trajectory.targets[target]
                            for trajectory in training.trajectories
                        ]
                    )
                )
            ),
            1e-8,
        )
        for target in targets
    }


def evaluate_equations(
    equations: Mapping[str, str],
    context: ValidationContext,
    split: DatasetSplit,
    scales: Mapping[str, float],
    *,
    identifier: str,
) -> EvaluationMetrics:
    """Compile untrusted equations and evaluate causal one-step rollouts."""
    candidate = candidate_from_equations(equations, context, identifier=identifier)
    compiled = compile_candidate(candidate, context)
    _, metrics = evaluate_fitted_candidate(
        compiled,
        split,
        global_parameters={},
        global_initial_conditions={},
        target_scales=scales,
        fit_trajectory_initial_conditions=False,
    )
    return metrics


def persistence_metrics(
    split: DatasetSplit, scales: Mapping[str, float]
) -> EvaluationMetrics:
    """Evaluate y[i-1] as the causal prediction for y[i]."""
    per_target: dict[str, float] = {}
    for target, scale in scales.items():
        pieces = [
            (trajectory.targets[target][1:] - trajectory.targets[target][:-1])
            / scale
            for trajectory in split.trajectories
        ]
        per_target[target] = float(np.mean(np.concatenate(pieces) ** 2))
    return EvaluationMetrics(float(np.mean(list(per_target.values()))), per_target)
