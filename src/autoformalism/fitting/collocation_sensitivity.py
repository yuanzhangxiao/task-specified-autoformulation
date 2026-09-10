"""Opt-in collocation initialization plus forward-sensitivity refinement.

This adapter transfers the verified synthetic fitter route to eligible frozen
single-target candidates.  It remains separate from the production default:
unsupported expression graphs fail closed instead of silently falling back to
finite differences.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler
from autoformalism.expressions import CompiledModel
from autoformalism.fitting.fitter import evaluate_fitted_candidate
from autoformalism.fitting.matching_probe import bounded_latent_start
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import (
    SensitivityContractError,
    SymbolicODE,
    SymbolicOracle,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.schemas import CandidateModel, ParameterDomain, ParameterRole
from autoformalism.schemas.base import StrictSchema


class CollocationSensitivityConfig(StrictSchema):
    """Frozen numerical budgets for one train-only fit and causal replay."""

    protocol: Literal["collocation-forward-sensitivity-1"] = (
        "collocation-forward-sensitivity-1"
    )
    initializer_seconds: float = Field(default=120.0, gt=0.0, le=300.0)
    refinement_seconds: float = Field(default=360.0, gt=0.0, le=900.0)
    maximum_function_evaluations: int = Field(default=80, ge=1, le=300)
    integration_method: Literal["Radau", "BDF"] = "Radau"
    relative_tolerance: float = Field(default=1e-7, gt=0.0)
    absolute_tolerance: float = Field(default=1e-9, gt=0.0)
    failure_penalty: float = Field(default=1e6, gt=0.0)

    def fit_config(self) -> FitConfig:
        """Build the common integration and runtime-domain policy."""
        return FitConfig(
            number_of_starts=1,
            integration_backend="solve_ivp",
            allow_derivative_regression=False,
            parameter_fit_strategy="bounded_nonlinear",
            integration_method=self.integration_method,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            maximum_function_evaluations=self.maximum_function_evaluations,
            maximum_wall_time_seconds=self.refinement_seconds,
            failure_penalty=self.failure_penalty,
        )


def fit_collocation_forward_sensitivity(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    config: CollocationSensitivityConfig,
    directory: Path,
    *,
    initial_parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit on training by collocation then exact forward sensitivities.

    Validation is opened only after the training optimizer has returned a
    parameter vector.  Collocation node states are discarded; all reported
    metrics are fresh causal rollouts from the candidate's fixed initials.
    """
    if training.name is not SplitName.TRAIN:
        raise SensitivityContractError(
            "collocation-sensitivity fitting requires training data"
        )
    if validation.name is not SplitName.VALIDATION:
        raise SensitivityContractError(
            "collocation-sensitivity scoring requires validation data"
        )
    if tuple(model.validated.context.targets) != ("v01",):
        raise SensitivityContractError(
            "current transfer adapter requires the single target v01"
        )
    system = SymbolicODE(model)
    settings = config.fit_config()
    scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
    if not np.isfinite(scale) or scale <= 0.0:
        raise SensitivityContractError(
            "training target scale must be positive and finite"
        )
    directory.mkdir(parents=True, exist_ok=True)
    layout = RolloutOracle(
        model,
        training,
        {"v01": scale},
        settings,
        directory / "layout",
        None,
    )
    start = dict(initial_parameters or _role_start(model.validated.candidate, training))
    theta = layout.vector(start)
    initializer = bounded_latent_start(
        system,
        training=training,
        lower=layout.lower,
        upper=layout.upper,
        start=theta,
        scale=scale,
        settings=settings,
        method="collocation_init",
        seconds=config.initializer_seconds,
        directory=directory / "collocation",
    )
    selected = initializer.get("parameters") if initializer.get("success") else start
    deadline = monotonic() + config.refinement_seconds
    oracle = SymbolicOracle(
        system,
        training,
        scale,
        settings,
        directory / "sensitivity_calls",
        deadline,
        sensitivities=True,
    )

    def optimizer(fun: Any, x: np.ndarray, **kwargs: Any) -> Any:
        kwargs["jac"] = oracle.jacobian
        return least_squares(fun, x, **kwargs)

    refinement = instrumented_fit(
        oracle,
        selected,
        diff_step=None,
        max_nfev=config.maximum_function_evaluations,
        settings=settings,
        optimizer=optimizer,
    )
    parameters = refinement.get("parameters")
    if not isinstance(parameters, dict):
        return {
            "schema_version": "collocation-forward-sensitivity-fit-1",
            "status": "fit_failed",
            "initializer": initializer,
            "refinement": refinement,
            "parameters": None,
            "training": None,
            "validation": None,
            "training_only_parameter_estimation": True,
            "validation_used_for_fitting": False,
        }
    train_initials, train_metrics = evaluate_fitted_candidate(
        model,
        training,
        global_parameters=parameters,
        global_initial_conditions={},
        target_scales={"v01": scale},
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    validation_initials, validation_metrics = evaluate_fitted_candidate(
        model,
        validation,
        global_parameters=parameters,
        global_initial_conditions={},
        target_scales={"v01": scale},
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    complete = not train_metrics.failed_trajectories and not (
        validation_metrics.failed_trajectories
    )
    return {
        "schema_version": "collocation-forward-sensitivity-fit-1",
        "status": "complete" if complete else "rollout_failed",
        "initializer": initializer,
        "refinement": refinement,
        "parameters": parameters,
        "training": _metrics_payload(train_metrics, train_initials),
        "validation": _metrics_payload(validation_metrics, validation_initials),
        "training_only_parameter_estimation": True,
        "validation_used_for_fitting": False,
        "collocation_states_used_for_final_score": False,
        "forward_sensitivity_jacobian_used": True,
    }


def _metrics_payload(metrics: Any, initials: Mapping[str, Mapping[str, float]]) -> dict:
    return {
        "normalized_mse": metrics.normalized_mse,
        "per_target_normalized_mse": dict(metrics.per_target_normalized_mse),
        "failed_trajectories": list(metrics.failed_trajectories),
        "trajectory_initial_conditions": {
            key: dict(value) for key, value in initials.items()
        },
    }


def _role_start(candidate: CandidateModel, training: DatasetSplit) -> dict[str, float]:
    """Choose finite runtime starts from roles and public training time scales."""
    span = max(
        float(np.median([row.time[-1] - row.time[0] for row in training.trajectories])),
        1e-6,
    )
    step = max(
        float(
            np.median([np.median(np.diff(row.time)) for row in training.trajectories])
        ),
        1e-6,
    )
    time_scale = max(step, span / 10.0)
    result: dict[str, float] = {}
    for parameter in candidate.parameters:
        if parameter.role is ParameterRole.TIME_CONSTANT:
            value = time_scale
        elif parameter.role is ParameterRole.RATE:
            value = 1.0 / time_scale
        elif parameter.role is ParameterRole.OFFSET:
            value = 0.0
        elif parameter.domain is ParameterDomain.REAL:
            value = 0.1
        else:
            value = 0.1
        if parameter.bounds is not None:
            value = min(max(value, parameter.bounds.lower), parameter.bounds.upper)
        if parameter.domain is ParameterDomain.POSITIVE:
            value = max(value, np.finfo(float).eps)
        elif parameter.domain is ParameterDomain.NONNEGATIVE:
            value = max(value, 0.0)
        result[parameter.name] = float(value)
    return result


__all__ = [
    "CollocationSensitivityConfig",
    "fit_collocation_forward_sensitivity",
]
