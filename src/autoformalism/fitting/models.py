"""Typed results and configuration for numerical fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class FitConfig(BaseModel):
    """Reproducible numerical integration and optimization controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number_of_starts: int = Field(default=1, ge=1)
    random_seed: int = 0
    integration_backend: Literal["fixed_rk4", "solve_ivp"] = "solve_ivp"
    allow_derivative_regression: bool = True
    integration_method: str = "RK45"
    fixed_step_substeps: int = Field(default=1, ge=1)
    relative_tolerance: float = Field(default=1e-7, gt=0.0)
    absolute_tolerance: float = Field(default=1e-9, gt=0.0)
    maximum_function_evaluations: int = Field(default=50, ge=1)
    maximum_wall_time_seconds: float | None = Field(default=None, gt=0.0)
    failure_penalty: float = Field(default=1e6, gt=0.0)
    bound_tolerance: float = Field(default=1e-5, gt=0.0)
    soft_constraint_penalty_weight: float = Field(default=1.0, ge=0.0)


@dataclass(frozen=True)
class SimulationResult:
    """One trajectory rollout or a recoverable numerical failure."""

    success: bool
    time: NDArray[np.float64]
    states: NDArray[np.float64] | None
    predictions: Mapping[str, NDArray[np.float64]]
    message: str | None = None

    def __post_init__(self) -> None:
        self.time.setflags(write=False)
        if self.states is not None:
            self.states.setflags(write=False)
        for values in self.predictions.values():
            values.setflags(write=False)
        object.__setattr__(
            self, "predictions", MappingProxyType(dict(self.predictions))
        )


@dataclass(frozen=True)
class EvaluationMetrics:
    """Normalized rollout error over a complete split."""

    normalized_mse: float
    per_target_normalized_mse: Mapping[str, float]
    failed_trajectories: tuple[str, ...] = ()
    soft_constraint_violations: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_target_normalized_mse",
            MappingProxyType(dict(self.per_target_normalized_mse)),
        )
        object.__setattr__(
            self,
            "soft_constraint_violations",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(value))
                    for key, value in self.soft_constraint_violations.items()
                }
            ),
        )


@dataclass(frozen=True)
class OptimizationDiagnostic:
    """Optimizer outcome and proximity to declared parameter bounds."""

    start_index: int
    success: bool
    status: int
    message: str
    cost: float
    function_evaluations: int
    integration_failures: int
    backend: str = "rollout_least_squares"
    integration_failure_messages: tuple[str, ...] = ()
    parameters_at_lower_bound: tuple[str, ...] = ()
    parameters_at_upper_bound: tuple[str, ...] = ()


@dataclass(frozen=True)
class FitResult:
    """Best bounded fit and train/validation rollout evaluation."""

    success: bool
    global_parameters: Mapping[str, float]
    global_initial_conditions: Mapping[str, float]
    training_trajectory_initial_conditions: Mapping[str, Mapping[str, float]]
    validation_trajectory_initial_conditions: Mapping[str, Mapping[str, float]]
    training_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    diagnostics: tuple[OptimizationDiagnostic, ...]
    best_start_index: int
    target_scales: Mapping[str, float]
    message: str | None = None

    def __post_init__(self) -> None:
        for attribute in (
            "global_parameters",
            "global_initial_conditions",
            "target_scales",
        ):
            object.__setattr__(
                self, attribute, MappingProxyType(dict(getattr(self, attribute)))
            )
        for attribute in (
            "training_trajectory_initial_conditions",
            "validation_trajectory_initial_conditions",
        ):
            nested = {
                key: MappingProxyType(dict(value))
                for key, value in getattr(self, attribute).items()
            }
            object.__setattr__(self, attribute, MappingProxyType(nested))


@dataclass
class FailureCounter:
    """Mutable internal counter shared by one optimizer residual function."""

    count: int = field(default=0)
    messages: list[str] = field(default_factory=list)

    def record(self, message: str | None) -> None:
        """Record a failure count and a bounded set of representative causes."""
        self.count += 1
        rendered = message or "simulation failed without a diagnostic"
        if rendered not in self.messages and len(self.messages) < 5:
            self.messages.append(rendered)
