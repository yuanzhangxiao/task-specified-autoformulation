"""Frozen-model and sealed-test contracts for classical Phase-B baselines."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.baselines.core import (
    combine_development_splits,
    evaluate_equations,
    numeric_feature_names,
    persistence_metrics,
    regression_table,
    target_scales,
)
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.baselines.sindy import fit_sindy
from autoformalism.data import DatasetSplit, DevelopmentDataset, SplitName
from autoformalism.expressions import ValidationContext

ClassicalBaselineMethod = Literal["persistence", "sindy", "pysr"]


class FrozenBaselineModel(BaseModel):
    """A final public-data model fixed before opening the test split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-frozen-baseline-model-1"] = (
        "phase-b-frozen-baseline-model-1"
    )
    task_index: int = Field(ge=0)
    method: ClassicalBaselineMethod
    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    seed: int = Field(ge=0)
    equations: dict[str, str]
    equations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_hyperparameters: dict[str, float | int | str]
    development_training_normalized_mse: float = Field(ge=0.0)
    development_validation_normalized_mse: float = Field(ge=0.0)
    normalization_scales: dict[str, float]
    train_fingerprint: str = Field(min_length=1)
    validation_fingerprint: str = Field(min_length=1)
    source_development_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_development_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalization_protocol: Literal[
        "causal_previous_observation",
        "selected_equations_preserved",
        "selected_threshold_refit_on_train_plus_validation",
    ]
    derivative_provenance: Literal["not_used", "estimated_numpy_gradient"]
    selection_frozen: Literal[True] = True
    test_data_opened: Literal[False] = False
    private_reference_opened: Literal[False] = False

    @model_validator(mode="after")
    def content_matches_protocol(self) -> FrozenBaselineModel:
        """Bind equations and method labels to the declared finalization rule."""
        if not self.equations or any(
            not value.strip() for value in self.equations.values()
        ):
            raise ValueError("frozen baseline equations must be nonempty")
        if self.equations_sha256 != equations_sha256(self.equations):
            raise ValueError("frozen baseline equation hash differs")
        if set(self.normalization_scales) != set(self.equations) or any(
            not isfinite(value) or value <= 0.0
            for value in self.normalization_scales.values()
        ):
            raise ValueError("normalization scales must match the target equations")
        expected = {
            "persistence": (
                "causal_previous_observation",
                "not_used",
            ),
            "sindy": (
                "selected_threshold_refit_on_train_plus_validation",
                "estimated_numpy_gradient",
            ),
            "pysr": (
                "selected_equations_preserved",
                "estimated_numpy_gradient",
            ),
        }[self.method]
        if (self.finalization_protocol, self.derivative_provenance) != expected:
            raise ValueError("baseline method and finalization protocol differ")
        if self.method == "sindy" and "threshold" not in self.selected_hyperparameters:
            raise ValueError("frozen SINDy model requires its selected threshold")
        return self


class BaselinePredictiveTestResult(BaseModel):
    """One separate predictive endpoint opened after the model freeze."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-baseline-predictive-test-1"] = (
        "phase-b-baseline-predictive-test-1"
    )
    task_index: int = Field(ge=0)
    method: ClassicalBaselineMethod
    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    seed: int = Field(ge=0)
    frozen_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["available", "failed"]
    evaluation_protocol: Literal[
        "causal_previous_observation",
        "causal_one_step_observed_state_reset",
    ]
    normalized_mse: float | None = Field(default=None, ge=0.0)
    per_target_normalized_mse: dict[str, float] = Field(default_factory=dict)
    trajectory_count: int = Field(ge=0)
    successful_trajectory_count: int = Field(ge=0)
    failed_trajectories: tuple[str, ...] = ()
    error_type: str | None = None
    error: str | None = None
    test_data_opened_after_freeze: Literal[True] = True
    private_reference_opened: Literal[False] = False

    @model_validator(mode="after")
    def values_match_status(self) -> BaselinePredictiveTestResult:
        """Keep unavailable scores distinct from numerical values."""
        expected_protocol = (
            "causal_previous_observation"
            if self.method == "persistence"
            else "causal_one_step_observed_state_reset"
        )
        if self.evaluation_protocol != expected_protocol:
            raise ValueError("test protocol does not match the baseline method")
        if self.successful_trajectory_count > self.trajectory_count:
            raise ValueError("successful trajectory count exceeds total")
        if self.status == "available":
            if (
                self.normalized_mse is None
                or not self.per_target_normalized_mse
                or self.failed_trajectories
                or self.error_type is not None
                or self.error is not None
                or self.successful_trajectory_count != self.trajectory_count
            ):
                raise ValueError("available predictive result is inconsistent")
        elif (
            self.normalized_mse is not None
            or self.per_target_normalized_mse
            or not self.error_type
            or not self.error
        ):
            raise ValueError("failed predictive result requires diagnostics only")
        return self


def equations_sha256(equations: dict[str, str]) -> str:
    """Hash a target-equation mapping canonically."""
    raw = json.dumps(
        equations,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze_baseline_model(
    *,
    task_index: int,
    result: BaselineDevelopmentResult,
    development: DevelopmentDataset,
    context: ValidationContext,
    source_development_result_sha256: str,
    source_development_freeze_sha256: str,
) -> FrozenBaselineModel:
    """Apply the predeclared final-fit rule using public development data only."""
    identity = (
        result.benchmark_id,
        result.tier,
    )
    if identity != (development.benchmark_id, development.tier):
        raise ValueError("development result and dataset identity differ")
    if result.method not in {"persistence", "sindy", "pysr"}:
        raise ValueError(f"unsupported classical method: {result.method}")
    if set(result.equations) != set(context.targets):
        raise ValueError("development equations differ from public targets")
    equations = dict(result.equations)
    if result.method == "sindy":
        raw_threshold = result.selected_hyperparameters.get("threshold")
        if isinstance(raw_threshold, bool) or not isinstance(
            raw_threshold,
            int | float,
        ):
            raise ValueError("selected SINDy threshold is not numeric")
        threshold = float(raw_threshold)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("selected SINDy threshold must be finite and positive")
        names = numeric_feature_names(development.train, context)
        combined = combine_development_splits(
            development.train,
            development.validation,
        )
        design, derivatives, expanded_names = regression_table(
            combined,
            names,
            context.targets,
        )
        equations = dict(
            fit_sindy(
                design,
                derivatives,
                expanded_names,
                context.targets,
                threshold=threshold,
            ).equations
        )
        protocol = "selected_threshold_refit_on_train_plus_validation"
        provenance = "estimated_numpy_gradient"
    elif result.method == "pysr":
        protocol = "selected_equations_preserved"
        provenance = "estimated_numpy_gradient"
    else:
        protocol = "causal_previous_observation"
        provenance = "not_used"
    scales = target_scales(development.train, context.targets)
    return FrozenBaselineModel(
        task_index=task_index,
        method=result.method,
        benchmark_id=result.benchmark_id,
        tier=result.tier,
        seed=result.seed,
        equations=equations,
        equations_sha256=equations_sha256(equations),
        selected_hyperparameters=dict(result.selected_hyperparameters),
        development_training_normalized_mse=result.training_normalized_mse,
        development_validation_normalized_mse=result.validation_normalized_mse,
        normalization_scales=scales,
        train_fingerprint=development.train.fingerprint,
        validation_fingerprint=development.validation.fingerprint,
        source_development_result_sha256=source_development_result_sha256,
        source_development_freeze_sha256=source_development_freeze_sha256,
        finalization_protocol=protocol,
        derivative_provenance=provenance,
    )


def evaluate_frozen_baseline_predictively(
    model: FrozenBaselineModel,
    test: DatasetSplit,
    context: ValidationContext,
    *,
    frozen_model_sha256: str,
) -> BaselinePredictiveTestResult:
    """Evaluate one frozen classical model without changing its selection."""
    if test.name is not SplitName.TEST:
        raise ValueError("predictive post-freeze evaluation requires the test split")
    if set(context.targets) != set(model.equations):
        raise ValueError("public context targets differ from the frozen model")
    trajectory_count = len(test.trajectories)
    protocol = (
        "causal_previous_observation"
        if model.method == "persistence"
        else "causal_one_step_observed_state_reset"
    )
    try:
        if model.method == "persistence":
            metrics = persistence_metrics(test, model.normalization_scales)
        else:
            metrics = evaluate_equations(
                model.equations,
                context,
                test,
                model.normalization_scales,
                identifier=(
                    f"frozen_{model.method}_{model.task_index}_predictive_test"
                ),
            )
        if metrics.failed_trajectories:
            failed = tuple(metrics.failed_trajectories)
            return failed_predictive_result(
                model,
                trajectory_count=trajectory_count,
                evaluation_protocol=protocol,
                error=RuntimeError(
                    "predictive rollout failed for " + ", ".join(failed)
                ),
                frozen_model_sha256=frozen_model_sha256,
                failed_trajectories=failed,
            )
    except Exception as exc:
        return failed_predictive_result(
            model,
            trajectory_count=trajectory_count,
            evaluation_protocol=protocol,
            error=exc,
            frozen_model_sha256=frozen_model_sha256,
        )
    return BaselinePredictiveTestResult(
        task_index=model.task_index,
        method=model.method,
        benchmark_id=model.benchmark_id,
        tier=model.tier,
        seed=model.seed,
        frozen_model_sha256=frozen_model_sha256,
        status="available",
        evaluation_protocol=protocol,
        normalized_mse=metrics.normalized_mse,
        per_target_normalized_mse=dict(metrics.per_target_normalized_mse),
        trajectory_count=trajectory_count,
        successful_trajectory_count=trajectory_count,
    )


def failed_predictive_result(
    model: FrozenBaselineModel,
    *,
    trajectory_count: int,
    evaluation_protocol: Literal[
        "causal_previous_observation",
        "causal_one_step_observed_state_reset",
    ],
    error: Exception,
    frozen_model_sha256: str,
    failed_trajectories: tuple[str, ...] = (),
) -> BaselinePredictiveTestResult:
    """Retain one scientific endpoint failure without failing the array task."""
    return BaselinePredictiveTestResult(
        task_index=model.task_index,
        method=model.method,
        benchmark_id=model.benchmark_id,
        tier=model.tier,
        seed=model.seed,
        frozen_model_sha256=frozen_model_sha256,
        status="failed",
        evaluation_protocol=evaluation_protocol,
        trajectory_count=trajectory_count,
        successful_trajectory_count=(
            max(0, trajectory_count - len(failed_trajectories))
            if failed_trajectories
            else 0
        ),
        failed_trajectories=failed_trajectories,
        error_type=type(error).__name__,
        error=str(error)[:4000],
    )
