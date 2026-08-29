"""Method-neutral post-freeze replay on held-out Phase-B trajectories."""

from __future__ import annotations

from time import monotonic
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    InterventionEndpoint,
    TargetPredictionEndpoint,
)
from autoformalism.rebuttal.intervention_evaluation import (
    qualitative_response_metrics,
)


class PostFreezeEvaluationOutcome(BaseModel):
    """Checkpointable result of opening test data for one frozen subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-postfreeze-outcome-1"] = (
        "phase-b-postfreeze-outcome-1"
    )
    subject_id: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["complete", "failed"]
    test_data_opened: Literal[True] = True
    target_status: Literal["available", "failed"]
    intervention_count: int = Field(ge=0)
    error_type: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def fields_match_status(self) -> PostFreezeEvaluationOutcome:
        """Require explicit diagnostics for failed test evaluation."""
        if self.status == "complete":
            if self.target_status != "available" or self.error_type or self.error:
                raise ValueError("complete post-freeze outcome is inconsistent")
        elif self.target_status != "failed" or not self.error_type or not self.error:
            raise ValueError("failed post-freeze outcome requires diagnostics")
        return self


def evaluate_subject_on_test(
    subject: FrozenEvaluationSubject,
    *,
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    fit_config: FitConfig | None = None,
) -> FrozenEvaluationSubject:
    """Replay a frozen candidate without fitting parameters or initial states."""
    if training_split.name is not SplitName.TRAIN:
        raise ValueError("normalization must be fitted on the training split")
    if test_split.name is not SplitName.TEST:
        raise ValueError("post-freeze replay requires the test split")
    if not test_split.trajectories:
        raise ValueError("post-freeze replay requires at least one test trajectory")
    if subject.target_prediction.status == "available":
        raise ValueError("subject already carries available target test metrics")
    trajectory_ids = tuple(item.trajectory_id for item in test_split.trajectories)
    if subject.parameterization.status not in {"available", "not_required"}:
        return _failed_subject(
            subject,
            trajectory_ids,
            "frozen parameterization is not replay-complete: "
            f"{subject.parameterization.status}",
        )
    settings = fit_config or FitConfig(
        integration_backend="solve_ivp",
        allow_derivative_regression=False,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        maximum_wall_time_seconds=300.0,
    )
    try:
        compiled = compile_candidate(subject.candidate, subject.validation_context)
    except Exception as exc:
        return _failed_subject(
            subject,
            trajectory_ids,
            f"candidate compilation failed: {type(exc).__name__}: {exc}",
        )
    scales = _target_scales(training_split, subject.validation_context.targets)
    squared: dict[str, list[np.ndarray]] = {
        name: [] for name in subject.validation_context.targets
    }
    interventions: list[InterventionEndpoint] = []
    failed: list[str] = []
    parameters = subject.parameterization.global_parameters
    initials = subject.parameterization.global_initial_conditions
    primary_target = subject.validation_context.targets[0]
    for trajectory in test_split.trajectories:
        deadline = (
            None
            if settings.maximum_wall_time_seconds is None
            else monotonic() + settings.maximum_wall_time_seconds
        )
        simulation = simulate_trajectory(
            compiled,
            trajectory,
            parameters,
            initials,
            settings,
            deadline=deadline,
            reset_observed_states=False,
        )
        case_id = f"test_trajectory:{trajectory.trajectory_id}"
        if not simulation.success:
            failed.append(trajectory.trajectory_id)
            interventions.append(
                InterventionEndpoint(
                    case_id=case_id,
                    status="failed",
                    message=simulation.message or "free rollout failed",
                )
            )
            continue
        for target in squared:
            residual = (
                simulation.predictions[target] - trajectory.targets[target]
            ) / scales[target]
            squared[target].append(np.square(residual))
        primary_prediction = simulation.predictions[primary_target]
        primary_reference = trajectory.targets[primary_target]
        case_nmse = float(
            np.mean(
                tuple(
                    float(
                        np.mean(
                            np.square(
                                simulation.predictions[target]
                                - trajectory.targets[target]
                            )
                        )
                        / scales[target] ** 2
                    )
                    for target in subject.validation_context.targets
                )
            )
        )
        direction, shape, timing = qualitative_response_metrics(
            trajectory.time,
            primary_prediction,
            primary_reference,
        )
        interventions.append(
            InterventionEndpoint(
                case_id=case_id,
                status="available",
                target_nmse=case_nmse,
                response_direction_correct=direction,
                response_shape_correlation=shape,
                peak_timing_error_fraction=timing,
            )
        )
    if failed:
        target = TargetPredictionEndpoint(
            status="failed",
            evaluation_protocol="unseen_condition_free_rollout",
            normalization_scales=scales,
            trajectory_count=len(test_split.trajectories),
            successful_trajectory_count=len(test_split.trajectories) - len(failed),
            failed_trajectories=tuple(failed),
            message="one or more held-out free rollouts failed",
        )
    else:
        per_target = {
            name: float(np.mean(np.concatenate(values)))
            for name, values in squared.items()
        }
        target = TargetPredictionEndpoint(
            status="available",
            evaluation_protocol="unseen_condition_free_rollout",
            normalized_mse=float(np.mean(tuple(per_target.values()))),
            per_target_normalized_mse=per_target,
            normalization_scales=scales,
            trajectory_count=len(test_split.trajectories),
            successful_trajectory_count=len(test_split.trajectories),
            message="frozen-parameter unseen-condition free rollout",
        )
    return _updated_subject(subject, target, tuple(interventions))


def outcome_for_subject(
    subject: FrozenEvaluationSubject,
    *,
    error: Exception | None = None,
) -> PostFreezeEvaluationOutcome:
    """Summarize one updated subject for resume and completion accounting."""
    target_status = (
        "available" if subject.target_prediction.status == "available" else "failed"
    )
    if target_status == "failed" and error is None:
        error_type = "TargetPredictionFailure"
        error_message = subject.target_prediction.message or "target replay failed"
    else:
        error_type = None if error is None else type(error).__name__
        error_message = None if error is None else str(error)
    return PostFreezeEvaluationOutcome(
        subject_id=subject.subject_id,
        candidate_sha256=subject.source_provenance.candidate_sha256,
        status="complete" if target_status == "available" else "failed",
        target_status=target_status,
        intervention_count=len(subject.interventions),
        error_type=error_type,
        error=error_message,
    )


def _target_scales(
    training_split: DatasetSplit,
    targets: tuple[str, ...],
) -> dict[str, float]:
    scales = TrainingScaler().fit(training_split).scales
    return {
        target: float(scales[f"target:{target}"].standard_deviation)
        for target in targets
    }


def _failed_subject(
    subject: FrozenEvaluationSubject,
    trajectory_ids: tuple[str, ...],
    message: str,
) -> FrozenEvaluationSubject:
    target = TargetPredictionEndpoint(
        status="failed",
        evaluation_protocol="unseen_condition_free_rollout",
        trajectory_count=len(trajectory_ids),
        successful_trajectory_count=0,
        failed_trajectories=trajectory_ids,
        message=message,
    )
    interventions = tuple(
        InterventionEndpoint(
            case_id=f"test_trajectory:{identifier}",
            status="failed",
            message=message,
        )
        for identifier in trajectory_ids
    )
    return _updated_subject(subject, target, interventions)


def _updated_subject(
    subject: FrozenEvaluationSubject,
    target: TargetPredictionEndpoint,
    interventions: tuple[InterventionEndpoint, ...],
) -> FrozenEvaluationSubject:
    existing_ids = {item.case_id for item in subject.interventions}
    overlap = existing_ids & {item.case_id for item in interventions}
    if overlap:
        raise ValueError(f"duplicate intervention cases: {sorted(overlap)}")
    payload = subject.model_dump(mode="json")
    payload.update(
        {
            "private_metrics_opened_after_freeze": True,
            "target_prediction": target.model_dump(mode="json"),
            "interventions": [
                item.model_dump(mode="json")
                for item in (*subject.interventions, *interventions)
            ],
        }
    )
    return FrozenEvaluationSubject.model_validate(payload)
