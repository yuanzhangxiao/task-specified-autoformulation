"""Versioned public fitting handoff, independent of experiment orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.candidate import CandidateModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FitProfile = Literal[
    "general-rollout-v1",
    "collocation-feasible-v1",
    "collocation-single-target-v2",
    "collocation-multi-target-v1",
]


class PublicFitSource(StrictSchema):
    """Upstream artifact identity; the exporter verifies the referenced artifact."""

    stage: Literal[
        "construction", "requirement_repair", "controller_revision", "synthetic_control"
    ]
    task_id: str = Field(min_length=1, max_length=512)
    artifact_sha256: Sha256


class PublicFitRequest(StrictSchema):
    """Base equations and public boundaries, before initialization lowering."""

    protocol: Literal["public-fit-1"] = "public-fit-1"
    base_candidate: CandidateModel
    context: ValidationContext
    initialization_plan: LatentInitializationPlan
    parameter_guesses: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)
    profile: FitProfile
    random_seed: int = Field(default=20260913, ge=0, le=2**32 - 1)
    source: PublicFitSource


class PublicTrajectory(StrictSchema):
    """Only declared observations and forcing; no hidden or derivative labels."""

    trajectory_id: str = Field(min_length=1)
    time: tuple[FiniteFloat, ...] = Field(min_length=2)
    targets: Mapping[Identifier, tuple[FiniteFloat, ...]]
    auxiliaries: Mapping[Identifier, tuple[FiniteFloat, ...]] = Field(
        default_factory=dict
    )
    external_inputs: Mapping[Identifier, tuple[FiniteFloat, ...]] = Field(
        default_factory=dict
    )
    fixed_covariates: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_arrays(self):
        if any(b <= a for a, b in zip(self.time[:-1], self.time[1:], strict=True)):
            raise ValueError("time must be strictly increasing")
        for channels in (self.targets, self.auxiliaries, self.external_inputs):
            if any(len(values) != len(self.time) for values in channels.values()):
                raise ValueError("channel length differs from time")
        return self


class PublicSplit(StrictSchema):
    """A development split whose actual content is hashed at preparation."""

    name: Literal["train", "val"]
    fingerprint: str = Field(min_length=1)
    rows: tuple[PublicTrajectory, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_rows(self):
        ids = [row.trajectory_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate trajectory identifier")
        return self


class PublicFitMetrics(StrictSchema):
    """Unavailable/failed rollouts never expose their failure penalty as NMSE."""

    available: bool = False
    normalized_mse: FiniteFloat | None = None
    per_target_normalized_mse: Mapping[Identifier, FiniteFloat] = Field(
        default_factory=dict
    )
    failed_trajectories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent_availability(self):
        if self.available:
            if self.normalized_mse is None or self.failed_trajectories:
                raise ValueError("available metrics require a complete finite rollout")
        elif self.normalized_mse is not None or self.per_target_normalized_mse:
            raise ValueError("unavailable metrics cannot contain penalty scores")
        return self


class PublicFitResult(StrictSchema):
    """Execution and numerical evidence, without a scientific acceptance verdict."""

    protocol: Literal["public-fit-result-1"] = "public-fit-result-1"
    status: Literal["complete", "fit_failed", "capability_unsupported", "interrupted"]
    profile: FitProfile
    identity: Sha256
    request_sha256: Sha256
    lowered_candidate_sha256: Sha256
    initialization_plan_sha256: Sha256
    training_content_sha256: Sha256
    validation_content_sha256: Sha256
    source: PublicFitSource
    parameters: Mapping[Identifier, FiniteFloat] | None = None
    training: PublicFitMetrics = Field(default_factory=PublicFitMetrics)
    validation: PublicFitMetrics = Field(default_factory=PublicFitMetrics)
    native_optimizer_converged: bool | None = None
    budget_exhausted: bool | None = None
    actual_residual_calls: int | None = Field(default=None, ge=0)
    independent_replay: Literal["not_performed"] = "not_performed"
    training_only_parameter_estimation: Literal[True] = True
    validation_initials_fitted: Literal[False] = False
    backend_result_sha256: Sha256 | None = None
    message: str
