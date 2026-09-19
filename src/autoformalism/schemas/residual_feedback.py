"""Measured training mismatches, without structural diagnoses or hidden labels."""

from typing import Literal

from pydantic import Field

from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256


class FeedbackSelection(StrictSchema):
    """Select the completed two-round parent, independently of its running sibling."""

    protocol: Literal["public-fit-residual-feedback-1"] = (
        "public-fit-residual-feedback-1"
    )
    continuation_identity: Sha256 = (
        "46e9914dd9d2097ba492adcf8edaecb638da6a8bd93404089afa5143d07c91bb"
    )
    continuation_backend_sha256: Sha256 = (
        "d92f087138f921ab7f634c9c94d9c87963b9a160b4930fa508e77045d6b9e986"
    )
    replay_seconds_per_point: Literal[300] = 300


class ResidualSettings(StrictSchema):
    """Deterministic presentation limits, not statistical significance thresholds."""

    maximum_details: int = Field(default=6, ge=1, le=8)
    maximum_samples: int = Field(default=12, ge=5, le=16)
    maximum_rows: int = Field(default=64, ge=1, le=128)
    absolute_tolerance: FiniteFloat = Field(default=1e-8, ge=0)
    relative_tolerance: FiniteFloat = Field(default=0.01, ge=0, le=0.1)


class SignalShape(StrictSchema):
    """Simple sampled shape, using one common tolerance for both signals."""

    initial: FiniteFloat
    final: FiniteFloat
    mean: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat
    range: FiniteFloat
    turning_point_count: int = Field(ge=0)
    common_change_tolerance: FiniteFloat = Field(ge=0)


class ResidualWindow(StrictSchema):
    """Equal-duration thirds; every sample belongs to exactly one nonempty window."""

    evidence_id: str
    label: Literal["early", "middle", "late"]
    first_index: int = Field(ge=0)
    last_index: int = Field(ge=0)
    first_time: FiniteFloat
    last_time: FiniteFloat
    sample_count: int = Field(ge=1)
    normalized_mse: FiniteFloat = Field(ge=0)
    normalized_signed_bias: FiniteFloat
    previous_normalized_mse: FiniteFloat | None = Field(default=None, ge=0)


class ResidualRow(StrictSchema):
    """One observed target and training trajectory at two retained parameter sets."""

    evidence_id: str
    trajectory_id: str
    target: str
    sample_count: int = Field(ge=2)
    minimum_time_step: FiniteFloat = Field(gt=0)
    maximum_time_step: FiniteFloat = Field(gt=0)
    sampled_input_regime: Literal["zero", "constant_nonzero", "varying", "no_inputs"]
    input_ranges: dict[str, tuple[FiniteFloat, FiniteFloat]]
    training_scale: FiniteFloat = Field(gt=0)
    normalized_mse: FiniteFloat = Field(ge=0)
    normalized_signed_bias: FiniteFloat
    maximum_absolute_normalized_residual: FiniteFloat = Field(ge=0)
    observed: SignalShape
    predicted: SignalShape
    previous_normalized_mse: FiniteFloat | None = Field(default=None, ge=0)
    previous_predicted: SignalShape | None = None
    windows: tuple[ResidualWindow, ...]


class ResidualSample(StrictSchema):
    """Exact common-grid samples; the residual sign is prediction minus observation."""

    index: int = Field(ge=0)
    time: FiniteFloat
    observed: FiniteFloat
    predicted: FiniteFloat
    normalized_residual: FiniteFloat
    previous_prediction: FiniteFloat | None = None
    inputs: dict[str, FiniteFloat]
    auxiliaries: dict[str, FiniteFloat]


class ResidualDetail(StrictSchema):
    evidence_id: str
    row_id: str
    reasons: tuple[str, ...]
    samples: tuple[ResidualSample, ...]
    omitted_samples: int = Field(ge=0)


class NumericalQualification(StrictSchema):
    """An allowlist that cannot hide held-out metrics in an arbitrary dictionary."""

    feedback_status: Literal[
        "budget_limited_unresolved",
        "numerical_failure_unresolved",
        "local_optimizer_stopped",
        "fixed_model_evaluated",
        "continuation_not_run",
    ] = "budget_limited_unresolved"
    native_optimizer_converged: bool | None = None
    budget_exhausted: bool | None = None
    residual_calls: int | None = Field(default=None, ge=0)
    numerical_seconds: FiniteFloat | None = Field(default=None, ge=0)
    numerical_seconds_scope: Literal["latest_completed_continuation"] = (
        "latest_completed_continuation"
    )
    residual_calls_scope: Literal["cumulative_parent_and_continuation"] = (
        "cumulative_parent_and_continuation"
    )
    independent_solver_replay: Literal[False] = False
    previous_residual_calls: int | None = Field(default=None, ge=0)
    recent_training_costs: tuple[FiniteFloat, ...] = ()
    relative_training_cost_drop: FiniteFloat | None = None
    structural_failure_established: Literal[False] = False
    uncertainty: Literal["at_retained_parameters_only"] = "at_retained_parameters_only"


class ResidualEvidence(StrictSchema):
    """Provider-safe packet: no validation, hidden state, or mandatory repair."""

    protocol: Literal["training-residual-evidence-1"] = "training-residual-evidence-1"
    split: Literal["train"] = "train"
    settings: ResidualSettings
    candidate_sha256: Sha256
    parameter_sha256: Sha256
    training_content_sha256: Sha256
    replay_sha256: Sha256
    normalized_mse: FiniteFloat = Field(ge=0)
    previous_normalized_mse: FiniteFloat | None = Field(default=None, ge=0)
    numerical_status: NumericalQualification
    total_rows: int = Field(ge=1)
    omitted_rows: int = Field(ge=0)
    rows: tuple[ResidualRow, ...]
    details: tuple[ResidualDetail, ...]
    interpretation: tuple[str, ...]
    packet_sha256: Sha256
