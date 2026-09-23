"""Training-only response measurements; no inferred causal mechanisms."""

from typing import Literal

from pydantic import Field

from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import Sha256


class ResponseShape(StrictSchema):
    initial: FiniteFloat
    final: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat
    excursion: FiniteFloat
    peak_time: FiniteFloat | None
    peak_status: Literal["flat", "interior", "boundary", "plateau"]
    half_return_elapsed: FiniteFloat | None
    half_return_status: Literal["observed", "not_reached", "undefined"]
    turning_points: int = Field(ge=0)


class InputResponse(StrictSchema):
    initial: FiniteFloat
    final: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat
    first_sampled_change: FiniteFloat | None
    departure_count: int = Field(ge=0)


class ResponseRow(StrictSchema):
    evidence_id: str
    trajectory_id: str
    target: str
    sample_count: int = Field(ge=2)
    normalized_mse: FiniteFloat = Field(ge=0)
    normalized_signed_bias: FiniteFloat
    window_nmse: dict[str, FiniteFloat]
    observed: ResponseShape
    predicted: ResponseShape
    peak_timing_error: FiniteFloat | None
    # Relative to first sampled departure, not identified input-output causality.
    input_peak_delays: dict[str, tuple[FiniteFloat | None, FiniteFloat | None]]


class ResponseEvidence(StrictSchema):
    protocol: Literal["training-response-evidence-1"] = "training-response-evidence-1"
    split: Literal["train"] = "train"
    candidate_sha256: Sha256
    parameter_sha256: Sha256
    training_content_sha256: Sha256
    source_packet_sha256: Sha256
    replayed_packet_sha256: Sha256
    replay_sha256: Sha256
    rows: tuple[ResponseRow, ...]
    inputs: dict[str, dict[str, InputResponse]]
    numerical_status: dict
    interpretation: tuple[str, ...]
