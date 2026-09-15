"""A diagnostic budget sweep, separate from the frozen fitting protocol."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicFitMetrics, Sha256


class ConvergenceSelection(StrictSchema):
    """Select one completed continuation and explicitly bound this experiment."""

    protocol: Literal["public-fit-convergence-diagnostic-1"] = (
        "public-fit-convergence-diagnostic-1"
    )
    continuation_identity: Sha256
    continuation_backend_sha256: Sha256
    maximum_windows: int = Field(default=20, ge=1, le=100)
    window_seconds: Literal[180] = 180
    window_residual_calls: Literal[240] = 240
    stationarity_tolerance: Literal[1e-6] = 1e-6
    stop_on_slow_progress: Literal[False] = False
    final_protocol_limit_decided: Literal[False] = False


class RetainedFit(StrictSchema):
    """The complete vector, objective and observed-output scores at a handoff."""

    parameters: Mapping[Identifier, FiniteFloat]
    training: PublicFitMetrics
    validation: PublicFitMetrics
    training_cost: FiniteFloat = Field(ge=0)
    actual_residual_calls: int | None = Field(default=None, ge=0)
