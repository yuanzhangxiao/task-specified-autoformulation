"""A separately authorized, capped public-fit continuation pilot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicFitMetrics, Sha256


class ContinuationSelection(StrictSchema):
    """Pin one parent artifact; these limits are not caller-tunable defaults."""

    protocol: Literal["public-fit-continuation-pilot-1"] = (
        "public-fit-continuation-pilot-1"
    )
    parent_lowered_candidate_sha256: Sha256
    parent_backend_result_sha256: Sha256
    additional_seconds: Literal[180] = 180
    additional_residual_calls: Literal[240] = 240
    maximum_extensions: Literal[1] = 1
    recent_iterations: Literal[4] = 4
    minimum_relative_cost_drop: Literal[0.01] = 0.01


class ContinuationResult(StrictSchema):
    """Separate execution, selected-fit quality and uncertainty for feedback."""

    protocol: Literal["public-fit-continuation-result-1"] = (
        "public-fit-continuation-result-1"
    )
    identity: Sha256
    parent_identity: Sha256
    parent_result_sha256: Sha256
    status: Literal["complete", "ineligible", "interrupted", "extension_failed"]
    selected: Literal["parent", "extension"] = "parent"
    parameters: Mapping[Identifier, FiniteFloat]
    training: PublicFitMetrics
    validation: PublicFitMetrics
    extension_budget_exhausted: bool | None = None
    extension_residual_calls: int | None = Field(default=None, ge=0)
    cumulative_residual_calls: int | None = Field(default=None, ge=0)
    extension_numerical_seconds: FiniteFloat | None = None
    native_optimizer_converged: bool | None = None
    feedback_status: Literal[
        "budget_limited_unresolved",
        "numerical_failure_unresolved",
        "local_optimizer_stopped",
        "continuation_not_run",
    ]
    structural_failure_established: Literal[False] = False
    training_only_selection: Literal[True] = True
    validation_initials_fitted: Literal[False] = False
    independent_replay: Literal["not_performed"] = "not_performed"
    automatic_followup: Literal[False] = False
    backend_result_sha256: Sha256 | None = None
    message: str
