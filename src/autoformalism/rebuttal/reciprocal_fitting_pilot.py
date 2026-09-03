"""Typed contract for the Phase-B reciprocal-coordinate fitting pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.proposer_finalist_evaluation import (
    ProposerFinalistCondition,
    ProposerFinalistSourceContract,
)


class ReciprocalFittingCell(BaseModel):
    """One public benchmark cell plus trusted simulator routing metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    family: Literal["dalla_man", "cstr", "alien_device"]
    task: str = Field(min_length=1)
    tier: Literal["easy", "hard"]
    dynamics: Literal["canonical", "perturbed"]
    semantic_variant: Literal["named", "obfuscated", "functional", "opaque"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReciprocalFittingCondition(BaseModel):
    """One common numerical fitting condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    parameter_fit_strategy: Literal[
        "bounded_nonlinear", "profiled_latent_basis_linear_ridge"
    ]
    allow_derivative_regression: bool
    use_certified_reciprocal_coordinates: bool
    number_of_starts: int = Field(ge=1)
    maximum_function_evaluations: int = Field(ge=1)
    maximum_wall_time_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def condition_is_coherent(self) -> ReciprocalFittingCondition:
        if self.parameter_fit_strategy == "bounded_nonlinear":
            if self.allow_derivative_regression:
                raise ValueError("bounded rollout condition cannot regress derivatives")
            if self.use_certified_reciprocal_coordinates:
                raise ValueError(
                    "bounded rollout condition has no reciprocal coordinate"
                )
        elif not self.allow_derivative_regression:
            raise ValueError("profiled condition requires derivative regression")
        return self


class ReciprocalFittingPilotPlan(BaseModel):
    """Frozen two-cell, three-seed fitting comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-reciprocal-fitting-pilot-1"]
    status: Literal["planned_before_derivative_overlay_or_fit"]
    development_only: Literal[True]
    test_data_opened: Literal[False]
    private_reference_available_to_fitter: Literal[False]
    exact_training_observed_derivatives_supplied: Literal[True]
    latent_values_supplied: Literal[False]
    latent_derivatives_supplied: Literal[False]
    weighted_overall_score_defined: Literal[False]
    source_replay: ProposerFinalistSourceContract
    source_candidate_condition: ProposerFinalistCondition
    cells: tuple[ReciprocalFittingCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    fit_conditions: tuple[ReciprocalFittingCondition, ...] = Field(min_length=2)
    reported_endpoints: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matrix_is_unique(self) -> ReciprocalFittingPilotPlan:
        cells = [item.benchmark_id for item in self.cells]
        conditions = [item.condition_id for item in self.fit_conditions]
        if len(cells) != len(set(cells)):
            raise ValueError("reciprocal fitting cells must be unique")
        if len(conditions) != len(set(conditions)):
            raise ValueError("reciprocal fitting conditions must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        required = {
            "bounded_rollout",
            "profiled_original_coordinate",
            "profiled_certified_reciprocal",
        }
        if set(conditions) != required:
            raise ValueError("reciprocal fitting pilot conditions differ")
        return self


def load_reciprocal_fitting_pilot_plan(path: Path) -> ReciprocalFittingPilotPlan:
    """Load the strict pilot plan."""
    return ReciprocalFittingPilotPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def canonical_reciprocal_fitting_plan_sha256(
    plan: ReciprocalFittingPilotPlan,
) -> str:
    """Hash the semantic plan independently of JSON whitespace."""
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def reciprocal_fitting_task_count(plan: ReciprocalFittingPilotPlan) -> int:
    """Return the number of independently runnable condition/cell/seed fits."""
    return len(plan.fit_conditions) * len(plan.cells) * len(plan.repetitions)


def reciprocal_fitting_task_identity(
    plan: ReciprocalFittingPilotPlan, task_index: int
) -> tuple[ReciprocalFittingCondition, ReciprocalFittingCell, int, int]:
    """Map task index to condition, cell, repetition, and shared candidate index."""
    per_condition = len(plan.cells) * len(plan.repetitions)
    if not 0 <= task_index < reciprocal_fitting_task_count(plan):
        raise ValueError(f"reciprocal fitting task index is out of range: {task_index}")
    condition_index, candidate_index = divmod(task_index, per_condition)
    cell_index, repetition_index = divmod(candidate_index, len(plan.repetitions))
    return (
        plan.fit_conditions[condition_index],
        plan.cells[cell_index],
        plan.repetitions[repetition_index],
        candidate_index,
    )
