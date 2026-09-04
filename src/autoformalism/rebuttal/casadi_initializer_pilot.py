"""Typed contract for the public CasADi-initializer fitting pilot."""

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


class CasadiInitializerCell(BaseModel):
    """One frozen public benchmark cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CasadiInitializerCondition(BaseModel):
    """One matched initialization condition and its total fitting budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: Literal[
        "runtime_owned_start", "casadi_multiple_shooting_start"
    ]
    nonlinear_initializer: Literal["none", "casadi_multiple_shooting"]
    nonlinear_initializer_failure_policy: Literal["continue", "raise"]
    number_of_starts: int = Field(ge=1)
    maximum_function_evaluations: int = Field(ge=1)
    core_fit_wall_time_seconds: float = Field(gt=0.0)
    initializer_wall_time_seconds: float = Field(ge=0.0)
    casadi_shooting_interval_count: int = Field(ge=1)
    casadi_maximum_intervals_per_trajectory: int = Field(ge=1)
    casadi_maximum_iterations: int = Field(ge=1)

    @model_validator(mode="after")
    def condition_is_coherent(self) -> CasadiInitializerCondition:
        """Bind condition identity to the backend under comparison."""
        uses_casadi = self.nonlinear_initializer == "casadi_multiple_shooting"
        if uses_casadi != self.condition_id.startswith("casadi_"):
            raise ValueError("initializer condition id and backend differ")
        if uses_casadi != (self.initializer_wall_time_seconds > 0.0):
            raise ValueError("only the CasADi arm has an initialization budget")
        return self

    @property
    def total_wall_time_budget_seconds(self) -> float:
        """Return the matched initializer-plus-core wall-time budget."""
        return self.core_fit_wall_time_seconds + self.initializer_wall_time_seconds


class CasadiInitializerPilotPlan(BaseModel):
    """Frozen two-cell, three-seed initializer comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-casadi-initializer-pilot-1"]
    status: Literal["planned_before_fitting"]
    development_only: Literal[True]
    test_data_opened: Literal[False]
    private_reference_available_to_fitter: Literal[False]
    observed_derivatives_supplied: Literal[False]
    latent_values_supplied: Literal[False]
    latent_derivatives_supplied: Literal[False]
    weighted_overall_score_defined: Literal[False]
    source_replay: ProposerFinalistSourceContract
    source_candidate_condition: ProposerFinalistCondition
    cells: tuple[CasadiInitializerCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    fit_conditions: tuple[CasadiInitializerCondition, ...] = Field(min_length=2)
    reported_endpoints: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matrix_is_matched(self) -> CasadiInitializerPilotPlan:
        """Require unique cells/seeds and exactly the two declared arms."""
        cell_ids = [item.benchmark_id for item in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("CasADi initializer pilot cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("pilot repetitions must be unique and nonnegative")
        condition_ids = {item.condition_id for item in self.fit_conditions}
        if condition_ids != {
            "runtime_owned_start",
            "casadi_multiple_shooting_start",
        }:
            raise ValueError("CasADi initializer pilot conditions differ")
        budgets = {
            item.total_wall_time_budget_seconds for item in self.fit_conditions
        }
        if len(budgets) != 1:
            raise ValueError("initializer arms require equal total wall-time budgets")
        return self


def load_casadi_initializer_pilot_plan(path: Path) -> CasadiInitializerPilotPlan:
    """Load the strict frozen plan."""
    return CasadiInitializerPilotPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def canonical_casadi_initializer_plan_sha256(
    plan: CasadiInitializerPilotPlan,
) -> str:
    """Hash the semantic plan independently of JSON formatting."""
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def casadi_initializer_task_count(plan: CasadiInitializerPilotPlan) -> int:
    """Return the number of independent fitting tasks."""
    return len(plan.fit_conditions) * len(plan.cells) * len(plan.repetitions)


def casadi_initializer_task_identity(
    plan: CasadiInitializerPilotPlan,
    task_index: int,
) -> tuple[CasadiInitializerCondition, CasadiInitializerCell, int, int]:
    """Map an array index to condition, cell, seed, and shared candidate."""
    per_condition = len(plan.cells) * len(plan.repetitions)
    if not 0 <= task_index < casadi_initializer_task_count(plan):
        raise ValueError(f"CasADi initializer task index is invalid: {task_index}")
    condition_index, candidate_index = divmod(task_index, per_condition)
    cell_index, repetition_index = divmod(candidate_index, len(plan.repetitions))
    return (
        plan.fit_conditions[condition_index],
        plan.cells[cell_index],
        plan.repetitions[repetition_index],
        candidate_index,
    )
