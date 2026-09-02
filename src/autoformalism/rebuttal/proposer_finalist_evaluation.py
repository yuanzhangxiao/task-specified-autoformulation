"""Frozen public-only evaluation of repaired proposer finalists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProposerFinalistCell(BaseModel):
    """One public benchmark cell and its immutable evaluation contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mechanism_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProposerFinalistCondition(BaseModel):
    """One repaired transport condition retained for paired comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: Literal["low", "medium"]
    max_output_tokens: int = Field(ge=1)

    @property
    def directory_name(self) -> str:
        """Return the condition directory used by the frozen replay bundle."""
        return f"{self.reasoning_effort}_{self.max_output_tokens:06d}"


class ProposerFinalistFitProfile(BaseModel):
    """One ordered, common numerical fitting budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    integration_backend: Literal["fixed_rk4"]
    number_of_starts: int = Field(ge=1)
    maximum_function_evaluations: int = Field(ge=1)
    maximum_wall_time_seconds: float = Field(gt=0.0)


class ProposerFinalistSourceContract(BaseModel):
    """Exact replay-bundle identity required before public fitting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: Literal[
        "phase-b-proposer-repair-replay-manifest-1"
    ]
    required_status: Literal["pass"]
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_result_count: int = Field(ge=1)


class ProposerFinalistEvaluationPlan(BaseModel):
    """Development-only paired public evaluation plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "phase-b-proposer-finalist-public-evaluation-plan-1"
    ]
    status: Literal["frozen_before_public_fits"]
    development_only: Literal[True]
    new_llm_calls_permitted: Literal[False]
    scientific_judge_called: Literal[False]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    weighted_overall_score_defined: Literal[False]
    automatic_operating_point_selection: Literal[False]
    source_replay: ProposerFinalistSourceContract
    conditions: tuple[ProposerFinalistCondition, ...] = Field(min_length=2)
    cells: tuple[ProposerFinalistCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    fit_profiles: tuple[ProposerFinalistFitProfile, ...] = Field(min_length=1)
    stop_after_first_success: Literal[True]
    reported_endpoints: tuple[
        Literal[
            "runtime_validity",
            "public_target_completeness",
            "public_mechanism_compliance",
            "model_complexity",
            "public_training_nmse",
            "public_validation_nmse",
            "fit_success",
            "numerical_failures",
            "fit_resource_use",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def matrix_is_unique(self) -> ProposerFinalistEvaluationPlan:
        """Require a complete unique paired matrix and ordered fit ladder."""
        conditions = [
            (item.reasoning_effort, item.max_output_tokens)
            for item in self.conditions
        ]
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        profiles = [item.profile_id for item in self.fit_profiles]
        if len(conditions) != len(set(conditions)):
            raise ValueError("finalist conditions must be unique")
        if len(cells) != len(set(cells)):
            raise ValueError("finalist cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        if len(profiles) != len(set(profiles)):
            raise ValueError("fit profile identifiers must be unique")
        if len(self.reported_endpoints) != len(set(self.reported_endpoints)):
            raise ValueError("reported endpoints must be unique")
        expected = len(self.conditions) * len(self.cells) * len(self.repetitions)
        if expected != self.source_replay.replay_result_count:
            raise ValueError("evaluation matrix differs from replay result count")
        return self


def load_proposer_finalist_evaluation_plan(
    path: Path,
) -> ProposerFinalistEvaluationPlan:
    """Load and validate the frozen public finalist plan."""
    return ProposerFinalistEvaluationPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def canonical_plan_sha256(plan: ProposerFinalistEvaluationPlan) -> str:
    """Hash the semantic plan independently of source JSON whitespace."""
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def finalist_task_count(plan: ProposerFinalistEvaluationPlan) -> int:
    """Return the number of independently runnable candidate fits."""
    return len(plan.conditions) * len(plan.cells) * len(plan.repetitions)


def task_identity(
    plan: ProposerFinalistEvaluationPlan,
    task_index: int,
) -> tuple[ProposerFinalistCondition, ProposerFinalistCell, int, int]:
    """Map one evaluation index to condition, cell, repetition, and source task."""
    per_condition = len(plan.cells) * len(plan.repetitions)
    if not 0 <= task_index < finalist_task_count(plan):
        raise ValueError(f"task index is out of range: {task_index}")
    condition_index, source_task_index = divmod(task_index, per_condition)
    cell_index, repetition_index = divmod(
        source_task_index, len(plan.repetitions)
    )
    return (
        plan.conditions[condition_index],
        plan.cells[cell_index],
        plan.repetitions[repetition_index],
        source_task_index,
    )
