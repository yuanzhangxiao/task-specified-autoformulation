"""Frozen development-only fit-recovery diagnostics for search candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FitRecoveryProfile(BaseModel):
    """One deterministic numerical budget in an ordered recovery ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    mode: Literal["screening", "final_refit"]
    integration_backend: Literal["fixed_rk4", "solve_ivp"]
    number_of_starts: int = Field(ge=1)
    maximum_function_evaluations: int = Field(ge=1)
    maximum_wall_time_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def backend_matches_mode(self) -> FitRecoveryProfile:
        """Keep the diagnostic faithful to the production search stages."""
        expected = {
            "screening": "fixed_rk4",
            "final_refit": "solve_ivp",
        }[self.mode]
        if self.integration_backend != expected:
            raise ValueError(f"{self.mode} requires {expected}")
        return self


class FitRecoveryCase(BaseModel):
    """One frozen search run and its selected checkpoint rounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    seed: int = Field(ge=0)
    mode: Literal["screening", "final_refit"]
    run_directory: str = Field(min_length=1)
    round_indices: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def rounds_are_unique(self) -> FitRecoveryCase:
        """Prevent repeated numerical work inside a frozen case."""
        if len(self.round_indices) != len(set(self.round_indices)) or any(
            item < 0 for item in self.round_indices
        ):
            raise ValueError("round indices must be unique and nonnegative")
        return self


class SearchFitRecoveryPlan(BaseModel):
    """Predeclared no-LLM recovery experiment over frozen failed candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-search-fit-recovery-plan-1"]
    status: Literal["frozen_before_refits"]
    development_only: Literal[True]
    new_llm_calls: Literal[False]
    test_data_opened: Literal[False]
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_path: str = Field(min_length=1)
    stop_after_first_success: Literal[True]
    cases: tuple[FitRecoveryCase, ...] = Field(min_length=1)
    profiles: tuple[FitRecoveryProfile, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def matrix_is_well_formed(self) -> SearchFitRecoveryPlan:
        """Require unique identifiers and a recovery ladder for every case mode."""
        case_ids = [item.case_id for item in self.cases]
        profile_ids = [item.profile_id for item in self.profiles]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case identifiers must be unique")
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("profile identifiers must be unique")
        modes = {item.mode for item in self.profiles}
        missing = {item.mode for item in self.cases} - modes
        if missing:
            raise ValueError(f"case modes lack profiles: {sorted(missing)}")
        return self

    def profiles_for(self, mode: str) -> tuple[FitRecoveryProfile, ...]:
        """Return the frozen ordered ladder for one search stage."""
        return tuple(item for item in self.profiles if item.mode == mode)


def load_search_fit_recovery_plan(path: Path) -> SearchFitRecoveryPlan:
    """Load and validate a frozen fit-recovery plan."""
    return SearchFitRecoveryPlan.model_validate_json(path.read_text(encoding="utf-8"))


def verify_source_plan(plan: SearchFitRecoveryPlan, search_root: Path) -> Path:
    """Bind diagnostics to the exact search plan that generated the candidates."""
    source = search_root / plan.source_plan_path
    if not source.is_file():
        raise ValueError(f"missing frozen source plan: {source}")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != plan.source_plan_sha256:
        raise ValueError(
            "frozen source plan differs: "
            f"observed={observed}, expected={plan.source_plan_sha256}"
        )
    return source


def recovery_task_count(plan: SearchFitRecoveryPlan) -> int:
    """Return the number of independently runnable cases."""
    return len(plan.cases)


def canonical_plan_sha256(plan: SearchFitRecoveryPlan) -> str:
    """Hash the validated semantic plan independently of JSON whitespace."""
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
