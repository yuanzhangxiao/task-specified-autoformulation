"""Typed contracts for checkpointed iterative model search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.fitting import EvaluationMetrics, FitConfig, FitResult
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import CandidateModel, JudgeAssessment
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.hybrid_pair import HybridPairJudgment

SelectionPolicy = Literal[
    "validation_only",
    "normalized_weighted_sum",
    "incumbent_relative_hybrid",
]


class IncumbentChallenge(StrictSchema):
    """One checkpointed fit/science challenge against the current incumbent."""

    incumbent_hash: str
    challenger_hash: str
    fit_preference_for_challenger: float = Field(ge=-1.0, le=1.0)
    science_preference_for_challenger: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    combined_preference_for_challenger: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    challenger_relative_score: float | None = Field(default=None, ge=0.0, le=1.0)
    incumbent_path_score_before: float = Field(ge=0.0)
    incumbent_path_score_after: float = Field(ge=0.0)
    selected_hash: str
    judgment: HybridPairJudgment | None = None


class SearchConfig(BaseModel):
    """Iteration, beam, stopping, judging, and checkpoint controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_directory: Path
    maximum_iterations: int = Field(default=5, ge=1)
    beam_size: int = Field(default=2, ge=1)
    stagnation_iterations: int = Field(default=3, ge=1)
    validation_mse_target: float = Field(default=0.0, ge=0.0)
    cheap_prefit_judge: bool = False
    use_judge: bool = True
    require_initial_proposer_cache_hit: bool = False
    selection_policy: SelectionPolicy = "validation_only"
    judge_weight: float = Field(default=0.25, ge=0.0)
    judge_score_epsilon: float = Field(default=0.05, gt=0.0)
    hybrid_science_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluate_test: bool = True
    proposer_system_prompt: str
    judge_system_prompt: str
    fit_config: FitConfig = FitConfig()
    fit_retry_config: FitConfig | None = None
    final_fit_config: FitConfig = FitConfig()
    final_fit_retry_config: FitConfig | None = None
    pruning_config: PruningConfig = PruningConfig()

    @model_validator(mode="after")
    def validate_selection_policy_contract(self) -> SearchConfig:
        """Enforce judge, beam, and split boundaries for selection policies."""
        if self.selection_policy != "validation_only" and not self.use_judge:
            raise ValueError(f"{self.selection_policy} requires use_judge=True")
        if self.require_initial_proposer_cache_hit and self.use_judge:
            raise ValueError(
                "initial proposer cache enforcement is reserved for no-judge arms"
            )
        if (
            self.selection_policy == "incumbent_relative_hybrid"
            and self.beam_size != 1
        ):
            raise ValueError("incumbent_relative_hybrid requires beam_size=1")
        if (
            self.selection_policy == "incumbent_relative_hybrid"
            and self.evaluate_test
        ):
            raise ValueError(
                "incumbent_relative_hybrid is development-only and requires "
                "evaluate_test=False"
            )
        for stage, primary, retry in (
            ("screening", self.fit_config, self.fit_retry_config),
            ("final", self.final_fit_config, self.final_fit_retry_config),
        ):
            if retry is None:
                continue
            if retry.integration_backend != primary.integration_backend:
                raise ValueError(
                    f"{stage} fit retry must retain the primary integration backend"
                )
            if retry.allow_derivative_regression != primary.allow_derivative_regression:
                raise ValueError(
                    f"{stage} fit retry must retain the primary fit objective"
                )
            if (
                retry.maximum_function_evaluations
                < primary.maximum_function_evaluations
            ):
                raise ValueError(
                    f"{stage} fit retry must not reduce maximum function evaluations"
                )
            if retry.number_of_starts < primary.number_of_starts:
                raise ValueError(
                    f"{stage} fit retry must not reduce the number of starts"
                )
            if retry.maximum_wall_time_seconds < primary.maximum_wall_time_seconds:
                raise ValueError(
                    f"{stage} fit retry must not reduce the wall-clock limit"
                )
        return self


@dataclass(frozen=True)
class CandidateRecord:
    """Completed candidate lineage and all non-test selection evidence."""

    round_index: int
    candidate: CandidateModel
    parent_candidate_id: str | None
    structural_hash: str
    fit: FitResult
    postfit_judge: JudgeAssessment
    pruned_candidate: CandidateModel
    pruned_fit: FitResult
    postpruning_judge: JudgeAssessment
    pruning_removed_terms: tuple[str, ...]
    pruning_removed_parameters: tuple[str, ...]
    pruning_contributions: dict[str, float]
    incumbent_challenge: IncumbentChallenge | None = None


@dataclass(frozen=True)
class FrozenSelection:
    """Validation-selected structure fixed before final fitting or test access."""

    selection_hash: str
    candidate: CandidateModel
    validation_mse: float
    round_index: int
    selection_policy: SelectionPolicy
    selection_objective: float
    normalized_log_validation: float
    normalized_judge_penalty: float
    judge_score: float
    incumbent_path_score: float | None = None
    hybrid_science_weight: float | None = None


@dataclass(frozen=True)
class FinalEvaluation:
    """Frozen selection, train-plus-validation refit, and one test evaluation."""

    frozen_selection: FrozenSelection
    final_fit: FitResult
    test_metrics: EvaluationMetrics | None
    test_trajectory_initial_conditions: dict[str, dict[str, float]] | None
    stopping_reason: Literal[
        "iteration_budget", "stagnation", "validation_target"
    ]
    completed_iterations: int
