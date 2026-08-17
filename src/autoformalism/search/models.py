"""Typed contracts for checkpointed iterative model search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.fitting import EvaluationMetrics, FitConfig, FitResult
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import CandidateModel, JudgeAssessment

SelectionPolicy = Literal["validation_only", "normalized_weighted_sum"]


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
    selection_policy: SelectionPolicy = "validation_only"
    judge_weight: float = Field(default=0.25, ge=0.0)
    judge_score_epsilon: float = Field(default=0.05, gt=0.0)
    evaluate_test: bool = True
    proposer_system_prompt: str
    judge_system_prompt: str
    fit_config: FitConfig = FitConfig()
    final_fit_config: FitConfig = FitConfig()
    pruning_config: PruningConfig = PruningConfig()

    @model_validator(mode="after")
    def require_judge_for_weighted_selection(self) -> SearchConfig:
        """Reject a judge-weighted policy when judge calls are disabled."""
        if self.selection_policy == "normalized_weighted_sum" and not self.use_judge:
            raise ValueError("normalized_weighted_sum requires use_judge=True")
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
