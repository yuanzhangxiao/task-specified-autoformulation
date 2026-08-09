"""Ratio and weighted-sum objective comparison on a frozen candidate pool."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import kendalltau, spearmanr

from autoformalism.rebuttal.artifacts import CandidateArtifact


class ObjectiveComparison(BaseModel):
    """Summary of one ratio-versus-weighted-sum ranking comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_multiplier: float
    lambda_value: float
    epsilon: float
    candidate_count: int
    spearman_correlation: float
    kendall_correlation: float
    top_k: int
    top_k_overlap: int
    ratio_selected_artifact_id: str
    weighted_sum_selected_artifact_id: str


SelectorPolicy = Literal[
    "validation_only",
    "normalized_weighted_sum",
    "pareto_compromise",
    "epsilon_constrained",
]


class FrozenSelectionResult(BaseModel):
    """Development-only selection and its auditable objective components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: SelectorPolicy
    artifact_id: str
    candidate_count: int
    eligible_count: int
    validation_mse: float
    judge_score: float
    term_count: int
    normalized_log_validation: float
    normalized_judge_penalty: float
    normalized_complexity: float
    objective_value: float
    judge_weight: float
    sparsity_weight: float
    epsilon_fraction: float


def ratio_objective(loss: float, judge_score: float, epsilon: float) -> float:
    """Compute validation loss divided by a guarded judge score."""
    return loss / (judge_score + epsilon)


def weighted_sum_objective(
    loss: float, judge_score: float, lambda_value: float, epsilon: float
) -> float:
    """Compute additive validation loss plus a negative-log judge penalty."""
    return loss + lambda_value * -math.log(judge_score + epsilon)


def compare_ratio_and_weighted_sum(
    candidates: Sequence[CandidateArtifact],
    *,
    lambda_multiplier: float,
    epsilon: float = 0.05,
    top_k: int = 5,
) -> ObjectiveComparison:
    """Compare rankings without using test metrics.

    Lambda is scaled by the median development loss so the judge penalty and
    normalized-MSE term remain commensurate across benchmark contexts.
    """
    eligible = [item for item in candidates if item.judge_score is not None]
    if len(eligible) < 2:
        raise ValueError("at least two judged candidates are required")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    losses = np.asarray([item.validation_mse for item in eligible], dtype=float)
    lambda_value = float(np.median(losses)) * lambda_multiplier
    ratio_values = [
        ratio_objective(item.validation_mse, float(item.judge_score), epsilon)
        for item in eligible
    ]
    weighted_values = [
        weighted_sum_objective(
            item.validation_mse,
            float(item.judge_score),
            lambda_value,
            epsilon,
        )
        for item in eligible
    ]
    ratio_order = np.argsort(ratio_values, kind="stable")
    weighted_order = np.argsort(weighted_values, kind="stable")
    ratio_ranks = _ranks(ratio_order)
    weighted_ranks = _ranks(weighted_order)
    bounded_k = min(top_k, len(eligible))
    overlap = len(
        set(ratio_order[:bounded_k].tolist())
        & set(weighted_order[:bounded_k].tolist())
    )
    return ObjectiveComparison(
        lambda_multiplier=lambda_multiplier,
        lambda_value=lambda_value,
        epsilon=epsilon,
        candidate_count=len(eligible),
        spearman_correlation=float(spearmanr(ratio_ranks, weighted_ranks).statistic),
        kendall_correlation=float(kendalltau(ratio_ranks, weighted_ranks).statistic),
        top_k=bounded_k,
        top_k_overlap=overlap,
        ratio_selected_artifact_id=eligible[int(ratio_order[0])].artifact_id,
        weighted_sum_selected_artifact_id=eligible[
            int(weighted_order[0])
        ].artifact_id,
    )


def select_frozen_candidate(
    candidates: Sequence[CandidateArtifact],
    *,
    policy: SelectorPolicy,
    judge_weight: float = 0.25,
    sparsity_weight: float = 0.1,
    epsilon_fraction: float = 0.05,
    score_epsilon: float = 0.05,
) -> FrozenSelectionResult:
    """Select from a frozen judged pool using development-only evidence.

    Every input candidate is assumed to have already passed deterministic
    runtime validation. Private mechanism references and test metrics are not
    represented by :class:`CandidateArtifact` and cannot enter this decision.
    """
    eligible = [item for item in candidates if item.judge_score is not None]
    if not eligible:
        raise ValueError("at least one judged candidate is required")
    if judge_weight < 0.0 or sparsity_weight < 0.0:
        raise ValueError("objective weights must be nonnegative")
    if epsilon_fraction < 0.0:
        raise ValueError("epsilon_fraction must be nonnegative")
    if score_epsilon <= 0.0:
        raise ValueError("score_epsilon must be positive")

    losses = np.asarray([item.validation_mse for item in eligible], dtype=float)
    if np.any(~np.isfinite(losses)) or np.any(losses <= 0.0):
        raise ValueError("validation losses must be finite and positive")
    scores = np.asarray([float(item.judge_score) for item in eligible], dtype=float)
    terms = np.asarray([item.term_count for item in eligible], dtype=float)
    log_losses = np.log(losses)
    judge_penalties = -np.log(scores + score_epsilon)
    complexities = np.log1p(terms)
    loss_z = _robust_standardize(log_losses)
    judge_z = _robust_standardize(judge_penalties)
    complexity_z = _robust_standardize(complexities)
    weighted = loss_z + judge_weight * judge_z + sparsity_weight * complexity_z

    if policy == "validation_only":
        eligible_indices = np.arange(len(eligible))
        objective = loss_z
    elif policy == "normalized_weighted_sum":
        eligible_indices = np.arange(len(eligible))
        objective = weighted
    elif policy == "pareto_compromise":
        eligible_indices = np.flatnonzero(
            _pareto_mask(np.column_stack((losses, judge_penalties, terms)))
        )
        objective = weighted
    elif policy == "epsilon_constrained":
        eligible_indices = np.flatnonzero(
            losses <= losses.min() * (1.0 + epsilon_fraction)
        )
        objective = weighted
    else:  # pragma: no cover - guarded by Literal in typed callers
        raise ValueError(f"unknown policy: {policy}")

    if policy == "epsilon_constrained":
        selected_index = min(
            eligible_indices,
            key=lambda index: (
                -scores[index],
                terms[index],
                losses[index],
                eligible[index].artifact_id,
            ),
        )
    else:
        selected_index = min(
            eligible_indices,
            key=lambda index: (
                objective[index],
                losses[index],
                -scores[index],
                terms[index],
                eligible[index].artifact_id,
            ),
        )
    selected = eligible[int(selected_index)]
    return FrozenSelectionResult(
        policy=policy,
        artifact_id=selected.artifact_id,
        candidate_count=len(candidates),
        eligible_count=len(eligible_indices),
        validation_mse=selected.validation_mse,
        judge_score=float(selected.judge_score),
        term_count=selected.term_count,
        normalized_log_validation=float(loss_z[selected_index]),
        normalized_judge_penalty=float(judge_z[selected_index]),
        normalized_complexity=float(complexity_z[selected_index]),
        objective_value=float(objective[selected_index]),
        judge_weight=judge_weight,
        sparsity_weight=sparsity_weight,
        epsilon_fraction=epsilon_fraction,
    )


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    """Center by the median and scale by IQR with a deterministic fallback."""
    median = float(np.median(values))
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    if scale <= np.finfo(float).eps:
        scale = float(np.ptp(values))
    if scale <= np.finfo(float).eps:
        scale = 1.0
    return (values - median) / scale


def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Return nondominated rows for objectives that are all minimized."""
    keep = np.ones(len(costs), dtype=bool)
    for index, point in enumerate(costs):
        dominated = np.all(costs <= point, axis=1) & np.any(costs < point, axis=1)
        dominated[index] = False
        keep[index] = not np.any(dominated)
    return keep


def _ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks
