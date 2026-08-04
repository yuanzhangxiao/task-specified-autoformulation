"""Ratio and weighted-sum objective comparison on a frozen candidate pool."""

from __future__ import annotations

import math
from collections.abc import Sequence

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


def _ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks
