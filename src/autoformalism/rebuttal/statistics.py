"""Deterministic paired statistics for frozen experiment results."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import wilcoxon


class PairedLogComparison(BaseModel):
    """Paired comparison on the log10 error ratio, first over second."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_count: int
    first_win_rate: float
    geometric_mean_ratio: float
    geometric_ratio_ci_low: float
    geometric_ratio_ci_high: float
    median_ratio: float
    standardized_paired_effect: float
    wilcoxon_p_value: float
    sign_flip_p_value: float


def paired_log_comparison(
    first: np.ndarray,
    second: np.ndarray,
    *,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 100_000,
    random_seed: int = 20260804,
) -> PairedLogComparison:
    """Compare strictly positive paired errors without sentinel imputation."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equal length")
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    log_ratio = np.log10(first[valid] / second[valid])
    if not len(log_ratio):
        raise ValueError("at least one finite positive pair is required")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(log_ratio), size=(bootstrap_samples, len(log_ratio)))
    boot = log_ratio[indices].mean(axis=1)
    low, high = np.quantile(boot, (0.025, 0.975))
    effect = 0.0
    if len(log_ratio) > 1 and float(np.std(log_ratio, ddof=1)) > 0:
        effect = float(np.mean(log_ratio) / np.std(log_ratio, ddof=1))
    wilcoxon_p = 1.0
    if np.any(log_ratio != 0):
        wilcoxon_p = float(wilcoxon(log_ratio, alternative="two-sided").pvalue)
    permutation_p = _sign_flip_p_value(log_ratio, permutation_samples, rng)
    return PairedLogComparison(
        pair_count=len(log_ratio),
        first_win_rate=float(np.mean(log_ratio < 0)),
        geometric_mean_ratio=float(10 ** np.mean(log_ratio)),
        geometric_ratio_ci_low=float(10**low),
        geometric_ratio_ci_high=float(10**high),
        median_ratio=float(10 ** np.median(log_ratio)),
        standardized_paired_effect=effect,
        wilcoxon_p_value=wilcoxon_p,
        sign_flip_p_value=permutation_p,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in input order."""
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Compute the two-sided 95% Wilson interval for a binomial proportion."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("successes and total define an invalid binomial count")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def _sign_flip_p_value(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(np.mean(differences)))
    if len(differences) <= 20:
        masks = np.arange(2 ** len(differences), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(differences), dtype=np.uint64)) & 1
        signs = bits.astype(float) * 2 - 1
        permuted = np.abs(np.mean(signs * differences, axis=1))
        return float(np.mean(permuted >= observed - 1e-15))
    exceedances = 0
    completed = 0
    batch = 10_000
    while completed < samples:
        current = min(batch, samples - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, len(differences)))
        permuted = np.abs(np.mean(signs * differences, axis=1))
        exceedances += int(np.sum(permuted >= observed - 1e-15))
        completed += current
    return (exceedances + 1) / (samples + 1)
