"""Post-selection hidden-mechanism affine-alignment metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.optimize import lsq_linear


class HiddenMechanismMetric(BaseModel):
    """Affine calibration fitted on training and scored on held-out test data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scale: float
    offset: float
    test_nmse: float
    allow_signed_scale: bool
    train_sample_count: int
    test_sample_count: int


def hidden_mechanism_nmse(
    train_candidate: NDArray[np.float64],
    train_reference: NDArray[np.float64],
    test_candidate: NDArray[np.float64],
    test_reference: NDArray[np.float64],
    *,
    allow_signed_scale: bool = False,
    epsilon: float = 1e-12,
) -> HiddenMechanismMetric:
    """Fit an evaluation-only affine alignment on train and score it on test."""
    arrays = tuple(
        np.asarray(item, dtype=float).reshape(-1)
        for item in (
            train_candidate,
            train_reference,
            test_candidate,
            test_reference,
        )
    )
    train_z, train_h, test_z, test_h = arrays
    if len(train_z) != len(train_h) or len(test_z) != len(test_h):
        raise ValueError("candidate/reference lengths must match within each split")
    if not len(train_z) or not len(test_z):
        raise ValueError("train and test arrays must be nonempty")
    if not all(np.isfinite(item).all() for item in arrays):
        raise ValueError("hidden-mechanism arrays must be finite")
    design = np.column_stack((train_z, np.ones_like(train_z)))
    lower = np.asarray([-np.inf if allow_signed_scale else 0.0, -np.inf])
    upper = np.asarray([np.inf, np.inf])
    fitted = lsq_linear(design, train_h, bounds=(lower, upper))
    scale, offset = (float(item) for item in fitted.x)
    predicted = scale * test_z + offset
    numerator = float(np.sum((predicted - test_h) ** 2))
    train_mean = float(np.mean(train_h))
    denominator = float(np.sum((test_h - train_mean) ** 2)) + epsilon
    return HiddenMechanismMetric(
        scale=scale,
        offset=offset,
        test_nmse=numerator / denominator,
        allow_signed_scale=allow_signed_scale,
        train_sample_count=len(train_z),
        test_sample_count=len(test_z),
    )
