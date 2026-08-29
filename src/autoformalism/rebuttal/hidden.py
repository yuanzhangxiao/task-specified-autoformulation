"""Post-selection hidden-mechanism alignment metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
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


class HiddenSubspaceMetric(BaseModel):
    """Training-aligned recovery of a private mechanism-response subspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claimed_dimension: int = Field(gt=0)
    rank_tolerance: float = Field(gt=0.0)
    reference_rank: int = Field(ge=0)
    candidate_direction_count: int = Field(ge=0)
    candidate_rank: int = Field(ge=0)
    candidate_rank_coverage: float = Field(ge=0.0, le=1.0)
    structurally_recovered: bool
    recovered: bool
    train_relative_residual: float | None = Field(default=None, ge=0.0)
    aligned_test_nmse: float | None = Field(default=None, ge=0.0)
    train_sample_count: int = Field(gt=0)
    test_sample_count: int = Field(gt=0)

    @model_validator(mode="after")
    def score_requires_recovery(self) -> HiddenSubspaceMetric:
        if self.recovered and self.aligned_test_nmse is None:
            raise ValueError("recovered subspace requires aligned test NMSE")
        if not self.recovered and self.aligned_test_nmse is not None:
            raise ValueError("unrecovered subspace cannot carry aligned test NMSE")
        return self


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


def hidden_subspace_nmse(
    train_candidate: NDArray[np.float64],
    train_reference_directions: NDArray[np.float64],
    test_candidate: NDArray[np.float64],
    test_reference_directions: NDArray[np.float64],
    *,
    claimed_dimension: int,
    structurally_recovered: bool,
    rank_tolerance: float = 1e-3,
    epsilon: float = 1e-12,
) -> HiddenSubspaceMetric:
    """Align candidate mechanism sensitivities on train and score them on test.

    Columns are local response directions and rows are normalized
    trajectory/time/target samples. The private training directions define the
    claimed right-singular subspace. A least-squares map from candidate
    directions to that subspace is fitted on training rows and applied without
    refitting to held-out rows.
    """
    if claimed_dimension <= 0:
        raise ValueError("claimed dimension must be positive")
    if rank_tolerance <= 0.0 or epsilon <= 0.0:
        raise ValueError("numeric tolerances must be positive")
    arrays = tuple(np.asarray(value, dtype=float) for value in (
        train_candidate,
        train_reference_directions,
        test_candidate,
        test_reference_directions,
    ))
    train_c, train_r, test_c, test_r = arrays
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("hidden-subspace direction arrays must be matrices")
    if train_c.shape[0] != train_r.shape[0]:
        raise ValueError("training candidate/reference sample counts differ")
    if test_c.shape[0] != test_r.shape[0]:
        raise ValueError("test candidate/reference sample counts differ")
    if train_c.shape[1] != test_c.shape[1]:
        raise ValueError("candidate direction counts differ across splits")
    if train_r.shape[1] != test_r.shape[1]:
        raise ValueError("reference direction counts differ across splits")
    if not train_c.shape[0] or not test_c.shape[0]:
        raise ValueError("train and test direction matrices must be nonempty")
    if claimed_dimension > train_r.shape[1]:
        raise ValueError("claimed dimension exceeds reference direction count")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("hidden-subspace direction arrays must be finite")

    _, singular_values, right_vectors = np.linalg.svd(train_r, full_matrices=False)
    leading_reference = max(float(singular_values[0]), epsilon)
    reference_rank = int(
        np.count_nonzero(singular_values >= rank_tolerance * leading_reference)
    )
    if reference_rank < claimed_dimension:
        raise ValueError(
            "private reference does not support its claimed subspace dimension"
        )
    private_basis = right_vectors[:claimed_dimension].T
    train_target = train_r @ private_basis
    test_target = test_r @ private_basis
    candidate_count = train_c.shape[1]
    if candidate_count:
        candidate_singular = np.linalg.svd(train_c, compute_uv=False)
        leading_candidate = max(float(candidate_singular[0]), epsilon)
        candidate_rank = int(
            np.count_nonzero(
                candidate_singular >= rank_tolerance * leading_candidate
            )
        )
        alignment, *_ = np.linalg.lstsq(train_c, train_target, rcond=rank_tolerance)
        train_prediction = train_c @ alignment
        train_residual = float(np.sum((train_prediction - train_target) ** 2)) / (
            float(np.sum(train_target**2)) + epsilon
        )
    else:
        candidate_rank = 0
        alignment = np.empty((0, claimed_dimension), dtype=float)
        train_residual = None
    recovered = structurally_recovered and candidate_rank >= claimed_dimension
    test_nmse = None
    if recovered:
        prediction = test_c @ alignment
        test_nmse = float(np.sum((prediction - test_target) ** 2)) / (
            float(np.sum(test_target**2)) + epsilon
        )
    return HiddenSubspaceMetric(
        claimed_dimension=claimed_dimension,
        rank_tolerance=rank_tolerance,
        reference_rank=reference_rank,
        candidate_direction_count=candidate_count,
        candidate_rank=candidate_rank,
        candidate_rank_coverage=min(candidate_rank / claimed_dimension, 1.0),
        structurally_recovered=structurally_recovered,
        recovered=recovered,
        train_relative_residual=train_residual,
        aligned_test_nmse=test_nmse,
        train_sample_count=train_c.shape[0],
        test_sample_count=test_c.shape[0],
    )
