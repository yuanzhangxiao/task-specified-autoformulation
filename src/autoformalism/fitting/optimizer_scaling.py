"""Bounded, objective-magnitude-invariant scaling in dimensionless coordinates."""

import numpy as np
from pydantic import Field, model_validator

from autoformalism.schemas.base import StrictSchema


class RelativeScaling(StrictSchema):
    """Frozen per search block; these are numerical safeguards, not fitted priors."""

    exponent: float = Field(default=0.5, ge=0, le=1)
    relative_floor: float = Field(default=1e-6, gt=0, le=1)
    minimum: float = Field(default=0.1, gt=0, le=1)
    maximum: float = Field(default=10.0, ge=1, le=100)

    @model_validator(mode="after")
    def finite(self):
        if not np.isfinite(list(self.model_dump().values())).all():
            raise ValueError("scaling settings must be finite")
        return self


def relative_scales(jacobian: np.ndarray, policy: RelativeScaling) -> dict:
    """Balance relative RMS column sensitivities without an arbitrary loss unit.

    Divide by a common matrix magnitude before norms to avoid overflow. That
    common factor cancels in all ratios. Zero-information matrices use identity
    scales, explicitly diagnosed; weak columns never acquire unbounded steps.
    """
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or 0 in j.shape or not np.isfinite(j).all():
        raise ValueError("scaling requires a finite nonempty Jacobian matrix")
    magnitude = float(np.max(np.abs(j)))
    norms = (
        np.linalg.norm(j / magnitude, axis=0) / np.sqrt(len(j))
        if magnitude
        else (np.zeros(j.shape[1]))
    )
    positive = norms[norms > 0]
    reference = float(np.median(positive)) if len(positive) else 0.0
    relative = norms / reference if reference else np.zeros_like(norms)
    scales = (
        np.clip(
            np.maximum(relative, policy.relative_floor) ** -policy.exponent,
            policy.minimum,
            policy.maximum,
        )
        if reference
        else np.ones_like(norms)
    )
    return {
        "protocol": "guarded-relative-scaling-1",
        "policy": policy.model_dump(mode="json"),
        "scales": scales.tolist(),
        "relative_column_norms": relative.tolist(),
        "weak_columns": np.flatnonzero(relative <= policy.relative_floor).tolist(),
        "status": "relative_sensitivity" if reference else "zero_information_identity",
        "frozen_within_block": True,
    }
