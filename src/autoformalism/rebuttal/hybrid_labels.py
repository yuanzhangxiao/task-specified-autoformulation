"""Human-auditable question-level labels for hybrid judge calibration."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    RelativeCriterion,
)
from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema


class ExpectedVerdict(str, Enum):
    """Gold absolute verdict, with an explicit unreviewed template state."""

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"
    UNLABELED = "unlabeled"


class ExpectedPairPreference(str, Enum):
    """Gold pairwise outcome normalized to baseline/mutated identities."""

    BASELINE = "baseline"
    MUTATED = "mutated"
    TIE = "tie"
    INDETERMINATE = "indeterminate"
    UNLABELED = "unlabeled"


class ExpectedAbsoluteLabel(StrictSchema):
    """Question-level truth for both members of one calibration pair."""

    criterion: AbsoluteCriterion
    subject_id: Identifier
    baseline: ExpectedVerdict
    mutated: ExpectedVerdict
    rationale: NonEmptyText
    label_source: NonEmptyText


class ExpectedComparativeLabel(StrictSchema):
    """Expected outcome for one irreducibly comparative question."""

    criterion: RelativeCriterion
    preference: ExpectedPairPreference
    rationale: NonEmptyText
    label_source: NonEmptyText


class HybridCalibrationLabels(StrictSchema):
    """Frozen item-level and overall truth for one blinded pair."""

    schema_version: Literal["hybrid-labels-1"] = "hybrid-labels-1"
    pair_id: Identifier
    overall_preference: ExpectedPairPreference
    absolute_labels: tuple[ExpectedAbsoluteLabel, ...] = Field(max_length=256)
    comparative_labels: tuple[ExpectedComparativeLabel, ...] = Field(
        max_length=len(RelativeCriterion)
    )

    @model_validator(mode="after")
    def unique_labels(self) -> HybridCalibrationLabels:
        """Reject ambiguous duplicate gold labels."""
        absolute = [
            (item.criterion, item.subject_id) for item in self.absolute_labels
        ]
        if len(absolute) != len(set(absolute)):
            raise ValueError("absolute gold-label keys must be unique")
        comparative = [item.criterion for item in self.comparative_labels]
        if len(comparative) != len(set(comparative)):
            raise ValueError("comparative gold-label criteria must be unique")
        return self


def expected_from_runtime(verdict: AbsoluteVerdict) -> ExpectedVerdict:
    """Convert one certified runtime verdict to a gold-label value."""
    return ExpectedVerdict(verdict.value)
