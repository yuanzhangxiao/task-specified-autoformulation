"""Strict structured judge-result schema."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, field_validator

from autoformalism.schemas.base import (
    Identifier,
    NonEmptyText,
    StrictSchema,
    UnitInterval,
)


class ActionPriority(str, Enum):
    """Priority assigned to a requested candidate edit."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class HardRedFlag(StrictSchema):
    """Blocking or severe specification violation."""

    code: Identifier
    description: NonEmptyText
    evidence: NonEmptyText


class ActionableEdit(StrictSchema):
    """Concrete edit the proposer can apply in a later milestone."""

    target: Identifier
    instruction: NonEmptyText
    priority: ActionPriority


ScoreMap = Annotated[
    dict[Identifier, UnitInterval],
    Field(min_length=1, max_length=64),
]


class JudgeResult(StrictSchema):
    """Structured task-compliance assessment independent of fit metrics."""

    schema_version: Literal["1"] = "1"
    hard_red_flags: tuple[HardRedFlag, ...] = Field(default=(), max_length=64)
    category_scores: ScoreMap
    aggregate_score: UnitInterval
    missing_requirements: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=128
    )
    actionable_edits: tuple[ActionableEdit, ...] = Field(
        default=(), max_length=128
    )

    @field_validator("category_scores")
    @classmethod
    def category_names_are_unique_after_normalization(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        """Return an ordinary dict after strict key/value validation."""
        return dict(value)

