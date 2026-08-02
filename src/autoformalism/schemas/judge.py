"""Strict structured judge-result schema."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

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
    description: NonEmptyText = "unspecified"
    evidence: NonEmptyText


class ActionableEdit(StrictSchema):
    """Concrete edit the proposer can apply in a later milestone."""

    target: Identifier
    instruction: NonEmptyText
    priority: ActionPriority


class CategoryScore(StrictSchema):
    """Numeric category assessment with an optional concise justification."""

    score: UnitInterval
    justification: NonEmptyText = "unspecified"


ScoreMap = Annotated[
    dict[Identifier, CategoryScore],
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

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_numeric_scores(cls, value: object) -> object:
        """Accept cached/mock V1 numeric scores while emitting the richer schema."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        scores = payload.get("category_scores")
        if isinstance(scores, dict):
            payload["category_scores"] = {
                name: ({"score": score} if isinstance(score, (int, float)) else score)
                for name, score in scores.items()
            }
        return payload

    @field_validator("category_scores")
    @classmethod
    def category_names_are_unique_after_normalization(
        cls, value: dict[str, CategoryScore]
    ) -> dict[str, CategoryScore]:
        """Return an ordinary dict after strict key/value validation."""
        return dict(value)

    @property
    def numeric_category_scores(self) -> dict[str, float]:
        """Return the score-only mapping used by ranking and proposer feedback."""
        return {name: item.score for name, item in self.category_scores.items()}
