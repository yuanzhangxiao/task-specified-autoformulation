"""Provenance-aware public scientific requirements and candidate claims."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema


class RequirementSource(str, Enum):
    """Authority that introduced a scientific requirement or claim."""

    DOMAIN_EXPERT = "domain_expert"
    BENCHMARK = "benchmark"
    PROPOSER = "proposer"
    RUNTIME = "runtime"


class RequirementEnforcement(str, Enum):
    """How a public requirement participates in scientific selection."""

    HARD = "hard"
    SOFT = "soft"
    DESCRIPTIVE = "descriptive"


class ScientificRequirement(StrictSchema):
    """One frozen requirement extracted only from the public task contract."""

    requirement_id: Identifier
    text: NonEmptyText
    source: RequirementSource
    enforcement: RequirementEnforcement
    weight: float = Field(default=1.0, gt=0.0, le=1.0)


class RequirementRegistry(StrictSchema):
    """Frozen public requirements shared by every candidate in one task."""

    requirements: tuple[ScientificRequirement, ...] = Field(
        default=(), max_length=128
    )

    @model_validator(mode="after")
    def public_authority_only(self) -> RequirementRegistry:
        """Reject duplicate IDs and candidate/runtime-owned task requirements."""
        identifiers = [item.requirement_id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("public requirement identifiers must be unique")
        invalid = [
            item.requirement_id
            for item in self.requirements
            if item.source
            not in {RequirementSource.DOMAIN_EXPERT, RequirementSource.BENCHMARK}
        ]
        if invalid:
            raise ValueError(
                f"public requirements cannot be proposer/runtime owned: {invalid}"
            )
        return self


class ProposerClaim(StrictSchema):
    """One candidate-owned mechanism claim with no task authority."""

    claim_id: Identifier
    subject_id: Identifier
    mechanism: Identifier
    source: RequirementSource = RequirementSource.PROPOSER

    @model_validator(mode="after")
    def proposer_authority_only(self) -> ProposerClaim:
        """Prevent a candidate claim from escalating its own authority."""
        if self.source is not RequirementSource.PROPOSER:
            raise ValueError("candidate claims must have proposer provenance")
        return self
