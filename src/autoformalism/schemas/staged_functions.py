"""Minimal provider replies for functions on a frozen scientific topology."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema
from autoformalism.schemas.candidate import ParameterRole


class FunctionParameter(StrictSchema):
    """A fitted identity and role, without scope, values or ranges."""

    name: Identifier
    role: ParameterRole


class InteractionFunctionReply(StrictSchema):
    """One scalar expression; the runtime owns its interaction binding."""

    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[FunctionParameter, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def unique_parameters(self) -> InteractionFunctionReply:
        """Reject ambiguous local declarations before compiling expressions."""
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter declaration")
        return self


class DeterministicFunctionRepair(StrictSchema):
    """One semantics-preserving repair certified from the restricted AST."""

    schema_version: Literal["deterministic-function-repair-1"] = (
        "deterministic-function-repair-1"
    )
    code: Literal["OUTER_GAIN_ROLE_CERTIFIED"]
    parameter: Identifier
    original_role: Literal["coefficient"] = "coefficient"
    repaired_role: Literal["nonnegative_coefficient"] = "nonnegative_coefficient"
    certificate: Literal["single_direct_outer_multiplicative_gain"] = (
        "single_direct_outer_multiplicative_gain"
    )


class EquationFunctionBatchReply(StrictSchema):
    """Ordered scalar functions for every term of one runtime-selected LHS."""

    functions: tuple[InteractionFunctionReply, ...] = Field(
        min_length=1,
        max_length=8,
    )


class InteractionFunctionObligation(StrictSchema):
    """Runtime-owned, syntax-auditable requirements for one frozen term."""

    schema_version: Literal["interaction-function-obligation-1"] = (
        "interaction-function-obligation-1"
    )
    requires_nonlinear_source_dependence: bool = False
    parameter_identity_policy: Literal["preserve", "interaction_local"] = "preserve"
    provenance: tuple[str, ...] = Field(default=(), max_length=8)


class FixedInitial(StrictSchema):
    """A fixed finite causal initial value."""

    fixed_value: FiniteFloat


class AnalyticInitial(StrictSchema):
    """A restricted expression evaluated only from allowed initial information."""

    expression: str = Field(min_length=1, max_length=4096)


class LatentInitialReply(StrictSchema):
    """Exactly one initialization mode for a runtime-selected latent state."""

    initial: FixedInitial | AnalyticInitial


class PrefitReviewCategory(str, Enum):
    """Fixed public-scientific questions reviewed before parameter fitting."""

    MECHANISM_TOPOLOGY = "mechanism_topology"
    DIMENSIONAL_CONSISTENCY = "dimensional_consistency"
    DYNAMIC_PLAUSIBILITY = "dynamic_plausibility"
    FUNCTIONAL_SEMANTICS = "functional_semantics"
    PARAMETER_PARSIMONY = "parameter_parsimony"
    LATENT_INITIALIZATION = "latent_initialization"


class PrefitReviewFinding(StrictSchema):
    """One bounded scientific finding anchored to runtime-owned identities."""

    category: PrefitReviewCategory
    status: Literal["pass", "fail", "uncertain"]
    interaction_ids: tuple[Identifier, ...] = Field(default=(), max_length=3)
    latent_states: tuple[Identifier, ...] = Field(default=(), max_length=2)
    finding: NonEmptyText
    suggested_change: NonEmptyText

    @model_validator(mode="after")
    def unique_anchors(self) -> PrefitReviewFinding:
        """Keep a finding small and unambiguous before runtime validation."""
        if len(self.interaction_ids) != len(set(self.interaction_ids)):
            raise ValueError("duplicate interaction anchor")
        if len(self.latent_states) != len(set(self.latent_states)):
            raise ValueError("duplicate latent-state anchor")
        return self


class PrefitScientificReviewReply(StrictSchema):
    """Exactly one assessment for every fixed pre-fitting category."""

    schema_version: Literal["prefit-scientific-review-1"] = "prefit-scientific-review-1"
    findings: tuple[PrefitReviewFinding, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def complete_unique_categories(self) -> PrefitScientificReviewReply:
        """Reject omitted, duplicated, or invented rubric categories."""
        categories = [item.category for item in self.findings]
        if len(categories) != len(set(categories)):
            raise ValueError("duplicate prefit review category")
        if set(categories) != set(PrefitReviewCategory):
            missing = sorted(
                item.value for item in set(PrefitReviewCategory) - set(categories)
            )
            raise ValueError(f"missing prefit review categories: {missing}")
        return self
