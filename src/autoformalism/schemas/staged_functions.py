"""Minimal provider replies for functions on a frozen scientific topology."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.candidate import ParameterRole
from autoformalism.schemas.staged_topology import OuterWeightSign


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


class OuterWeightDomainDerivation(StrictSchema):
    """Runtime record for one sign-derived outer magnitude domain."""

    parameter: Identifier
    requested_role: ParameterRole
    effective_role: ParameterRole
    outer_weight_sign: OuterWeightSign


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
