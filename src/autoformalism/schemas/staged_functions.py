"""Minimal provider replies for functions on a frozen scientific topology."""

from __future__ import annotations

from pydantic import Field, FiniteFloat, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.candidate import ParameterRole


class FunctionParameter(StrictSchema):
    """A shared fitted identity and role, without scope, values or ranges."""

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


class FixedInitial(StrictSchema):
    """A fixed finite causal initial value."""

    fixed_value: FiniteFloat


class AnalyticInitial(StrictSchema):
    """A restricted expression evaluated only from allowed initial information."""

    expression: str = Field(min_length=1, max_length=4096)


class LatentInitialReply(StrictSchema):
    """Exactly one initialization mode for a runtime-selected latent state."""

    initial: FixedInitial | AnalyticInitial
