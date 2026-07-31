"""Shared strict schema primitives."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
    ),
]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=10_000)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrictSchema(BaseModel):
    """Immutable model that rejects undeclared fields and nonfinite numbers."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )

