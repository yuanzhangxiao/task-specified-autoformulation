"""Lightweight M43 model and policy schemas, independent of numerical backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.initialization_graph import InitializationGraph
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.candidate import CandidateModel


class PrefixFitProblem(StrictSchema):
    """Serializable model boundary with no validation/test data or legacy plan."""

    protocol: Literal["prefix-fit-problem-1"] = "prefix-fit-problem-1"
    base_candidate: CandidateModel
    context: ValidationContext
    graph: InitializationGraph
    parameter_guesses: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)


class PrefixPolicy(StrictSchema):
    """Predeclared prefix duration and separately authorized future forcing."""

    duration: FiniteFloat = Field(ge=0)
    future_inputs: tuple[Identifier, ...] = ()
    future_auxiliaries: tuple[Identifier, ...] = ()

    @model_serializer(mode="wrap")
    def serialize_policy(self, handler):
        """Keep historical fingerprints unchanged unless auxiliaries are enabled."""
        result = handler(self)
        if not self.future_auxiliaries:
            result.pop("future_auxiliaries", None)
        return result

    @model_validator(mode="after")
    def unique_inputs(self):
        if len(self.future_inputs) != len(set(self.future_inputs)):
            raise ValueError("duplicate future input")
        if len(self.future_auxiliaries) != len(set(self.future_auxiliaries)):
            raise ValueError("duplicate future auxiliary")
        if set(self.future_inputs) & set(self.future_auxiliaries):
            raise ValueError("future input and auxiliary roles overlap")
        return self
