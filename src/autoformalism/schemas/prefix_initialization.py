"""Lightweight M43 model and policy schemas, independent of numerical backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

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
    """Predeclared elapsed prefix duration and genuinely known future inputs."""

    duration: FiniteFloat = Field(ge=0)
    future_inputs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_inputs(self):
        if len(self.future_inputs) != len(set(self.future_inputs)):
            raise ValueError("duplicate future input")
        return self
