"""Strict proposer-candidate schema."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import (
    FiniteFloat,
    Identifier,
    NonEmptyText,
    StrictSchema,
)


class StateKind(str, Enum):
    """Whether a modeled state is measured or inferred."""

    OBSERVED = "observed"
    LATENT = "latent"


class ParameterScope(str, Enum):
    """Sharing scope for a fitted scalar."""

    GLOBAL = "global"
    TRAJECTORY_SPECIFIC = "trajectory_specific"


class ConstraintKind(str, Enum):
    """Machine-readable constraint categories."""

    NONNEGATIVE = "nonnegative"
    POSITIVE = "positive"
    BOUNDED = "bounded"
    MONOTONE_INCREASING = "monotone_increasing"
    MONOTONE_DECREASING = "monotone_decreasing"
    CONSERVATION = "conservation"
    CUSTOM = "custom"


class ValueRange(StrictSchema):
    """Closed finite numeric interval."""

    lower: FiniteFloat
    upper: FiniteFloat

    @model_validator(mode="after")
    def lower_precedes_upper(self) -> ValueRange:
        """Require a nondegenerate interval."""
        if self.lower >= self.upper:
            raise ValueError("range lower must be less than upper")
        return self

    def contains(self, other: ValueRange) -> bool:
        """Return whether another interval lies inside this interval."""
        return self.lower <= other.lower and other.upper <= self.upper


class StateSpec(StrictSchema):
    """Observed or latent dynamic state declaration."""

    name: Identifier
    kind: StateKind
    unit: NonEmptyText
    description: NonEmptyText


class ProcessSpec(StrictSchema):
    """Algebraic generated process evaluated from declared symbols."""

    name: Identifier
    expression: NonEmptyText
    unit: NonEmptyText
    description: NonEmptyText


class StateEquation(StrictSchema):
    """Right-hand side for exactly one declared state."""

    state: Identifier
    rhs: NonEmptyText


class ObservationMapping(StrictSchema):
    """Mapping from model symbols to one measured target channel."""

    channel: Identifier
    expression: NonEmptyText
    unit: NonEmptyText


class ParameterSpec(StrictSchema):
    """Bounded scalar parameter and its initialization interval."""

    name: Identifier
    scope: ParameterScope
    bounds: ValueRange
    initialization_range: ValueRange
    unit: NonEmptyText
    description: NonEmptyText

    @model_validator(mode="after")
    def initialization_is_within_bounds(self) -> ParameterSpec:
        """Keep every proposed optimizer start inside declared bounds."""
        if not self.bounds.contains(self.initialization_range):
            raise ValueError("parameter initialization_range must lie within bounds")
        return self


class InitialConditionSpec(StrictSchema):
    """Bounded initialization range for a modeled state."""

    state: Identifier
    scope: ParameterScope
    initialization_range: ValueRange


class ConstraintSpec(StrictSchema):
    """Constraint attached to a declared model symbol."""

    subject: Identifier
    kind: ConstraintKind
    description: NonEmptyText
    bounds: ValueRange | None = None

    @model_validator(mode="after")
    def bounds_match_kind(self) -> ConstraintSpec:
        """Require bounds exactly for bounded constraints."""
        if self.kind is ConstraintKind.BOUNDED and self.bounds is None:
            raise ValueError("bounded constraint requires bounds")
        if self.kind is not ConstraintKind.BOUNDED and self.bounds is not None:
            raise ValueError("bounds are only valid for a bounded constraint")
        return self


class CandidateModel(StrictSchema):
    """Complete machine-readable proposer candidate."""

    schema_version: Literal["1"] = "1"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None
    change_summary: NonEmptyText
    states: tuple[StateSpec, ...] = Field(min_length=1, max_length=64)
    processes: tuple[ProcessSpec, ...] = Field(default=(), max_length=256)
    state_equations: tuple[StateEquation, ...] = Field(
        min_length=1, max_length=64
    )
    observation_mappings: tuple[ObservationMapping, ...] = Field(
        min_length=1, max_length=64
    )
    parameters: tuple[ParameterSpec, ...] = Field(default=(), max_length=256)
    initial_conditions: tuple[InitialConditionSpec, ...] = Field(
        min_length=1, max_length=64
    )
    constraints: tuple[ConstraintSpec, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_model_references(self) -> CandidateModel:
        """Validate uniqueness, equation closure, and declaration references."""
        state_names = self._unique_names("state", self.states)
        process_names = self._unique_names("process", self.processes)
        parameter_names = self._unique_names("parameter", self.parameters)
        observation_names = self._unique_attributes(
            "observation channel", self.observation_mappings, "channel"
        )
        namespaces = {
            "states": state_names,
            "processes": process_names,
            "parameters": parameter_names,
        }
        all_names: set[str] = set()
        for namespace, names in namespaces.items():
            overlap = all_names & names
            if overlap:
                raise ValueError(
                    f"{namespace} reuse declared names: {sorted(overlap)}"
                )
            all_names.update(names)

        equation_states = self._unique_attributes(
            "state equation", self.state_equations, "state"
        )
        if equation_states != state_names:
            missing = sorted(state_names - equation_states)
            unknown = sorted(equation_states - state_names)
            raise ValueError(
                f"state equations must cover every state; "
                f"missing={missing}, unknown={unknown}"
            )

        initial_states = self._unique_attributes(
            "initial condition", self.initial_conditions, "state"
        )
        if initial_states != state_names:
            missing = sorted(state_names - initial_states)
            unknown = sorted(initial_states - state_names)
            raise ValueError(
                f"initial conditions must cover every state; "
                f"missing={missing}, unknown={unknown}"
            )
        state_by_name = {state.name: state for state in self.states}
        for initial in self.initial_conditions:
            if (
                initial.scope is ParameterScope.TRAJECTORY_SPECIFIC
                and state_by_name[initial.state].kind is not StateKind.LATENT
            ):
                raise ValueError(
                    "trajectory-specific initial conditions are allowed only "
                    f"for latent states: {initial.state}"
                )

        constraint_subjects = all_names | observation_names
        unknown_subjects = sorted(
            {
                constraint.subject
                for constraint in self.constraints
                if constraint.subject not in constraint_subjects
            }
        )
        if unknown_subjects:
            raise ValueError(
                f"constraints reference undeclared subjects: {unknown_subjects}"
            )
        return self

    @staticmethod
    def _unique_names(label: str, values: tuple[object, ...]) -> set[str]:
        return CandidateModel._unique_attributes(label, values, "name")

    @staticmethod
    def _unique_attributes(
        label: str,
        values: tuple[object, ...],
        attribute: str,
    ) -> set[str]:
        names = [str(getattr(value, attribute)) for value in values]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate {label}: {duplicates}")
        return set(names)

