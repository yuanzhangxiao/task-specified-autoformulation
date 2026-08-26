"""Strict proposer-candidate schema."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

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


class ConstraintSource(str, Enum):
    """Runtime-owned provenance for a constraint declaration."""

    UNSPECIFIED = "unspecified"
    PROPOSER = "proposer"
    BENCHMARK = "benchmark"
    RUNTIME = "runtime"
    DETERMINISTIC = "deterministic"


class ConstraintEnforcement(str, Enum):
    """Whether violation blocks execution or contributes soft evidence."""

    HARD = "hard"
    SOFT = "soft"


class ValueRange(StrictSchema):
    """Closed finite numeric interval."""

    lower: FiniteFloat
    upper: FiniteFloat

    @model_validator(mode="after")
    def lower_precedes_upper(self) -> ValueRange:
        """Require an ordered interval; equality represents a fixed value."""
        if self.lower > self.upper:
            raise ValueError("range lower must not exceed upper")
        return self

    def contains(self, other: ValueRange) -> bool:
        """Return whether another interval lies inside this interval."""
        return self.lower <= other.lower and other.upper <= self.upper


class StateSpec(StrictSchema):
    """Observed or latent dynamic state declaration."""

    name: Identifier
    kind: StateKind
    unit: NonEmptyText = "unspecified"
    description: NonEmptyText = "unspecified"
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class ProcessSpec(StrictSchema):
    """Algebraic generated process evaluated from declared symbols."""

    name: Identifier
    expression: NonEmptyText
    unit: NonEmptyText = "unspecified"
    description: NonEmptyText = "unspecified"
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class StateEquation(StrictSchema):
    """Canonical derivative equation for exactly one declared state."""

    state: Identifier
    rhs: NonEmptyText


class ProposedEquation(StrictSchema):
    """Proposer-facing equation whose left-hand side defines its semantics.

    ``dx/dt`` or ``d(x)/dt`` is a derivative equation. A plain identifier,
    including ``x_rate``, is an algebraic process unless ``derivative_of``
    explicitly links that process to a declared dynamic state.
    """

    lhs: NonEmptyText
    rhs: NonEmptyText
    derivative_of: Identifier | None = None


class ObservationMapping(StrictSchema):
    """Mapping from model symbols to one measured target channel."""

    channel: Identifier
    expression: NonEmptyText
    unit: NonEmptyText = "unspecified"


class ParameterSpec(StrictSchema):
    """Bounded scalar parameter and its initialization interval."""

    name: Identifier
    scope: ParameterScope
    bounds: ValueRange
    initialization_range: ValueRange
    unit: NonEmptyText = "unspecified"
    description: NonEmptyText = "unspecified"

    @model_validator(mode="after")
    def initialization_is_within_bounds(self) -> ParameterSpec:
        """Keep every proposed optimizer start inside declared bounds."""
        if self.bounds.lower == self.bounds.upper:
            raise ValueError("parameter bounds must be nondegenerate")
        if not self.bounds.contains(self.initialization_range):
            raise ValueError("parameter initialization_range must lie within bounds")
        return self


class InitialConditionSpec(StrictSchema):
    """One explicit initialization rule for a modeled state."""

    state: Identifier
    scope: ParameterScope
    fixed_value: FiniteFloat | None = None
    expression: NonEmptyText | None = None
    initialization_range: ValueRange | None = None

    @model_validator(mode="after")
    def exactly_one_initialization_mode(self) -> InitialConditionSpec:
        """Reject ambiguous modes; contextual validation handles a missing mode."""
        modes = (
            self.fixed_value is not None,
            self.expression is not None,
            self.initialization_range is not None,
        )
        if sum(modes) > 1:
            raise ValueError(
                "initial condition allows at most one of fixed_value, "
                "expression, or initialization_range"
            )
        return self


class ConstraintSpec(StrictSchema):
    """Constraint attached to a declared model symbol."""

    subject: Identifier
    kind: ConstraintKind
    description: NonEmptyText = "unspecified"
    bounds: ValueRange | None = None
    source: ConstraintSource = ConstraintSource.UNSPECIFIED
    enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD

    @model_validator(mode="after")
    def bounds_match_kind(self) -> ConstraintSpec:
        """Require explicit bounds for bounded constraints."""
        if self.kind is ConstraintKind.BOUNDED and self.bounds is None:
            raise ValueError("bounded constraint requires bounds")
        return self


class CandidateModel(StrictSchema):
    """Complete machine-readable proposer candidate."""

    schema_version: Literal["1"] = "1"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None
    change_summary: NonEmptyText = "unspecified"
    states: tuple[StateSpec, ...] = Field(min_length=1, max_length=64)
    processes: tuple[ProcessSpec, ...] = Field(default=(), max_length=256)
    state_equations: tuple[StateEquation, ...] = Field(
        default=(),
        max_length=64,
        description=(
            "Canonical/legacy derivative equations; proposers should use equations."
        ),
    )
    equations: tuple[ProposedEquation, ...] = Field(
        default=(),
        max_length=320,
        description=(
            "Explicit-LHS proposer equations. Derivatives use dx/dt or d(x)/dt. "
            "Plain identifiers, including x_rate, are algebraic unless linked "
            "with derivative_of. Normalized on input."
        ),
    )
    observation_mappings: tuple[ObservationMapping, ...] = Field(
        min_length=1, max_length=64
    )
    parameters: tuple[ParameterSpec, ...] = Field(default=(), max_length=256)
    initial_conditions: tuple[InitialConditionSpec, ...] = Field(
        default=(), max_length=64
    )
    constraints: tuple[ConstraintSpec, ...] = Field(default=(), max_length=256)

    @model_validator(mode="before")
    @classmethod
    def normalize_explicit_equations(cls, value: Any) -> Any:
        """Normalize explicit-LHS equations into the canonical representation."""
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        states = [dict(item) for item in payload.get("states", ())]
        state_names = {str(item.get("name")) for item in states}
        identity_channels: dict[str, str] = {}
        for mapping in payload.get("observation_mappings", ()):
            item = mapping.model_dump(mode="json") if isinstance(
                mapping, ObservationMapping
            ) else dict(mapping)
            expression = re.sub(r"\s+", "", str(item.get("expression", "")))
            if expression in state_names:
                identity_channels[expression] = str(item.get("channel"))
        initials: list[Any] = []
        for original in payload.get("initial_conditions", ()):
            item = (
                original.model_dump(mode="json")
                if isinstance(original, InitialConditionSpec)
                else dict(original)
            )
            channel = identity_channels.get(str(item.get("state")))
            missing_mode = (
                item.get("fixed_value") is None
                and item.get("expression") is None
                and item.get("initialization_range") is None
            )
            if channel is not None and (
                item.get("scope") == ParameterScope.TRAJECTORY_SPECIFIC.value
                or missing_mode
            ):
                # Identity-mapped states are reset from the measured channel at
                # every one-step interval. The data contract is authoritative:
                # never fit, preserve, or invent a separate initial value for
                # them.
                item["scope"] = ParameterScope.GLOBAL.value
                item["fixed_value"] = None
                item["initialization_range"] = None
                item["expression"] = channel
            initials.append(item)
        payload["initial_conditions"] = initials

        if not payload.get("equations"):
            return payload

        processes = [dict(item) for item in payload.get("processes", ())]
        state_equations = [
            dict(item) for item in payload.get("state_equations", ())
        ]
        identifier = r"[A-Za-z_][A-Za-z0-9_]*"

        for equation in payload["equations"]:
            item = equation.model_dump(mode="json") if isinstance(
                equation, ProposedEquation
            ) else dict(equation)
            lhs = re.sub(r"\s+", "", str(item["lhs"]))
            rhs = item["rhs"]
            derivative = re.fullmatch(rf"d\(({identifier})\)/dt", lhs)
            if derivative is None:
                derivative = re.fullmatch(rf"d({identifier})/dt", lhs)
            if derivative is not None:
                state = derivative.group(1)
                derivative_of = item.get("derivative_of")
                if derivative_of is not None and derivative_of != state:
                    raise ValueError(
                        f"derivative_of={derivative_of!r} conflicts with "
                        f"left-hand side derivative of {state!r}"
                    )
                if state not in state_names:
                    raise ValueError(
                        "derivative left-hand side references undeclared state: "
                        f"{state}"
                    )
                state_equations.append({"state": state, "rhs": rhs})
                continue

            if re.fullmatch(identifier, lhs) is None:
                raise ValueError(f"unsupported equation left-hand side: {item['lhs']}")
            derivative_of = item.get("derivative_of")
            if derivative_of is not None:
                if derivative_of not in state_names:
                    raise ValueError(
                        "derivative_of references undeclared state: "
                        f"{derivative_of}"
                    )
                if lhs in state_names:
                    if lhs == derivative_of:
                        raise ValueError(
                            "a derivative process must have a different name "
                            f"from its state: {lhs}"
                        )
                    # ``derivative_of`` is authoritative. Weak proposers often
                    # redundantly list the named rate itself as a latent state;
                    # keeping both would introduce unintended higher-order
                    # dynamics.
                    states = [
                        state for state in states if state.get("name") != lhs
                    ]
                    state_names.remove(lhs)
                    state_equations = [
                        equation
                        for equation in state_equations
                        if equation.get("state") != lhs
                    ]
                    payload["initial_conditions"] = [
                        initial
                        for initial in payload.get("initial_conditions", ())
                        if (
                            initial.state
                            if isinstance(initial, InitialConditionSpec)
                            else initial.get("state")
                        )
                        != lhs
                    ]
                processes.append(
                    {
                        "name": lhs,
                        "expression": rhs,
                        "description": f"Derivative rate for {derivative_of}.",
                    }
                )
                state_equations.append({"state": derivative_of, "rhs": lhs})
                continue

            if lhs in state_names:
                # The explicit LHS wins: plain ``x`` is algebraic, even if a
                # weaker proposer redundantly listed x as a state.
                states = [item for item in states if item.get("name") != lhs]
                state_names.remove(lhs)
                state_equations = [
                    item for item in state_equations if item.get("state") != lhs
                ]
                payload["initial_conditions"] = [
                    item
                    for item in payload.get("initial_conditions", ())
                    if (
                        item.state if isinstance(item, InitialConditionSpec)
                        else item.get("state")
                    )
                    != lhs
                ]
            processes.append({"name": lhs, "expression": rhs})

        payload["states"] = states
        unique_processes: list[dict[str, Any]] = []
        process_expressions: dict[str, str] = {}
        for process in processes:
            name = str(process.get("name"))
            expression = str(process.get("expression"))
            previous = process_expressions.get(name)
            if previous is not None:
                if previous != expression:
                    raise ValueError(
                        f"conflicting algebraic definitions for process {name}"
                    )
                continue
            process_expressions[name] = expression
            unique_processes.append(process)
        payload["processes"] = unique_processes
        payload["parameters"] = [
            item
            for item in payload.get("parameters", ())
            if (
                item.name if isinstance(item, ParameterSpec) else item.get("name")
            )
            not in process_expressions
        ]
        payload["state_equations"] = state_equations
        # Canonical candidates store only the normalized representation. This
        # avoids stale duplicate equations during pruning and checkpoint resume.
        payload["equations"] = []
        return payload

    @model_validator(mode="after")
    def validate_model_references(self) -> CandidateModel:
        """Validate uniqueness, equation closure, and declaration references."""
        state_names = self._unique_names("state", self.states)
        self._unique_names("process", self.processes)
        self._unique_names("parameter", self.parameters)
        self._unique_attributes(
            "observation channel", self.observation_mappings, "channel"
        )
        self._unique_attributes(
            "state equation", self.state_equations, "state"
        )
        # Equation closure depends on the benchmark channel contract. It is
        # checked by CandidateValidator after supplied-channel declarations
        # have been repaired at the runtime boundary.

        self._unique_attributes(
            "initial condition", self.initial_conditions, "state"
        )
        state_by_name = {state.name: state for state in self.states}
        identity_mapped_states = {
            mapping.expression.strip()
            for mapping in self.observation_mappings
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", mapping.expression.strip()
            )
            and mapping.expression.strip() in state_names
        }
        for initial in self.initial_conditions:
            if (
                initial.scope is ParameterScope.TRAJECTORY_SPECIFIC
                and initial.state in state_by_name
                and state_by_name[initial.state].kind is not StateKind.LATENT
                and initial.state not in identity_mapped_states
            ):
                raise ValueError(
                    "trajectory-specific initial conditions are allowed only "
                    f"for latent states: {initial.state}"
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
