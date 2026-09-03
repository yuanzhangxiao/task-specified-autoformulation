"""Compact proposer-facing contract and deterministic canonical enrichment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import (
    FiniteFloat,
    Identifier,
    NonEmptyText,
    StrictSchema,
)
from autoformalism.schemas.candidate import (
    CandidateModel,
    ConstraintEnforcement,
    ConstraintKind,
    ConstraintSource,
    ConstraintSpec,
    InitialConditionSpec,
    ObservationMapping,
    ParameterScope,
    ParameterSpec,
    ProcessSpec,
    StateEquation,
    StateKind,
    StateSpec,
    ValueRange,
)


class EmbeddedConstraint(StrictSchema):
    """Constraint embedded in the declaration of the symbol it governs."""

    kind: ConstraintKind
    bounds: ValueRange | None = None
    description: NonEmptyText = "unspecified"

    @model_validator(mode="after")
    def bounds_match_kind(self) -> EmbeddedConstraint:
        if self.kind is ConstraintKind.BOUNDED and self.bounds is None:
            raise ValueError("bounded constraint requires bounds")
        return self


class ProposedState(StrictSchema):
    """A dynamic state with its equation and constraints embedded."""

    name: Identifier
    kind: StateKind
    rhs: NonEmptyText
    initial_value: FiniteFloat | None = None
    initial_expression: NonEmptyText | None = None
    initial_scope: ParameterScope = ParameterScope.GLOBAL
    unit: NonEmptyText = "unspecified"
    description: NonEmptyText = "unspecified"
    constraints: tuple[EmbeddedConstraint, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def one_initialization_mode(self) -> ProposedState:
        """A state may have a fixed or analytic initialization, but not both."""
        if self.initial_value is not None and self.initial_expression is not None:
            raise ValueError("state allows only one initialization mode")
        return self


class ProposedProcess(StrictSchema):
    """A named algebraic expression."""

    name: Identifier
    expression: NonEmptyText
    constraints: tuple[EmbeddedConstraint, ...] = Field(default=(), max_length=16)


class ProposedObservation(StrictSchema):
    """Prediction mapping for a measured target channel."""

    channel: Identifier
    expression: NonEmptyText


class ProposedParameter(StrictSchema):
    """A fitted scalar identity; numerical values belong to the runtime."""

    name: Identifier
    scope: ParameterScope = ParameterScope.GLOBAL
    # Accepted only so historical stored responses remain readable. These are
    # excluded from serialization, enrichment, and the provider-facing schema.
    bounds: ValueRange | None = Field(default=None, exclude=True, repr=False)
    initialization_range: ValueRange | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_ranges(cls, value: Any) -> Any:
        """Read old payloads without preserving obsolete proposer metadata."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.pop("bounds", None)
        normalized.pop("initialization_range", None)
        return normalized

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> dict[str, Any]:
        """Hide obsolete numeric ranges from every structured-output provider."""
        schema = handler(core_schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("bounds", None)
            properties.pop("initialization_range", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                item
                for item in required
                if item not in {"bounds", "initialization_range"}
            ]
        return schema


class ProposerCandidate(StrictSchema):
    """Small LLM-facing model contract; canonical details are derived locally."""

    schema_version: Literal["1"] = "1"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    states: tuple[ProposedState, ...] = Field(min_length=1, max_length=64)
    processes: tuple[ProposedProcess, ...] = Field(default=(), max_length=256)
    observations: tuple[ProposedObservation, ...] = Field(min_length=1, max_length=64)
    parameters: tuple[ProposedParameter, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def declarations_are_unique(self) -> ProposerCandidate:
        """Reject ambiguous duplicate declarations inside the small contract."""
        for label, names in (
            ("state", [item.name for item in self.states]),
            ("process", [item.name for item in self.processes]),
            ("parameter", [item.name for item in self.parameters]),
            ("observation", [item.channel for item in self.observations]),
        ):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label}: {duplicates}")
        return self


def enrich_proposal(proposal: ProposerCandidate) -> CandidateModel:
    """Convert compact scientific intent to the complete canonical schema."""
    initials = []
    for state in proposal.states:
        if state.initial_value is None and state.initial_expression is None:
            continue
        initials.append(
            InitialConditionSpec(
                state=state.name,
                scope=(
                    ParameterScope.GLOBAL
                    if state.kind is StateKind.OBSERVED
                    else state.initial_scope
                ),
                fixed_value=state.initial_value,
                expression=state.initial_expression,
            )
        )
    parameters = [
        ParameterSpec(name=parameter.name, scope=parameter.scope)
        for parameter in proposal.parameters
    ]
    return CandidateModel(
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.parent_candidate_id,
        change_summary=proposal.change_summary,
        states=tuple(
            StateSpec(
                name=state.name,
                kind=state.kind,
                unit=state.unit,
                description=state.description,
            )
            for state in proposal.states
        ),
        processes=tuple(
            ProcessSpec(name=item.name, expression=item.expression)
            for item in proposal.processes
        ),
        state_equations=tuple(
            StateEquation(state=state.name, rhs=state.rhs)
            for state in proposal.states
        ),
        observation_mappings=tuple(
            ObservationMapping(channel=item.channel, expression=item.expression)
            for item in proposal.observations
        ),
        parameters=tuple(parameters),
        initial_conditions=tuple(initials),
        constraints=tuple(
            ConstraintSpec(
                subject=declaration.name,
                kind=constraint.kind,
                bounds=constraint.bounds,
                description=constraint.description,
                source=ConstraintSource.PROPOSER,
                enforcement=ConstraintEnforcement.SOFT,
            )
            for declaration in (*proposal.states, *proposal.processes)
            for constraint in declaration.constraints
        ),
    )


class ProposedInitialValue(StrictSchema):
    """Fixed or analytic initialization for one latent state."""

    fixed_value: FiniteFloat | None = None
    expression: NonEmptyText | None = None

    @model_validator(mode="after")
    def exactly_one_mode(self) -> ProposedInitialValue:
        if (self.fixed_value is None) == (self.expression is None):
            raise ValueError(
                "initial requires exactly one of fixed_value or expression"
            )
        return self


class ProposedStateV2(StrictSchema):
    """V2 state: dynamics, data identity, initialization, and semantics together."""

    name: Identifier
    kind: StateKind
    rhs: NonEmptyText
    observed_channel: Identifier | None = None
    initial: ProposedInitialValue | None = None
    constraints: tuple[EmbeddedConstraint, ...] = Field(default=(), max_length=16)
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode="before")
    @classmethod
    def discard_observed_initializer(cls, value: Any) -> Any:
        """Infer routine observed metadata and discard inert initialization."""
        if isinstance(value, dict) and value.get("kind") == StateKind.OBSERVED.value:
            normalized = dict(value)
            if normalized.get("observed_channel") is None and normalized.get("name"):
                normalized["observed_channel"] = normalized["name"]
            normalized.pop("initial", None)
            return normalized
        return value

    @model_validator(mode="after")
    def initialization_matches_kind(self) -> ProposedStateV2:
        if self.kind is StateKind.OBSERVED:
            if self.observed_channel is None:
                raise ValueError("observed state requires observed_channel")
            if self.initial is not None:  # defensive for non-dict model inputs
                raise ValueError("observed state must omit initial")
        else:
            if self.observed_channel is not None:
                raise ValueError("latent state must omit observed_channel")
            if self.initial is None:
                raise ValueError("latent state requires initial")
        return self


class ProposedAlgebraicV2(StrictSchema):
    """Instantaneous named expression, optionally producing a target channel."""

    name: Identifier
    expression: NonEmptyText
    constraints: tuple[EmbeddedConstraint, ...] = Field(default=(), max_length=16)
    mechanisms: tuple[Identifier, ...] = Field(default=(), max_length=32)


class ProposerCandidateV2(StrictSchema):
    """Minimal executable proposer contract for deterministic ODE systems."""

    schema_version: Literal["2"] = "2"
    candidate_id: Identifier
    parent_candidate_id: Identifier | None = None
    change_summary: NonEmptyText = "unspecified"
    states: tuple[ProposedStateV2, ...] = Field(min_length=1, max_length=64)
    algebraics: tuple[ProposedAlgebraicV2, ...] = Field(default=(), max_length=256)
    parameters: tuple[ProposedParameter, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def declarations_and_outputs_are_unique(self) -> ProposerCandidateV2:
        namespaces = {
            "state": [item.name for item in self.states],
            "algebraic": [item.name for item in self.algebraics],
            "parameter": [item.name for item in self.parameters],
        }
        for label, names in namespaces.items():
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label}: {duplicates}")
        declared = [name for names in namespaces.values() for name in names]
        collisions = sorted({name for name in declared if declared.count(name) > 1})
        if collisions:
            raise ValueError(f"declaration names collide: {collisions}")
        return self


@dataclass(frozen=True)
class ProposerPreSchemaRepair:
    """One lossless normalization applied before strict V2 validation."""

    code: Literal[
        "removed_exact_duplicate_parameter",
        "removed_legacy_parameter_ranges",
        "removed_protected_parameter",
    ]
    parameter_name: str
    removed_count: int

    def as_json(self) -> dict[str, object]:
        """Return a stable JSON record for logs and frozen replay reports."""
        return asdict(self)


def normalize_proposer_candidate_v2_payload(
    payload: object,
    *,
    protected_parameter_names: tuple[str, ...] = (),
) -> tuple[object, tuple[ProposerPreSchemaRepair, ...]]:
    """Remove only semantically predetermined or byte-equivalent parameters.

    A protected public channel can never be a fitted parameter. Repeated
    parameter declarations are removed only when their complete JSON values are
    identical. Conflicting declarations remain untouched so strict validation
    can reject them and request a targeted repair.
    """
    if not isinstance(payload, dict):
        return payload, ()
    parameters = payload.get("parameters")
    if not isinstance(parameters, list):
        return payload, ()

    protected = set(protected_parameter_names)
    retained: list[object] = []
    protected_counts: dict[str, int] = {}
    range_counts: dict[str, int] = {}
    exact_duplicate_counts: dict[str, int] = {}
    seen: set[str] = set()
    for item in parameters:
        if isinstance(item, dict):
            item = dict(item)
            removed_ranges = sum(
                key in item for key in ("bounds", "initialization_range")
            )
            if removed_ranges:
                label = str(item.get("name", "<invalid>"))
                range_counts[label] = range_counts.get(label, 0) + removed_ranges
                item.pop("bounds", None)
                item.pop("initialization_range", None)
        name = item.get("name") if isinstance(item, dict) else None
        if isinstance(name, str) and name in protected:
            protected_counts[name] = protected_counts.get(name, 0) + 1
            continue
        canonical = json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical in seen:
            label = name if isinstance(name, str) else "<invalid>"
            exact_duplicate_counts[label] = exact_duplicate_counts.get(label, 0) + 1
            continue
        seen.add(canonical)
        retained.append(item)

    if not protected_counts and not exact_duplicate_counts and not range_counts:
        return payload, ()
    normalized = dict(payload)
    normalized["parameters"] = retained
    repairs = tuple(
        ProposerPreSchemaRepair(
            code="removed_protected_parameter",
            parameter_name=name,
            removed_count=count,
        )
        for name, count in sorted(protected_counts.items())
    ) + tuple(
        ProposerPreSchemaRepair(
            code="removed_legacy_parameter_ranges",
            parameter_name=name,
            removed_count=count,
        )
        for name, count in sorted(range_counts.items())
    ) + tuple(
        ProposerPreSchemaRepair(
            code="removed_exact_duplicate_parameter",
            parameter_name=name,
            removed_count=count,
        )
        for name, count in sorted(exact_duplicate_counts.items())
    )
    return normalized, repairs


def enrich_proposal_v2(
    proposal: ProposerCandidateV2,
    target_channels: tuple[str, ...] = (),
) -> CandidateModel:
    """Convert the V2 proposer contract into the canonical internal model."""
    if not target_channels:
        components = (*proposal.states, *proposal.algebraics)
        if len(components) != 1:
            raise ValueError(
                "target channels are required when a proposal has multiple components"
            )
        target_channels = (components[0].name,)
    observations = []
    for target in target_channels:
        matches: dict[tuple[str, str], str] = {}
        for state in proposal.states:
            if state.name == target or state.observed_channel == target:
                matches[("state", state.name)] = state.name
        for item in proposal.algebraics:
            if item.name == target:
                matches[("algebraic", item.name)] = item.name
        if len(matches) != 1:
            names = sorted(name for _, name in matches)
            raise ValueError(
                f"target {target} must match exactly one observed channel or "
                f"same-named state/algebraic; matches={names}"
            )
        observations.append(
            ObservationMapping(
                channel=target,
                expression=next(iter(matches.values())),
            )
        )
    initials = [
        InitialConditionSpec(
            state=state.name,
            scope=ParameterScope.GLOBAL,
            fixed_value=state.initial.fixed_value,
            expression=state.initial.expression,
        )
        for state in proposal.states
        if state.initial is not None
    ]
    initials.extend(
        InitialConditionSpec(
            state=state.name,
            scope=ParameterScope.GLOBAL,
            expression=state.observed_channel,
        )
        for state in proposal.states
        if state.observed_channel is not None
    )
    declarations = (*proposal.states, *proposal.algebraics)
    return CandidateModel(
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.parent_candidate_id,
        change_summary=proposal.change_summary,
        states=tuple(
            StateSpec(
                name=state.name,
                kind=state.kind,
                mechanisms=state.mechanisms,
            )
            for state in proposal.states
        ),
        processes=tuple(
            ProcessSpec(
                name=item.name,
                expression=item.expression,
                mechanisms=item.mechanisms,
            )
            for item in proposal.algebraics
        ),
        state_equations=tuple(
            StateEquation(state=state.name, rhs=state.rhs)
            for state in proposal.states
        ),
        observation_mappings=tuple(observations),
        parameters=tuple(
            ParameterSpec(
                name=parameter.name,
                scope=parameter.scope,
            )
            for parameter in proposal.parameters
        ),
        initial_conditions=tuple(initials),
        constraints=tuple(
            ConstraintSpec(
                subject=declaration.name,
                kind=constraint.kind,
                bounds=constraint.bounds,
                description=constraint.description,
                source=ConstraintSource.PROPOSER,
                enforcement=ConstraintEnforcement.SOFT,
            )
            for declaration in declarations
            for constraint in declaration.constraints
        ),
    )
