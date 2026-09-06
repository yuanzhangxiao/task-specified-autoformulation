"""Small scientific contracts for staged variable and equation construction."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, create_model, model_validator

from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema

Definition = Literal["supplied", "differential", "algebraic", "unused"]


class ModelingLimits(StrictSchema):
    """Declared construction limits shared by schema and controller."""

    generated_variables: int = Field(default=12, ge=1, le=64)
    terms_per_equation: int = Field(default=8, ge=1, le=32)
    total_terms: int = Field(default=32, ge=1, le=512)


class PublicVariable(StrictSchema):
    """A public data identity, independent of its candidate definition."""

    name: Identifier
    data_role: Literal["target", "auxiliary", "external_input", "covariate", "time"]


class ScientificRequirement(StrictSchema):
    """Positive obligations derived only from the public task."""

    id: Identifier
    public_requirement: NonEmptyText
    targets: tuple[Identifier, ...]
    drivers: tuple[Identifier, ...]
    positive_requirements: tuple[NonEmptyText, ...] = ()


class TargetDependency(StrictSchema):
    """An explicit public composition requirement, independent of representation."""

    target: Identifier
    acceptable_sources: tuple[Identifier, ...] = Field(min_length=1)
    public_requirement: NonEmptyText


class PublicScientificBrief(StrictSchema):
    """Scientific input without transport metadata or hidden reference information."""

    scientific_context: str = Field(min_length=1, max_length=60000)
    public_variables: tuple[PublicVariable, ...] = Field(min_length=1, max_length=256)
    requirements: tuple[ScientificRequirement, ...] = Field(max_length=32)
    target_dependencies: tuple[TargetDependency, ...] = ()
    limits: ModelingLimits = ModelingLimits()

    @model_validator(mode="after")
    def unique_public_names(self) -> PublicScientificBrief:
        """Reject ambiguous context before a provider call."""
        names = [item.name for item in self.public_variables]
        if len(names) != len(set(names)):
            raise ValueError("public variable names must be unique")
        if not any(item.data_role == "target" for item in self.public_variables):
            raise ValueError("a public target is required")
        identifiers = [item.id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scientific requirement IDs must be unique")
        references = {
            name
            for item in self.requirements
            for name in (*item.targets, *item.drivers)
        }
        references.update(
            name
            for item in self.target_dependencies
            for name in (item.target, *item.acceptable_sources)
        )
        if references - set(names):
            raise ValueError(
                "scientific requirements reference undeclared public names"
            )
        return self


class ScientificVariable(StrictSchema):
    """One candidate variable choice, with no runtime-owned data role."""

    name: Identifier
    definition: Definition
    scientific_role: str = Field(min_length=1, max_length=1000)


class VariableReply(StrictSchema):
    """Variables proposed or explicitly reused for one scientific agenda item."""

    variables: tuple[ScientificVariable, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def names_are_unique(self) -> VariableReply:
        """Reject ambiguous same-response declarations."""
        names = [item.name for item in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique within a response")
        return self


class EquationTerm(StrictSchema):
    """An RHS contribution; outer assembly sign is not response behavior."""

    sources: tuple[Identifier, ...] = Field(max_length=64)
    outer_sign: Literal["add", "subtract"]
    scientific_role: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def sources_are_unique(self) -> EquationTerm:
        """A grouped source set may be empty, but may not repeat a variable."""
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("term sources must be unique")
        return self


class InventoryRevision(StrictSchema):
    """Explicit unresolved request to revisit a scientific variable choice."""

    variable: ScientificVariable
    reason: str = Field(min_length=1, max_length=1000)


class EquationReply(StrictSchema):
    """Either a definition for the selected LHS or an inventory revision request."""

    terms: tuple[EquationTerm, ...] = Field(max_length=32)
    inventory_revision: InventoryRevision | None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> EquationReply:
        """Keep revision requests distinct from an accepted equation."""
        if bool(self.terms) == (self.inventory_revision is not None):
            raise ValueError(
                "return nonempty terms OR an inventory_revision, never both"
            )
        return self


class EquationDefinition(StrictSchema):
    """Runtime-selected LHS with its accepted scientific terms."""

    name: Identifier
    definition: Literal["differential", "algebraic"]
    terms: tuple[EquationTerm, ...] = Field(min_length=1, max_length=32)


def equation_reply_model(
    allowed_sources: tuple[str, ...], *, maximum_terms: int
) -> type[EquationReply]:
    """Derive the actual provider schema and validator from one active inventory."""
    if not allowed_sources or len(set(allowed_sources)) != len(allowed_sources):
        raise ValueError("allowed sources must be nonempty and unique")
    source_type = Literal[allowed_sources]  # type: ignore[valid-type]
    term_model = create_model(
        "SelectedEquationTerm",
        __base__=EquationTerm,
        sources=(tuple[source_type, ...], Field(max_length=len(allowed_sources))),
    )
    return create_model(
        "SelectedEquationReply",
        __base__=EquationReply,
        terms=(tuple[term_model, ...], Field(max_length=maximum_terms)),
    )
