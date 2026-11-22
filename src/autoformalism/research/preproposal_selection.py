"""Present a train-only minimal source portfolio for a fixed differential target."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, create_model, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema

Option = Annotated[str, Field(pattern=r"^(history|initial):[A-Za-z_][A-Za-z0-9_]*$")]
Token = Annotated[str, Field(min_length=1, max_length=128)]
Reason = Annotated[str, Field(min_length=1, max_length=1600)]
MAX_CHOICES = 3


class DeclaredVariable(StrictSchema):
    name: Identifier
    kind: Literal["differential", "algebraic"]


class ScreenCandidate(StrictSchema):
    candidate_id: Token
    sources: tuple[Option, ...]


class SelectionCase(StrictSchema):
    """Only these compact scientific fields can reach a selection request."""

    case_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
    target: Identifier
    inventory: tuple[DeclaredVariable, ...]
    public_task: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_summary: str = Field(min_length=1)
    required_options: tuple[Option, ...]
    candidates: tuple[ScreenCandidate, ...]

    @model_validator(mode="after")
    def validate_menu(self):
        names = [v.name for v in self.inventory]
        if self.target not in names or len(names) != len(set(names)):
            raise ValueError(
                "one inventory declaration is required for the fixed target"
            )
        ids = [c.candidate_id for c in self.candidates]
        supports = [frozenset(c.sources) for c in self.candidates]
        if len(set(ids)) != len(ids) or len(set(supports)) != len(supports):
            raise ValueError("menu candidates need unique IDs and distinct supports")
        if len(set(self.required_options)) != len(self.required_options):
            raise ValueError("required options must be unique")
        for row, sources in zip(self.candidates, supports, strict=True):
            if len(row.sources) != len(sources) or sources.intersection(
                self.required_options
            ):
                raise ValueError(
                    "an added source repeats a required or optional source"
                )
        if any(other < current for current in supports for other in supports):
            raise ValueError("menu contains a nonminimal support")
        return self


class SelectionIntake(StrictSchema):
    cases: tuple[SelectionCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case ID")
        return self


class RankedCandidate(StrictSchema):
    candidate_id: Token
    rationale: Reason


class SelectionChoice(RankedCandidate):
    """Canonical branch handoff with the algorithm-owned support restored."""

    sources: tuple[Option, ...]


class SelectionReply(StrictSchema):
    selections: tuple[RankedCandidate, ...] = Field(
        min_length=1, max_length=MAX_CHOICES
    )


def needs_selection(case: SelectionCase) -> bool:
    """An empty support represents a vacuous screen and warrants no LLM call."""
    return any(c.sources for c in case.candidates)


def skipped_outcome(case: SelectionCase) -> dict:
    """Distinguish a vacuous equality test from an unsatisfied catalogue."""
    if needs_selection(case):
        raise ValueError("a nonempty portfolio is available for selection")
    return {
        "status": "no_added_information_required"
        if case.candidates
        else "no_certified_cover",
        "selected": [],
    }


def selection_schema(case: SelectionCase) -> type[SelectionReply]:
    """Restrict the live JSON grammar to IDs from this specific inventory."""
    if not needs_selection(case):
        raise ValueError("no decision required for this case")
    ids = tuple(candidate.candidate_id for candidate in case.candidates)
    choice = create_model(
        "ScientificChoice",
        __base__=RankedCandidate,
        candidate_id=(Literal[ids], ...),
    )
    return create_model(
        "ScientificSelections",
        __base__=SelectionReply,
        selections=(
            tuple[choice, ...],
            Field(min_length=1, max_length=min(len(ids), MAX_CHOICES)),
        ),
    )


def messages(case: SelectionCase) -> tuple[str, str]:
    """Freeze the target and scientific choice, independently of later topology."""
    if not needs_selection(case):
        raise ValueError("no decision required for this case")
    system = (
        f"Choose scientific explanations for the observed dynamics of {case.target}. "
        f"The target is fixed to {case.target}. The runtime has screened the listed "
        "source sets against public training trajectories. Each permits differences "
        "to reach this target in time. Function, mechanism and fit remain open. "
        "Rank one to three distinct listed candidate IDs by scientific preference. "
        "Choose fewer if fewer are plausible or available. Explain how the sources in "
        "each choice could affect this target, including required mechanisms "
        "and quantities available at prediction time. history:X is supplied X through "
        "time. initial:X is a permitted baseline reading and requires a causal route "
        "through initialization. Return each ID with its scientific rationale."
    )
    packet = case.model_dump(
        mode="json", exclude={"case_id", "target", "evidence_sha256"}
    )
    return system, json.dumps({"target": case.target, **packet}, ensure_ascii=False)


def assess(case: SelectionCase, reply: SelectionReply) -> dict:
    """Resolve distinct selected IDs back to the original immutable sources."""
    allowed = {candidate.candidate_id: candidate for candidate in case.candidates}
    ids = [choice.candidate_id for choice in reply.selections]
    if not needs_selection(case):
        raise ValueError("received a selection for a case requiring none")
    if len(set(ids)) != len(ids):
        return {
            "status": "invalid_selection",
            "reason": "repeated candidate ID",
            "selected": [],
        }
    if unknown := sorted(set(ids) - allowed.keys()):
        return {
            "status": "invalid_selection",
            "reason": f"unknown candidate IDs: {unknown}",
            "selected": [],
        }
    return {
        "status": "selected",
        "selected": [
            {
                "candidate_id": choice.candidate_id,
                "sources": list(allowed[choice.candidate_id].sources),
                "rationale": choice.rationale,
            }
            for choice in reply.selections
        ],
    }
