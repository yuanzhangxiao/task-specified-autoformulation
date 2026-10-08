"""Explicit public graph predicates; never infer requirements from model prose."""

from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
)


class GraphObligation(StrictSchema):
    """A reviewed public interpretation, not a proposer-created scientific rule."""

    id: Identifier
    kind: Literal["dynamic_feedback", "target_feedback", "forbidden_path"]
    target: Identifier
    source: Identifier | None = None
    public_quote: NonEmptyText
    interpretation: NonEmptyText

    @model_validator(mode="after")
    def endpoints_match_kind(self):
        """Feedback searches dynamic ancestors; forbidden paths have one source."""
        if (self.kind == "forbidden_path") != (self.source is not None):
            raise ValueError("only forbidden_path requires a source")
        if self.source == self.target:
            raise ValueError("forbidden_path needs distinct endpoints")
        return self


class PublicGraphContract(StrictSchema):
    """Versioned, public-only contract frozen separately from historical briefs."""

    policy: Literal["reviewed-public-graph-1", "reviewed-public-graph-2"] = (
        "reviewed-public-graph-1"
    )
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    obligations: tuple[GraphObligation, ...] = ()
    deferred_scientific_checks: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def unique_ids(self):
        """Each obligation has exactly one auditable identity."""
        ids = [rule.id for rule in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate public graph obligation ID")
        return self

    def validate_public(self, brief: PublicScientificBrief) -> None:
        """Check provenance quotes/endpoints; this is not semantic interpretation."""
        public = {v.name for v in brief.public_variables}
        targets = {v.name for v in brief.public_variables if v.data_role == "target"}
        text = " ".join(brief.scientific_context.split())
        for rule in self.obligations:
            if rule.target not in targets or (
                rule.source is not None and rule.source not in public
            ):
                raise ValueError("public graph obligation endpoint is not public")
            if " ".join(rule.public_quote.split()) not in text:
                raise ValueError(
                    "public graph obligation quote differs from public task"
                )


def _path(graph: dict[str, set[str]], start: str, end: str) -> list[str] | None:
    """Find a deterministic nonempty directed path, including a genuine cycle."""
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for neighbor in sorted(graph.get(node, ())):
            if neighbor == end:
                return [*path, neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    return None


def target_feedback_evidence(
    equations: tuple[EquationDefinition, ...] | None,
    target: str,
    feedback_coordinates: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """Separate coordinate declarations from actual cycle and readout witnesses."""
    if equations is None:
        return {
            "graph_available": False,
            "binding_status": "unassessed",
            "target_definition": None,
            "coordinate_checks": [],
            "passed": None,
            "witness": None,
        }
    definitions = {e.name: e.definition for e in equations}
    graph: dict[str, set[str]] = {}
    for equation in equations:
        for term in equation.terms:
            for source in term.sources:
                graph.setdefault(source, set()).add(equation.name)
    states = {n for n, d in definitions.items() if d == "differential"}
    implicit = target in states
    coordinates = (
        (target,) if implicit else (feedback_coordinates or {}).get(target, ())
    )
    readout_graph = {
        source: {end for end in ends if end not in states}
        for source, ends in graph.items()
    }
    checks, witnesses = [], []
    for state in coordinates:
        cycle = _path(graph, state, state) if state in states else None
        readout = [state] if state == target else _path(readout_graph, state, target)
        checks.append(
            {
                "state": state,
                "declared": state in definitions,
                "differential": state in states,
                "feedback_cycle_exists": cycle is not None,
                "algebraic_readout_path_exists": readout is not None,
                "feedback_cycle": cycle,
                "target_path": readout,
            }
        )
        if cycle and readout:
            witnesses.append({"state": state, "cycle": cycle, "target_path": readout})
    passed = bool(coordinates) and len(witnesses) == len(coordinates)
    return {
        "graph_available": True,
        "binding_status": "implicit_target_coordinate"
        if implicit
        else "present"
        if coordinates
        else "missing",
        "target_definition": definitions.get(target),
        "coordinate_checks": checks,
        "passed": passed,
        "witness": witnesses if passed else None,
    }


def check(
    contract: PublicGraphContract,
    equations: tuple[EquationDefinition, ...] | None,
    *,
    feedback_coordinates: dict[str, tuple[str, ...]] | None = None,
) -> list[dict]:
    """Assess a compiled topology; None means unavailable, never a failed path."""
    graph: dict[str, set[str]] = {}
    for equation in equations or ():
        for term in equation.terms:
            for source in term.sources:
                graph.setdefault(source, set()).add(equation.name)
    states = sorted(e.name for e in equations or () if e.definition == "differential")
    rows = []
    for rule in contract.obligations:
        witness = None
        passed = None
        reason = None
        if equations is not None:
            if rule.kind == "forbidden_path":
                witness = _path(graph, rule.source, rule.target)
                passed = witness is None
            elif rule.kind == "target_feedback":
                # A differential target is already its own dynamic coordinate.
                # For an algebraic readout the proposer owns that assignment;
                # an arbitrary upstream ancestor is not a storage realization.
                evidence = target_feedback_evidence(
                    equations, rule.target, feedback_coordinates
                )
                passed, witness = evidence["passed"], evidence["witness"]
                reason = (
                    None
                    if passed
                    else (
                        "For a differential target, include a feedback cycle through "
                        "that target. For an algebraic target, declare "
                        "feedback_bindings "
                        "with its actual dynamic storage/energy coordinates: each must "
                        "be differential, have feedback, and reach the target through "
                        "algebraic readouts only. An upstream cycle alone "
                        "is insufficient."
                    )
                )
            else:
                for state in states:
                    cycle = _path(graph, state, state)
                    target_path = (
                        [state]
                        if state == rule.target
                        else _path(graph, state, rule.target)
                    )
                    if cycle and target_path:
                        witness = {
                            "state": state,
                            "cycle": cycle,
                            "target_path": target_path,
                        }
                        break
                passed = witness is not None
        rows.append(
            {
                **rule.model_dump(mode="json"),
                "status": "unavailable_graph"
                if passed is None
                else "pass"
                if passed
                else "fail",
                "passed": passed,
                "witness": witness,
                "reason": reason,
                "scope": (
                    "Declared dependencies only; physical identity, signs, "
                    "stability and laws are unverified."
                ),
            }
        )
    return rows
