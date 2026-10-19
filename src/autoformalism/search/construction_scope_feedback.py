"""Scope-specific repair guidance without changing initializer acceptance tests."""

from __future__ import annotations

from autoformalism.expressions import ValidationContext
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import _closure, _graph
from autoformalism.search.topology_initialization import _initial_scope


def _invalid_targets(
    draft: Draft, contract: GraphConstructionContract, previous: dict | None
) -> list[dict]:
    """Classify attempted targets by declarations, never by guessed naming rules."""
    raw = (previous or {}).get("attempted_records")
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        return []
    kinds = {variable.name: variable.definition for variable in draft.variables}
    processes = {process.name for process in draft.processes}
    rows = []
    seen = set()
    for record in raw["records"][:64]:
        if not isinstance(record, dict) or not isinstance(record.get("state"), str):
            continue
        state = record["state"]
        if state in seen:
            continue
        seen.add(state)
        if state in contract.observed_initial_channels:
            kind = "fixed_observed_initial"
            message = (
                f"{state}(t0) retains its recorded observed identity. Do not return "
                "an initializer record for this state."
            )
        elif state in contract.known_initials:
            rule = record.get("rule")
            if isinstance(rule, dict) and rule.get("mode") == "known":
                continue
            kind = "fixed_known_initial"
            message = (
                f"{state}(t0) is runtime-known. Retain its mode=known record; "
                "its value or scope cannot be changed."
            )
        elif kinds.get(state) == "algebraic":
            kind = "algebraic_output"
            message = (
                f"{state} is computed by its algebraic equation, including at t0; "
                "it cannot have an independent initializer."
            )
        elif state in processes:
            kind = "process"
            message = (
                f"{state} is a process, not a differential state with an initial "
                "value. An initializer record cannot create a new state."
            )
        elif state not in kinds:
            kind = "undeclared_differential_state"
            message = (
                f"{state} is not a declared differential state. An initializer "
                "record cannot add a variable, dynamics parameter or ODE term. "
                "If you intended a coefficient, that is outside this repair scope."
            )
        else:
            continue
        if state in contract.permitted_public_initials:
            message += (
                f" Its permitted recorded reading may still be used as "
                f"public_initials=[{state}]; that is not an initializer for {state}."
            )
        rows.append({"state": state, "kind": kind, "message": message})
    return rows


def _conflict_scope(
    draft: Draft,
    context: ValidationContext,
    current: GraphDependencyHandoff | None,
    gate: dict | None,
    editable: set[str],
) -> list[dict]:
    """Trace relevant state initials and transitive maps, without choosing a fix."""
    graph, kinds, _ = _graph(draft, context)
    records = (
        {record.state: record.rule for record in current.records} if current else {}
    )
    rows = []
    for conflict in (gate or {}).get("conflicts", ()):
        target, property_ = conflict["target"], conflict["property"]
        relevant: set[str] = set()
        if property_ == "closed_subsystem":
            initial_only = all(
                point["sample_index"] == 0 for point in conflict["observations"]
            )
            scope = (
                _initial_scope(target, graph, kinds)
                if initial_only
                else _closure(target, graph)
            )
            states = {name for name in scope if kinds.get(name) == "differential"}
            pending = list(states)
            while pending:
                rule = records.get(pending.pop())
                for parent in getattr(rule, "state_initials", ()):
                    if parent not in states:
                        states.add(parent)
                        pending.append(parent)
            relevant = states & editable
        rows.append(
            {
                "target": target,
                "property": property_,
                "editable_initializers": sorted(relevant),
                "current_rules": {
                    state: records[state].model_dump(mode="json")
                    for state in sorted(relevant & records.keys())
                },
                "instruction": (
                    "These existing initializer rules contribute to the displayed "
                    "equality implication and are editable. A schema-valid rule "
                    "is not protected when it causes a model-data conflict. Choose "
                    "a scientifically justified permitted change; this list does "
                    "not prescribe one or guarantee that it will resolve the conflict."
                    if relevant
                    else "No editable initializer is identified for this current "
                    "conflict. Do not invent a root or coefficient; a dynamics or "
                    "output change may be needed outside this repair scope."
                ),
            }
        )
    return rows


def scoped_feedback(
    draft: Draft,
    context: ValidationContext,
    contract: GraphConstructionContract,
    current: GraphDependencyHandoff | None,
    gate: dict | None,
    previous: dict | None,
) -> dict:
    """Keep record validity distinct from the last verified model-data conflict."""
    editable = (
        set(contract.state_declaration_sha256)
        - set(contract.observed_initial_channels)
        - set(contract.known_initials)
    )
    blocked = bool(gate and gate["blocked"])
    return {
        "editable_initializers": sorted(editable),
        "invalid_record_targets": _invalid_targets(draft, contract, previous),
        "conflict_repair_scope": _conflict_scope(
            draft, context, current, gate, editable
        ),
        "acceptance_obligation": (
            "The model cannot be accepted while the displayed model-data conflict "
            "remains. Removing an invalid record only fixes that record error; "
            "keeping the original conflicting initializer does not fix the model. "
            "Every replacement is rechecked against the evidence."
            if blocked
            else "The complete replacement must satisfy declaration validity and "
            "the deterministic model-data checks. Deferred evidence is not a "
            "scientific pass."
        ),
        "record_replacement_rule": (
            "Return the complete latent initializer list. Retain unaffected "
            "records and fixed known records; revise conflict-causing rules even "
            "if their syntax and dependencies are valid. Do not initialize an "
            "algebraic output independently or add a new dynamics parameter."
        ),
    }
