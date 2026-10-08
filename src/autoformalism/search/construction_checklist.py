"""Declaration checks on every turn; pathway and scientific claims remain separate."""

from __future__ import annotations

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-variable-checklist-1"
INSTRUCTION = """The runtime checklist describes the accepted declarations, not the
last rejected reply. Resolve missing or inconsistent variable-stage items before
stage_complete=true. Partial declarations are allowed. A memory assignment names
your chosen differential state(s); it does not establish the mechanism's pathways.
Items deferred to topology do not block this stage. Types are constrained only by
explicit public rules and your structured assignments, never by variable names or
explanatory prose. You may assign an existing state without declaring it again."""


def variable_checklist(
    brief: PublicScientificBrief,
    draft: ledger.Draft,
    target_definitions: dict[str, str],
    graph_contract: PublicGraphContract | None = None,
) -> dict:
    """Check declaration obligations without requiring equations or choosing states.

    Missing references can be supplied by later partial replies. They block only
    completion. The mediator restriction mirrors the existing final memory check;
    it is not a general rule that every memory mechanism needs a separate state.
    """
    definitions = {v.name: v.definition for v in draft.variables}
    definitions.update((p.name, "algebraic") for p in draft.processes)
    declared = {v.name for v in draft.variables}
    items = []

    def add(code: str, name: str, status: str, **details) -> None:
        items.append(
            {
                "id": f"{code}:{name}",
                "code": code,
                "name": name,
                "status": status,
                **details,
            }
        )

    targets = sorted(v.name for v in brief.public_variables if v.data_role == "target")
    missing = [name for name in targets if name not in declared]
    for name in targets:
        add(
            "public_target",
            name,
            "missing" if name in missing else "satisfied",
            definition=definitions.get(name),
            action="Declare this target in variables and choose its type."
            if name in missing
            else None,
        )
    for name, expected in sorted(target_definitions.items()):
        actual = definitions.get(name)
        add(
            "explicit_type",
            name,
            "missing"
            if actual is None
            else "satisfied"
            if actual == expected
            else "inconsistent",
            expected=expected,
            actual=actual,
            action=None
            if actual == expected
            else f"Use the explicit public definition {expected} for {name}.",
        )
    forbidden = {
        v.name: v.data_role
        for v in brief.public_variables
        if v.data_role in {"external_input", "covariate", "time"}
    }
    conflicts = sorted(forbidden.keys() & definitions.keys())
    add(
        "source_roles",
        "public",
        "inconsistent" if conflicts else "satisfied",
        generated_public_sources=conflicts,
        action=None
        if not conflicts
        else "Supplied inputs, covariates and time need no generated declaration; "
        "intended RHS use remains permitted.",
    )

    requirements = {r.id: r for r in brief.requirements}
    bindings = {b.requirement_id: b.memory_states for b in draft.mechanism_bindings}
    for name in sorted(
        set(bindings) | {r.id for r in brief.requirements if r.requires_dynamic_memory}
    ):
        requirement = requirements.get(name)
        states = bindings.get(name, ())
        if requirement is None:
            add(
                "memory_assignment",
                name,
                "inconsistent",
                states=list(states),
                action="Unknown requirement ID. Remove this binding explicitly or "
                "replace it with an intended public requirement assignment.",
                removal={"remove_bindings": [name]},
            )
            continue
        excluded = set(requirement.drivers) | set(requirement.targets)
        checks = [
            {
                "state": state,
                "declared": state in definitions,
                "definition": definitions.get(state),
                "differential": definitions.get(state) == "differential",
                "allowed_mediator": not requirement.requires_dynamic_memory
                or state not in excluded,
            }
            for state in states
        ]
        valid = bool(states) and all(
            c["differential"] and c["allowed_mediator"] for c in checks
        )
        add(
            "memory_assignment",
            name,
            "satisfied" if valid else "missing" if not states else "inconsistent",
            required=requirement.requires_dynamic_memory,
            states=list(states),
            state_checks=checks,
            excluded_mediator_endpoints=sorted(excluded)
            if requirement.requires_dynamic_memory
            else [],
            action=None
            if valid
            else "In mechanism_bindings, choose existing or newly declared "
            "differential states carrying this memory. For this required mediator "
            "contract, use states distinct from its listed drivers and targets."
            if requirement.requires_dynamic_memory
            else "Correct this optional assignment to differential states, or remove "
            "it explicitly with remove_bindings.",
        )

    applicable = (
        {o.target for o in graph_contract.obligations if o.kind == "target_feedback"}
        if graph_contract
        else set()
    )
    feedback = {b.target: b.states for b in draft.feedback_bindings}
    for target in sorted(applicable | feedback.keys()):
        definition = definitions.get(target)
        states = feedback.get(target, ())
        implicit = definition == "differential" and target in applicable
        if implicit and not states:
            states = (target,)
        valid = target in applicable and definition in {"algebraic", "differential"}
        valid &= bool(states) and all(
            definitions.get(s) == "differential" for s in states
        )
        valid &= not implicit or states == (target,)
        add(
            "feedback_coordinates",
            target,
            "satisfied" if valid else "missing" if not states else "inconsistent",
            definition=definition,
            states=list(states),
            implicit_target=implicit,
            state_definitions={s: definitions.get(s) for s in states},
            action=None
            if valid
            else "This differential target is its own coordinate; remove any "
            "conflicting feedback binding."
            if implicit
            else "Unknown target-feedback assignment; remove it with "
            "remove_feedback_bindings."
            if target not in applicable
            else "For this algebraic target, identify your differential coordinates "
            "in feedback_bindings. Cycles and readout paths are checked later.",
        )

    for requirement in brief.requirements:
        add(
            "mechanism_topology",
            requirement.id,
            "deferred",
            drivers=list(requirement.drivers),
            targets=list(requirement.targets),
            assigned_memory_states=list(bindings.get(requirement.id, ())),
            reason="Topology must connect the relevant ordinary contributions "
            "and/or shared processes. An assignment alone is not a pathway pass.",
        )
    for obligation in graph_contract.obligations if graph_contract else ():
        add(
            "public_graph",
            obligation.id,
            "deferred",
            kind=obligation.kind,
            target=obligation.target,
            reason="Checked on the assembled topology.",
        )
    blocked = [r["id"] for r in items if r["status"] in {"missing", "inconsistent"}]
    return {
        "policy": POLICY,
        "status": "incomplete" if blocked else "ready",
        "items": items,
        "blocking_items": blocked,
        "missing_public_targets": missing,
        "explicit_type_rule_count": len(target_definitions),
        "scope": "Declaration consistency only. Deferred checks do not block "
        "variable completion; scientific sufficiency is not established.",
    }


def variable_snapshot(draft: ledger.Draft) -> dict:
    """Display variable-stage records without premature missing-equation warnings."""
    return {
        "declarations": draft.model_dump(mode="json"),
        "editing_contract": "Return changes only; omitted declarations remain. "
        "Use remove_variables, remove_bindings or remove_feedback_bindings to "
        "delete the corresponding named entry. No topology is required yet.",
    }
