"""Proposer-owned memory assignments; the runtime checks references, not prose."""

from __future__ import annotations

from pydantic import Field, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
    VariableReply,
)
from autoformalism.staged_topology import merge_variable_reply_partially

POLICY = "explicit-variable-memory-bindings-1"
INSTRUCTION = """
Also return mechanism_bindings: a list of {requirement_id, memory_states}.
For each displayed requirement requiring dynamic memory, YOU identify the
differential state(s) carrying that memory. Use exact requirement and variable
names. You may bind an EXISTING differential variable without returning it again:
variables=[] is valid when only a binding is needed. Do not invent an extra state
just to acknowledge an existing one. A mediator must be distinct from that
requirement's drivers and targets. Each listed binding replaces the previous
binding for that requirement; omitted bindings are retained. You may explicitly
correct an earlier binding during variable identification. The runtime does not
infer an assignment from scientific_role, variable order, or names. Bindings are
scientific proposals, not certified pathways; topology must subsequently connect
the required drivers through the chosen memory to the targets.
The displayed target_definitions are enforced both here and before fitting.
An absent target definition means differential OR algebraic is allowed.
"""


class MechanismBinding(StrictSchema):
    """One explicit replacement of a mechanism's memory-state set."""

    requirement_id: Identifier
    memory_states: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_states(self):
        if len(set(self.memory_states)) != len(self.memory_states):
            raise ValueError("memory_states must be unique")
        return self


class BoundVariableReply(VariableReply):
    """Keep variable declarations unchanged and add a separate scientific decision."""

    mechanism_bindings: tuple[MechanismBinding, ...]

    @model_validator(mode="after")
    def unique_requirements(self):
        ids = [b.requirement_id for b in self.mechanism_bindings]
        if len(set(ids)) != len(ids):
            raise ValueError("one binding per requirement per reply")
        return self


def binding_context(brief, bindings, target_definitions):
    """Expose the whole current assignment and its exact public obligations."""
    return {
        "policy": POLICY,
        "requirements": [
            r.model_dump(mode="json")
            for r in brief.requirements
            if r.requires_dynamic_memory
        ],
        "current_bindings": {k: sorted(v) for k, v in sorted(bindings.items())},
        "target_definitions": target_definitions,
        "instruction": INSTRUCTION,
    }


def merge(
    brief: PublicScientificBrief,
    inventory: tuple[ScientificVariable, ...],
    reply: VariableReply,
    bindings: dict[str, set[str]],
    target_definitions: dict[str, str],
) -> tuple[tuple[ScientificVariable, ...], tuple[dict, ...], list[dict]]:
    """Retain valid variables; apply explicit bindings atomically if well formed.

    Invalid bindings never erase the previous assignment. A partially accepted
    inventory is shown on retry, so an already accepted variable can be bound
    without repeating its declaration.
    """
    decisions = []
    for variable in reply.variables:
        required = target_definitions.get(variable.name)
        if required and variable.definition != required:
            decisions.append(
                {
                    "name": variable.name,
                    "accepted": False,
                    "error": f"public target definition requires {required}",
                }
            )
        else:
            inventory, decision = merge_variable_reply_partially(
                brief, inventory, VariableReply(variables=(variable,))
            )
            decisions.extend(decision)
    definitions = {v.name: v.definition for v in inventory}
    requirements = {r.id: r for r in brief.requirements if r.requires_dynamic_memory}
    proposed = getattr(reply, "mechanism_bindings", ())
    errors = []
    updates = {}
    for binding in proposed:
        requirement = requirements.get(binding.requirement_id)
        if requirement is None:
            reason = "unknown requirement or requirement does not request memory"
        elif any(definitions.get(n) != "differential" for n in binding.memory_states):
            reason = "memory must reference existing differential variables"
        elif set(binding.memory_states) & {*requirement.drivers, *requirement.targets}:
            reason = (
                "memory mediator must differ from this requirement's drivers/targets"
            )
        else:
            updates[binding.requirement_id] = set(binding.memory_states)
            continue
        errors.append({"requirement_id": binding.requirement_id, "error": reason})
    if not errors:
        bindings.update(updates)
    return inventory, tuple(decisions), errors
