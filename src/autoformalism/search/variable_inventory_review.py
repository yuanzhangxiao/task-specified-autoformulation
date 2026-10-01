"""Explicit observation choices and atomic, proposer-owned inventory review."""

from __future__ import annotations

from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
    VariableReply,
)
from autoformalism.search import variable_bindings
from autoformalism.staged_topology import freeze_inventory, merge_variable_reply

POLICY = "explicit-observation-inventory-review-1"
COMMON_INSTRUCTION = """
All measured channels are observable variables. Their public data roles differ:
`target` means prediction target; the legacy label `auxiliary` means a measured
non-target observation. Neither label specifies differential versus algebraic.
Availability in the public brief is NOT selection in the candidate inventory.
To use a measured non-target trajectory as an input, explicitly declare that
variable supplied. Alternatively declare it differential/algebraic to generate
it with the candidate's own equation, or unused to deliberately omit it.
Supplied variables do not get candidate equations; generated variables do.
An undeclared available channel cannot be used silently in later equations.

requires_dynamic_memory=true requires the displayed pathway's distinct memory
mediator. false means no such EXTRA mediator is required for that pathway; it
does NOT imply that its targets or the physical system have no state dynamics.
Independently honor all public storage/accumulation requirements. A target may
be algebraic if appropriate differential states carry the necessary dynamics.
The absence of a fixed target definition is freedom of representation, not
permission to omit physical dynamics required by the public task.
"""

INSTRUCTION = (
    COMMON_INSTRUCTION
    + """
After the agenda, a separate whole-inventory review will let you revise earlier
definitions and bindings before any equations are constructed. Local agenda
replies remain additions/reuse; do not silently change their accepted types.
"""
)

REVIEW_SYSTEM = (
    """You review the complete variable inventory before equations.
Return only variables and mechanism_bindings in the required schema. This reply
is a COMPLETE replacement, not additions. Include every prediction target,
every measured non-target observation (including unused ones), and every internal
variable you retain or add. Omitted internal variables are removed. The runtime
retains external inputs, covariates and time as supplied unless you explicitly
mark them unused. You may now change earlier definitions and scientific roles.
Return all required memory bindings, including unchanged bindings; none are
inferred or inherited. Each binding names differential states distinct from
that requirement's drivers and targets. Do not add redundant states merely to
acknowledge existing memory. Review the entire public task, including physical
storage, and explain variable choices only in concise scientific_role fields.
Do not return equations, dependencies, parameters, initial conditions or a
generic approval flag. Runtime checks establish mechanical consistency, not
scientific adequacy. On retry, the displayed pre-review inventory is unchanged;
return a complete replacement again, using the rejection feedback.
"""
    + COMMON_INSTRUCTION
)


def review_binding_context(brief, bindings, target_definitions) -> dict:
    """Reuse the binding evidence without the incremental-update instruction."""
    context = variable_bindings.binding_context(brief, bindings, target_definitions)
    context["instruction"] = (
        "Return complete bindings; omitted bindings are NOT retained in this review. "
        "Target definitions remain enforced; an absent definition permits either "
        "differential or algebraic representation subject to the public task."
    )
    return context


def observation_choices(
    brief: PublicScientificBrief, inventory: tuple[ScientificVariable, ...]
) -> list[dict]:
    """Expose availability separately from the current candidate declaration."""
    selected = {v.name: v for v in inventory}
    return [
        {
            "name": v.name,
            "observation_role": (
                "prediction_target"
                if v.data_role == "target"
                else "non_target_observation"
            ),
            "allowed_definitions": ["differential", "algebraic"]
            if v.data_role == "target"
            else ["supplied", "differential", "algebraic", "unused"],
            "current_definition": selected[v.name].definition
            if v.name in selected
            else "not_declared",
        }
        for v in brief.public_variables
        if v.data_role in {"target", "auxiliary"}
    ]


def replace_inventory(
    brief: PublicScientificBrief,
    reply: variable_bindings.BoundVariableReply,
    target_definitions: dict[str, str],
) -> tuple[tuple[ScientificVariable, ...], dict[str, set[str]]]:
    """Validate a full replacement without modifying any prior draft or binding."""
    declared = {v.name for v in reply.variables}
    required = {
        v.name for v in brief.public_variables if v.data_role in {"target", "auxiliary"}
    }
    if missing := required - declared:
        raise ValueError(f"explicit observation decisions missing: {sorted(missing)}")
    seed = tuple(
        ScientificVariable(
            name=v.name,
            definition="supplied",
            scientific_role=f"runtime-registered public {v.data_role}",
        )
        for v in brief.public_variables
        if v.data_role in {"external_input", "covariate", "time"}
        and v.name not in declared
    )
    # Reuse the normal name/role/limit and public-source checks, atomically.
    candidate = merge_variable_reply(
        brief, seed, VariableReply(variables=reply.variables)
    )
    candidate = freeze_inventory(brief, candidate)
    if any(
        target_definitions.get(v.name, v.definition) != v.definition for v in candidate
    ):
        raise ValueError(
            f"public target definitions must be retained: {target_definitions}"
        )
    bindings: dict[str, set[str]] = {}
    _, _, errors = variable_bindings.merge(
        brief,
        candidate,
        variable_bindings.BoundVariableReply(
            variables=(), mechanism_bindings=reply.mechanism_bindings
        ),
        bindings,
        target_definitions,
    )
    required_bindings = {r.id for r in brief.requirements if r.requires_dynamic_memory}
    if errors or required_bindings - set(bindings):
        raise ValueError(
            f"invalid complete memory bindings: errors={errors}; "
            f"missing={sorted(required_bindings - set(bindings))}"
        )
    return candidate, bindings


def decision_record(before, after, old_bindings, new_bindings) -> dict:
    """Preserve the two inventories and explicit changes for human inspection."""
    previous = {v.name: v.model_dump(mode="json") for v in before}
    current = {v.name: v.model_dump(mode="json") for v in after}
    return {
        "policy": POLICY,
        "status": "accepted",
        "before": list(previous.values()),
        "after": list(current.values()),
        "changes": [
            {"name": name, "before": previous.get(name), "after": current.get(name)}
            for name in sorted(previous.keys() | current.keys())
            if previous.get(name) != current.get(name)
        ],
        "bindings_before": {k: sorted(v) for k, v in old_bindings.items()},
        "bindings_after": {k: sorted(v) for k, v in new_bindings.items()},
        "scientific_adequacy": "not_assessed",
    }
