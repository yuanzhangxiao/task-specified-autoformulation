"""Opt-in LHS/RHS clarification within the existing variable agenda."""

from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
    VariableReply,
)
from autoformalism.search import variable_bindings

POLICY = "explicit-variable-equation-usage-1"
INSTRUCTION = """
Decide equation usage explicitly. LHS means the generated variable (or its time
derivative) on the left of an equation; RHS means a source used to compute it.
- supplied: NO generated LHS; MAY appear on the RHS using the measured trajectory.
- differential: generate a derivative equation; MAY appear on other RHSs.
- algebraic: generate an algebraic equation; MAY appear on other RHSs, without loops.
- unused: NEITHER LHS NOR RHS. It means deliberately exclude this channel from
  the candidate equations, NOT merely 'do not model this channel dynamically'.
For example, using a measured channel z in dx/dt = f(x,z,u) needs z=supplied,
not z=unused. This example specifies usage only, not a scientific dependency.
Both supplied and unused avoid adding an equation. Choose between them according
to whether the measured trajectory should be available on the RHS.

In the first agenda reply, also decide every measured non-target observation
whose current_definition is not_declared. Include each as supplied, unused,
differential or algebraic. Existing runtime-registered supplied channels already
have a displayed selection. Do not assume that public availability selects a
channel. A missing decision is a format gap; the runtime never chooses for you.
For each choice, give its intended scientific role, rather than only its data
label. If omitting a channel, explain why the task can still be represented.

Consider these scientific questions using only the public task:
1. What carries each required accumulation or delayed response? Does an output
   have independent dynamics or read out the chosen states instantaneously?
2. If a repair adds a memory mediator, reconcile the affected output's role with
   that mediator. You may update scientific_role for an unchanged name/type.
3. Can the chosen states accommodate the public preparation and varying measured
   initial outputs? Explain any relevant state-coordinate interpretation in the
   role; actual causal initializers are constructed later.
These questions are advisory. Do not invent extra states just to answer them.
Do not return equations, dependencies or initial values in this variable reply.
No additional self-review call follows this agenda. Later topology establishes
actual RHS use and pathways; descriptions are not scientific acceptance tests.
"""


def missing_choices(
    brief: PublicScientificBrief, inventory: tuple[ScientificVariable, ...]
) -> tuple[str, ...]:
    """Require explicit channel choices without choosing whether to use them."""
    declared = {v.name for v in inventory}
    return tuple(
        f"observation_choice:{v.name}"
        for v in brief.public_variables
        if v.data_role == "auxiliary" and v.name not in declared
    )


def binding_context(brief, bindings, target_definitions) -> dict:
    """Make the eligible memory-ID list and the empty-list case unambiguous."""
    context = variable_bindings.binding_context(brief, bindings, target_definitions)
    ids = [r["id"] for r in context["requirements"]]
    return {
        **context,
        "allowed_requirement_ids": ids,
        "binding_instruction": (
            "Use only allowed_requirement_ids; one entry per requirement. "
            "Do not bind a target merely because it has a differential equation. "
            "A binding denotes a distinct required memory mediator, not every "
            "mechanism or state. Omitted existing bindings are retained."
            if ids
            else "No distinct memory binding is requested. Return "
            "mechanism_bindings=[]; physical storage may still require states."
        ),
    }


def refresh_roles(
    inventory: tuple[ScientificVariable, ...], reply: VariableReply
) -> tuple[ScientificVariable, ...]:
    """Accept wording updates only; never infer or change scientific structure."""
    returned = {v.name: v for v in reply.variables}
    return tuple(
        returned[v.name]
        if v.name in returned and returned[v.name].definition == v.definition
        else v
        for v in inventory
    )


def usage_table(inventory: list[dict]) -> list[dict]:
    """Report declaration permissions, explicitly not observed equation use."""
    return [
        {
            "name": v["name"],
            "definition": v["definition"],
            "generated_lhs_required": v["definition"] in {"differential", "algebraic"},
            "rhs_allowed": v["definition"] != "unused",
            "actual_rhs_occurrences": None,
        }
        for v in inventory
    ]
