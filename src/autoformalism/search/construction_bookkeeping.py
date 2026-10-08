"""Opt-in current-prompt bookkeeping; scientific choices remain structured edits."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import shared_process_contract, signed_processes

Policy = Literal["legacy", "current-bookkeeping-1", "current-bookkeeping-2"]
POLICY = "current-bookkeeping-1"
FEEDBACK_POLICY = "current-bookkeeping-2"
POLICIES = ("legacy", POLICY, FEEDBACK_POLICY)

EDITING = """Only current_draft.declarations contains editable records. The other
views are read-only previews rebuilt from those records. Edit a named process
through processes (drivers, kind, uses); never copy its generated definition into
equations. Each ordinary contribution's sources jointly drive ONE later function,
not a list of separate additive effects. Shared uses are already inserted.

Check the displayed left-hand side: d(x)/dt means a differential state; x means
an algebraic value. Explanatory text does not change this type. To change the
representation, return the explicit variables definition and the complete new
ordinary topology in equations together, with any affected process/binding edits.
The separate schedule permits such inventory changes during overall repair;
joint schedules permit them during topology too. No type is inferred from prose.

An equations entry replaces the entire ordinary list for that variable. To change
only B in [A, B], return [A, C]; returning [C] removes both old entries. Omitted
entries remain unchanged. An invalid transaction saves none of its edits."""


def validate_policy(policy: str, prompt_family: str | None = None) -> None:
    """Do not silently apply current-prompt fixes to the matched wording study."""
    if policy not in POLICIES:
        raise ValueError("unknown construction bookkeeping policy")
    if policy != "legacy" and prompt_family is not None:
        raise ValueError("bookkeeping fixes are separate from the prompt comparison")


def lhs(name: str, definition: str | None) -> str:
    """Render only the structured type, never the scientific description."""
    if definition == "differential":
        return f"d({name})/dt"
    return name if definition == "algebraic" else f"{name} [type pending]"


def balance_view(draft: ledger.Draft, name: str) -> dict:
    """Keep ordinary joint functions and automatic process uses visibly distinct."""
    variable = next((v for v in draft.variables if v.name == name), None)
    definition = variable.definition if variable else None
    equation = next((e for e in draft.equations if e.name == name), None)
    terms = equation.terms if equation else ()
    uses = [
        {"process": p.name, **u.model_dump(mode="json")}
        for p in draft.processes
        for u in p.uses
        if u.target == name
    ]
    contributions, represented = [], []
    for i, term in enumerate(terms):
        sign = term.outer_weight_sign.value
        # The existing assembler already folds these exact identity references
        # into the declared use. Do not display them as a second contribution.
        # They can survive in a partial draft when the consumer was undeclared.
        if any(term.sources == (u["process"],) and sign == u["sign"] for u in uses):
            represented.append(i)
            continue
        factor = f"c{i}" if sign == "unrestricted" else f"g{i}"
        contributions.append(
            f"{'-' if sign == 'negative' else '+'} {factor} * "
            f"phi{i}({', '.join(term.sources)})"
        )
    for i, use in enumerate(uses):
        conversion = f"({use['conversion']}) * " if use["conversion"] else ""
        contributions.append(
            f"{'-' if use['sign'] == 'negative' else '+'} gP{i} * "
            f"{conversion}{use['process']}"
        )
    if equation is None:
        contributions.append("+ [ordinary topology not declared]")
    return {
        "name": name,
        "definition": definition,
        "lhs": lhs(name, definition),
        "ordinary_rhs_declared": equation is not None,
        "ordinary_terms": [t.model_dump(mode="json") for t in terms],
        "ordinary_indices_already_represented_by_process_uses": represented,
        "runtime_inserted_process_uses": uses,
        "balance_topology": f"{lhs(name, definition)} = "
        + (" ".join(contributions).removeprefix("+ ") or "[no contributions]"),
    }


def snapshot(brief: PublicScientificBrief, draft: ledger.Draft) -> dict:
    """Show one editable ledger and labeled previews, including incomplete drafts."""
    old = ledger.snapshot(brief, draft)
    return {
        "declarations": old["declarations"],
        "editing_contract": EDITING,
        "read_only_generated_process_definitions": [
            {
                "name": p.name,
                "edit_via": "processes",
                "definition": shared_process_contract.definition(
                    signed_processes.binding_for(p)
                ).model_dump(mode="json"),
            }
            for p in draft.processes
        ],
        "read_only_balances": [
            balance_view(draft, v.name)
            for v in draft.variables
            if v.name not in {p.name for p in draft.processes}
        ],
        "preview_notation": (
            "phi is an unspecified later interaction, g a nonnegative magnitude, "
            "c an unrestricted coefficient. These are display placeholders, not "
            "function proposals or parameter declarations. Each process denotes "
            "one shared value. Unknown conversions remain unknown."
        ),
        **{
            key: old[key]
            for key in (
                "assembly_pending_reason",
                "pending",
                "process_usage",
                "potential_contribution_overlaps",
            )
        },
    }


def definition_conflicts(draft: ledger.Draft, patch: ledger.DraftPatch) -> list[dict]:
    """Name competing structured definitions after harmless repeats are removed."""
    processes = {
        p.name: p for p in draft.processes if p.name not in patch.remove_processes
    }
    processes.update((p.name, p) for p in patch.processes)
    equations = {
        e.name: e for e in draft.equations if e.name not in patch.remove_equations
    }
    equations.update((e.name, e) for e in patch.equations)
    return [
        {
            "code": "conflicting_process_definition",
            "process": name,
            "ordinary_definition": equations[name].model_dump(mode="json"),
            "runtime_definition": shared_process_contract.definition(
                signed_processes.binding_for(processes[name])
            ).model_dump(mode="json"),
            "instruction": (
                "Choose the drivers in processes and omit this name from equations. "
                "To replace a previously ordinary quantity by a named process, "
                "explicitly remove its existing ordinary topology. No driver, sign "
                "or scientific meaning is chosen by the runtime."
            ),
        }
        for name in sorted(processes.keys() & equations.keys())
    ]


def assessment_context(
    check: dict, draft: ledger.Draft, *, policy: Policy = POLICY
) -> dict:
    """Add targeted questions without changing any scientific acceptance decision."""
    result = deepcopy(check)
    for issue in result["errors"]:
        if issue["code"] != "process_consumer_decision":
            continue
        name, target = issue["process"], issue["target"]
        uses = issue["declared_uses_for_target"]
        issue["current_balance"] = balance_view(draft, target)
        issue["question"] = (
            f"The displayed balance already includes the declared use of {name}. "
            f"Does the ordinary term using {name} restate that balance, or represent "
            "a separate joint interaction?"
            if uses
            else f"Is the ordinary term using {name} a signed use of that same "
            "process, or a distinct joint interaction? No signed use for this "
            "target has been declared."
        )
        issue["clarification_actions"] = [
            {
                "if": "restatement of already inserted process use",
                "action": (
                    "Replace this variable's COMPLETE ordinary list with only the "
                    "remaining intended contributions. Keep other ordinary terms "
                    "and the intended process uses. Splitting a joint sources list "
                    "into additive terms is your scientific decision."
                ),
            },
            {
                "if": "a distinct joint interaction",
                "action": (
                    "Declare the distinct law with its chosen drivers and consumer "
                    "uses in processes, and remove the replaced ordinary reference. "
                    "Preserve other effects; do not invent consumers."
                ),
            },
            {
                "if": "the declared use, sign or conversion is wrong",
                "action": (
                    "Replace the process declaration explicitly, preserving its "
                    "other consumers, and coordinate affected ordinary topologies."
                ),
            },
        ]
    if policy == FEEDBACK_POLICY:
        from autoformalism.search import construction_feedback

        result = construction_feedback.explain_feedback(result, draft)
    return result
