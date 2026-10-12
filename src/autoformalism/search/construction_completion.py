"""Targeted completion questions; scientific assignments remain proposer edits."""

from __future__ import annotations

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_readable as readable
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-completion-feedback-1"
SYSTEM = readable.SYSTEM
EDITING = readable.EDITING
model_view = readable.model_view
stage_text = readable.stage_text


def _binding_question(issue: dict, draft: ledger.Draft) -> dict:
    """Separate a state declaration from the missing structured assignment."""
    name = issue["name"]
    states = issue.get("type_eligible_states", [])
    return {
        "code": "confirm_memory_assignment",
        "requirement_id": name,
        "field_to_edit": "mechanism_bindings",
        "retained_entry": next(
            (
                b.model_dump(mode="json")
                for b in draft.mechanism_bindings
                if b.requirement_id == name
            ),
            None,
        ),
        "question": f"Which declared differential state(s) carry {name}? "
        "Answer by returning a mechanism_bindings entry, not by declaring the "
        "same variable again. A matching state name does not record an assignment.",
        "choices_by_type_only": states,
        "conditional_answers": issue.get("conditional_examples", []),
        "alternative": "If none is scientifically suitable, declare the intended "
        "state(s) and return their assignment together. You may choose several "
        "states. The runtime does not select one.",
        "completion": "An omitted or empty mechanism_bindings list keeps the "
        "existing assignment unchanged. Keep stage_complete=false until this "
        "required assignment is supplied; pathways are checked after topology.",
    }


def _binding_destination(
    issue: dict, brief: PublicScientificBrief, draft: ledger.Draft
) -> dict:
    """Expose the exact retained assignment and an optional explicit withdrawal."""
    name = issue.get("requirement")
    requirement = next((r for r in brief.requirements if r.id == name), None)
    binding = next(
        (b for b in draft.mechanism_bindings if b.requirement_id == name), None
    )
    if requirement is None or binding is None:
        return issue
    row = dict(issue)
    row["assignment_decision"] = {
        "field_to_edit": "mechanism_bindings",
        "retained_entry": binding.model_dump(mode="json"),
        "question": "Is this state assignment intended? If yes, repair the "
        "displayed pathway through these states. If not, replace this exact "
        "entry with your intended state(s). Changing another entry or returning "
        "an empty list leaves this assignment unchanged.",
        "memory_assignment_required": requirement.requires_dynamic_memory,
    }
    if not requirement.requires_dynamic_memory:
        row["assignment_decision"]["if_optional_assignment_was_unintended"] = {
            "edit": {"remove_bindings": [name]},
            "meaning": "Withdraw only this optional memory assignment. Its public "
            "mechanism and required pathways still apply; no state or process is "
            "deleted by this edit.",
        }
    return row


def _empty_equations(brief: PublicScientificBrief, draft: ledger.Draft) -> list[dict]:
    """Point at the exact missing LHS without proposing a scientific dependency."""
    equations = {e.name: e for e in draft.equations}
    process_definitions = {p.name for p in draft.processes}
    targets = {v.name for v in brief.public_variables if v.data_role == "target"}
    supplied_observations = {
        v.name for v in brief.public_variables if v.data_role == "auxiliary"
    }
    rows = []
    for variable in draft.variables:
        name = variable.name
        # A variable-stage declaration may subsequently acquire a process-owned
        # definition. The ledger assembles that RHS; asking for an ordinary
        # equation here would contradict the duplicate-definition validator.
        if name in process_definitions:
            continue
        equation = equations.get(name)
        uses = [
            {"process": p.name, **u.model_dump(mode="json")}
            for p in draft.processes
            for u in p.uses
            if u.target == name
        ]
        if equation is not None and (equation.terms or uses):
            continue
        row = {
            "code": "equation_destination",
            "field_to_edit": "equations",
            "exact_name": name,
            "declared_type": variable.definition,
            "retained_entry": equation.model_dump(mode="json") if equation else None,
            "already_inserted_uses": uses,
            "action": f"Return an equations entry with name={name!r}. "
            "An equation that merely uses this quantity on its RHS does not "
            "define it; editing a different name will not resolve this item.",
        }
        if uses:
            row["if_shared_uses_are_the_entire_intended_rhs"] = {
                "equations": [{"name": name, "terms": []}]
            }
        else:
            row["if_intended_constant"] = {
                "instruction": "Only if scientifically intended, a constant "
                "contribution has sources=[] inside a term. Choose its outer "
                "sign and meaning. terms=[] instead means no local contribution.",
                "exact_name": name,
            }
        if name in supplied_observations - targets:
            edit = {"remove_variables": [name]}
            if equation is not None:
                edit["remove_equations"] = [name]
            row["if_you_intended_to_use_the_supplied_observation"] = {
                "edit": edit,
                "condition": "Only if generating this observed quantity was "
                "unintended. Keep it available as a supplied RHS source; revise "
                "any process uses or memory assignments that require it generated. "
                "This alternative never applies to a public target.",
            }
        rows.append(row)
    return rows


def _flow_decision(process: ledger.ProcessDeclaration, draft: ledger.Draft) -> dict:
    """Offer classification choices without inventing a receiver or changing signs."""
    value = process.model_dump(mode="json")
    return {
        "code": "flow_classification_decision",
        "field_to_edit": "processes",
        "process": process.name,
        "retained_entry": value,
        "question": "Does this quantity transfer between two modeled balances, "
        "or enter/leave the modeled system at its boundary? A supplied forcing "
        "can drive a boundary flow without receiving a contribution.",
        "if_internal_pairwise_transfer": "Keep kind=transfer and explicitly "
        "choose two distinct generated variables with opposite contributions. "
        "Declare any scientifically intended new variable and its topology. "
        "Do not invent a second receiver merely to satisfy this format.",
        "generated_variables": [v.name for v in draft.variables],
        "if_boundary_flow_or_other_nontransfer_effect": {
            "edit": {"processes": [{**value, "kind": "influence"}]},
            "scope": "Changes only the classification. Keeps every dependency, "
            "sign, conversion and use exactly as declared. Other errors still "
            "need explicit edits: duplicate/supplied receivers are not fixed by "
            "this choice. One valid receiver is allowed for a local process.",
        },
        "preservation": "Review scientific meaning before choosing. Neither a "
        "sign flip nor deletion of an intended effect is required by a kind change.",
    }


def payload(
    stage: str,
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    enriched_brief: dict,
    draft: ledger.Draft,
    contract: PublicGraphContract | None,
    last: dict | None = None,
    continuation: dict | None = None,
) -> dict:
    """Extend the readable view only where its current checks identify a gap."""
    packet = readable.payload(
        stage,
        brief,
        context,
        target_definitions,
        enriched_brief,
        draft,
        contract,
        last,
        continuation,
    )
    feedback = packet["feedback"]
    issues = []
    for issue in feedback["issues"]:
        if (
            issue["code"] == "memory_assignment"
            and issue.get("memory_assignment_required")
            and issue["status"] == "missing"
        ):
            issues.append(_binding_question(issue, draft))
        else:
            issues.append(_binding_destination(issue, brief, draft))
    if stage in {"equations", "repair"}:
        issues.extend(_empty_equations(brief, draft))
    if stage != "variables":
        affected = {
            i.get("process")
            for i in issues
            if i["code"]
            in {"transfer_signs", "duplicate_receiver", "supplied_receiver"}
        }
        issues.extend(
            _flow_decision(p, draft)
            for p in draft.processes
            if p.kind == "transfer" and p.name in affected
        )
    feedback["issues"] = issues
    return packet


def repair_budget(
    packet: dict,
    maximum: int,
    attempt: int,
    token_limit: int,
    charged: int,
    repair_tokens: int,
) -> None:
    """Show the active cap, not unused construction tokens as extra repair budget."""
    readable.repair_budget(packet, maximum, attempt)
    packet["feedback"]["repair_budget"].update(
        maximum_global_repair_tokens=repair_tokens,
        active_token_ceiling=token_limit,
        charged_before_call=charged,
        available_before_reservation=max(0, token_limit - charged),
        reservation_rule="Each new call must fit request UTF-8 bytes plus the "
        "output allowance before generation. This conservative budget reservation "
        "is separate from the serving-token context check. Actual reported usage "
        "replaces the reservation after a response; unknown usage stays reserved.",
    )
