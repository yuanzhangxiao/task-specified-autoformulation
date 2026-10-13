"""Separate memory claims from scientific review without inferring model edits."""

from __future__ import annotations

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-public-coverage-2"
SYSTEM = coverage.SYSTEM
EDITING = coverage.EDITING
model_view = coverage.model_view
repair_budget = coverage.repair_budget
MEMORY_ISSUES = frozenset({"memory_driver_path", "memory_target_path", "memory_type"})

MEMORY_RULES = """mechanism_bindings assigns the differential states carrying a
mechanism's history; it is not a checklist of represented mechanisms. Required
memory choices and existing assignments are listed in memory_binding_options.
Other requirements need no assignment unless you intentionally realize them
through memory states. For example, a direct input u -> target y process needs
no assignment to a separate state m merely because m exists in the model.
Every assignment claims the corresponding input-to-state and state-to-target
pathways. A false memory_assignment_required means this extra assignment is
optional, not that the public mechanism is optional. Omitted entries remain;
remove_bindings withdraws an unintended assignment, not its public requirement,
states or processes. Required memory assignments must instead be supplied or
corrected."""

VARIABLES = coverage.stage_text("variables").replace(
    "Record required memory-state choices in mechanism_bindings; paths are checked\n"
    "after topology.",
    "Record required memory-state choices in mechanism_bindings, not a binding\n"
    "for every mechanism. Paths are checked after topology.",
)
SHARED = coverage.SHARED.replace(
    "Choose signs consistently with the declared rate meaning.",
    "State what a positive process value means, then choose each use's sign.\n"
    "For example, a nonnegative inflow into x has a positive use in d(x)/dt;\n"
    "a nonnegative outflow has a negative use. A signed net rate needs its\n"
    "direction convention stated in scientific_meaning.",
)
TOPOLOGY = coverage.TOPOLOGY.replace(
    "Before finishing, read each public requirement against the assembled model:\n"
    "where is each required mechanism represented, and how does it affect its\n"
    "target(s)? A declaration or mechanism name alone is insufficient.",
    "Before finishing, locate each public mechanism in the actual processes and\n"
    "their paths to its targets. This is not a request to add memory bindings.\n"
    "Use scientific_review questions to check meanings against the displayed\n"
    "equations. A declaration or mechanism name alone is insufficient.",
)
REPAIR = coverage.REPAIR.replace(
    "Check the assembled\n"
    "model against every public requirement before declaring completion, including\n"
    "those whose scientific interpretation is not decided by a graph predicate.",
    "Use memory_assignment_decisions when an optional history claim is in doubt;\n"
    "adding bindings is not a way to mark mechanisms covered. Use scientific_review\n"
    "questions for physical meaning and balance consistency. Correct the existing\n"
    "declarations, rather than adding an explanation that leaves them unchanged.",
)


def stage_text(stage: str) -> str:
    """Keep one reviewed instruction per stage for both live calls and previews."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def _optional_decisions(
    brief: PublicScientificBrief, draft: ledger.Draft, feedback: dict
) -> list[dict]:
    """Expose optional-claim choices even when complete graph checks are unavailable."""
    affected = {
        row.get("requirement")
        for row in feedback["issues"]
        if row["code"] in MEMORY_ISSUES
    }
    affected.update(
        row.get("requirement")
        for row in feedback.get("provisional_paths", [])
        if row["kind"] in {"driver_to_assigned_memory", "assigned_memory_to_target"}
        and row["status"] == "missing_in_draft"
    )
    optional = {r.id for r in brief.requirements if not r.requires_dynamic_memory}
    return [
        {
            "requirement_id": binding.requirement_id,
            "retained_entry": binding.model_dump(mode="json"),
            "memory_assignment_required": False,
            "question": "Does this mechanism intentionally use these states to "
            "store its history, or was this entry only meant to say the mechanism "
            "is represented? The public mechanism remains required either way.",
            "if_history_is_intended": "Keep or explicitly replace this assignment "
            "using differential states and implement its displayed pathways "
            "using your intended processes. "
            "Do not add an unrelated path merely to satisfy the annotation.",
            "if_only_coverage_was_intended": {
                "edit": {"remove_bindings": [binding.requirement_id]},
                "meaning": "Withdraw only this optional memory claim; preserve "
                "the mechanism's processes and required input-to-target effect. "
                "The runtime does not choose this edit for you.",
            },
            "evidence_scope": "See the checked state types and checked or "
            "provisional missing paths. "
            "Provisional observations are not a new rejection or proof of failure.",
        }
        for binding in draft.mechanism_bindings
        if binding.requirement_id in affected & optional
    ]


def _scientific_review(brief: PublicScientificBrief, draft: ledger.Draft) -> dict:
    """Ask general questions about the displayed model, never classify its prose."""
    questions = []
    processes = [p.name for p in draft.processes]
    local_rhs = [e.name for e in draft.equations if e.terms]
    if processes or local_rhs:
        questions.append(
            {
                "topic": "rate_directions",
                "processes": processes,
                "local_rhs": local_rhs,
                "question": "For these processes and local terms in model_view, "
                "what does a positive value mean? Do every displayed sign and "
                "receiving equation match that meaning? For a physical flow, "
                "identify what loses and gains the quantity, including outside "
                "the modeled system. For other effects, no transfer is implied. "
                "State a signed-net-rate convention explicitly if intended; "
                "changing kind alone does not correct a physical direction.",
            }
        )
    questions.append(
        {
            "topic": "related_quantities",
            "prediction_targets": [
                v.name for v in brief.public_variables if v.data_role == "target"
            ],
            "question": "Read these target definitions in public_brief and the "
            "generated quantities in model_view. Where a quantity represents a "
            "rate or effect in another modeled balance, which existing local "
            "term or shared use realizes it there? Is its definition consistent "
            "with those uses? Distinguish defining a rate from contributing to "
            "its time derivative. Allow equivalent inline or shared forms; "
            "unrelated quantities need no artificial connection or shared process.",
        }
    )
    return {
        "status": "questions_for_proposer_not_runtime_verdicts",
        "reference": "Use the single model_view and public_brief already shown.",
        "questions": questions,
        "response": "Use the existing reply fields only. If a meaning is "
        "ambiguous, clarify scientific_meaning or scientific_role and make "
        "consistent topology edits. If already consistent, leave it unchanged. "
        "Do not return this question table or use mechanism_bindings as answers.",
        "scope": "These questions apply across benchmarks. They do not infer "
        "relationships, prescribe edges or signs, or certify scientific coverage. "
        "Function and behavioral assessment remains later work.",
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
    """Clarify scientific decisions without changing schemas, evidence or checks."""
    packet = coverage.payload(
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
    packet["stage_instructions"] = stage_text(stage)
    packet["memory_assignment_rules"] = MEMORY_RULES
    assigned = {b.requirement_id for b in draft.mechanism_bindings}
    packet["memory_binding_options"] = [
        row
        for row in packet["memory_binding_options"]
        if row["memory_assignment_required"] or row["requirement_id"] in assigned
    ]
    if stage != "variables":
        feedback = packet["feedback"]
        decisions = _optional_decisions(brief, draft, feedback)
        feedback["memory_assignment_decisions"] = decisions
        decided = {row["requirement_id"] for row in decisions}
        for issue in feedback["issues"]:
            if issue.get("requirement") not in decided:
                continue
            issue.pop("assignment_decision", None)
            if issue["code"] in MEMORY_ISSUES:
                issue["assignment_choice"] = (
                    "First decide whether this optional memory claim is intended; "
                    "see memory_assignment_decisions for this requirement."
                )
                if "repair_options" in issue:
                    issue["if_assignment_retained"] = issue.pop("repair_options")
        feedback["scientific_review"] = _scientific_review(brief, draft)
    return packet
