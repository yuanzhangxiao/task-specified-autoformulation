"""Qualify rejected process edits using retained differential-state ownership."""

from __future__ import annotations

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_edit_feedback as editing
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-edit-feedback-2"
SYSTEM = editing.SYSTEM
EDITING = editing.EDITING
model_view = editing.model_view
stage_text = editing.stage_text
repair_budget = editing.repair_budget


def _reply_names(raw: dict, field: str, *, entries: bool = False) -> set[str]:
    """Read exact names without coercing malformed proposer fields."""
    values = raw.get(field)
    if not isinstance(values, list):
        return set()
    if entries:
        return {
            row["name"]
            for row in values
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
    return {name for name in values if isinstance(name, str)}


def _ownership_issues(
    draft: ledger.Draft, last: dict | None, issues: list[dict]
) -> list[dict]:
    """Describe retained ownership without treating a rejected edit as installed."""
    if not last or not last.get("error"):
        return []
    raw = last.get("rejected_reply")
    raw = raw if isinstance(raw, dict) else {}
    proposed = _reply_names(raw, "processes", entries=True)
    removed = _reply_names(raw, "remove_processes")
    diagnosed = {
        name
        for row in issues
        for name in [
            row.get("process")
            if row.get("code") == "conflicting_process_definition"
            else row.get("name")
            if row.get("code") == "process_name_collision"
            else None
        ]
        if isinstance(name, str)
    }
    process_names = {p.name for p in draft.processes}
    states = {
        v.name for v in draft.variables if v.definition == "differential"
    } - process_names
    equations = {e.name for e in draft.equations}
    return [
        {
            "code": "state_process_ownership",
            "name": name,
            "retained_owner": "differential_state",
            "ordinary_equation_retained": name in equations,
            "process_retained": False,
            "attempted_process_addition": name in proposed or name in diagnosed,
            "attempted_process_removal": name in removed,
            "if_retaining_state": {
                "field_to_edit": f"equations[{name}]",
                "action": "Keep the differential variable. Retain its ordinary "
                "equation entry if present; otherwise add it. Complete or revise "
                "its intended RHS local terms here; existing process uses are "
                "inserted separately. Omit the "
                "same-name processes entry and omit this name from "
                "remove_processes and remove_equations. Do not delete the state "
                "equation merely to resolve the rejected process collision.",
                "if_distinct_instantaneous_law_is_intended": "Choose a distinct "
                "unused process name with scientifically intended depends_on "
                "and signed uses. A function of this state uses its name as a "
                "driver, not as the new process name. Preserve intended effects "
                "without duplicating them as both local terms and process uses.",
            },
            "if_changing_scientific_type": "A different scientific type requires "
            "an explicit variables edit and coordinated topology, bindings and "
            "references, subject to the existing checks. Do not change the state "
            "type merely to resolve a naming collision.",
            "action": "The retained model has no process with this name, so "
            "remove_processes cannot delete it. A rejected process addition was "
            "never installed. If abandoning the uncommitted addition, omit its "
            "proposed entry and its removal key. No edits from the rejected "
            "reply were committed; "
            "resubmit other intended edits. No name, type or scientific law "
            "has been changed automatically.",
        }
        for name in sorted(states & (proposed | removed | diagnosed))
    ]


def _qualify_instructions(issues: list[dict], names: set[str]) -> None:
    """Keep diagnostic evidence while replacing inconsistent corrective advice."""
    for issue in issues:
        code = issue.get("code")
        name = (
            issue.get("process")
            if code == "conflicting_process_definition"
            else (issue.get("name"))
        )
        if isinstance(name, str) and name in names:
            pointer = (
                f"See state_process_ownership for {name!r}: the retained name "
                "belongs to a differential state, not an installed process. "
                "Choose edits conditional on retaining that state or explicitly "
                "changing its scientific type."
            )
            if code == "conflicting_process_definition":
                issue["instruction"] = pointer
            elif code == "process_name_collision":
                issue["repair_options"] = [pointer]
        if code == "process_replace_remove_conflict":
            affected = names.intersection(issue["names"])
            if affected:
                issue["state_owned_names"] = sorted(affected)
                issue["other_process_names"] = sorted(set(issue["names"]) - names)
                advice = "For state_owned_names, follow state_process_ownership. "
                issue["action"] = advice + (
                    "For other_process_names only: " + issue["action"]
                    if issue["other_process_names"]
                    else "No edits from the rejected reply were committed; "
                    "resubmit other intended edits."
                )


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
    """Add ownership-aware advice without changing validation or previous policies."""
    packet = editing.payload(
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
    if stage == "variables":
        return packet
    feedback = packet["feedback"]
    additions = _ownership_issues(draft, last, feedback["issues"])
    if not additions:
        return packet
    _qualify_instructions(feedback["issues"], {row["name"] for row in additions})
    feedback["issues"].extend(additions)
    if "repair_agenda" in feedback:
        feedback["repair_agenda"]["representation_issues"].append(
            "state_process_ownership"
        )
    return packet
