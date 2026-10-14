"""Opt-in reply-control validation and exact retained-entry removal guidance."""

from __future__ import annotations

from copy import deepcopy

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger

# Only unambiguous reply controls are reserved. Generic schema words such as
# name, terms, variables and processes are not reserved scientific identifiers.
CONTROL_NAMES = frozenset(
    {
        "stage_complete",
        "remove_variables",
        "remove_equations",
        "remove_processes",
        "remove_bindings",
        "remove_feedback_bindings",
        "remove_overlap_confirmations",
    }
)
_REMOVALS = {
    "remove_variables": ("variables", "name"),
    "remove_equations": ("equations", "name"),
    "remove_processes": ("processes", "name"),
    "remove_bindings": ("mechanism_bindings", "requirement_id"),
    "remove_feedback_bindings": ("feedback_bindings", "target"),
    "remove_overlap_confirmations": ("overlap_confirmations", "overlap_id"),
}
_DECLARATIONS = ("variables", "equations", "processes")


def validate_patch(
    patch: ledger.DraftPatch, *, brief: PublicScientificBrief | None = None
) -> None:
    """Reject exact control names as declarations before committing any edits."""
    conflicts = [
        f"{field}[{entry.name}]"
        for field in _DECLARATIONS
        for entry in getattr(patch, field)
        if entry.name in CONTROL_NAMES
    ]
    if not conflicts:
        return
    public = {v.name for v in brief.public_variables} if brief is not None else set()
    collisions = sorted(
        {
            entry.name
            for field in _DECLARATIONS
            for entry in getattr(patch, field)
            if entry.name in CONTROL_NAMES & public
        }
    )
    if collisions:
        raise ValueError(
            "public scientific names collide with reserved reply controls: "
            f"{collisions}. "
            "This is an ambiguous public/schema namespace, not a scientific type "
            "error. Do not automatically rename or delete a public quantity. "
            "No part of this reply was committed."
        )
    raise ValueError(
        f"reply-control fields are not scientific declarations: {conflicts}. "
        "stage_complete is a top-level boolean; remove_* fields are top-level "
        "edit operations. Omit these accidental declarations, keep their controls "
        "at the top level and resubmit all other intended edits. A rejected "
        "declaration was never installed, so do not request its removal. "
        "No part of this reply was committed."
    )


def _raw_names(raw: dict, field: str, *, declarations: bool = False) -> set[str]:
    """Read only exact strings from potentially malformed rejected replies."""
    values = raw.get(field)
    if not isinstance(values, list):
        return set()
    if declarations:
        return {
            entry["name"]
            for entry in values
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
    return {name for name in values if isinstance(name, str)}


def _owners(draft: ledger.Draft) -> dict[str, list[dict]]:
    """Keep distinct scientific namespaces distinct, including overlapping keys."""
    result: dict[str, list[dict]] = {}
    for removal, (field, key) in _REMOVALS.items():
        for entry in getattr(draft, field):
            name = getattr(entry, key)
            result.setdefault(name, []).append(
                {"field": field, "key": key, "removal_if_intended": {removal: [name]}}
            )
    return result


def _removal_issues(draft: ledger.Draft, raw: dict) -> list[dict]:
    """Identify wrong removal destinations without deciding what should be removed."""
    owners = _owners(draft)
    issues = []
    for removal, (field, key) in _REMOVALS.items():
        available = {getattr(entry, key) for entry in getattr(draft, field)}
        for name in sorted(_raw_names(raw, removal) - available):
            retained = owners.get(name, [])
            issues.append(
                {
                    "code": "removal_namespace_mismatch",
                    "name": name,
                    "attempted_field": removal,
                    "not_retained_in": field,
                    "retained_entries": retained,
                    "action": f"Omit {name!r} from {removal}; no such entry is "
                    "retained in that category. "
                    + (
                        "If deleting an actual retained entry is intended, choose "
                        "its listed removal field. These are separate entries: "
                        "do not delete every listed owner automatically. Preserve "
                        "intended effects and update affected references explicitly. "
                        if retained
                        else "The name is absent from every retained edit category. "
                        "A rejected addition was never installed; abandoning it "
                        "requires omitting it, not deleting it. "
                    )
                    + "The rejected transaction committed nothing; resubmit its "
                    "other intended edits. No removal has been applied automatically.",
                }
            )
    return issues


def augment_payload(
    packet: dict,
    draft: ledger.Draft,
    last: dict | None,
    *,
    brief: PublicScientificBrief | None = None,
) -> dict:
    """Explain control fields and wrong-category removals in any construction stage."""
    result = deepcopy(packet)
    raw = (last or {}).get("rejected_reply")
    raw = raw if (last or {}).get("error") and isinstance(raw, dict) else {}
    owners = _owners(draft)
    rejected = {
        name
        for field in _DECLARATIONS
        for name in _raw_names(raw, field, declarations=True)
        if name in CONTROL_NAMES
    }
    retained = {
        entry.name
        for field in _DECLARATIONS
        for entry in getattr(draft, field)
        if entry.name in CONTROL_NAMES
    }
    public = {v.name for v in brief.public_variables} if brief is not None else set()
    ambiguous = (retained | rejected) & public
    additions = [
        {
            "code": "reply_control_is_not_quantity",
            "name": name,
            "retained_entries": owners.get(name, []),
            "attempted_declaration_rejected": name in rejected,
            "action": "This is a reply control, not a quantity or coefficient. "
            "stage_complete belongs only at the reply's top level as a boolean; "
            "remove_* are top-level edit operations. "
            + (
                "If this accidental entry is already retained, remove it using "
                "the exact listed owner field and update affected references "
                "explicitly while preserving intended scientific effects. "
                if name in retained
                else "Its rejected declaration was never installed. Omit the "
                "declaration and do not request its removal. "
            )
            + "Resubmit other intended edits from the rejected transaction.",
        }
        for name in sorted((retained | rejected) - public)
    ]
    additions.extend(
        {
            "code": "public_control_namespace_conflict",
            "name": name,
            "action": "This exact public quantity name also names a reply control. "
            "The public/schema namespaces require explicit resolution; do not "
            "rename or delete the public quantity automatically, or change its "
            "scientific type to evade this conflict. No edit was committed.",
        }
        for name in sorted(ambiguous)
    )
    additions.extend(_removal_issues(draft, raw))
    if not additions:
        return result
    feedback = result["feedback"]
    # Existing generic quantity/coefficient choices are inappropriate for exact
    # protocol controls; preserve every unrelated scientific question and error.
    questions = feedback.get("symbol_role_questions")
    if isinstance(questions, list):
        feedback["symbol_role_questions"] = [
            row for row in questions if row.get("name") not in CONTROL_NAMES - public
        ]
    for issue in feedback["issues"]:
        if issue.get("code") == "undeclared_lhs_entries":
            for row in issue["entries"]:
                if row.get("exact_name") in CONTROL_NAMES - public:
                    row["choices"] = [
                        {
                            "action": "Follow reply_control_is_not_quantity "
                            "for this name."
                        }
                    ]
    feedback["issues"] = additions + feedback["issues"]
    if "repair_agenda" in feedback:
        agenda = feedback["repair_agenda"]["representation_issues"]
        feedback["repair_agenda"]["representation_issues"] = list(
            dict.fromkeys([row["code"] for row in additions] + agenda)
        )
    return result
