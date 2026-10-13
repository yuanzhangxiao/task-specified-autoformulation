"""Explain existing topology edit failures without changing accepted models."""

from __future__ import annotations

from collections import Counter

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_decisions as decisions
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-edit-feedback-1"
SYSTEM = decisions.SYSTEM
EDITING = decisions.EDITING
model_view = decisions.model_view
stage_text = decisions.stage_text
repair_budget = decisions.repair_budget


def _undeclared_lhs(
    brief: PublicScientificBrief,
    draft: ledger.Draft,
    target_definitions: dict[str, str],
) -> list[dict]:
    """List all undeclared LHS entries, distinguishing public roles and ownership."""
    generated = {v.name for v in draft.variables} | {p.name for p in draft.processes}
    public = {v.name: v.data_role for v in brief.public_variables}
    entries = []
    for equation in draft.equations:
        name = equation.name
        if name in generated:
            continue
        role = public.get(name)
        row = {"exact_name": name, "public_role": role, "choices": []}
        if role in {"external_input", "covariate", "time", "auxiliary"}:
            row["choices"].append(
                {
                    "if": "Keep this observed quantity supplied."
                    if role == "auxiliary"
                    else "This public quantity must remain supplied.",
                    "edit": {"remove_equations": [name]},
                    "effect": "Remove only this LHS entry; retain intended RHS "
                    "references to the supplied quantity. Omission or terms=[] "
                    "does not remove an existing entry. Any process uses targeting "
                    "it need separate explicit edits; keep the other intended uses.",
                }
            )
        if role not in {"external_input", "covariate", "time"}:
            row["choices"].append(
                {
                    "if": "This public target must be generated."
                    if role == "target"
                    else "You intend to generate this quantity.",
                    "field_to_edit": "variables",
                    "name": name,
                    "allowed_types": [target_definitions[name]]
                    if name in target_definitions
                    else ["differential", "algebraic"],
                    "action": "Declare its intended type and scientific_role, "
                    "then complete its topology. A sources=[] term is a "
                    "constant contribution; terms=[] needs existing process uses "
                    "to supply the RHS. The runtime chooses neither type nor law.",
                }
            )
        if role is None:
            row["choices"].append(
                {
                    "if": "This name was only a mistaken label, not a quantity.",
                    "action": "Return the intended local topology under the exact "
                    "generated-variable name and remove this mistaken entry. "
                    "Review references and preserve intended effects.",
                    "remove_mistaken_entry": {"remove_equations": [name]},
                }
            )
        entries.append(row)
    if not entries:
        return []
    return [
        {
            "code": "undeclared_lhs_entries",
            "entries": entries,
            "action": "Resolve all these LHS entries together. The compiler may "
            "show only the first one. A supplied RHS source needs no equation; "
            "an intentionally generated observable needs an explicit declaration. "
            "No entry has been removed or declared automatically.",
        }
    ]


def _duplicate_receivers(draft: ledger.Draft) -> dict[str, dict]:
    """Report duplicate signed uses without identifying/cancelling physical effects."""
    rows = {}
    for process in draft.processes:
        counts = Counter(u.target for u in process.uses)
        duplicate = {n for n, count in counts.items() if count > 1}
        if not duplicate:
            continue
        rows[process.name] = {
            "field_to_edit": f"processes[{process.name}].uses",
            "retained_duplicate_uses": [
                u.model_dump(mode="json") for u in process.uses if u.target in duplicate
            ],
            "repair_options": [
                "If these entries are one intended use of one common quantity, "
                "return this process with one scientifically chosen signed use "
                "per receiving variable; preserve its other intended uses.",
                "If they represent different laws, represent them as distinct "
                "local terms or distinct processes with appropriate names and "
                "dependencies. Use remove_processes only if replacing the old "
                "process; coordinate its consumers and references in the same reply.",
            ],
            "warning": "Changing kind or merely flipping a sign does not remove "
            "a duplicate receiver. This rule also applies to influence. Opposite "
            "uses are not automatically cancelled: their fitted magnitudes, "
            "conversions and scientific intent need not be identical.",
        }
    return rows


def _string_names(raw: dict, field: str) -> set[str]:
    """Read names from a rejected, potentially malformed reply without coercion."""
    values = raw.get(field)
    return (
        {v for v in values if isinstance(v, str)} if isinstance(values, list) else set()
    )


def _process_edit_destinations(draft: ledger.Draft, last: dict | None) -> list[dict]:
    """Route rejected process edits to their owning field; never retry them here."""
    raw = (last or {}).get("rejected_reply")
    if not (last or {}).get("error") or not isinstance(raw, dict):
        return []
    processes = {p.name: p for p in draft.processes}
    equations = {e.name for e in draft.equations}
    wrong = _string_names(raw, "remove_equations") & (processes.keys() - equations)
    issues = []
    for name in sorted(wrong):
        issues.append(
            {
                "code": "process_edit_destination",
                "name": name,
                "attempted_field": "remove_equations",
                "owner": "processes",
                "choices": {
                    "keep_or_revise": f"Return the complete processes entry for "
                    f"{name!r}; omit its name from both removal lists.",
                    "if_removal_is_intended": {"remove_processes": [name]},
                },
                "affected_consumers": sorted({u.target for u in processes[name].uses}),
                "action": "This process has a runtime-generated definition, not "
                "an ordinary equations entry. Omit this name from "
                "remove_equations in either choice; preserve unrelated intended "
                "equation removals. remove_processes removes that "
                "definition and all its inserted uses. Preserve intended effects "
                "as replacement local terms/processes and update references in "
                "the same reply. A separate variables declaration, if retained, "
                "will still need topology. The previous reply committed nothing; "
                "resubmit its other intended edits too.",
            }
        )
    returned = raw.get("processes")
    replacements = (
        {
            p["name"]
            for p in returned
            if isinstance(p, dict) and isinstance(p.get("name"), str)
        }
        if isinstance(returned, list)
        else set()
    )
    conflicts = replacements & _string_names(raw, "remove_processes")
    if conflicts:
        issues.append(
            {
                "code": "process_replace_remove_conflict",
                "names": sorted(conflicts),
                "retained_process_names": sorted(conflicts & processes.keys()),
                "uncommitted_process_names": sorted(conflicts - processes.keys()),
                "action": "Choose one operation per process: to keep/revise, "
                "return its complete processes entry and omit it from "
                "remove_processes. To remove a retained process, list it only "
                "in remove_processes. To abandon an uncommitted process, omit "
                "both its proposed entry and its removal key: it was never "
                "installed. No edits from the rejected reply were committed. "
                "Resubmit other intended edits, including any needed "
                "local-topology removal.",
            }
        )
    return issues


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
    """Project existing errors into concrete, conditional edits under a new policy."""
    packet = decisions.payload(
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
    duplicates = _duplicate_receivers(draft)
    for issue in feedback["issues"]:
        name = issue.get("process")
        if name in duplicates and issue["code"] == "duplicate_receiver":
            issue.update(duplicates[name])
            issue["action"] = "Resolve the repeated uses before rechecking signs/kind."
        elif name in duplicates and issue["code"] == "process_uses":
            issue.pop("repair_options", None)
            issue.pop("warning", None)
            issue["action"] = (
                f"See duplicate_receiver for {name!r}: changing kind or a sign "
                "alone does not fix repeated uses of the same receiver."
            )
    additions = _process_edit_destinations(draft, last)
    if stage in {"equations", "repair"}:
        lhs = _undeclared_lhs(brief, draft, target_definitions)
        additions.extend(lhs)
        if lhs:
            for issue in feedback["issues"]:
                if issue["code"] == "compiler" and "LHS needs" in issue.get(
                    "reason", ""
                ):
                    issue["action"] = (
                        "See undeclared_lhs_entries for all affected names and "
                        "their permitted edits, not only this first failure."
                    )
    feedback["issues"].extend(additions)
    if "repair_agenda" in feedback:
        feedback["repair_agenda"]["representation_issues"].extend(
            dict.fromkeys(row["code"] for row in additions)
        )
    return packet
