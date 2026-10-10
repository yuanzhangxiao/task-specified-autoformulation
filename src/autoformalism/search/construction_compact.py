"""Lossless prompt deduplication and explicit state/process collision feedback."""

from __future__ import annotations

from copy import deepcopy

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger

POLICY = "minimal-deferred-interaction-2"


def naming_conflicts(
    brief: PublicScientificBrief, draft: ledger.Draft, patch: ledger.DraftPatch
) -> list[dict]:
    """Explain structured conflicts without renaming a scientific quantity."""
    variables = {
        v.name: v.definition
        for v in draft.variables
        if v.name not in patch.remove_variables
    }
    variables.update((v.name, v.definition) for v in patch.variables)
    public = {v.name: v.data_role for v in brief.public_variables}
    rows = []
    for p in patch.processes:
        role = (
            "differential state"
            if variables.get(p.name) == "differential"
            else f"public {public[p.name]} source"
            if p.name in public
            and public[p.name] != "target"
            and p.name not in variables
            else None
        )
        if role is None:
            continue
        rows.append(
            {
                "code": "process_name_collision",
                "name": p.name,
                "existing_role": role,
                "proposed_drivers": list(p.depends_on),
                "proposed_receivers": [u.model_dump(mode="json") for u in p.uses],
                "explanation": (
                    f"{p.name} already denotes a {role}. processes.name defines a "
                    "new instantaneous quantity, not the existing driver or its "
                    "derivative. depends_on names its drivers; uses names its "
                    "receivers."
                ),
                "repair_options": [
                    "If you intend a new named law, choose an unused process name "
                    "and retain the scientifically intended drivers and signed uses. "
                    "For example, P_X depends_on=[X] means P_X=phi(X); it does not "
                    "redeclare the state X as X=phi(X).",
                    "If you intend an ordinary contribution, omit this process "
                    "addition and include its drivers and sign in the receiving "
                    "variable's ordinary topology at the topology stage. A name "
                    "already in current_draft.declarations.processes needs an explicit "
                    "remove_processes edit if you intend to remove it.",
                    "If you intend to change a variable's scientific type, make "
                    "that explicit in variables and revise affected declarations; "
                    "do not change a state type merely to resolve a name collision.",
                ],
                "runtime_renamed_anything": False,
            }
        )
    return rows


def _parent(payload: dict, path: str) -> tuple[dict | None, str]:
    parts = path.split(".")
    node = payload
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
    return (node if isinstance(node, dict) else None), parts[-1]


def compact_payload(payload: dict) -> dict:
    """Keep canonical fields once; only remove exactly equal redundant copies."""
    result = deepcopy(payload)
    result["bookkeeping_policy"] = POLICY
    aliases = []
    for duplicate, canonical in (
        ("runtime_diagnostics", "last_edit_result"),
        ("runtime_diagnostics.last_edit_result", "last_edit_result"),
        (
            "last_edit_result.repair_receipt.committed_edits",
            "last_edit_result.edit_effects",
        ),
        ("last_edit_result.shared_process_checklist", "shared_process_checklist"),
        ("runtime_diagnostics.shared_process_checklist", "shared_process_checklist"),
        ("last_edit_result.stage_completion", "shared_process_checklist"),
        ("runtime_diagnostics.stage_completion", "shared_process_checklist"),
        (
            "runtime_diagnostics.remaining_topology_failures",
            "runtime_diagnostics.structural_failures",
        ),
    ):
        source, source_key = _parent(result, duplicate)
        target, target_key = _parent(result, canonical)
        if source is None or target is None:
            continue
        value = source.get(source_key)
        if (
            isinstance(value, (dict, list))
            and value
            and value == target.get(target_key)
        ):
            del source[source_key]
            aliases.append({"omitted_copy": duplicate, "same_content_at": canonical})
    if aliases:
        result["payload_references"] = {
            "meaning": "Exact duplicate fields are displayed once at the paths below.",
            "aliases": aliases,
        }
    return result


def expand_payload(payload: dict) -> dict:
    """Reconstruct exact original fields for offline verification, not LLM input."""
    result = deepcopy(payload)
    aliases = result.pop("payload_references", {}).get("aliases", [])
    for row in reversed(aliases):
        parent, key = _parent(result, row["omitted_copy"])
        source, source_key = _parent(result, row["same_content_at"])
        if parent is None or source is None or source_key not in source:
            raise ValueError("invalid prompt reference")
        parent[key] = deepcopy(source[source_key])
    return result
