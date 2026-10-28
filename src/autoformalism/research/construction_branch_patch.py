"""Explicit edits to one unadopted branch patch, never to a sibling or model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.research.construction_revision_edits import LawEdit
from autoformalism.research.construction_structural_revision import (
    InitializationEdit,
    NewLatentState,
    NewProcess,
)
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.staged_topology import content_hash

POLICY = "explicit-branch-patch-delta-1"
IDENTICAL_ACTION_POLICY = "identical-pending-action-normalization-1"
ACTION_KEYS = {
    "edits": "component",
    "new_states": "name",
    "new_processes": "name",
    "new_parameters": "name",
    "initialization_edits": "state",
    "remove_states": None,
    "remove_processes": None,
}
_LIMITS = dict(zip(ACTION_KEYS, (32, 8, 16, 64, 16, 8, 16), strict=True))


class BranchLawEdit(LawEdit):
    """Expressions determine dependencies; omitted metadata is not ambiguous."""

    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)


class BranchPatchFields(StrictSchema):
    """Sparse patch edits; a supplied list entry replaces that whole entry."""

    decision: Literal["revise", "retain", "requires_wider_revision"] | None = None
    hypothesis: str | None = Field(default=None, min_length=1, max_length=3000)
    evidence_ids: tuple[str, ...] | None = Field(
        default=None, min_length=1, max_length=16
    )
    expected_observable_effect: str | None = Field(
        default=None, min_length=1, max_length=3000
    )
    edits: tuple[BranchLawEdit, ...] = Field(default=(), max_length=32)
    new_states: tuple[NewLatentState, ...] = Field(default=(), max_length=8)
    new_processes: tuple[NewProcess, ...] = Field(default=(), max_length=16)
    new_parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=64)
    initialization_edits: tuple[InitializationEdit, ...] = Field(
        default=(), max_length=16
    )
    remove_states: tuple[Identifier, ...] = Field(default=(), max_length=8)
    remove_processes: tuple[Identifier, ...] = Field(default=(), max_length=16)


class DropAction(StrictSchema):
    """Delete a pending intention; this does not delete a parent component."""

    field: Literal[
        "edits",
        "new_states",
        "new_processes",
        "new_parameters",
        "initialization_edits",
        "remove_states",
        "remove_processes",
    ]
    key: Identifier


class BranchRepairReply(StrictSchema):
    """Pin the selected branch and choose delta, whole replacement, or withdrawal."""

    mode: Literal["delta", "replace", "withdraw"]
    base_candidate_id: str = Field(min_length=1, max_length=128)
    base_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch: BranchPatchFields | None = None
    drop_actions: tuple[DropAction, ...] = Field(default=(), max_length=160)

    @model_validator(mode="after")
    def mode_fields(self) -> BranchRepairReply:
        if self.mode == "withdraw":
            if self.patch is not None or self.drop_actions:
                raise ValueError("withdraw requires patch=null and no drop_actions")
        elif self.patch is None:
            raise ValueError("delta/replace requires patch")
        if self.mode != "delta" and self.drop_actions:
            raise ValueError("drop_actions is supported only for mode=delta")
        return self


def complete_patch(candidate: dict) -> Any:
    """Return the exact stored intention, including scientifically invalid edits."""
    return deepcopy(candidate.get("composed_patch", candidate.get("raw_reply")))


def reference(candidate: dict) -> dict:
    """Bind edits to the selected candidate's complete pending intention."""
    return {
        "base_candidate_id": candidate["id"],
        "base_patch_sha256": content_hash(complete_patch(candidate)),
    }


def _identity(field: str, row: Any) -> str:
    key = ACTION_KEYS[field]
    value = (
        row.get(key) if key and isinstance(row, dict) else row if key is None else None
    )
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field} entry has no unambiguous {key or 'name'}; use replace"
        )
    return value


def _entries(
    field: str,
    rows: Any,
    *,
    normalizations: list[dict] | None = None,
    location: str = "",
) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) > _LIMITS[field]:
        raise ValueError(f"invalid or excessive {field} list; use replace")
    result = {}
    first_indices = {}
    for index, row in enumerate(rows):
        key = _identity(field, row)
        if key in result:
            if normalizations is not None and content_hash(row) == content_hash(
                result[key]
            ):
                normalizations.append(
                    {
                        "location": location,
                        "field": field,
                        "key": key,
                        "retained_index": first_indices[key],
                        "removed_index": index,
                        "entry_sha256": content_hash(row),
                    }
                )
                continue
            raise ValueError(
                f"ambiguous repeated {field} key {key}; "
                "use replace or explicitly drop it"
            )
        result[key] = deepcopy(row)
        first_indices[key] = index
    return result


def compose(
    raw: Any, intended: dict, *, repair_contract: bool = False
) -> tuple[dict | None, dict]:
    """Compose only explicit same-branch actions; all scientific checks run later."""
    normalized_input = deepcopy(raw)
    normalizations = [] if repair_contract else None
    if repair_contract and isinstance(normalized_input, dict):
        supplied_patch = normalized_input.get("patch")
        if isinstance(supplied_patch, dict):
            # Compare original JSON records before schema defaults can make two
            # different declarations look identical. Raw list limits still apply.
            for field in ACTION_KEYS:
                if field in supplied_patch:
                    supplied_patch[field] = list(
                        _entries(
                            field,
                            supplied_patch[field],
                            normalizations=normalizations,
                            location="submitted_patch",
                        ).values()
                    )
    reply = BranchRepairReply.model_validate(normalized_input)
    expected = reference(intended)
    if (
        reply.base_candidate_id != expected["base_candidate_id"]
        or reply.base_patch_sha256 != expected["base_patch_sha256"]
    ):
        raise ValueError(
            "branch reference differs; use the displayed selected candidate "
            "and patch hash, never a sibling"
        )
    prior = complete_patch(intended)
    provenance = {
        "policy": POLICY,
        "mode": reply.mode,
        **expected,
        "branch_id": intended["branch_id"],
        "parent_unchanged": True,
        "model_adopted": False,
    }
    if repair_contract:
        provenance["identical_action_normalization"] = {
            "policy": IDENTICAL_ACTION_POLICY,
            "removed_entries": normalizations,
            "submitted_reply_sha256": content_hash(raw),
        }
    if reply.mode == "withdraw":
        return None, provenance
    supplied = reply.patch.model_dump(mode="json", exclude_unset=True)
    # Null optional metadata means no replacement. Empty action lists add nothing
    # in delta mode; clearing an intention always requires an explicit drop.
    supplied = {key: value for key, value in supplied.items() if value is not None}
    defaults = []
    for row in supplied.get("edits", []):
        if "sources" not in row:
            row["sources"] = []
            defaults.append({"component": row["component"], "field": "sources"})
    provenance["derived_dependency_metadata_defaults"] = defaults
    if reply.mode == "replace":
        result = deepcopy(supplied)
        for field in ACTION_KEYS:
            _entries(field, result.get(field, []))
        provenance.update(retained_actions=[], replaced_actions=[], dropped_actions=[])
    else:
        if not isinstance(prior, dict):
            raise ValueError("stored branch is not a patch object; use mode=replace")
        unknown = set(prior) - set(BranchPatchFields.model_fields)
        if unknown:
            raise ValueError("stored patch has unknown fields; use mode=replace")
        result = deepcopy(prior)
        drops = [(row.field, row.key) for row in reply.drop_actions]
        if len(drops) != len(set(drops)):
            raise ValueError("duplicate drop_actions")
        retained, replaced = [], []
        for field in ACTION_KEYS:
            old = result.get(field, [])
            if not isinstance(old, list) or len(old) > _LIMITS[field]:
                raise ValueError(f"invalid stored {field}; use replace")
            field_drops = {key for group, key in drops if group == field}
            old_names = [_identity(field, row) for row in old]
            if not field_drops <= set(old_names):
                raise ValueError(f"drop_actions names no stored {field} intention")
            # Explicitly dropping an ambiguous key removes all its competing rows.
            previous = _entries(
                field,
                [row for row in old if _identity(field, row) not in field_drops],
                normalizations=normalizations,
                location="stored_patch_after_explicit_drops",
            )
            updates = _entries(field, supplied.get(field, []))
            if field_drops & set(updates):
                raise ValueError(
                    f"cannot both drop and replace {field} key in one reply"
                )
            for key, value in previous.items():
                if key not in updates:
                    retained.append(
                        {"field": field, "key": key, "sha256": content_hash(value)}
                    )
            replaced.extend({"field": field, "key": key} for key in updates)
            previous.update(updates)
            if len(previous) > _LIMITS[field]:
                raise ValueError(f"composed {field} exceeds its bounded list limit")
            if field in result or field in supplied or field_drops:
                result[field] = list(previous.values())
        result.update(
            {key: value for key, value in supplied.items() if key not in ACTION_KEYS}
        )
        provenance.update(
            retained_actions=retained,
            replaced_actions=replaced,
            dropped_actions=[{"field": field, "key": key} for field, key in drops],
        )
    provenance["composed_patch_sha256"] = content_hash(result)
    return result, provenance


def literal_changes(candidate: dict) -> dict:
    """Display actual attempted RHSs/initializers rather than only edited names."""
    patch = complete_patch(candidate)
    if not isinstance(patch, dict):
        return {}
    result = {
        field: deepcopy(patch[field]) for field in ACTION_KEYS if patch.get(field)
    }
    for field, keys in {
        "edits": ("component", "kind", "expression"),
        "new_states": ("name", "rhs", "initial"),
        "new_processes": ("name", "expression"),
    }.items():
        rows = result.get(field)
        if isinstance(rows, list):
            result[field] = [
                {key: row[key] for key in keys if key in row}
                if isinstance(row, dict)
                else row
                for row in rows
            ]
    return result
