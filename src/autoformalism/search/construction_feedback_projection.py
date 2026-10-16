"""Opt-in lossless factoring of repeated structural repair feedback.

Only exact common fields in diagnostic/checklist rows are shared. Row identity,
status, order and multiplicity remain visible; training evidence and compatibility
witnesses are untouched. Historical payload builders never call this module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy

from autoformalism.search import construction_readable as readable

POLICY = "factored-structural-feedback-1"
_ROW_IDENTITY = frozenset(
    {"code", "id", "name", "status", "requirement", "target", "checked_at"}
)
_TABLE_PATHS = (
    ("feedback", "issues"),
    ("requirements_checklist",),
)
_EXPLANATION = (
    "A row's shared_details entries name exact additional fields in "
    "feedback_details. Read them together with that row; its own identity and "
    "status remain separate. All distinct findings remain present. model_view "
    "is the authoritative current topology, not an edit request."
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _rows(packet: dict) -> list[dict]:
    """Select structural records only, excluding numerical/advisory evidence."""
    rows = []
    for path in _TABLE_PATHS:
        value = packet
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def factor(packet: dict) -> dict:
    """Factor repeated fields only when sharing reduces JSON bytes materially.

    expand(factor(packet)) equals packet exactly. No scientific association is
    inferred: fields share storage only when both key and complete value match.
    """
    if "feedback_projection" in packet or "feedback_details" in packet:
        raise ValueError("feedback projection is already applied or names collide")
    result = deepcopy(packet)
    rows = _rows(result)
    if any("shared_details" in row for row in rows):
        raise ValueError("shared_details is reserved for feedback projection")
    occurrences: dict[str, list[int]] = defaultdict(list)
    fields = {}
    for index, row in enumerate(rows):
        for key, value in row.items():
            if key in _ROW_IDENTITY or key.endswith("_status"):
                continue
            encoded = _json({key: value})
            occurrences[encoded].append(index)
            fields[encoded] = (key, value)
    # Fields repeated in the same rows form one readable shared record, instead
    # of placing a separate reference on every sentence or list element.
    groups: dict[tuple[int, ...], dict] = defaultdict(dict)
    for encoded, indices in occurrences.items():
        if len(indices) > 1:
            key, value = fields[encoded]
            groups[tuple(indices)][key] = value
    details = {}
    for indices, shared in groups.items():
        name = f"details_{len(details) + 1}"
        repeated_size = len(indices) * len(_json(shared).encode())
        shared_size = len(_json({name: shared}).encode()) + len(indices) * len(
            _json({"shared_details": [name]}).encode()
        )
        if repeated_size - shared_size < 80:
            continue
        details[name] = shared
        for index in indices:
            for key in shared:
                del rows[index][key]
            rows[index].setdefault("shared_details", []).append(name)
    if not details:
        return result
    result["feedback_details"] = details
    result["feedback_projection"] = {"policy": POLICY, "reading_rule": _EXPLANATION}
    # The explanatory contract itself costs bytes: do not expand small packets.
    return (
        result
        if len(_json(result).encode()) < len(_json(packet).encode())
        else deepcopy(packet)
    )


def expand(packet: dict) -> dict:
    """Restore factored fields for offline integrity checks; never parse prose."""
    result = deepcopy(packet)
    marker = result.pop("feedback_projection", None)
    if marker is None:
        if "feedback_details" in result:
            raise ValueError("feedback details require a projection marker")
        return result
    if marker.get("policy") != POLICY:
        raise ValueError("unsupported feedback projection policy")
    details = result.pop("feedback_details")
    for row in _rows(result):
        for name in row.pop("shared_details", []):
            shared = details[name]
            if set(shared) & set(row):
                raise ValueError("factored feedback fields conflict")
            row.update(deepcopy(shared))
    return result


def project(packet: dict, draft) -> dict:
    """Apply qualified consumer advice then factor the corrected opt-in packet."""
    result = deepcopy(packet)
    feedback = result.get("feedback", {})
    if "issues" in feedback:
        feedback["issues"] = [
            readable.qualified_consumer_feedback(issue, draft)
            for issue in feedback["issues"]
        ]
    return factor(result)
