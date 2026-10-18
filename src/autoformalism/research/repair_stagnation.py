"""Scientific-attempt fingerprints independent of reply-control toggles."""

from __future__ import annotations

from copy import deepcopy

from autoformalism.search import construction_actionable_repair as actionable
from autoformalism.search.construction_protocol import substantive
from autoformalism.staged_topology import content_hash

_NONSCIENTIFIC = frozenset(
    {"explanation", "reference_sha256", "compatibility_decisions", "stage_complete"}
)


def scientific(value: object) -> object:
    """Canonicalize declarations, retaining multiplicity, signs and dependencies."""
    if isinstance(value, dict):
        value = {
            key: scientific(item)
            for key, item in value.items()
            if key not in _NONSCIENTIFIC
        }
    elif isinstance(value, (list, tuple)):
        value = [scientific(item) for item in value]
    return substantive(value)


def model_signature(draft, check: dict, report: dict) -> str:
    """Ignore presentation changes while retaining the unresolved obligations."""
    return content_hash(
        [
            scientific(draft.model_dump(mode="json")),
            scientific(check["errors"]),
            check["eligible"],
            sorted(q["question_id"] for q in report["advisory_questions"]),
        ]
    )


def attempt_signature(
    draft, check: dict, report: dict, *, error: str | None, normalized, raw
) -> str:
    """Keep genuinely different rejected alternatives; collapse accepted/no-op repeats.

    An unchanged full restatement and an empty completion reply have the same
    scientific effect. A rejected edit with different mathematical content keeps
    its own fingerprint; failure alone must not exhaust useful alternatives.
    """
    alternative = None
    if error and not error.startswith("no draft change;"):
        proposed = normalized if normalized is not None else raw
        if isinstance(proposed, dict):
            alternative = scientific(
                {
                    key: value
                    for key, value in proposed.items()
                    if key in actionable.DraftPatch.model_fields
                    and key not in _NONSCIENTIFIC
                    and value not in (None, [], ())
                }
            )
        else:
            alternative = proposed
    return content_hash([model_signature(draft, check, report), alternative])


def initializer_semantics(value: object) -> object:
    """Ignore handoff record order, retaining multiplicity and ordered arguments.

    This opt-in comparison does not validate or repair a reply. Invalid records
    remain distinct alternatives; only the order of state records is immaterial.
    """
    result = deepcopy(value)
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        result["records"] = sorted(result["records"], key=content_hash)
    return result


def initializer_attempt_signature(
    base: str, retained: dict, proposed: object, *, error: str | None
) -> str:
    """Hash accepted initializer effects, while preserving rejected alternatives.

    An accepted full restatement and null both retain the same handoff. Rejected
    initializers remain part of the attempt so a genuinely different alternative
    can still be tried. Historical policies keep their original raw-payload hash.
    """
    effective = initializer_semantics(retained)
    alternative = None
    if error and not error.startswith("no draft change;") and proposed is not None:
        candidate = initializer_semantics(proposed)
        if candidate != effective:
            alternative = candidate
    return content_hash([base, effective, alternative])
