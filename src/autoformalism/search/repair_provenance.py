"""Stage-specific repair evidence without rewriting historical construction facts."""

from __future__ import annotations

import copy

from autoformalism.search.requirement_feedback import repair_failure_feedback
from autoformalism.staged_topology import content_hash

STAGE_AWARE = "stage-aware-1"


def attempt_feedback(bundle: dict, record: dict, raw: object, error: Exception) -> dict:
    """Separate provider delivery from rejection of an actual mathematical reply."""
    choices = (record.get("raw_response") or {}).get("choices", [])
    reason = (
        choices[0].get("finish_reason")
        if isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        else None
    )
    if record.get("status") != "responded" or reason != "stop" or raw is None:
        truncated = record.get("status") == "responded" and reason == "length"
        return {
            "schema_version": "repair-attempt-feedback-1",
            "stage": "provider_response",
            "code": "PROVIDER_RESPONSE_TRUNCATED"
            if truncated
            else "PROVIDER_RESPONSE_UNAVAILABLE",
            "finish_reason": reason,
            "error": str(error)[:6000],
            "rejected_reply": None,
            "transaction": "rolled_back",
            "mathematical_reply_evaluated": False,
            "next_action": (
                "No complete usable final JSON reply was received. Return one concise "
                "complete JSON object with interaction_id, expression and parameters. "
                "Keep the same model context; this is not evidence "
                "that the selected interaction or equation is invalid."
            ),
            "budget_policy": (
                "Retry only within the existing remaining request/token budget."
            ),
        }
    return {
        **repair_failure_feedback(bundle, raw, error),
        "schema_version": "repair-attempt-feedback-1",
        "stage": "function_contract",
        "mathematical_reply_evaluated": True,
    }


def repair_provenance(bundle: dict, result: dict) -> dict:
    """Compare this requirement repair against its immediate accepted parent."""
    selected = result["selected_function"]
    original = next(
        s
        for s in bundle["slots"]
        if s["interaction_id"] == result["selected_interaction"]
    )
    before, after = original["canonical_function"], selected["canonical_function"]
    return {
        "schema_version": "requirement-repair-provenance-1",
        "stage": "requirement_repair",
        "candidate_hash_convention": "legacy-content-hash-1",
        "interaction_id": result["selected_interaction"],
        "before_candidate_sha256": content_hash(bundle["candidate"]),
        "after_candidate_sha256": content_hash(result["candidate"]),
        "before_rhs": before["expression"],
        "after_rhs": after["expression"],
        "rhs_changed": before["expression"] != after["expression"],
        "parameter_declarations_changed": before["parameters"] != after["parameters"],
        "selected_slot_preserved_without_call": False,
        "protected_slots_preserved": result["protected_slots_preserved"],
        "initialization_plan_preserved": bundle["initialization"]["plan"]
        == result["initialization"]["plan"],
        "outer_sign_normalizations": result.get("outer_sign_normalizations", []),
        "comparison": "Exact canonical structure; parameter names may be rebound.",
    }


def with_repair_provenance(bundle: dict, result: dict) -> dict:
    """Separate old construction counters from the newly accepted requirement edit."""
    revised = copy.deepcopy(result)
    original = revised["selected_function"]
    revised["selected_function"] = {
        key: original[key]
        for key in (
            "interaction_id",
            "selected_term",
            "accepted_reply",
            "canonical_function",
        )
    }
    revised["selected_function"]["construction_provenance"] = {
        key: value
        for key, value in original.items()
        if key not in revised["selected_function"]
    }
    revised["requirement_repair_provenance"] = repair_provenance(bundle, result)
    return revised
