"""Explain frozen judge evidence without changing its requests or verdicts."""

from __future__ import annotations

import re

from autoformalism.judging import build_atomic_evidence_plan
from autoformalism.judging.hybrid import LEGACY_SIGN_POLICY
from autoformalism.schemas import CandidateModel


def named_references(
    parent: CandidateModel,
    candidate: CandidateModel,
    observation: str,
    *,
    sign_policy: str = LEGACY_SIGN_POLICY,
) -> dict:
    """Resolve only exact atomic IDs on the current candidate in both orientations.

    Reconstruct the same syntax plan as PairedHybridJudge. Parent-only and unknown
    IDs remain unresolved; no fuzzy matching, scientific sign inference or score
    changes are permitted. Certified signs are added only to proposer feedback,
    never to the sign-blinded judge request.
    """
    index = {}
    for orientation, left, right, side in (
        ("forward", parent, candidate, "candidate_b"),
        ("reverse", candidate, parent, "candidate_a"),
    ):
        plan = build_atomic_evidence_plan(left, right, sign_policy=sign_policy)
        for item in plan.occurrences:
            if item.candidate_side == side:
                index[item.occurrence_id] = {
                    "reference_id": item.occurrence_id,
                    "orientation": orientation,
                    "component": item.governed_quantity,
                    "location": item.equation_location,
                    "unsigned_expression": item.unsigned_expression,
                    "actual_outer_polarity": item.actual_polarity,
                }
        for item in plan.repeat_candidates:
            if item.candidate_side == side:
                index[item.repeat_pair_id] = {
                    "reference_id": item.repeat_pair_id,
                    "orientation": orientation,
                    "component": item.governed_quantity,
                    "location": item.equation_location,
                    "unsigned_expression": item.unsigned_expression,
                    "occurrence_ids": list(item.occurrence_ids),
                }
    ids = sorted(
        set(re.findall(r"\b(?:occurrence|repeat)_[A-Za-z0-9_]+\b", observation))
    )
    return {
        "named_references": [index[key] for key in ids if key in index],
        "unresolved_reference_ids": [key for key in ids if key not in index],
        "interpretation": (
            "Outer polarity describes equation syntax, not a state's sign or a "
            "term's value. The expected scientific direction is the judge's "
            "advisory interpretation, not an explicit public requirement unless "
            "supported by the public task. Do not infer a prescribed repair sign."
        ),
    }


def review_status(review: dict | None) -> dict:
    """Keep provider validation failures distinct from scientific concerns.

    Terminal messages remain verbatim. Classification is a reporting aid only;
    missing or duplicated units are never filled, dropped or treated as approval.
    """
    if review is None:
        return {"status": "not_requested", "advice_available": False}
    status = review.get("status", "unknown")
    errors = list((review.get("comparison") or {}).get("prior_terminal_failures", []))
    if review.get("error"):
        errors.append(review["error"])
    diagnostics = []
    for message in errors:
        code = (
            "JUDGE_ATOMIC_UNIT_SET_MISMATCH"
            if "atomic assessment units differ" in message
            else "JUDGE_DUPLICATE_OCCURRENCE_ID"
            if "signed occurrence identifiers must be unique" in message
            else "JUDGE_RESPONSE_UNAVAILABLE"
        )
        diagnostics.append({"code": code, "message": message})
    return {
        "status": status,
        "advice_available": status == "reviewed",
        "scientific_finding_count": len(review.get("findings", [])),
        "physical_requests": (review.get("cost") or {}).get("physical_requests"),
        "observed_total_tokens": (review.get("cost") or {}).get(
            "observed_total_tokens"
        ),
        "no_findings_does_not_imply_approval": True,
        "diagnostics": diagnostics,
        "interpretation": (
            "Review availability is a pipeline status, not evidence against the "
            "candidate. An indeterminate/interrupted review supplied no reliable "
            "paired advice. Scientific disagreement and provider failure are distinct."
        ),
    }
