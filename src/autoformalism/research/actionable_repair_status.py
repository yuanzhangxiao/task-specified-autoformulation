"""Track conditional training conflicts without equating answers with repairs.

This is an opt-in construction stopping policy, not a model-validity predicate.
It consumes already-qualified diagnostics and never chooses scientific edits or
changes the frozen evidence. A prior conflict remains visible when a revision
moves the model outside the diagnostic's supported assumptions.
"""

from __future__ import annotations

from copy import deepcopy

from autoformalism.search.topology_compatibility import Witness
from autoformalism.search.topology_initialization import ClosedWitnessRecheck

_NO_COMPARISON = "no qualifying observed comparison"
_EXPLICIT_TIME = "explicit time dependence is not autonomous"


def _key(row: dict) -> tuple[str, str]:
    """Identify the same necessary-condition test across model revisions."""
    return row["target"], row["property"]


def _actionable(question: dict) -> bool:
    """Only a witness beyond known error bounds is a clear repair request."""
    return any(
        witness.get("error_intervals_disjoint") is True
        for witness in question.get("witnesses", ())
    )


def _prefix_resolution(
    question: dict, diagnostic: dict | None, report: dict, previous: list[dict]
) -> tuple[str, bool, str] | None:
    """Resolve only complete, current checks of every retained paired witness.

    Initializer dependencies may distinguish trajectories without changing the
    topology's ancestor scope. The optional prefix receipt establishes that
    distinction at each saved observation time. Missing or stale receipts cannot
    fall back to a weaker whole-horizon comparison and erase an earlier conflict.
    """
    if (
        question["property"] != "closed_subsystem"
        or "saved_witness_recheck" not in report
    ):
        return None
    unresolved = (
        "deferred_after_revision",
        False,
        "Not every retained paired witness has a current, validated distinction "
        "in initial information or forcing through its observation time.",
    )
    raw = report["saved_witness_recheck"]
    if (
        not isinstance(raw, dict)
        or raw.get("protocol") != "closed-witness-prefix-recheck-1"
    ):
        return unresolved
    try:
        recheck = ClosedWitnessRecheck.model_validate(raw)
    except (ValueError, TypeError):
        return unresolved
    full = report.get("full_report", {})
    conditioned_evidence = report.get(
        "initialization_evidence_sha256", report["evidence_sha256"]
    )
    if (
        recheck.evidence_sha256 != conditioned_evidence
        or full.get("evidence_sha256") != conditioned_evidence
        or recheck.topology_sha256 != report.get("topology_sha256")
        or full.get("topology_sha256") != report.get("topology_sha256")
        or diagnostic is None
        or diagnostic.get("status") not in {"deferred", "no_conflict_found"}
        or set(diagnostic.get("unavailable", ())) - {_NO_COMPARISON}
        or any(_key(row.model_dump()) == _key(question) for row in recheck.diagnostics)
    ):
        return unresolved

    expected = set()
    for old in previous:
        if _key(old) != _key(question):
            continue
        identifier = old.get("question_id")
        witnesses = old.get("witnesses")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(witnesses, (list, tuple))
            or not witnesses
        ):
            return unresolved
        for value in witnesses:
            try:
                witness = Witness.model_validate(value)
            except (ValueError, TypeError):
                return unresolved
            if (
                any(
                    len(values) != 2
                    for values in (
                        witness.trajectories,
                        witness.sample_indices,
                        witness.times,
                        witness.observations,
                    )
                )
                or witness.trajectories[0] == witness.trajectories[1]
                or witness.times[0] != witness.times[1]
                or any(index < 0 for index in witness.sample_indices)
            ):
                return unresolved
            expected.add(
                (
                    identifier,
                    witness.trajectories,
                    witness.sample_indices,
                    witness.times[0],
                )
            )
    observed = set()
    for check in recheck.checks:
        if check.get("target") != question["target"]:
            continue
        if check.get("status") != "conditioning_distinction":
            return unresolved
        try:
            coordinates = (
                check["previous_question_id"],
                tuple(check["trajectories"]),
                tuple(check["sample_indices"]),
                check["conditioning_end_time"],
            )
            observed.add(coordinates)
        except (KeyError, TypeError):
            return unresolved
    if not expected or not expected <= observed:
        return unresolved
    return (
        "conditioning_distinction_verified",
        True,
        "Every retained paired witness was rechecked against current initial "
        "information and forcing through its observation time. Their conditioning "
        "differs, removing this equality restriction even if the graph scope is "
        "unchanged; the revised functions and predictive accuracy remain unverified.",
    )


def _resolution(question: dict, diagnostic: dict | None) -> tuple[str, bool, str]:
    """Distinguish checked removal of a restriction from unsupported disappearance.

    The diagnostics use the exact reason strings below as a deliberately narrow
    interface. Unknown/new reasons abstain. A comparison-free dynamic diagnostic
    with no other unavailable item means all complete conditioning was available,
    but none of the recorded pairs had identical conditioning. An old witness with
    the same immutable evidence supplied at least one pair, so a changed dependency
    scope can explain why that equality-based necessary condition no longer applies.
    """
    if diagnostic is None:
        return "deferred_after_revision", False, "The previous test is unavailable."
    unavailable = set(diagnostic.get("unavailable", ()))
    if (
        diagnostic["status"] == "no_conflict_found"
        and diagnostic.get("comparisons", 0) > 0
        and not unavailable
    ):
        return (
            "no_conflict_found",
            True,
            "The current test performed supported comparisons and found no "
            "conflict; this does not establish fitting quality.",
        )
    changed_scope = set(diagnostic.get("scope", ())) != set(question.get("scope", ()))
    if (
        diagnostic["status"] == "deferred"
        and changed_scope
        and unavailable == {_NO_COMPARISON}
        and question["property"] in {"closed_subsystem", "scalar_autonomous"}
    ):
        return (
            "conditioning_distinction_verified",
            True,
            "The dependency scope changed and complete conditioning was "
            "available, but no original equality/constant-forcing comparison "
            "qualifies for the current model. The old restriction is removed; "
            "the revised functions and trajectories remain unverified.",
        )
    if (
        diagnostic["status"] == "deferred"
        and question["property"] == "scalar_autonomous"
        and changed_scope
        and unavailable == {_EXPLICIT_TIME}
    ):
        return (
            "autonomous_restriction_removed",
            True,
            "The graph now explicitly depends on time, so the autonomous-scalar "
            "restriction does not apply. Public permission for this dependency "
            "still belongs to the structural checks; behavior remains unverified.",
        )
    return (
        "deferred_after_revision",
        False,
        "The previous conflict is no longer reported, but unsupported graph, "
        "initialization, conditioning, or model-class assumptions prevent a "
        "verified resolution. A changed or invalid model is not itself evidence "
        "that the original witness was addressed.",
    )


def build_state(
    report: dict,
    baseline: dict | None,
    structural_eligible: bool,
    decisions: dict[str, dict] | None = None,
) -> dict:
    """Build an independent repair status from current and previously seen tests.

    Pass the previous returned state as ``baseline`` to retain conflicts first
    encountered during repair. A raw initial compatibility report is also valid.
    Question answers are receipts only. In particular, an approximation or an
    unverified assumption dispute cannot clear a current or prior conflict.
    """
    if baseline and baseline["evidence_sha256"] != report["evidence_sha256"]:
        raise ValueError("actionable repair evidence changed")
    answers = decisions or {}
    current = list(report.get("advisory_questions", ()))
    previous = list(
        (baseline or {}).get(
            "tracked_conflicts", (baseline or {}).get("advisory_questions", ())
        )
    )
    tracked = {_key(question): deepcopy(question) for question in previous}
    for question in current:
        tracked.setdefault(_key(question), deepcopy(question))
    current_by_key = {_key(question): question for question in current}
    diagnostics = {
        _key(row): row for row in report.get("full_report", {}).get("diagnostics", ())
    }
    pending = []
    for question in current:
        row = deepcopy(question)
        row["repair_status"] = (
            "repair_required" if _actionable(row) else "uncertainty_limited"
        )
        answer = answers.get(row["question_id"])
        if answer:
            row["recorded_decision"] = deepcopy(answer)
            row["answer_resolves_conflict"] = False
            row["reminder"] = (
                "Your previous answer is recorded. The same conditional conflict "
                "remains; an explanation alone does not change the model or the "
                "frozen evidence."
            )
        pending.append(row)

    outcomes = []
    for key, old in sorted(tracked.items()):
        diagnostic = diagnostics.get(key)
        present = current_by_key.get(key)
        if not structural_eligible:
            status, resolved, reason = (
                "structurally_blocked",
                False,
                "The current model does not pass the structural checks; "
                "missing compatibility questions cannot establish resolution.",
            )
        elif present:
            status, resolved, reason = (
                "repair_required" if _actionable(present) else "uncertainty_limited",
                False,
                "The current diagnostic still reports this conditional conflict.",
            )
        else:
            prefix = _prefix_resolution(old, diagnostic, report, previous)
            status, resolved, reason = prefix or _resolution(old, diagnostic)
        outcomes.append(
            {
                "target": key[0],
                "property": key[1],
                "previous_question_id": old["question_id"],
                "current_question_id": present["question_id"] if present else None,
                "current_status": diagnostic["status"] if diagnostic else "unavailable",
                "previously_actionable": _actionable(old),
                "status": status,
                "resolved": resolved,
                "reason": reason,
            }
        )
    actionable = [q for q in pending if q["repair_status"] == "repair_required"]
    uncertain = [q for q in pending if q["repair_status"] == "uncertainty_limited"]
    unresolved = [row for row in outcomes if not row["resolved"]]
    status = (
        "structurally_blocked"
        if not structural_eligible
        else "repair_required"
        if actionable
        else "review_deferred"
        if uncertain or unresolved
        else "checks_clear"
    )
    return {
        **deepcopy(report),
        "pending_questions": pending,
        "actionable_questions": actionable,
        "uncertain_questions": uncertain,
        "tracked_conflicts": [tracked[key] for key in sorted(tracked)],
        "baseline_outcomes": outcomes,
        "decisions": [deepcopy(answers[key]) for key in sorted(answers)],
        "repair_status": status,
        "review_status": status,
        "completion_allowed": status == "checks_clear",
        "structural_eligible": structural_eligible,
        "scientific_correctness": "unverified",
        "scope": "Resolution concerns only these necessary-condition witnesses, "
        "not fit quality or universal scientific compatibility. Answer records "
        "do not modify the frozen assumptions or satisfy a repair request.",
    }
