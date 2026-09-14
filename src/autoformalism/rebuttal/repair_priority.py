"""Evidence-strength routing; support is ordinal and claim-specific, not probability."""

from __future__ import annotations

import math
from typing import Literal

from autoformalism.rebuttal.repair_evidence import Finding, model_hash
from autoformalism.rebuttal.repair_fit_reporting import fit_outcomes
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "repair-evidence-priority-1"
RoutingPolicy = Literal["legacy_category", "evidence_strength"]


class EvidenceSupport(StrictSchema):
    """What supports a narrowly stated claim; never a model-validity probability."""

    strength: Literal["unresolved", "observed", "corroborated", "certified"]
    scope: Literal[
        "contract",
        "domain",
        "scientific_absolute",
        "scientific_preference",
        "selected_fit",
        "numerical_search",
    ]
    basis: tuple[str, ...]
    limitation: str


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def numerical_reliability(fit: dict) -> dict:
    """Read public fit evidence; exhausted search does not establish impossibility."""
    refinement = fit.get("refinement") or {}
    training = fit.get("training") or {}
    outcomes = fit_outcomes(fit)
    verified = (
        fit.get("status") == "complete"
        and refinement.get("production_training_rollout_verified") is True
        and _finite(training.get("normalized_mse"))
        and training.get("failed_trajectories") == []
    )
    derivative_difficulty = (
        refinement.get("state_integration_succeeded") is True
        and refinement.get("augmented_integration_failed") is True
    )
    strength = (
        "corroborated"
        if verified and outcomes["selected_optimizer_converged"] is True
        else "observed"
        if verified or derivative_difficulty
        else "unresolved"
    )
    basis = []
    if verified:
        basis.append("finite_production_training_replay_at_selected_point")
        if outcomes["selected_optimizer_converged"] is True:
            basis.append("verified_local_optimizer_termination_at_selected_point")
    if derivative_difficulty:
        basis.append("state_only_success_and_augmented_failure_at_evaluated_points")
    if not basis:
        basis.append("selected_point_verification_unavailable")
    opportunity_keys = (
        "actual_residual_calls",
        "valid_residual_evaluations",
        "fit_seconds",
        "shared_evaluation_budget",
        "budget_exhausted",
        "state_integration_succeeded",
        "augmented_integration_failed",
        "integration_failures",
        "rejected_sensitivity_trials",
    )
    opportunity = {key: refinement.get(key) for key in opportunity_keys}
    opportunity["screen_count"] = (
        len(refinement["screens"])
        if isinstance(refinement.get("screens"), list)
        else None
    )
    return {
        "support": EvidenceSupport(
            strength=strength,
            scope="selected_fit" if verified else "numerical_search",
            basis=tuple(basis),
            limitation=(
                "Replay and local convergence describe the selected parameters. "
                "They do not prove global optimality, scientific validity, or "
                "structural infeasibility. State/sensitivity flags concern evaluated "
                "points and need not describe the selected point."
            ),
        ).model_dump(mode="json"),
        "selected_training_replay_verified": verified,
        "selected_optimizer_converged": outcomes["selected_optimizer_converged"],
        "training_nmse": training.get("normalized_mse"),
        "training_failed_trajectories": training.get("failed_trajectories"),
        "search_opportunity": opportunity,
        "optimizer_stages": outcomes["optimizer_stages"],
        "collocation_status": {
            key: (fit.get("initializer") or {}).get(key)
            for key in ("success", "message", "optimizer_started")
        },
        "validation_used_for_confidence": False,
        "global_infeasibility_established": False,
    }


def scientific_support(finding: Finding, review: dict | None) -> EvidenceSupport:
    """Use exact matched-pair provenance, never confidence asserted in LLM prose."""
    review = review or {}
    pair = review.get("comparison") or {}
    consensus = pair.get("consensus_result") or {}
    absolute = finding.category == "scientific_requirement"
    records = consensus.get(
        "absolute_assessments" if absolute else "comparative_assessments", []
    )
    matched = [
        item
        for item in records
        if item.get("criterion") == finding.code
        and (not absolute or item.get("subject_id") == finding.component)
    ]
    valid = (
        (
            review.get("status") == "reviewed"
            and bool(review.get("request_sha256"))
            and finding.evidence.get("request_sha256") == review["request_sha256"]
            and len(pair.get("request_hashes", [])) == 4
            and len(matched) == 1
            and (
                (
                    (matched[0].get("candidate_b") or {}).get("verdict") == "fail"
                    and (matched[0].get("candidate_b") or {}).get("evidence")
                    == finding.observation
                )
                if absolute
                else (
                    matched[0].get("verdict") == "candidate_a"
                    and matched[0].get("evidence") == finding.observation
                )
            )
        )
        if matched
        else False
    )
    # The frozen default uses strict question consensus. A different aggregation
    # must establish its own support semantics rather than inherit this certificate.
    valid = valid and pair.get("protocol_version") == (
        "incumbent-hybrid-question-consensus-4-blinded-metadata"
    )
    distinct_requests = len(set(pair.get("request_hashes", []))) == 4
    return EvidenceSupport(
        strength="corroborated"
        if valid and absolute and distinct_requests
        else "observed"
        if valid
        else "unresolved",
        scope="scientific_absolute" if absolute else "scientific_preference",
        basis=(
            "matched_orientation_question_consensus"
            if distinct_requests
            else "same_requests_reused_across_orientations",
        )
        if valid
        else ("paired_support_not_verified",),
        limitation=(
            "Orientation agreement is repeatable advice, not calibrated correctness. "
            "This judge has not been validated for unpruned pre-fit models. "
            "A comparative preference is not an absolute scientific defect."
        ),
    )


def route_evidence(
    candidate: CandidateModel,
    runtime: list[Finding],
    science: list[Finding],
    numerical: list[Finding],
    *,
    fit: dict,
    scientific_review: dict | None,
) -> dict:
    """Keep certified blockers first; expose ties instead of inventing confidence."""
    identity = model_hash(candidate)
    reliability = numerical_reliability(fit)
    ranked, excluded = [], []
    for finding in [*runtime, *science, *numerical]:
        if finding.candidate_sha256 != identity:
            excluded.append(
                {
                    "finding": finding.model_dump(mode="json"),
                    "reason": "different_candidate",
                }
            )
            continue
        if finding.source == "runtime":
            blocking = finding.blocking and finding.certainty in {
                "certified",
                "observed",
            }
            support = EvidenceSupport(
                strength="certified" if blocking else "unresolved",
                scope="contract" if blocking else "domain",
                basis=(
                    "deterministic_contract_failure"
                    if blocking
                    else "unresolved_static_risk",
                ),
                limitation="A possible domain violation is not a reachable failure.",
            )
            category = "runtime_contract" if blocking else "domain_uncertainty"
        elif finding.source == "scientific_judge":
            support = scientific_support(finding, scientific_review)
            category = (
                "scientific_requirement"
                if (support.scope == "scientific_absolute")
                else "scientific_comparison"
            )
        else:
            # Phase status messages remain visible but are not independent votes.
            if finding.code not in {"FINITE_ROLLOUT", "FIT_UNAVAILABLE"}:
                continue
            support = EvidenceSupport.model_validate(reliability["support"])
            category = (
                "prediction_refinement"
                if (reliability["selected_training_replay_verified"])
                else "numerical_feasibility"
            )
        ranked.append(
            {
                "finding": finding.model_dump(mode="json"),
                "support": support.model_dump(mode="json"),
                "category": category,
            }
        )
    levels = {"unresolved": 0, "observed": 1, "corroborated": 2, "certified": 3}
    best = max((levels[r["support"]["strength"]] for r in ranked), default=-1)
    selected = [r for r in ranked if levels[r["support"]["strength"]] == best]
    categories = sorted({r["category"] for r in selected})
    return {
        "schema_version": "repair-decision-report-4",
        "objective_category": categories[0]
        if len(categories) == 1
        else "joint_review"
        if categories
        else "model_review",
        "focus": [r["finding"] for r in selected],
        "priority_evidence": {
            "protocol": PROTOCOL,
            "ranked_findings": ranked,
            "excluded_findings": excluded,
            "selected_categories": categories,
            "reason": (
                "Certified deterministic blockers first; otherwise compare support "
                "for each limited claim. Equal support retains joint attention. "
                "Strength levels are a prespecified routing policy, not probabilities."
            ),
            "scientific_or_numerical_findings_block": False,
        },
        "numerical_reliability": reliability,
    }
