"""M38: measured discrepancies and conditional, testable investigation drafts.

Public entry points verify the M37 packet against its diagnostic source. This
layer does not execute models, infer scientific obligations, or apply revisions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import interaction_mismatches as mismatches
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import Sha256


class Policy(StrictSchema):
    """Presentation settings, never statistical significance or fit criteria."""

    protocol: Literal["interaction-advice-policy-1"] = "interaction-advice-policy-1"
    extraction: mismatches.Policy = Field(default_factory=mismatches.Policy)
    investigations_per_target: int = Field(default=3, ge=1, le=4)
    maximum_user_bytes: int = Field(default=96000, ge=1024, le=300000)


class Readiness(StrictSchema):
    """Search evidence supports scoped decisions, not a universal 'fit done'."""

    preferred_stage: Literal["numerical_checks", "fitting", "interaction", "retain"]
    hypothesis_status: Literal["permitted", "conditional", "withheld"]
    search_scope: Literal[
        "unverified",
        "accurate_output",
        "improving",
        "incomplete",
        "retained_point_stationary",
        "restricted_candidate_optimum",
    ]
    rationale: str
    alternative_action: str
    fitting_globally_solved: Literal[False] = False
    automatic_stage_transition: Literal[False] = False


class Investigation(StrictSchema):
    """A question to investigate, with competing explanations and a test."""

    target: str
    family: Literal["response_strength", "timing_recovery", "balance", "time_window"]
    evidence_ids: tuple[str, ...]
    candidate_law_refs: tuple[str, ...]
    affected_targets: tuple[str, ...]
    primary_stage: Literal["interaction"] = "interaction"
    conditional_hypothesis: str
    alternatives: tuple[str, ...]
    additional_evidence_needed: tuple[str, ...]
    expected_observable_change: str
    disconfirmation: str
    broader_stage_condition: str
    fault_localized: Literal[False] = False
    edit_authorized: Literal[False] = False


class Identity(StrictSchema):
    source_packet_sha256: Sha256
    policy_sha256: Sha256
    draft_source_sha256: str


class Advice(StrictSchema):
    protocol: Literal["interaction-advice-1"] = "interaction-advice-1"
    identity: Identity
    policy: Policy
    readiness: Readiness
    mismatches: mismatches.MismatchSummary
    parameter_information: str
    investigations: tuple[Investigation, ...]
    omitted_investigations: dict[str, int]
    withheld_investigation_reasons: dict[str, str]
    preservation_requirements: tuple[str, ...]
    missing_contract: Literal["authoritative_constructor_source_sign_obligations"] = (
        "authoritative_constructor_source_sign_obligations"
    )
    observation_interpretation: str = (
        "Compare predictions with observed training readings; these are not private "
        "hidden-state truth. No measurement-error diagnosis is made."
    )
    interpretation: tuple[str, ...]
    optimization_calls: Literal[0] = 0
    rollout_calls: Literal[0] = 0
    live_llm_calls: Literal[0] = 0
    validation_used: Literal[False] = False
    test_data_opened: Literal[False] = False
    model_changed: Literal[False] = False


class ProposedInvestigation(feedback.Hypothesis):
    """Optional future proposer reply; this module never applies the answer."""

    target_stage: Literal[
        "interaction", "variable", "shared_process_topology", "fitting", "retain"
    ]
    alternative_explanations: tuple[str, ...] = Field(min_length=1, max_length=4)
    disconfirmation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def stage_matches_action(self):
        allowed = {
            "function_hypothesis": {"interaction"},
            "topology_revision_needed": {"variable", "shared_process_topology"},
            "no_change": {"fitting", "retain"},
        }
        if self.target_stage not in allowed[self.action]:
            raise ValueError("proposed stage differs from advisory action")
        return self


def readiness(packet: feedback.Packet) -> Readiness:
    """Use a positive route list; availability is not optimization convergence."""
    facts = packet.feedback
    if (
        packet.status != "ready_for_advisory_review"
        or packet.response.status != "available"
        or facts.output_status != "verified"
        or facts.solver_agreement is not True
    ):
        return Readiness(
            preferred_stage="numerical_checks",
            hypothesis_status="withheld",
            search_scope="unverified",
            rationale="Independent output evidence is unavailable or unverified.",
            alternative_action=(
                "Complete the missing numerical check before interpreting "
                "discrepancies."
            ),
        )
    if facts.diagnosis in {
        "accurate_local_evidence",
        "accurate_ambiguity",
        "accurate_unresolved",
    }:
        return Readiness(
            preferred_stage="retain",
            hypothesis_status="withheld",
            search_scope="accurate_output",
            rationale=(
                "The numerical output target was reached; do not force an "
                "error-driven rewrite."
            ),
            alternative_action=(
                "Retain the model and characterize unresolved parameter "
                "information; check public mechanism obligations separately."
            ),
        )
    witness_routes = {
        "stationary_poor_fit": (
            "local_stationarity",
            "retained_point_local_search",
            "retained_point_stationary",
            (
                "The poor fit has retained-point local stationarity evidence; "
                "another basin of attraction remains possible."
            ),
            (
                "Compare a conditional function hypothesis with genuinely different"
                " fitting starts."
            ),
        ),
        "solved_restricted_problem_poor_fit": (
            "restricted_output_optimum",
            "fixed_candidate_output",
            "restricted_candidate_optimum",
            (
                "An optimum witness is bound to this retained candidate and "
                "restricted domain; its output fit is poor."
            ),
            (
                "Inspect that function family and its allowed dependencies without "
                "rejecting the public specification."
            ),
        ),
    }
    if facts.diagnosis in witness_routes:
        kind, scope, label, rationale, alternative = witness_routes[facts.diagnosis]
        if (
            facts.witness_bound_to_retained_vector
            and facts.witness_kind == kind
            and facts.witness_scope == scope
        ):
            return Readiness(
                preferred_stage="interaction",
                hypothesis_status="permitted",
                search_scope=label,
                rationale=rationale,
                alternative_action=alternative,
            )
        return Readiness(
            preferred_stage="fitting",
            hypothesis_status="withheld",
            search_scope="unverified",
            rationale=(
                "The stronger search assessment lacks its matching retained-vector "
                "witness."
            ),
            alternative_action=(
                "Verify the search evidence before making an equation-review "
                "recommendation."
            ),
        )
    if facts.diagnosis in {"search_improving", "search_incomplete"}:
        improving = bool(
            facts.diagnosis == "search_improving"
            and facts.witness_bound_to_retained_vector
            and facts.witness_kind == "improvement"
            and facts.witness_scope == "retained_point_local_search"
        )
        return Readiness(
            preferred_stage="fitting",
            hypothesis_status="conditional",
            search_scope="improving" if improving else "incomplete",
            rationale=(
                "Verified trajectory discrepancies are available, but optimization "
                "is still improving."
            )
            if improving
            else (
                "Verified trajectory discrepancies are available, but search "
                "completion is unestablished."
            ),
            alternative_action=(
                "Continue the incumbent within budget. If the controller "
                "independently routes a function-review visit, treat hypotheses as "
                "provisional."
            )
            if improving
            else (
                "Extend or diversify fitting. An independently allocated "
                "function-review visit may test a hypothesis without claiming "
                "structural failure."
            ),
        )
    return Readiness(
        preferred_stage="numerical_checks",
        hypothesis_status="withheld",
        search_scope="unverified",
        rationale=(
            "The diagnosis is unsupported or inconsistent with verified-output routing."
        ),
        alternative_action=(
            "Inspect diagnostic availability and provenance; no equation advice is "
            "authorized."
        ),
    )


# These are investigation templates, not diagnoses inferred from residuals.
_TEMPLATES = {
    "response_strength": (
        (
            "The existing response-strength dependence may be too strong, too weak,"
            " or have an inconsistent excursion direction in the cited response. "
            "Inspect admissible forms along the displayed pathway."
        ),
        (
            "Existing coefficients may still explain the discrepancy.",
            (
                "Initialization or coupled feedback can alter excursion without "
                "changing the suspected law."
            ),
        ),
        (
            (
                "Compare additional training responses under controlled input "
                "amplitudes, initials and covariates before proposing saturation."
            ),
            "Selected worst/best trajectories are not a controlled amplitude contrast.",
        ),
        (
            "Reduce the cited excursion discrepancy and output NMSE after "
            "refitting; preserve the correctly reproduced excursion direction and "
            "other targets."
        ),
    ),
    "timing_recovery": (
        (
            "The existing response dynamics may build up or relax at the wrong "
            "rate. Inspect their allowed nonlinear dependencies before asserting "
            "missing memory."
        ),
        (
            "Rate coefficients and permitted initial values may explain the timing.",
            (
                "Coupled dynamics, repeated forcing or multiple peaks can explain "
                "apparent recovery differences."
            ),
        ),
        (
            (
                "Check the full cited input history and comparable response "
                "segments before assigning a delay to any process."
            ),
            (
                "A new memory state needs evidence that existing state dynamics and"
                " initialization cannot explain the pattern."
            ),
        ),
        (
            "Reduce the cited peak-time or sampled half-return discrepancy after "
            "refitting, while preserving excursion, output NMSE and all other "
            "targets."
        ),
    ),
    "balance": (
        (
            "The existing source/sink response balance may leave a systematic "
            "prediction offset in the cited trajectory. Inspect allowed function "
            "contributions without treating a parameter sign as a measured "
            "mechanism."
        ),
        (
            "Fitting or permitted initialization may explain the bias.",
            (
                "Different functions can compensate for each other; an output bias "
                "does not locate a faulty term."
            ),
        ),
        (
            (
                "Separate early transient offset from bias persisting across input "
                "regimes."
            ),
            (
                "Any physical source/sink or conservation obligation must come from"
                " the actual public construction contract."
            ),
        ),
        (
            "Reduce the cited signed bias and NMSE without trading bias for larger "
            "transient error or harming another target."
        ),
    ),
    "time_window": (
        (
            "A cited time window has residual error above the presentation "
            "threshold. Inspect its current forcing and existing "
            "transient/relaxation dynamics."
        ),
        (
            "Early errors may arise from the permitted initialization rule.",
            (
                "Late errors may reflect ongoing forcing or rate coefficients "
                "rather than missing memory."
            ),
        ),
        (
            (
                "Inspect the cited interval and its input history; "
                "early/middle/late thirds are not event-aligned causal windows."
            ),
        ),
        (
            "Reduce the cited window NMSE and full-trajectory error after refitting"
            " without transferring the error into another interval or output."
        ),
    ),
}
_FAMILY = {
    "excursion_presence": "response_strength",
    "excursion_direction": "response_strength",
    "excursion_magnitude": "response_strength",
    "peak_timing": "timing_recovery",
    "half_return": "timing_recovery",
    "signed_bias": "balance",
    "window_error": "time_window",
}


def _investigations(
    packet: feedback.Packet,
    extraction: mismatches.MismatchSummary,
    policy: Policy,
    route: Readiness,
) -> tuple[tuple[Investigation, ...], dict[str, int], dict[str, str]]:
    results, omitted, withheld = [], {}, {}
    for target in packet.response.targets:
        groups = {}
        for finding in extraction.findings:
            if finding.target == target and finding.availability == "measured":
                groups.setdefault(_FAMILY[finding.code], []).append(finding)
        families = [family for family in _TEMPLATES if family in groups]
        laws = tuple(
            law
            for law in packet.laws
            if law.kind != "observation_mapping" and target in law.affected_targets
        )
        selected = (
            families[: policy.investigations_per_target]
            if route.hypothesis_status != "withheld" and laws
            else []
        )
        omitted[target] = len(families) - len(selected)
        if families and route.hypothesis_status == "withheld":
            withheld[target] = "Numerical/search readiness withholds function advice."
        elif families and not laws:
            withheld[target] = (
                "No displayed state equation or process law affects this target. "
                "Inspect the observation or construction contract before proposing "
                "a function edit; an observation mismatch alone does not authorize "
                "a new state or dependency."
            )
        for family in selected:
            hypothesis, alternatives, missing, prediction = _TEMPLATES[family]
            results.append(
                Investigation(
                    target=target,
                    family=family,
                    evidence_ids=tuple(
                        sorted(
                            {
                                ref
                                for fact in groups[family]
                                for ref in fact.evidence_ids
                            }
                        )
                    ),
                    candidate_law_refs=tuple(law.location_id for law in laws),
                    affected_targets=tuple(
                        sorted({t for law in laws for t in law.affected_targets})
                    ),
                    conditional_hypothesis=hypothesis,
                    alternatives=alternatives,
                    additional_evidence_needed=missing,
                    expected_observable_change=prediction,
                    disconfirmation=(
                        "If the predicted response change does not occur under a "
                        "verified refit, the hypothesis is unsupported. A lower "
                        "total error alone does not confirm its mechanism."
                    ),
                    broader_stage_condition=(
                        "New states or initialization rules require variable-stage "
                        "review; new sources, consumers or shared-process roles "
                        "require topology review. No such change is inferred from "
                        "this discrepancy alone."
                    ),
                )
            )
    return tuple(results), omitted, withheld


def draft(
    packet: feedback.Packet, policy: Policy | None = None, *, source: str
) -> Advice:
    """Pure projection; caller must verify the packet with M37 completed()."""
    policy = policy or Policy()
    route = readiness(packet)
    extracted = mismatches.extract(packet, policy.extraction)
    investigations, omitted, withheld = _investigations(
        packet, extracted, policy, route
    )
    return Advice(
        identity=Identity(
            source_packet_sha256=public.content_sha256(packet.model_dump(mode="json")),
            policy_sha256=public.content_sha256(policy.model_dump(mode="json")),
            draft_source_sha256=source,
        ),
        policy=policy,
        readiness=route,
        mismatches=extracted,
        parameter_information=packet.feedback.parameter_information,
        investigations=investigations,
        omitted_investigations=omitted,
        withheld_investigation_reasons=withheld,
        preservation_requirements=(
            (
                "Keep every observed target and inspect per-target changes, not "
                "aggregate error alone."
            ),
            (
                "Preserve the public specification and causal "
                "observation/initialization contract."
            ),
            (
                "Bind the constructor's actual allowed sources, signs and "
                "obligations before applying an edit."
            ),
            "A shared process law changes all its consumers consistently.",
            (
                "Preserve the parent on invalid edits, failed fitting or unverified"
                " improvement."
            ),
            (
                "Use matched, predetermined fitting budgets and a continuation "
                "control to assess a functional revision."
            ),
            (
                "Retain all attempt outcomes; keep held-out evaluation separate "
                "from this training-only request."
            ),
        ),
        interpretation=(
            (
                "Presentation thresholds are heuristic effect-size filters, not "
                "statistical significance or new fitting tolerances."
            ),
            (
                "Observed discrepancies and conditional explanations are separate; "
                "candidate-law references are navigation, not fault localization."
            ),
            (
                "Accurate output, local rank and a restricted optimum do not "
                "certify scientific correctness or global identifiability."
            ),
            (
                "No automatic inference of saturation, missing input, missing "
                "memory or physical conservation is performed."
            ),
            (
                "The selected trajectory examples are not a controlled comparison. "
                "No private reference formula, sensor correction or loss change is "
                "supplied."
            ),
        ),
    )


SYSTEM_PROMPT = (
    feedback.SYSTEM_PROMPT
    + """

The draft separates measured discrepancies from conditional investigations.
Read readiness first: it recommends effort allocation but does not certify a
globally solved fit or trigger a stage transition. If the controller allocated
function review despite incomplete fitting, test a provisional hypothesis rather
than treating budget exhaustion as proof or as the sole reason for no_change.
When readiness withholds hypotheses, return no_change with the missing evidence;
do not turn the numerical prerequisite into a functional or structural diagnosis.
Retain accurate models unless separately established task obligations require work.
State competing explanations and what result would fail to support the proposed
explanation. Do not infer saturation, a missing input or missing memory from the
selected extreme trajectories. References to complete laws are not edit contracts.
Return ProposedInvestigation; reference spelling is not scientific verification.
"""
)


def messages(packet: feedback.Packet, advice: Advice) -> list[dict[str, str]]:
    """Retain the complete all-target source alongside the bounded draft."""
    if advice.identity.source_packet_sha256 != public.content_sha256(
        packet.model_dump(mode="json")
    ):
        raise ValueError("draft source packet differs")
    content = json.dumps(
        {
            "source": packet.model_dump(mode="json"),
            "draft": advice.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(content.encode("utf-8")) > advice.policy.maximum_user_bytes:
        raise ValueError("interaction advice exceeds prompt byte cap; no call issued")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def render(advice: Advice) -> str:
    """Readable view of the exact sealed facts, without adding causal conclusions."""
    route = advice.readiness
    lines = [
        "# Draft numerical feedback",
        "",
        advice.observation_interpretation,
        "",
        f"Preferred stage: **{route.preferred_stage}**; "
        f"hypotheses: **{route.hypothesis_status}**.",
        route.rationale,
        route.alternative_action,
        "",
        f"Parameter information: {advice.parameter_information}.",
        "",
        "## Measured discrepancies and unavailable comparisons",
        "",
    ]
    for finding in advice.mismatches.findings:
        lines.append(
            f"- {json.dumps(finding.target)} / {finding.code} "
            f"({finding.availability}): {finding.interpretation} "
            f"References: {', '.join(finding.evidence_ids)}."
        )
        lines.append(
            f"  Observed={finding.observed}; predicted={finding.predicted}; "
            f"difference={finding.difference}; unit={finding.unit}; "
            f"threshold={finding.threshold}."
        )
        for caveat in finding.caveats:
            lines.append(f"  {caveat}")
    if not advice.mismatches.findings:
        lines.append(
            "No discrepancy passed the presentation filters, or output evidence"
            " is unavailable; consult readiness and the complete target "
            "overview."
        )
    lines.extend(["", "## Conditional investigations", ""])
    for item in advice.investigations:
        law_refs = ", ".join(item.candidate_law_refs)
        lines.extend(
            [
                f"- Target {json.dumps(item.target)}; {item.family}. "
                f"{item.conditional_hypothesis}",
                f"  Law references: {law_refs}.",
                f"  Evidence: {', '.join(item.evidence_ids)}.",
                "  Alternatives: " + " ".join(item.alternatives),
                "  Additional evidence: " + " ".join(item.additional_evidence_needed),
                "  Predicted improvement: " + item.expected_observable_change,
                "  Disconfirmation: " + item.disconfirmation,
                "  Stage boundary: " + item.broader_stage_condition,
            ]
        )
    if not advice.investigations:
        lines.append(
            "No equation investigation is drafted for this numerical state and "
            "evidence."
        )
    for target, reason in advice.withheld_investigation_reasons.items():
        lines.append(f"- Target {json.dumps(target)}: {reason}")
    lines.extend(["", "## Preserve and verify", ""])
    lines.extend(f"- {item}" for item in advice.preservation_requirements)
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in advice.interpretation)
    lines.extend(
        [
            "",
            (
                "Full target coverage, omission counts, exact thresholds and source"
                " identities are in result.json; the complete model and source "
                "packet are in request.json."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _publish_text(path: Path, content: str) -> None:
    """Create a deterministic text view atomically; never overwrite a mismatch."""
    if path.exists():
        if path.read_text() != content:
            raise ValueError("saved feedback text differs")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_advice(
    diagnostic_folder: Path,
    packet_folder: Path,
    output_folder: Path,
    policy: Policy | None = None,
) -> Advice:
    """Verify M37, then export sealed advice/request and a readable feedback draft."""
    packet = feedback.completed(diagnostic_folder, packet_folder)
    advice = draft(packet, policy, source=public._source_identity())
    request = {
        "messages": messages(packet, advice),
        "response_schema": ProposedInvestigation.model_json_schema(),
    }
    with public._lock(output_folder):
        seal(output_folder / "result.json", advice.model_dump(mode="json"))
        seal(output_folder / "request.json", request)
        _publish_text(output_folder / "feedback.md", render(advice))
    return advice


def completed(
    diagnostic_folder: Path, packet_folder: Path, output_folder: Path
) -> Advice:
    """Rebuild against original verified evidence; no numerical or provider work."""
    saved = Advice.model_validate(read_seal(output_folder / "result.json"))
    packet = feedback.completed(diagnostic_folder, packet_folder)
    expected = draft(packet, saved.policy, source=saved.identity.draft_source_sha256)
    if saved != expected:
        raise ValueError("interaction advice differs from verified source")
    request = {
        "messages": messages(packet, saved),
        "response_schema": ProposedInvestigation.model_json_schema(),
    }
    if read_seal(output_folder / "request.json") != request:
        raise ValueError("interaction advice request differs")
    if (output_folder / "feedback.md").read_text() != render(saved):
        raise ValueError("interaction advice text differs")
    return saved
