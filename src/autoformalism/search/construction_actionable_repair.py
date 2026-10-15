"""Isolated saved-draft repair prompts with exact contribution handles.

The stored scientific draft and structural validators are unchanged. Diagnosed
conflicts are repair obligations; prose dispositions do not discharge them. The
caller owns revalidation, uncertainty qualification and bounded completion.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, create_model, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.search import construction_evidence as core
from autoformalism.search import construction_evidence_repair as previous
from autoformalism.search import construction_realizations as declarations
from autoformalism.staged_topology import content_hash

POLICY = "saved-actionable-topology-repair-1"
SYSTEM = previous.SYSTEM
Draft = core.Draft
BoundaryContract = core.BoundaryContract
assess = core.assess
project = core.project
model_view = core.model_view
edit_effects = core.edit_effects
normalize_consumers_legacy = core.normalize_consumers
lost_required_predicates = previous.lost_required_predicates
REPAIR_REQUESTS = previous.REPAIR_REQUESTS
REPAIR_TOKENS = previous.REPAIR_TOKENS

EvidenceText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]


class CompatibilityDecision(StrictSchema):
    """A planned correction or evidenced challenge; neither closes a conflict."""

    question_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
    ]
    disposition: Literal["revise", "dispute_assumption"]
    explanation: EvidenceText
    supporting_evidence: tuple[EvidenceText, ...] = Field(default=(), max_length=6)

    @model_validator(mode="after")
    def challenge_needs_evidence(self):
        """An unsupported assertion is not an evidence-backed assumption challenge."""
        if self.disposition == "dispute_assumption" and not self.supporting_evidence:
            raise ValueError("dispute_assumption requires supporting_evidence")
        return self


class ExternalExchangeSelection(StrictSchema):
    """Select one exact displayed contribution without choosing its storage field."""

    obligation_id: Identifier
    contribution_handle: str = Field(pattern=r"^contribution-[0-9a-f]{64}$")
    explanation: EvidenceText


class DraftPatch(core.DraftPatch):
    """Canonical internal patch: legacy selectors plus unreleased challenge evidence."""

    compatibility_decisions: tuple[CompatibilityDecision, ...] = Field(
        default=(), max_length=16
    )


def response_model(stage: str):
    """Use handle-only selectors in repair; preserve all historical initial schemas."""
    if stage != "repair":
        return previous.response_model(stage)
    fields = set(core.response_model(stage).model_fields)
    definitions = {
        name: (field.annotation, deepcopy(field))
        for name, field in DraftPatch.model_fields.items()
        if name in fields
    }
    definitions["external_exchanges"] = (
        tuple[ExternalExchangeSelection, ...],
        Field(default=()),
    )
    return create_model("ActionableRepairReply", __base__=StrictSchema, **definitions)


def _legacy_patch(patch: DraftPatch) -> core.DraftPatch:
    value = patch.model_dump(mode="json")
    for decision in value["compatibility_decisions"]:
        decision.pop("supporting_evidence")
    return core.DraftPatch.model_validate(value)


def apply_patch(brief, draft: Draft, patch: DraftPatch) -> Draft:
    """Apply only explicit canonical edits, never an assumption or role inference."""
    return core.apply_patch(brief, draft, _legacy_patch(patch))


def normalize_consumers(draft: Draft, patch: DraftPatch, previous_draft: Draft):
    """Reuse the exact historical duplicate-use normalization after a typed edit."""
    return core.normalize_consumers(draft, _legacy_patch(patch), previous_draft)


def external_handle_options(brief, draft: Draft, boundary_contract=None) -> list[dict]:
    """Expose content-bound handles and their transparent existing-field mapping."""
    rows = previous.external_reference_choices(brief, draft, boundary_contract)
    for row in rows:
        options = []
        for option in row.pop("reference_choices"):
            selector = option["selector"]
            choice = core.ExternalExchangeChoice(
                **selector, explanation="Reference identity only."
            )
            reference = core._reference(draft, choice)
            handle = "contribution-" + content_hash(
                {"selector": selector, "reference": reference}
            )
            options.append(
                {
                    "contribution_handle": handle,
                    "receiver": selector["receiver"],
                    "declared_meaning": option["declared_meaning"],
                    "arguments": option["arguments"],
                    "stored_reference": selector,
                }
            )
        row["contributions"] = options
    return rows


def normalize_reply(
    brief, raw, *, draft: Draft, boundary_contract=None, **kwargs
) -> tuple[dict, list[dict]]:
    """Translate a current displayed handle to its exact canonical selector.

    Parse the returned dictionary using DraftPatch, not response_model. Handles
    address unchanged contributions in this request's predecessor. Introducing or
    changing an effect requires selecting its newly displayed handle next turn.
    """
    normalized, changes = core.normalize_reply(brief, raw, draft=draft, **kwargs)
    reply = response_model("repair").model_validate(normalized)
    options = {
        (row["obligation_id"], item["contribution_handle"]): item
        for row in external_handle_options(brief, draft, boundary_contract)
        for item in row["contributions"]
    }
    value = reply.model_dump(mode="json")
    canonical = []
    selected = []
    for selection in reply.external_exchanges:
        option = options.get((selection.obligation_id, selection.contribution_handle))
        if option is None:
            raise ValueError(
                "external contribution handle is absent or stale; choose a "
                "handle displayed for this obligation in the current request. "
                "A newly added effect receives a handle in the next request."
            )
        choice = core.ExternalExchangeChoice(
            **option["stored_reference"], explanation=selection.explanation
        )
        canonical.append(choice.model_dump(mode="json"))
        selected.append((selection, choice))
    value["external_exchanges"] = canonical
    patch = DraftPatch.model_validate(value)
    if selected:
        # Verify indices do not move and meanings are not silently retargeted by
        # another edit in the same response, including duplicate-use normalization.
        proposed = core.apply_patch(brief, draft, _legacy_patch(patch))
        proposed, _ = core.normalize_consumers(proposed, _legacy_patch(patch), draft)
        for selection, choice in selected:
            if core._reference(draft, choice) != core._reference(proposed, choice):
                raise ValueError(
                    "the selected contribution changed in this reply; commit "
                    "the intended model edit without this external assignment, "
                    "then select its new displayed handle in the next request"
                )
            changes.append(
                {
                    "code": "external_contribution_handle_resolved",
                    "contribution_handle": selection.contribution_handle,
                    "stored_reference": choice.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "scientific_role_inferred": False,
                }
            )
    return value, changes


def stage_text(stage: str) -> str:
    """Request model repair without a prose acceptance shortcut for conflicts."""
    if stage != "repair":
        return previous.stage_text(stage)
    return declarations.REPAIR + (
        "\nTreat actionable topology/data conflicts as concrete repair work. "
        "Use the displayed observations, matched conditioning and assumptions "
        "to decide the scientifically missing distinction; no particular new "
        "edge is prescribed. Return the actual model edits. A statement that "
        "you will revise is not an edit.\n"
        "An uncertain comparison is not a proven error. If an assumption is "
        "inapplicable, identify it and provide supporting_evidence in a "
        "dispute_assumption decision. The challenge remains unresolved until "
        "runtime revalidation establishes what can be concluded; neither a "
        "challenge nor accepting an approximation clears a conflict.\n"
        "Record corrections do not require deleting their model components. "
        "For external effects, copy the exact contribution_handle for the "
        "scientifically intended effect. A new or changed contribution receives "
        "its own handle in the next request. Set stage_complete=true when your "
        "edits are ready for rechecking; it does not override an outstanding "
        "conflict or unresolved assessment."
    )


def _question_view(question: dict) -> dict:
    """Keep witnesses and assumptions; remove the prior approximation-choice text."""
    row = deepcopy(question)
    status = row.get("repair_status", "uncertain_conflict")
    status = {
        "repair_required": "actionable_conflict",
        "uncertainty_limited": "uncertain_conflict",
    }.get(status, status)
    row["repair_status"] = status
    row["question"] = {
        "closed_subsystem": "The displayed subsystem has matched permitted initial "
        "information and complete forcing, yet its target observations differ. "
        "Identify and implement the scientifically justified missing distinction "
        "or give evidence that a displayed assumption is inapplicable.",
        "algebraic_map": "The generated algebraic quantity receives the same "
        "listed arguments, yet its target observations differ. Identify and "
        "implement the scientifically justified missing dependence or give "
        "evidence that a displayed assumption is inapplicable.",
        "scalar_autonomous": "The scalar autonomous subsystem has a recorded "
        "reversal under constant conditioning. Check the displayed uniqueness "
        "and observation assumptions, then implement the scientifically needed "
        "model distinction or provide a supported assumption challenge.",
    }.get(
        row.get("property"),
        "Address the displayed diagnostic using its "
        "witnesses and assumptions; the runtime does not choose a scientific edit.",
    )
    row["interpretation"] = (
        "Actionable conditional conflict under the displayed uncertainty bounds "
        "and assumptions. Repair and revalidate; this does not prescribe an edge "
        "or claim universal scientific correctness."
        if status == "actionable_conflict"
        else "Unresolved assumption challenge; the reported reasoning does not "
        "clear the conflict without runtime revalidation."
        if status == "assumption_challenged"
        else "Uncertain conditional evidence, not an established model error. "
        "Do not force an edit from this comparison alone. Inspect the stated "
        "bounds and assumptions; an unresolved assessment is not a scientific pass."
    )
    return row


def payload(
    stage,
    brief,
    context,
    target_definitions,
    enriched_brief,
    draft: Draft,
    contract,
    last=None,
    continuation=None,
    *,
    boundary_contract=None,
    training_highlights=None,
    compatibility_questions=(),
    compatibility_state=None,
    repetition_feedback=None,
):
    """Render concrete repair work with qualified evidence and exact edit handles."""
    packet = previous.payload(
        stage,
        brief,
        context,
        target_definitions,
        enriched_brief,
        draft,
        contract,
        last,
        continuation,
        boundary_contract=boundary_contract,
        training_highlights=training_highlights,
        compatibility_questions=compatibility_questions,
        compatibility_state=compatibility_state,
    )
    if stage != "repair":
        return packet
    packet["stage_instructions"] = stage_text(stage)
    packet["response_template"] = response_model(stage)(
        stage_complete=False
    ).model_dump(mode="json")
    packet["editing_rules"] = declarations.EDITING + (
        "\nexternal_exchanges selections contain obligation_id, contribution_handle "
        "and explanation. Handles map transparently to stored_reference fields "
        "shown for inspection; return the handle, not those internal fields. "
        "compatibility_decisions contains question_id, disposition (revise or "
        "dispute_assumption), explanation and supporting_evidence. A challenge "
        "requires concrete evidence; all decision text remains unverified."
    )
    questions = [_question_view(row) for row in compatibility_questions]
    if questions:
        packet["compatibility_questions"] = questions
    feedback = packet["feedback"]
    feedback.pop("advisory_reply_choices", None)
    feedback.pop("completion_example", None)
    work = feedback["repair_work"]
    work.pop("pending_advisory_questions", None)
    work.pop("pending_questions_previously_answered_revise", None)
    work.pop("advisory_disposition_scope", None)
    work["actionable_conflicts"] = [
        row["question_id"]
        for row in questions
        if row["repair_status"] == "actionable_conflict"
    ]
    work["uncertain_or_challenged_assessments"] = [
        {"question_id": row["question_id"], "status": row["repair_status"]}
        for row in questions
        if row["repair_status"] != "actionable_conflict"
    ]
    for decision in work.pop("accepted_advisory_dispositions", []):
        decision["question_status"] = "unresolved_pending_revalidation"
        work.setdefault("previous_unverified_responses", []).append(decision)
    work["mode"] = (
        "repair_required"
        if work["actionable_conflicts"] or not work["structural_eligibility"]
        else "assessment_unresolved"
        if work["uncertain_or_challenged_assessments"]
        else "recheck_completion"
    )
    state = compatibility_state or {}
    unresolved = [
        {
            key: deepcopy(row[key])
            for key in (
                "target",
                "property",
                "previous_question_id",
                "status",
                "resolved",
                "reason",
                "current_status",
            )
            if key in row
        }
        for row in state.get("baseline_outcomes", [])
        if not row.get("resolved")
    ]
    work["unresolved_previous_diagnostics"] = unresolved
    if "completion_allowed" in state:
        work["runtime_revalidation"] = {
            "repair_status": state.get("repair_status"),
            "completion_allowed": state["completion_allowed"],
        }
        if (
            not state["completion_allowed"]
            and not questions
            and work["structural_eligibility"]
        ):
            work["mode"] = "assessment_unresolved"
    work["instruction"] = (
        "Make explicit edits for concrete model errors and actionable conflicts. "
        "Correct record-only issues without deleting intended model effects. "
        "Keep uncertain evidence and challenges unresolved until revalidated. "
        "Neither stage_complete=true nor an explanatory response overrides these "
        "obligations. Retaining a known incompatible approximation is not a "
        "completion option in this experiment."
    )
    if feedback.get("continuation"):
        feedback["continuation"]["instruction"] = work["instruction"]
    if repetition_feedback:
        feedback["repeated_repair_feedback"] = {
            # The paired historical arm carries its own continuation prose.
            # Preserve factual receipts/counts, not that arm's stopping advice.
            "observations": {
                key: deepcopy(value)
                for key, value in repetition_feedback.items()
                if key not in {"continuation", "instruction"}
            },
            "instruction": "The displayed errors recur after the prior reply. "
            "Compare the actual retained model and structured edit, then address "
            "the same concrete failing condition. Repeating a description or a "
            "revise decision does not change dependencies or contribution "
            "references. Preserve other intended mechanisms; no automatic "
            "deletion or scientific replacement is inferred.",
        }
    if boundary_contract and boundary_contract.obligations:
        packet.pop("external_reference_choices", None)
        packet["external_effect_options"] = external_handle_options(
            brief, draft, boundary_contract
        )
        packet["external_effect_rules"] = (
            "Choose the intended existing external effect using only its displayed "
            "contribution_handle, obligation_id and physical explanation. A "
            "handle refers to one exact contribution in this retained model; "
            "the runtime maps it to stored_reference without selecting its "
            "scientific role. Inserted process uses are separate from ordinary "
            "equation terms, regardless of display order or number of receivers. "
            "For a missing or changed effect, first return the model edit, then "
            "select the new handle on the next turn."
        )
        equations = {e.name: e for e in draft.equations}
        for row in [*feedback["issues"], *packet["requirements_checklist"]]:
            if row.get("code") != "external_exchange_coverage" and not str(
                row.get("id", "")
            ).startswith("external_exchange:"):
                continue
            row["action"] = packet["external_effect_rules"]
            assignment = row.get("assignment") or {}
            receiver = assignment.get("receiver")
            if receiver:
                count = len(equations[receiver].terms) if receiver in equations else 0
                row["reference_namespace_facts"] = {
                    "receiver": receiver,
                    "ordinary_term_indices": list(range(count)),
                    "separate_process_uses": [
                        p.name
                        for p in draft.processes
                        if any(u.target == receiver for u in p.uses)
                    ],
                    "instruction": "These are stored-reference facts, not the "
                    "displayed equation's term positions. Select a current "
                    "contribution_handle; the runtime performs the exact mapping.",
                }
    return packet
