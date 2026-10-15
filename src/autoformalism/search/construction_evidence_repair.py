"""Opt-in repair presentation with unchanged scientific checks and initial prompts.

The v1 draft/schema and validators remain authoritative. This facade changes only
repair instructions and bookkeeping; it neither chooses scientific edits nor
equates a structural pass with scientific adequacy.
"""

from __future__ import annotations

from copy import deepcopy

from autoformalism.search import construction_evidence as v1

POLICY = "minimal-evidence-feedback-2"
POLICIES = (v1.POLICY, POLICY)
SYSTEM = v1.SYSTEM
REPAIR_REQUESTS = v1.REPAIR_REQUESTS
REPAIR_TOKENS = v1.REPAIR_TOKENS
REPEATED_REPAIRS = v1.REPEATED_REPAIRS
repair_budget = v1.repair_budget
repair_signature = v1.repair_signature

# Deliberately retain the actual v1 schemas, including their names. Changing a
# schema title would also change first-stage requests in this repair-only study.
BoundaryContract = v1.BoundaryContract
Draft = v1.Draft
DraftPatch = v1.DraftPatch
apply_patch = v1.apply_patch
assess = v1.assess
base = v1.base
edit_effects = v1.edit_effects
final_handoff = v1.final_handoff
from_legacy = v1.from_legacy
interaction_handoff = v1.interaction_handoff
load_draft = v1.load_draft
model_view = v1.model_view
normalize_consumers = v1.normalize_consumers
normalize_reply = v1.normalize_reply
pending = v1.pending
project = v1.project
response_model = v1.response_model
snapshot = v1.snapshot
variable_checklist = v1.variable_checklist
variable_snapshot = v1.variable_snapshot

_REFERENCE_ONLY = frozenset(
    {
        "unknown_public_boundary_obligation",
        "missing_external_exchange_assignment",
        "contribution_reference_unresolved",
        "contribution_reference_changed",
    }
)


def stage_text(stage: str) -> str:
    """Keep initial construction instructions exact; clarify repair controls only."""
    text = v1.stage_text(stage)
    if stage == "repair":
        text += (
            "\nUse repair_work to distinguish model/structural work, record "
            "corrections and advisory questions. A record correction does not "
            "require deleting its referenced model component. Correct the record "
            "alone when the intended effect already exists; otherwise make the "
            "scientifically needed model edits. Structural readiness is not "
            "scientific approval: revise a structurally accepted but inadequate "
            "model when the evidence calls for it.\n"
            "Answer pending questions and set stage_complete=true when your "
            "chosen edits and answers are ready for checking, including a reply "
            "with only decisions or completion. Use stage_complete=false only "
            "when you intentionally request further work. A revise decision alone "
            "does not change topology or resolve an unchanged compatibility "
            "question. Retaining an approximation or disputing an assumption "
            "requires your scientific explanation and does not certify accuracy."
        )
    return text


def external_reference_choices(
    brief, draft: Draft, contract: BoundaryContract | None
) -> list[dict]:
    """Enumerate structurally eligible exact selectors, without choosing a loss.

    Every option is checked by the existing validator in a disposable draft.
    Processes with a single use still require the schema's shared_process key;
    they are never counted as an ordinary local_term_index.
    """
    if contract is None:
        return []
    contract.validate_public(brief)
    equations = {equation.name: equation for equation in draft.equations}
    rows = []
    for obligation in contract.obligations:
        choices = []
        for variable in draft.variables:
            if variable.definition != "differential":
                continue
            candidates = [
                (
                    {"local_term_index": index},
                    term.scientific_role,
                    list(term.sources),
                )
                for index, term in enumerate(
                    equations[variable.name].terms if variable.name in equations else ()
                )
            ]
            candidates.extend(
                (
                    {"shared_process": process.name},
                    process.scientific_meaning,
                    list(process.depends_on),
                )
                for process in draft.processes
                if any(use.target == variable.name for use in process.uses)
            )
            for selector, meaning, arguments in candidates:
                choice = v1.ExternalExchangeChoice(
                    obligation_id=obligation.id,
                    receiver=variable.name,
                    explanation="Syntax candidate; scientific role not inferred.",
                    **selector,
                )
                bound = v1._bind(draft, choice)
                trial = draft.model_copy(update={"external_exchanges": (bound,)})
                check = next(
                    row
                    for row in v1.boundary_checks(brief, trial, contract)
                    if row["id"] == obligation.id
                )
                if check["status"] == "reference_covered":
                    choices.append(
                        {
                            "selector": {
                                "obligation_id": obligation.id,
                                "receiver": variable.name,
                                **selector,
                            },
                            "declared_meaning": meaning,
                            "arguments": arguments,
                        }
                    )
        rows.append(
            {
                "obligation_id": obligation.id,
                "public_target": obligation.target,
                "reference_choices": choices,
                "scope": "Valid reference syntax and balance/readout structure "
                "only. Choose a listed contribution only if it is the intended "
                "external loss. If none is appropriate, propose the missing "
                "effect; do not relabel an internal transfer.",
            }
        )
    return rows


def _predicate_rows(check: dict) -> dict[tuple, dict]:
    """Index explicit required predicates independently of their changing witnesses."""
    result = {}
    for group in ("public_structure_checks", "reviewed_public_graph_checks"):
        for row in check.get(group, []):
            identity = {
                key: row[key]
                for key in (
                    "id",
                    "requirement",
                    "kind",
                    "source",
                    "driver",
                    "memory",
                    "target",
                )
                if key in row
            }
            key = (group, *sorted(identity.items()))
            result[key] = {
                "predicate": group,
                **identity,
                "status": "passed"
                if row.get("passed") is True
                else "failed"
                if row.get("passed") is False
                else "unavailable",
            }
    for row in check.get("memory_binding_checks", []):
        if row.get("mandatory"):
            result[("required_memory", row["requirement"])] = {
                "predicate": "required_memory",
                "requirement": row["requirement"],
                "status": row["status"],
            }
    for row in check.get("mechanism_realization_checks", []):
        for field in ("declaration_status", "structural_status"):
            status = row.get(field, "unavailable")
            result[(field, row["requirement"])] = {
                "predicate": f"mechanism_realization:{field}",
                "requirement": row["requirement"],
                "status": "passed" if status == "satisfied" else status,
            }
    for row in check.get("boundary_checks", []):
        result[("boundary_reference", row["id"])] = {
            "predicate": "boundary_reference",
            "id": row["id"],
            "target": row.get("target"),
            "status": "passed"
            if row["status"] == "reference_covered"
            else row["status"],
        }
    return result


def lost_required_predicates(before: dict, after: dict) -> list[dict]:
    """Report lost checks after an edit, never a scientific regression verdict.

    An unavailable graph stays unavailable; it is not called a failed pathway.
    Witness changes with the same passed verdict are not reported as losses.
    """
    previous, current = _predicate_rows(before), _predicate_rows(after)
    rows = []
    for key, old in previous.items():
        if old["status"] != "passed":
            continue
        new = current.get(key)
        if new is not None and new["status"] == "passed":
            continue
        rows.append(
            {
                **{k: v for k, v in old.items() if k != "status"},
                "before_status": "passed",
                "after_status": new["status"] if new else "not_reported",
            }
        )
    return rows


def _record_issue(issue: dict) -> bool:
    """Identify record work without hiding an accompanying missing-path verdict."""
    return (
        issue["code"] == "mechanism_realization_source_only"
        or (
            issue["code"] == "mechanism_realization_declaration"
            and "unknown_requirement" in issue.get("issues", ())
        )
        or (
            issue["code"] == "external_exchange_coverage"
            and set(issue.get("issues", ())) <= _REFERENCE_ONLY
        )
    )


def _issue_summary(issue: dict, feedback: dict) -> dict:
    return {
        key: deepcopy(issue[key])
        for key in (
            "code",
            "requirement",
            "id",
            "kind",
            "driver",
            "memory",
            "memories",
            "target",
            "public_path_status",
        )
        if key in issue
    } | {
        "feedback_issue_indices": [
            index
            for index, row in enumerate(feedback["issues"])
            if row["code"] == issue["code"]
            and all(
                row.get(key) == issue.get(key)
                for key in ("requirement", "id", "kind", "driver", "memory", "target")
                if key in issue
            )
        ]
    }


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
):
    """Separate repair obligations while preserving every non-repair v1 payload."""
    packet = v1.payload(
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
    )
    if stage != "repair":
        return packet
    packet["stage_instructions"] = stage_text(stage)
    check = assess(
        brief,
        context,
        target_definitions,
        draft,
        graph_contract=contract,
        boundary_contract=boundary_contract,
    )
    feedback = packet["feedback"]
    for issue in feedback["issues"]:
        if issue["code"].startswith("mechanism_realization") and (
            "unknown_requirement" in issue.get("issues", ())
        ):
            name = issue.get("requirement", issue.get("name"))
            issue["action"] = (
                f"{name!r} is not a public requirement ID. Remove only this "
                "realization record with the displayed remove_realizations edit, "
                "or replace it with an explicitly chosen valid public record. "
                "The error does not request removing its variables, processes "
                "or equations. Preserve their intended effects; model edits are "
                "needed only if you separately judge those effects incorrect. "
                "All actual public requirements remain required."
            )
            issue["record_only_removal"] = {"remove_realizations": [name]}
            issue["allowed_requirement_ids"] = [r.id for r in brief.requirements]
    questions = list(compatibility_questions)
    question_ids = {row["question_id"] for row in questions}
    decisions = (compatibility_state or {}).get("decisions", [])
    current_questions = {
        row["question_id"]: row
        for row in (compatibility_state or {}).get("advisory_questions", questions)
    }
    work = {
        "structural_eligibility": bool(check["eligible"]),
        "model_or_structural_corrections": [
            _issue_summary(row, feedback)
            for row in check["errors"]
            if not _record_issue(row)
        ],
        "record_corrections": [
            _issue_summary(row, feedback)
            for row in check["errors"]
            if _record_issue(row)
        ],
        "pending_advisory_questions": [row["question_id"] for row in questions],
        "pending_questions_previously_answered_revise": [
            row["question_id"]
            for row in decisions
            if row["question_id"] in question_ids and row["disposition"] == "revise"
        ],
        "accepted_advisory_dispositions": [
            {
                "question_id": row["question_id"],
                "target": current_questions.get(row["question_id"], {}).get("target"),
                "property": current_questions.get(row["question_id"], {}).get(
                    "property"
                ),
                "disposition": row["disposition"],
                "explanation": row["explanation"],
                "question_status": "pending"
                if row["question_id"] in question_ids
                else "answered",
                "scientific_status": "unverified",
            }
            for row in decisions
        ],
        "advisory_disposition_scope": "These are accepted answer records, not "
        "evidence that conditional conflicts were resolved or assumptions "
        "established. A retained approximation still has the acknowledged "
        "model-class limitation; an answered question is not scientific approval.",
        "mode": "model_or_record_repair"
        if not check["eligible"]
        else "advisory_review"
        if questions
        else "completion_confirmation",
        "previous_accepted_stage_complete": (last or {}).get("stage_complete")
        if (last or {}).get("transaction_applied")
        else None,
        "instruction": "Correct the listed checks and answer the conditional "
        "questions. Record edits alone leave equations unchanged. Use scientific "
        "judgment if changing dependencies is needed, even with structural "
        "eligibility. When your chosen work is ready, set stage_complete=true; "
        "the runtime still checks the result. Explicit false requests another "
        "turn and is never silently overridden.",
        "scientific_adequacy": "unverified",
    }
    if not check["eligible"] and not work["model_or_structural_corrections"]:
        work["instruction"] += (
            " The currently failed checks concern records. If the intended "
            "implementations already exist, repair only those records; no "
            "equation deletion is requested. Missing scientific effects may "
            "still require model edits."
        )
    if work["mode"] == "completion_confirmation":
        work["instruction"] += (
            " No structural error or unanswered compatibility question is "
            "currently listed. You may finish with empty edit lists and "
            "stage_complete=true, or explicitly revise a scientific inadequacy."
        )
    feedback["repair_work"] = work
    feedback.pop("repair_agenda", None)
    # Historical stagnation guidance may demand another topology edit after all
    # requested work is resolved. Preserve its facts while removing that demand.
    if feedback.get("continuation"):
        feedback["continuation"] = {
            **feedback["continuation"],
            "instruction": work["instruction"],
        }
    if boundary_contract and boundary_contract.obligations:
        packet["external_reference_choices"] = external_reference_choices(
            brief, draft, boundary_contract
        )
        packet["external_effect_rules"] += (
            " A process in the processes list always uses shared_process:name "
            "as its selector, even when the model view calls it a one-receiver "
            "local process. local_term_index counts only entries of that "
            "receiver's equations.terms, never inserted process uses. Copy the "
            "appropriate selector from external_reference_choices and add your "
            "physical explanation; the choices do not identify which effect "
            "scientifically represents the loss."
        )
    if last and last.get("transaction_applied"):
        lost = last.get("required_predicates_lost", [])
        if lost:
            feedback["previously_passed_checks_changed"] = {
                "checks": deepcopy(lost),
                "action": "These previously passed checks no longer pass after "
                "the last accepted edit. This is a bookkeeping/structural fact, "
                "not evidence that the previous model was scientifically correct. "
                "Implement the required predicates in the intended corrected "
                "model; do not restore old equations automatically. Unavailable "
                "checks need reassessment after representation errors are fixed.",
                "automatic_rollback": False,
                "additional_acceptance_gate": False,
            }
    if questions:
        feedback["advisory_reply_choices"] = {
            "question_ids": [row["question_id"] for row in questions],
            "choices": [
                {
                    "disposition": "revise",
                    "action": "Explain the scientific revision and include its "
                    "actual variables/equations/processes/record edits. A decision "
                    "alone leaves the same model and the question pending.",
                },
                {
                    "disposition": "retain_approximation",
                    "action": "Only if intentionally accepting this model-class "
                    "limitation, explain the scientific reason. This acknowledges "
                    "the conditional conflict; it does not fix or approve the model.",
                },
                {
                    "disposition": "dispute_assumption",
                    "action": "Identify the unsupported diagnostic assumption "
                    "and explain why it does not apply to the intended model.",
                },
            ],
            "completion_control": {"stage_complete": True},
            "instruction": "Choose an explained disposition for every pending "
            "question. After your chosen edits and answers are ready for checking, "
            "set stage_complete=true, including an answer-only reply when no model "
            "change is intended. A pending revise still requires model correction. "
            "Set false if further work is intended; no choice is preselected.",
        }
    elif check["eligible"]:
        feedback["completion_example"] = {
            "if": "If you intend no further edits and all listed work is resolved, "
            "this completion-only reply requests the final check.",
            "reply": response_model("repair")(stage_complete=True).model_dump(
                mode="json"
            ),
            "scope": "Syntax example only. Empty edit lists preserve the model. "
            "This is not approval of its science or a request to choose an "
            "approximation. Scientific corrections remain allowed.",
        }
    return packet
