"""Opt-in focused presentation of saved initialization/topology repair requests.

This module changes no model, schema, scientific evidence or acceptance rule.
It presents current work first and preserves immutable public context in the same
stateless request. Exact repeated records in a *previous reply* may be displayed
with occurrence counts; they are never merged into an accepted edit.
"""

from __future__ import annotations

import json
from copy import deepcopy

POLICY = "focused-initialization-feedback-1"
REPETITION_POLICY = "ordered-identical-reply-records-1"

SYSTEM = """Construct ODEs and algebraic equations for the supplied scientific task.
The stages are variables, topology, then interaction functions and initialization
formulas. Follow the current stage and return JSON in the supplied format. The
runtime retains accepted declarations and displays the current model.

Terms:
Variables are quantities. Generated variables have ODEs (differential states) or
algebraic equations. Public target outputs must be generated variables too.
A process is a function of listed quantities contributing to an equation. A shared
process defines one algebraic value for reuse. sources/depends_on list function
arguments. uses.target names the generated variable whose equation receives the
contribution, including a declared public target output. A supplied-only input or
observation is available as an argument, not as a contribution's target.
A pathway is a chain of dependencies through variables and processes.
mechanism_realizations records which components the proposer chooses for each
public requirement. It is an attribution record, not a record of satisfied
requirements: the displayed checks separately assess declarations, paths and
behavior. memory_states names differential states carrying history; algebraic
readouts can depend on that history. stage_complete and remove_* are reply
controls, not scientific quantities."""

REPAIR_INSTRUCTIONS = """Repair the current model using feedback.repair_work and
the corresponding issues or compatibility questions. Inspect their observations
and assumptions. Return explicit, scientifically justified edits; the runtime
does not choose missing dependencies. Keep other intended effects and public
requirements represented. Use feedback.repair_guidance for the issues present in
this request. Set stage_complete=true when ready for rechecking; this requests a
check and does not override an unresolved requirement or assessment."""

EDITING_RULES = """Return JSON changes; omitted entries retain their accepted values.
variables declares differential or algebraic generated quantities. equations.name
is the quantity, not its derivative. Each term.sources lists the arguments of ONE
local process; [x, z] means a joint function. A processes entry defines a shared
process and its signed uses; the runtime supplies its definition and those uses.
Edit that entry through processes, not a second equations definition.

An equations entry replaces all local terms for its variable: to change B to C in
[A, B], return [A, C]; [C] removes A too. A processes entry replaces all its fields
and uses. remove_* deletes named entries; do not replace and delete the same entry.
The whole reply commits or none of it commits. Function formulas and fitted
coefficients belong to interaction; conversion is a known fixed public factor or
null, not a fitted coefficient.

initialization=null retains the handoff; a non-null record replaces its complete
editable-state list. Keep intended records, remove records for deleted or
now-algebraic states, and initialize new differential states. Fixed runtime rules
remain supplied and cannot be replaced. Initialization and topology commit together.
ODE arguments do not determine initialization arguments."""

REALIZATION_RULES = """mechanism_realizations maps each public requirement_id to
the chosen components and optional memory_states. Components may be generated
variables, shared processes or supplied quantities; for a local process name its
generated variable. Required paths must actually connect them. memory_states must
be differential components and are required only where public requirements demand
history. Editing a record replaces both complete lists; omission retains it.
remove_realizations removes a record, not the public requirement. feedback_bindings
separately identifies coordinates for the displayed target-feedback checks."""

MEASURED_RESERVATION_RULE = (
    "Each new call reserves measured serving input tokens plus the output "
    "allowance and the configured safety margin before generation. The serving "
    "context limit and episode token allowance are checked separately. Provider "
    "usage replaces the reservation after a response; unknown usage stays reserved."
)


def _encoded_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _reply_containers(packet: dict) -> list[dict]:
    """Locate only factual previous-reply receipts, never current model records."""
    repeated = packet.get("feedback", {}).get("repeated_repair_feedback", {})
    if not isinstance(repeated, dict):
        return []
    return [
        container
        for container in (repeated, repeated.get("observations", {}))
        if isinstance(container, dict) and "previous_reply" in container
    ]


def compact_repeated_replies(packet: dict) -> dict:
    """Encode identical consecutive declaration records only when it saves bytes.

    Each array preserves its original ordering, multiplicity and complete records.
    Only top-level lists of records in an earlier reply are eligible. The explicit
    envelope and recorded field names make expansion unambiguous.
    """
    result = deepcopy(packet)
    for container in _reply_containers(result):
        if "previous_reply_presentation" in container:
            raise ValueError("previous reply presentation is already applied")
        original = container["previous_reply"]
        if not isinstance(original, dict):
            continue
        compact = deepcopy(original)
        fields = []
        for name, records in original.items():
            if (
                not isinstance(records, list)
                or not records
                or not all(isinstance(row, dict) for row in records)
            ):
                continue
            runs = []
            for record in records:
                # Canonical JSON distinguishes e.g. True from 1; no approximate
                # equality, argument normalization or scientific merging occurs.
                encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if runs and encoded == runs[-1]["encoded"]:
                    runs[-1]["count"] += 1
                else:
                    runs.append(
                        {"record": deepcopy(record), "count": 1, "encoded": encoded}
                    )
            if len(runs) == len(records):
                continue
            replacement = {
                "original_count": len(records),
                "ordered_runs": [
                    {"record": row["record"], "count": row["count"]} for row in runs
                ],
            }
            if _encoded_size(replacement) < _encoded_size(records):
                compact[name] = replacement
                fields.append(name)
        if not fields:
            continue
        marker = {
            "policy": REPETITION_POLICY,
            "record_lists": fields,
            "reading_rule": "Only these previous-reply lists use ordered_runs: "
            "each complete record occurred count times consecutively. This is "
            "a display of the rejected or previous reply, not accepted model "
            "edits and not permission to merge scientific effects.",
        }
        if _encoded_size(compact) + _encoded_size(marker) >= _encoded_size(original):
            continue
        container["previous_reply"] = compact
        container["previous_reply_presentation"] = marker
    return result


def expand_repeated_reply(packet: dict) -> dict:
    """Restore exact prior reply arrays for integrity tests and offline inspection."""
    result = deepcopy(packet)
    for container in _reply_containers(result):
        marker = container.pop("previous_reply_presentation", None)
        if marker is None:
            continue
        if marker.get("policy") != REPETITION_POLICY:
            raise ValueError("unsupported previous reply presentation")
        reply = container["previous_reply"]
        fields = marker["record_lists"]
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate compressed reply fields")
        for name in fields:
            replacement = reply[name]
            records = []
            for run in replacement["ordered_runs"]:
                count = run["count"]
                if (
                    type(count) is not int
                    or count <= 0
                    or not isinstance(run["record"], dict)
                ):
                    raise ValueError("invalid previous reply run")
                records.extend(deepcopy(run["record"]) for _ in range(count))
            if len(records) != replacement["original_count"]:
                raise ValueError("previous reply record count mismatch")
            reply[name] = records
    return result


def _guidance(packet: dict) -> list[dict]:
    """Attach historical special-case rules only to applicable current findings."""
    feedback = packet.get("feedback", {})
    issues = feedback.get("issues", [])
    codes = {row.get("code") for row in issues if isinstance(row, dict)}
    error = str((packet.get("previous_edit") or {}).get("error") or "")
    guidance = []
    if "removal_namespace_mismatch" in codes or "cannot remove unknown" in error:
        guidance.append(
            {
                "for": "rejected deletion",
                "instruction": "Use the removal field of the displayed retained "
                "category. An uncommitted addition needs no deletion. Correcting a "
                "record does not require deleting its intended scientific effect.",
            }
        )
    if (
        codes
        & {
            "mechanism_realization",
            "mechanism_realization_declaration",
            "required_memory_path_revisits_driver",
            "mechanism_memory_component_mismatch",
        }
        or "memory_states must be a subset of components" in error
    ):
        guidance.append(
            {
                "for": "mechanism attribution or memory/path findings",
                "instruction": "Make mechanism_realizations match your intended "
                "variables, processes and paths. Include each memory state in "
                "components. An optional history claim may be withdrawn with "
                "memory_states=[] while retaining the mechanism's intended components; "
                "explicitly required memory and its paths still need a realization.",
            }
        )
    if codes & {"transfer_signs", "duplicate_receiver", "flow_classification_decision"}:
        guidance.append(
            {
                "for": "shared-process sign or classification findings",
                "instruction": "Choose signs and process classification from the "
                "physical meaning, and revise affected uses together. Changing a "
                "label alone does not justify inconsistent balances.",
            }
        )
    if "a shared process already has one runtime-generated definition" in error:
        process_names = {
            row["name"]
            for row in packet.get("model_view", {}).get("processes", [])
            if isinstance(row, dict) and "name" in row
        }
        conflicts = []
        for container in _reply_containers(packet):
            reply = container.get("previous_reply", {})
            if not isinstance(reply, dict):
                continue
            process_names.update(
                row["name"]
                for row in reply.get("processes", [])
                if isinstance(row, dict) and "name" in row
            )
            conflicts.extend(
                row["name"]
                for row in reply.get("equations", [])
                if isinstance(row, dict) and row.get("name") in process_names
            )
        guidance.append(
            {
                "for": "shared-process definition collision",
                "conflicting_equations_entries": sorted(set(conflicts)),
                "instruction": "A processes entry already supplies that process's "
                "algebraic definition. Edit its depends_on/uses in processes and omit "
                "the conflicting equations entry from this reply. Do not delete a "
                "retained process merely to remove an uncommitted duplicate "
                "definition. A different scientific representation needs explicit "
                "coordinated edits.",
            }
        )
    if "duplicate keys in processes" in error:
        guidance.append(
            {
                "for": "repeated process declarations in one reply",
                "instruction": "Return one complete processes record per process "
                "name, retaining all intended uses in that record. Identical repeats "
                "are a reply-format error; no copies were committed. Distinct intended "
                "scientific processes need distinct names and meanings.",
            }
        )
    questions = packet.get("compatibility_questions", [])
    if questions:
        guidance.append(
            {
                "for": "displayed topology/data compatibility questions",
                "instruction": "For an actionable conditional conflict, use its "
                "witnesses and assumptions to choose and implement the missing "
                "scientific distinction. A revise explanation alone is not an edit. "
                "Uncertain evidence alone does not establish an error. If an "
                "assumption does not apply, return a dispute_assumption decision with "
                "concrete "
                "supporting_evidence; runtime revalidation determines what remains "
                "unresolved. Neither acknowledgement nor retaining a known "
                "incompatible approximation closes a conflict.",
            }
        )
    if packet.get("external_effect_options"):
        guidance.append(
            {
                "for": "external-effect selections",
                "instruction": "Copy the intended displayed contribution_handle, "
                "obligation_id and explanation into external_exchanges. Stored "
                "reference fields are inspection information, not response fields. "
                "A changed/new contribution must be committed before its new handle "
                "can be selected in the following request.",
            }
        )
    return guidance


def _ordered(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value} | {
        key: item for key, item in value.items() if key not in keys
    }


COMPATIBILITY_POLICY = "shared-compatibility-details-1"
_QUESTION_IDENTITY = ("target", "question_id", "witnesses", "status", "repair_status")


def _question_rows(questions: list[dict]) -> tuple[list[dict], list[dict], list[int]]:
    """Separate nested explanations without changing any complete field value."""
    rows = deepcopy(questions)
    explanations = []
    indices = []
    for index, row in enumerate(rows):
        why = row.pop("why_this_still_conflicts", None)
        if why is not None:
            if not isinstance(why, dict):
                raise ValueError("compatibility explanation must be an object")
            explanations.append(why)
            indices.append(index)
    return rows, explanations, indices


def _join_question_rows(
    rows: list[dict], explanations: list[dict], indices: list[int]
) -> list[dict]:
    for index, explanation in zip(indices, explanations, strict=True):
        rows[index]["why_this_still_conflicts"] = explanation
    return rows


def factor_compatibility_details(packet: dict) -> dict:
    """Share only exact repeated fields; each target/witness/status stays inline.

    Reuse the qualified exact-field projection, with an independent table so it
    cannot collide with existing structural-feedback references. Splitting nested
    explanations lets their common rule be shown once without conflating distinct
    observed differences or assuming any two questions are interchangeable.
    """
    from autoformalism.search import construction_feedback_projection as projection

    if "compatibility_details" in packet or "compatibility_projection" in packet:
        raise ValueError("compatibility projection is already applied")
    original = deepcopy(packet)
    questions = original.get("compatibility_questions", [])
    if len(questions) < 2:
        return original
    rows, explanations, indices = _question_rows(questions)
    identities = [
        {key: row.pop(key) for key in _QUESTION_IDENTITY if key in row} for row in rows
    ]
    shell = {"feedback": {"issues": rows}, "requirements_checklist": explanations}
    compact = projection.factor(shell)
    if "feedback_details" not in compact:
        return original
    rows = compact["feedback"]["issues"]
    for row, identity in zip(rows, identities, strict=True):
        row.update(identity)
    original["compatibility_questions"] = _join_question_rows(
        rows, compact["requirements_checklist"], indices
    )
    original["compatibility_details"] = compact["feedback_details"]
    original["compatibility_projection"] = {
        "policy": COMPATIBILITY_POLICY,
        "reading_rule": "A question's shared_details, including those inside "
        "why_this_still_conflicts, reference exact complete fields in "
        "compatibility_details. Read them with that question. Each target, "
        "question ID, witness and status remains separate; sharing context "
        "does not merge findings or discard evidence.",
    }
    return (
        original
        if _encoded_size(original) < _encoded_size(packet)
        else deepcopy(packet)
    )


def expand_compatibility_details(packet: dict) -> dict:
    """Restore complete question fields for exact offline round-trip verification."""
    from autoformalism.search import construction_feedback_projection as projection

    result = deepcopy(packet)
    marker = result.pop("compatibility_projection", None)
    if marker is None:
        if "compatibility_details" in result:
            raise ValueError("compatibility details require a projection marker")
        return result
    if marker.get("policy") != COMPATIBILITY_POLICY:
        raise ValueError("unsupported compatibility projection policy")
    rows, explanations, indices = _question_rows(result["compatibility_questions"])
    shell = {
        "feedback": {"issues": rows},
        "requirements_checklist": explanations,
        "feedback_details": result.pop("compatibility_details"),
        "feedback_projection": {"policy": projection.POLICY},
    }
    expanded = projection.expand(shell)
    result["compatibility_questions"] = _join_question_rows(
        expanded["feedback"]["issues"], expanded["requirements_checklist"], indices
    )
    return result


def project(packet: dict, draft=None) -> dict:
    """Present focused instructions and exact feedback without editing a draft.

    Call after budget presentation. The caller must serialize this policy's packet
    in insertion order; sorted JSON would erase the requested reading priority.
    """
    del draft  # Accepted for parity with earlier presentation projections.
    if packet.get("stage") not in {"initialization", "repair"}:
        raise ValueError("focused initialization supports initialization or repair")
    result = deepcopy(packet)
    if result["stage"] == "repair":
        result["stage_instructions"] = REPAIR_INSTRUCTIONS
        result["editing_rules"] = EDITING_RULES
        result["mechanism_realization_rules"] = REALIZATION_RULES
        result["feedback"]["repair_guidance"] = _guidance(result)
    budget = result.get("feedback", {}).get("repair_budget")
    if isinstance(budget, dict):
        budget["reservation_rule"] = MEASURED_RESERVATION_RULE
    result = compact_repeated_replies(result)
    result = factor_compatibility_details(result)
    result["feedback"] = _ordered(
        result["feedback"],
        (
            "repair_work",
            "issues",
            "repair_guidance",
            "rejected_initialization",
            "repeated_repair_feedback",
            "initialization_status",
        ),
    )
    background = {
        key: result.pop(key)
        for key in ("training_highlights", "public_brief")
        if key in result
    }
    result = _ordered(
        result,
        (
            "stage",
            "stage_instructions",
            "feedback",
            "previous_edit",
            "model_view",
            "initialization",
            "compatibility_questions",
            "compatibility_details",
            "compatibility_projection",
            "requirements_checklist",
            "receiver_choices",
            "editing_rules",
            "mechanism_realization_rules",
            "response_template",
        ),
    )
    result.update(background)
    return result
