"""Prospective action-specific advice for rejected pending branch patches.

This presentation layer neither edits a patch nor assesses scientific validity.
Actions are explicit choices grounded in a diagnostic and existing declarations;
an unresolved symbol's scientific role is never inferred from its spelling.
"""

from __future__ import annotations

from copy import deepcopy

from autoformalism.research import construction_branch_patch as branch

POLICY = "typed-branch-action-feedback-1"


def _rows(patch: dict, field: str) -> list[dict]:
    """Read only existing object records; do not reinterpret malformed fields."""
    values = patch.get(field, [])
    return (
        [row for row in values if isinstance(row, dict)]
        if isinstance(values, list)
        else []
    )


def _drop(field: str, name: str) -> dict:
    """A response fragment the proposer may explicitly select, never apply here."""
    return {"drop_actions": [{"field": field, "key": name}]}


def _declarations(parent: dict, patch: dict) -> tuple[dict, dict]:
    base = parent.get("problem", {}).get("base_candidate", {})
    parameters = {
        row["name"]: deepcopy(row)
        for row in [*base.get("parameters", []), *_rows(patch, "new_parameters")]
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    kinds = {
        row["state"]: "dynamic"
        for row in base.get("state_equations", [])
        if isinstance(row, dict) and isinstance(row.get("state"), str)
    }
    kinds.update(
        {
            row["name"]: "algebraic"
            for row in base.get("processes", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
    )
    return parameters, kinds


def _parameter_edit(name: str, declaration: dict, pending: dict, kinds: dict) -> dict:
    return {
        "category": "parameter_used_as_equation_target",
        "name": name,
        "invalid_action": {"field": "edits", "key": name},
        "existing_parameter_declaration": declaration,
        "explanation": f"The diagnosed equation target is {name}, which is already "
        "declared as a parameter. This error does not prohibit editing existing "
        "ODEs or algebraic equations with their existing kinds.",
        "choices": [
            {
                "if": "You intend the existing trainable parameter declaration "
                "and the equation entry is accidental, not a fixed-value condition.",
                "reply_fragment": _drop("edits", name),
                "preserve": "Keep the parameter declaration and other still-intended "
                "equation edits. Omitting the bad edit in a delta does not delete it.",
            },
            {
                "if": "You intend a generated quantity or a fixed parameter value.",
                "instruction": "Choose its scientific meaning explicitly. A new "
                "generated quantity uses new_states with an initializer or "
                "new_processes, with a non-conflicting name and updated uses. "
                "Do not encode a fixed parameter value as a state equation; use "
                "the documented wider-revision/withdrawal outcome if unsupported.",
            },
        ],
        "other_existing_equation_edits_with_matching_kind": [
            row["component"]
            for row in _rows(pending, "edits")
            if row.get("component") in kinds
            and row.get("kind") == kinds[row["component"]]
        ],
        "remaining_checks": "These locations are permitted edit categories, not "
        "scientific approvals. The full composed patch must still pass all checks.",
    }


def _unknown_symbol(name: str, pending: dict, parameters: dict, kinds: dict) -> dict:
    definitions = [
        row for row in _rows(pending, "edits") if row.get("component") == name
    ]
    declared = {
        row.get("name")
        for field in ("new_states", "new_processes")
        for row in _rows(pending, field)
    }
    if (
        len(definitions) == 1
        and name not in kinds
        and name not in declared
        and name not in parameters
    ):
        kind = definitions[0].get("kind")
        destination = (
            "new_processes"
            if kind == "algebraic"
            else "new_states"
            if kind == "dynamic"
            else None
        )
        if destination:
            return {
                "category": "new_definition_placed_in_existing_edits",
                "name": name,
                "current_location": {"field": "edits", "key": name},
                "proposed_kind": kind,
                "if": "This supplied equation is intended to introduce a new quantity.",
                "instruction": f"Declare {name} under {destination}, carrying the "
                "complete scientific equation you intend. A new state also needs "
                "its typed initializer. edits can only replace existing components.",
                "required_explicit_cleanup": _drop("edits", name),
                "preserve": "Keep still-intended uses and parameters. Adding the new "
                "declaration does not remove the old invalid pending edit.",
            }
    return {
        "category": "unresolved_formula_symbol",
        "name": name,
        "instruction": f"The expression uses {name}, but has no resolved declaration. "
        "Choose its scientific role; its spelling does not determine that role.",
        "choices": [
            "If a new shared dynamics coefficient, declare its name and chosen "
            "admissible role in patch.new_parameters. No equation is needed.",
            "If a new generated quantity, declare its complete scientific definition "
            "in patch.new_processes or patch.new_states; the latter needs "
            "an initializer.",
            "If a typo or unintended term, replace the affected whole RHS explicitly.",
        ],
        "no_automatic_repair": "No coefficient role, value, formula or new dependence "
        "has been selected by the runtime.",
    }


def _initializer(name: str, pending: dict, details: dict, kinds: dict) -> dict:
    declarations = [
        row for row in _rows(pending, "new_states") if row.get("name") == name
    ]
    eligible = details.get("editable_initial_states", [])
    eligible = (
        [value for value in eligible if isinstance(value, str)]
        if isinstance(eligible, list)
        else []
    )
    result = {
        "category": "initializer_location_or_authority",
        "state": name,
        "eligible_existing_initializer_targets": eligible,
        "existing_state": kinds.get(name) == "dynamic",
        "pending_new_state": bool(declarations),
        "instruction": f"The initializer edit for {name} is not allowed at its "
        "current location. This does not prohibit permitted edits to other equations.",
        "locations": {
            "existing_eligible_latent_state": "patch.initialization_edits",
            "new_differential_state": "patch.new_states[].initial, together with "
            "that state's complete RHS; not initialization_edits",
        },
        "preserve": "Recorded/fixed initial identities remain fixed. Do not remove "
        "unrelated dynamics or mechanisms to repair an initializer location error.",
    }
    if any(row.get("state") == name for row in _rows(pending, "initialization_edits")):
        result["if_retracting_or_moving_this_pending_initializer"] = _drop(
            "initialization_edits", name
        )
    if name not in kinds and any(
        row.get("component") == name for row in _rows(pending, "edits")
    ):
        result["if_introducing_this_new_state"] = {
            "instruction": "Move your intended complete state equation and initializer "
            "into one new_states declaration; explicitly remove both obsolete pending "
            "actions instead of merely omitting them.",
            "drop_actions": [
                {"field": "edits", "key": name},
                *(
                    [{"field": "initialization_edits", "key": name}]
                    if any(
                        row.get("state") == name
                        for row in _rows(pending, "initialization_edits")
                    )
                    else []
                ),
            ],
        }
    return result


def view(parent: dict, intended: dict) -> dict:
    """Return only advice anchored to this selected branch and diagnosed records."""
    diagnostic = intended.get("diagnostic") or {}
    code = diagnostic.get("code", "")
    details = diagnostic.get("details") or {}
    details = details if isinstance(details, dict) else {}
    pending = branch.complete_patch(intended)
    pending = pending if isinstance(pending, dict) else {}
    parameters, kinds = _declarations(parent, pending)
    cards = []
    name = details.get("component")
    if (
        code == "COMPONENT_IDENTITY"
        and isinstance(name, str)
        and name in parameters
        and any(row.get("component") == name for row in _rows(pending, "edits"))
    ):
        cards.append(_parameter_edit(name, parameters[name], pending, kinds))
    if code == "UNKNOWN_FORMULA_SYMBOL":
        names = details.get("unknown", [])
        if isinstance(names, list):
            cards.extend(
                _unknown_symbol(name, pending, parameters, kinds)
                for name in names
                if isinstance(name, str)
            )
    if code == "INITIALIZATION_IDENTITY":
        names = details.get("unavailable", [])
        if isinstance(names, list):
            cards.extend(
                _initializer(name, pending, details, kinds)
                for name in names
                if isinstance(name, str)
            )
    result = {
        "policy": POLICY,
        "selected_candidate": intended["id"],
        "diagnostic_code": code,
        "action_choices": cards,
        "patch_changed_by_runtime": False,
        "scientific_role_selected_by_runtime": False,
    }
    if code == "QUALIFIED_MODEL_DATA_CONFLICT":
        result["repair_goal"] = (
            "This pending patch was already tested. It passed normalization and "
            "static validation, but the displayed qualified model-data witness "
            "still rejects its mathematical model. Change the relevant equation, "
            "dependence, function family or permitted initializer premise. "
            "Reapplying these actions, changing only their explanation, or changing "
            "serialization cannot remove this same witness. Identify which actual "
            "mathematical premise your new edits change; do not just apply the "
            "pending patch."
        )
        result["already_tested_scientific_actions"] = branch.literal_changes(intended)
        result["repeated_candidate_ids"] = list(
            intended.get("repeats_candidate_ids", [])
        )
    elif code == "UNKNOWN_EVIDENCE":
        result["repair_goal"] = (
            "Correct the citation metadata using allowed_evidence_ids. This trial "
            "has not reached scientific assessment. A citation-only delta retains "
            "the scientific patch for its first complete check."
        )
    else:
        result["repair_goal"] = (
            "Repair the diagnosed pending action using the choices below and "
            "repair_feedback.instruction. Keep other still-intended actions. "
            "If deleting an action, send its explicit drop_actions entry; omission "
            "in a delta retains it. The corrected full trial must pass all checks."
        )
    return result


def apply(packet: dict, parent: dict, intended: dict) -> None:
    """Update known instruction fields only; preserve evidence and quoted model data."""
    report = view(parent, intended)
    repair = packet["repair_feedback"]
    repair["action_feedback"] = report
    guidance = repair["guidance"]
    if (
        report["action_choices"]
        or report["diagnostic_code"]
        in {"QUALIFIED_MODEL_DATA_CONFLICT", "UNKNOWN_EVIDENCE"}
        or guidance.get("specific_feedback", "").startswith(
            "Correct the listed locations in one complete patch."
        )
    ):
        guidance["specific_feedback"] = report["repair_goal"]
    packet["search_instruction"]["instruction"] = (
        "Repair the displayed selected pending branch, which has already failed "
        "its recorded check. Return one BranchRepairReply using the specific "
        "action choices and repair_feedback.instruction. The task is to correct "
        "that failure, not to re-emit or install the pending patch."
    )
