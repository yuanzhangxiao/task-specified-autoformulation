"""Prospective action-specific advice for rejected pending branch patches.

This presentation layer neither edits a patch nor assesses scientific validity.
Actions are explicit choices grounded in a diagnostic and existing declarations;
an unresolved symbol's scientific role is never inferred from its spelling.
"""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy

from pydantic import ValidationError

from autoformalism.expressions import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research import construction_branch_patch as branch
from autoformalism.research import construction_repair_influence as influence
from autoformalism.research.construction_structural_revision import (
    InitialMapChoice,
    editable_initial_states,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem

POLICY = "typed-branch-action-feedback-1"
REPAIR_POLICY = "typed-branch-action-feedback-2"


def _example_sources(definition: dict) -> list[str]:
    """Keep supplied source names once; expressions remain authoritative."""
    sources = definition.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(x, str) for x in sources):
        return []
    return list(dict.fromkeys(sources))


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


def _state_declaration_choice(parent: dict, pending: dict, name: str) -> dict | None:
    """Separate an existing state's law from its explicitly proposed initial rule."""
    definitions = [
        row for row in _rows(pending, "new_states") if row.get("name") == name
    ]
    if not definitions:
        return None
    eligible = None
    with suppress(ValueError, TypeError, KeyError):
        eligible = editable_initial_states(
            PrefixFitProblem.model_validate(parent["problem"]),
            parent.get("scientific_aliases"),
        )
    result = {
        "category": "existing_state_in_new_declarations",
        "name": name,
        "existing_kind": "dynamic",
        "instruction": f"{name} already exists. Choose its intended complete RHS "
        "in edits with kind=dynamic, then explicitly drop its new_states entry. "
        "Treat that entry's initializer as a separate intention; do not silently "
        "discard it or overwrite a protected initial identity.",
        "eligible_existing_initializer_targets": eligible,
        "required_explicit_cleanup": _drop("new_states", name),
        "initializer_choices": "Explicitly retain the parent's current initial "
        "rule, or move the intended complete rule to initialization_edits only "
        "for an eligible state under the allowed observation policy. A required "
        "change to a protected initial identity needs a wider-revision outcome.",
    }
    if len(definitions) == 1:
        definition = definitions[0]
        edit = {
            "component": name,
            "kind": "dynamic",
            "expression": definition.get("rhs"),
            "sources": _example_sources(definition),
        }
        try:
            branch.BranchPatchFields.model_validate({"edits": [edit]})
        except (ValueError, TypeError):
            return result
        result["if_retaining_the_parent_initial_rule"] = {
            "patch": {"edits": [edit]},
            "drop_actions": [
                {"field": "new_states", "key": name},
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
        if eligible is not None and name in eligible:
            replacement = {
                "edits": [deepcopy(edit)],
                "initialization_edits": [
                    {"state": name, "initial": deepcopy(definition.get("initial"))}
                ],
            }
            try:
                branch.BranchPatchFields.model_validate(replacement)
            except (ValueError, TypeError):
                pass
            else:
                result["if_preserving_the_supplied_law_and_initializer"] = {
                    "patch": replacement,
                    **_drop("new_states", name),
                    "qualification": "This preserves your supplied rule; its "
                    "scope, aliases and scientific consistency still require "
                    "validation.",
                }
        if name in (pending.get("remove_states") or []):
            result["if_retaining_this_state_also_drop_its_removal"] = _drop(
                "remove_states", name
            )
    return result


def _declaration_choices(
    parent: dict, pending: dict, details: dict, kinds: dict
) -> list[dict]:
    """Explain exact category moves without choosing a scientific definition."""
    cards = []
    issues = details.get("declaration_issues", [])
    if not isinstance(issues, list):
        return cards
    for issue in issues[:16]:
        if not isinstance(issue, dict) or not isinstance(issue.get("name"), str):
            continue
        name = issue["name"]
        if issue.get("category") == "existing_state" and kinds.get(name) == "dynamic":
            card = _state_declaration_choice(parent, pending, name)
            if card is not None:
                cards.append(card)
        definitions = [
            r for r in _rows(pending, "new_processes") if r.get("name") == name
        ]
        if (
            issue.get("category") == "existing_process"
            and kinds.get(name) == "algebraic"
            and definitions
        ):
            card = {
                "category": "existing_process_in_new_declarations",
                "name": name,
                "existing_kind": "algebraic",
                "instruction": f"{name} already exists in the unchanged parent. "
                "Choose the complete expression you intend and place it in edits "
                "with kind=algebraic. Explicitly drop the pending new_processes "
                "entry; adding the edit alone leaves the invalid declaration present.",
                "required_explicit_cleanup": _drop("new_processes", name),
                "preserve": "Preserve unrelated state declarations and initializer "
                "choices. This category correction does not choose a new equation "
                "or authorize changing any initial value.",
            }
            if len(definitions) == 1 and isinstance(
                definitions[0].get("expression"), str
            ):
                definition = definitions[0]
                card["if_this_supplied_definition_is_your_intended_law"] = {
                    "patch": {
                        "edits": [
                            {
                                "component": name,
                                "kind": "algebraic",
                                "expression": definition["expression"],
                                "sources": _example_sources(definition),
                            }
                        ]
                    },
                    **_drop("new_processes", name),
                }
            removals = pending.get("remove_processes", [])
            if isinstance(removals, list) and name in removals:
                card["if_retaining_this_process_also_drop_its_removal"] = _drop(
                    "remove_processes", name
                )
            cards.append(card)
        if issue.get("category") == "parameter_component_collision":
            parameters = [
                r for r in _rows(pending, "new_parameters") if r.get("name") == name
            ]
            fields = [
                f
                for f in ("new_states", "new_processes")
                if any(r.get("name") == name for r in _rows(pending, f))
            ]
            if not parameters or not fields:
                continue
            cards.append(
                {
                    "category": "parameter_and_component_share_identity",
                    "name": name,
                    "choices": [
                        {
                            "if": "This identity is intended to be the declared "
                            "trainable coefficient.",
                            "reply_fragment": {
                                "drop_actions": [
                                    {"field": field, "key": name} for field in fields
                                ]
                            },
                            "instruction": "Keep its parameter declaration and "
                            "intended uses. A parameter needs no equation. If that "
                            "equation was also intended, give the generated quantity "
                            "a distinct name and update its uses explicitly.",
                        },
                        {
                            "if": "This identity is intended to be the "
                            "generated quantity.",
                            "reply_fragment": _drop("new_parameters", name),
                            "instruction": "Keep one complete generated definition. "
                            "Any independent coefficient needs a distinct declared "
                            "name and a scientifically chosen admissible role.",
                        },
                    ],
                    "no_automatic_choice": "The runtime selects neither meaning, "
                    "formula, coefficient domain nor initializer.",
                }
            )
    return cards


def _initializer_alias_choices(parent: dict, pending: dict) -> list[dict]:
    """Show the proposed local namespace when map references are inconsistent."""
    cards = []
    try:
        model = PrefixFitProblem.model_validate(parent["problem"])
        public = (
            set(model.context.targets)
            | set(model.context.forcing_channels)
            | {model.context.time_symbol}
        )
        states = {row.name for row in model.base_candidate.states}
        states |= {row.get("name") for row in _rows(pending, "new_states")}
    except (ValueError, TypeError, KeyError):
        public, states = None, None
    for field, key in (("initialization_edits", "state"), ("new_states", "name")):
        for row in _rows(pending, field)[:16]:
            initial = row.get("initial")
            if not isinstance(initial, dict) or initial.get("mode") != "map":
                continue
            aliases = [
                (f, r)
                for f in ("state_initials", "public_initials")
                for r in _rows(initial, f)
            ]
            if len(aliases) > 128 or len(_rows(initial, "parameters")) > 32:
                continue
            names = [r.get("alias") for _, r in aliases]
            names += [r.get("name") for r in _rows(initial, "parameters")]
            if any(not isinstance(n, str) for n in names) or len(names) != len(
                set(names)
            ):
                continue
            try:
                symbols = set(
                    RestrictedParser()
                    .parse(
                        initial.get("expression", ""), location="pending_initializer"
                    )
                    .symbols
                )
            except (ValueError, TypeError, ModelValidationError, RecursionError):
                continue
            declared = set(names)
            if symbols == declared:
                continue
            card = {
                "category": "initializer_local_alias_mismatch",
                "state": row.get(key),
                "location": f"patch.{field}",
                "expression": initial.get("expression"),
                "public_initials": deepcopy(_rows(initial, "public_initials")),
                "state_initials": deepcopy(_rows(initial, "state_initials")),
                "local_parameters": deepcopy(_rows(initial, "parameters")),
                "undeclared_symbols": sorted(symbols - declared),
                "unused_local_names": sorted(declared - symbols),
                "instruction": "The map expression must use its own declared "
                "aliases exactly. A runtime-generated alias from another map "
                "does not carry over. Confirm which initial reading or state "
                "you intend; replace this map coherently using its local names "
                "or explicitly revise the bindings. No new mapping is inferred.",
                "authority": "This corrects local syntax only. It does not make "
                "a protected initial value editable or choose a scientific map.",
            }
            if aliases:
                alias_field, reference = aliases[0]
                available = (
                    reference.get("channel") in public
                    if alias_field == "public_initials" and public is not None
                    else reference.get("state") in states
                    if alias_field == "state_initials" and states is not None
                    else False
                )
                example = {
                    "mode": "map",
                    "expression": reference["alias"],
                    "parameters": [],
                    "state_initials": [],
                    "public_initials": [],
                    alias_field: [deepcopy(reference)],
                }
                try:
                    InitialMapChoice.model_validate(example)
                except (ValidationError, ValueError, TypeError):
                    example = None
                if example is not None and available:
                    card["syntax_only_example_using_one_of_your_bindings"] = {
                        "initial": example,
                        "condition": "Use this identity map only if that single "
                        "declared initial value is the scientific initializer you "
                        "intend. The example is not an automatic replacement.",
                    }
            cards.append(card)
    return cards[:16]


def view(parent: dict, intended: dict, *, repair_contract: bool = False) -> dict:
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
    if repair_contract and code == "DECLARATION_IDENTITY":
        cards.extend(_declaration_choices(parent, pending, details, kinds))
    if repair_contract and code == "REVISION_VALIDATION":
        cards.extend(_initializer_alias_choices(parent, pending))
    result = {
        "policy": REPAIR_POLICY if repair_contract else POLICY,
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
        if repair_contract:
            result["new_component_dependencies"] = influence.view(parent, intended)
    elif repair_contract and code == "OPTION_DECISION_CONTRACT":
        result["repair_goal"] = (
            "This option failed its reply decision contract before scientific "
            "assessment. Initial options must propose decision=revise. The "
            "contract already supports adding latent states, replacing equations "
            "with their existing kinds, and editing eligible latent initializers "
            "within its stated constraints. These do not inherently require "
            "wider revision. "
            "If the listed changes are the in-scope scientific patch you intend, "
            "explicitly set patch.decision=revise in a delta; other pending actions "
            "remain. If an essential change is outside the displayed scope, "
            "withdraw this branch or explicitly replace it with a no-actions "
            "requires_wider_revision response under the documented contract. "
            "The runtime does not change your decision or accept this trial."
        )
        result["action_choices"] = [
            {
                "if": "You intend to submit the existing in-scope actions "
                "for validation.",
                "reply_fragment": {"patch": {"decision": "revise"}},
                "preserve": "A delta retains the other pending actions. This corrects "
                "the decision field only; all static and scientific checks apply.",
            }
        ]
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


def apply(
    packet: dict, parent: dict, intended: dict, *, repair_contract: bool = False
) -> None:
    """Update known instruction fields only; preserve evidence and quoted model data."""
    report = view(parent, intended, repair_contract=repair_contract)
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
