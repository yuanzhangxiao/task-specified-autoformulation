"""Opt-in prompts for explicit mechanism realizations and optional history claims."""

from __future__ import annotations

import json
from copy import deepcopy

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_control_fields as controls
from autoformalism.search import construction_ownership_feedback as ownership
from autoformalism.search import construction_protocol as protocol
from autoformalism.search import mechanism_realizations as realizations
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-mechanism-realizations-1"
REPAIR_REQUESTS = protocol.REPAIR_REQUESTS
REPAIR_TOKENS = protocol.REPAIR_TOKENS
REPEATED_REPAIRS = protocol.REPEATED_REPAIRS
repair_budget = protocol.repair_budget
response_model = realizations.response_model

_MEMBERSHIP_FIELDS = frozenset(
    {
        "components",
        "memory_states",
        "memories",
        "components_on_path",
        "disconnected_components",
        "unknown_components",
        "nondifferential_memory_states",
    }
)


def _canonical_error_members(value: object, field: str | None = None) -> object:
    """Normalize member order while preserving path order and scientific text."""
    if isinstance(value, dict):
        return {key: _canonical_error_members(item, key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_canonical_error_members(item) for item in value]
        return (
            sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
            if field in _MEMBERSHIP_FIELDS
            else items
        )
    return value


def repair_signature(
    draft: realizations.Draft, last: dict | None, errors: list
) -> dict:
    """Keep harmless realization-member reorderings within the same repair attempt."""
    signature = protocol.repair_signature(draft, last, errors)
    # Diagnostic emission order is immaterial to repetition; nested path witness
    # order remains meaningful. This changes only the signature, never feedback.
    signature["errors"] = sorted(
        (_canonical_error_members(error) for error in errors),
        key=lambda error: json.dumps(error, sort_keys=True),
    )
    return signature


DICTIONARY = """Variable: a quantity; generated variables are differential states
governed by ODEs or algebraic variables defined by instantaneous equations.
Process: a function of listed quantities contributing to an equation. A shared
process defines one algebraic value reused across equations. sources/depends_on
list arguments; uses.target names the generated quantity receiving a contribution.
Pathway: a chain of declared dependencies through variables and processes.
Mechanism: a required scientific effect realized through variables, processes
and pathways; it is not a variable type. Memory is history carried by differential
states; algebraic readouts can depend on that history."""

SYSTEM = (
    """Construct ODEs and algebraic equations for the supplied scientific task.
Work through variables, topology, then interaction functions.
Follow the current stage's instructions. The runtime retains accepted declarations
and shows the current model. Return JSON in the supplied format. A realization
record identifies the proposed components; it does not prove scientific coverage.

Terms:
"""
    + DICTIONARY
)

REALIZATION_RULES = """In mechanism_realizations, record one entry for every public
requirement: requirement_id, components and memory_states. components names the
supplied quantities, generated variables or shared processes chosen to realize it.
Several components may cooperate. For a local contribution, name the generated
variable whose equation contains it; do not invent term identifiers. At variables,
record the chosen quantities; extend or revise the record as processes and pathways
are added. The runtime derives component types from declarations.

memory_states identifies the differential states chosen to carry this mechanism's
history. It can be empty unless the public requirement explicitly requires memory.
Include each chosen memory state in components. Algebraic components can express
instantaneous effects without a memory assignment. Components alone do not create
dependencies; topology must implement applicable paths. Declaration consistency,
structural checks and scientific behavior are reported separately.

An edited record replaces its complete components and memory_states lists. An
omitted record stays unchanged. To withdraw an optional history claim, replace
that record with memory_states=[] and preserve or revise its intended components.
remove_realizations deletes a record by requirement_id; its public requirement
remains required. feedback_bindings separately supplies coordinates for listed
target-feedback checks, not mechanism realizations."""

EDITING = (
    ownership.EDITING
    + """
Use mechanism_realizations for the proposed components of every public mechanism.
stage_complete and remove_* are reply controls, never scientific quantities.
Set stage_complete at the reply's top level, outside all model lists."""
)

VARIABLES = """Declare every prediction target, then other generated quantities
needed for the public mechanisms. Use supplied observations where sufficient;
otherwise introduce differential states or algebraic variables. Give each name,
type and meaning. Supplied quantities need no generated declaration.
For every public mechanism, record its chosen realization components now in
mechanism_realizations. Use the chosen quantities at this stage; extend or revise
the record when processes are introduced. memory_states may be empty unless the
public requirement explicitly requires memory. Coefficients and initialization
formulas belong to interaction. Resolve declaration checklist items, then set
stage_complete=true at the reply's top level. No topology is needed yet."""

SHARED = (
    protocol.SHARED
    + """
Extend or revise mechanism_realizations when a shared process is part of a chosen
realization. Keep its components consistent with the accepted variables and uses;
recording a process name does not itself establish a pathway."""
)

TOPOLOGY = """Complete every generated variable's topology. Shared contributions
are already included in model_view; return only additional local processes in
equations. Each term gives sources, outer sign and meaning. sources=[x, z] means
one joint function, not two additive functions. Give no formulas or coefficients.
You may edit related variables together or add a scientifically needed variable.

Locate every public mechanism in its chosen variables, processes and pathways to
its targets. Keep mechanism_realizations consistent with these choices, including
algebraic realizations with no memory states. A local contribution is identified
through the variable it defines. Keep related definitions and uses consistent
across balances; shared and equivalent local forms are allowed. Use the public
science and training observations; no particular textbook names are required.
Checklist passes cover only the stated declaration and structural predicates;
function and behavior checks remain later work. Set stage_complete=true when the
whole topology is ready for checking."""

REPAIR = """Repair the accepted model using the issues and public requirements.
For a rejected deletion, check which retained category owns the entry; an
uncommitted addition needs no deletion. Keep intended mechanisms when changing
their variables, processes and pathways. Revise mechanism_realizations to match
the chosen representation. An optional history claim can be withdrawn by replacing
its record with memory_states=[]; the mechanism and its components still need a
realization. Explicitly required memory must be supplied or corrected.

Fix signs, types and related uses consistently with scientific meanings. You may
coordinate several edits. Return complete replacement lists for edited entries,
preserving other intended terms and components. Explanations alone do not change
topology. Coefficients belong to interaction. Set stage_complete=true when ready
for the full check."""


def stage_text(stage: str) -> str:
    """Use identical stage wording in live requests and the prompt review."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def model_view(draft: realizations.Draft) -> dict:
    """Keep the equation display and show one generalized realization mapping."""
    view = ownership.model_view(realizations.project(draft))
    view.pop("memory_assignments")
    view["mechanism_realizations"] = [
        row.model_dump(mode="json") for row in draft.mechanism_realizations
    ]
    return view


_REMOVALS = {
    "remove_variables": ("variables", "name"),
    "remove_equations": ("equations", "name"),
    "remove_processes": ("processes", "name"),
    "remove_realizations": ("mechanism_realizations", "requirement_id"),
    "remove_feedback_bindings": ("feedback_bindings", "target"),
    "remove_overlap_confirmations": ("overlap_confirmations", "overlap_id"),
}
_CONTROL_NAMES = controls.CONTROL_NAMES | {"remove_realizations"}
_OLD_MEMORY_ISSUES = frozenset(
    {
        "memory_assignment",
        "confirm_memory_assignment",
        "unknown_memory_requirement",
        "memory_type",
        "memory_driver_path",
        "driver_memory_path",
        "memory_target_path",
        "required_memory_bindings",
    }
)


def _control_issues(
    draft: realizations.Draft, last: dict | None, brief: PublicScientificBrief
) -> list[dict]:
    """Name the retained owner of rejected controls without choosing a deletion."""
    raw = (last or {}).get("rejected_reply")
    raw = raw if (last or {}).get("error") and isinstance(raw, dict) else {}
    owners: dict[str, list[dict]] = {}
    for removal, (field, key) in _REMOVALS.items():
        for entry in getattr(draft, field):
            owners.setdefault(getattr(entry, key), []).append(
                {
                    "field": field,
                    "key": key,
                    "removal_if_intended": {removal: [getattr(entry, key)]},
                }
            )
    issues = []
    for removal, (field, key) in _REMOVALS.items():
        entries = raw.get(removal)
        if not isinstance(entries, list):
            continue
        available = {getattr(row, key) for row in getattr(draft, field)}
        for name in sorted({s for s in entries if isinstance(s, str)} - available):
            retained = owners.get(name, [])
            issues.append(
                {
                    "code": "removal_namespace_mismatch",
                    "name": name,
                    "attempted_field": removal,
                    "not_retained_in": field,
                    "retained_entries": retained,
                    "action": f"Omit {name!r} from {removal}; that category has no "
                    "retained entry with this key. "
                    + (
                        "If deleting a retained entry is intended, use its listed "
                        "removal field. These entries are separate; do not remove "
                        "every owner automatically. Preserve intended effects and "
                        "update affected references explicitly. "
                        if retained
                        else "A rejected addition was never installed; omit it "
                        "instead of requesting deletion. "
                    )
                    + "The rejected reply committed nothing; resubmit its other "
                    "intended edits.",
                }
            )
    retained_controls = {
        row.name
        for field in ("variables", "equations", "processes")
        for row in getattr(draft, field)
        if row.name in _CONTROL_NAMES
    }
    rejected_controls = {
        row["name"]
        for field in ("variables", "equations", "processes")
        for row in (raw.get(field) if isinstance(raw.get(field), list) else [])
        if isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and row["name"] in _CONTROL_NAMES
    }
    public = {row.name for row in brief.public_variables}
    for name in sorted(retained_controls | rejected_controls):
        issues.append(
            {
                "code": "public_control_namespace_conflict"
                if name in public
                else "reply_control_is_not_quantity",
                "name": name,
                "retained_entries": owners.get(name, []),
                "attempted_declaration_rejected": name in rejected_controls,
                "action": "This public quantity also names a reply control. "
                "Resolve the ambiguous namespace explicitly; do not automatically "
                "rename or delete a public quantity or change its scientific type."
                if name in public
                else "This is a top-level reply control, not a scientific quantity. "
                "stage_complete is a boolean and remove_* fields are edit lists. "
                "Remove an already-retained accidental entry only through its "
                "listed owner field and explicitly update its references. A "
                "rejected declaration was never installed: omit it without a "
                "deletion, and resubmit other intended edits.",
            }
        )
    old = sorted({"mechanism_bindings", "remove_bindings"} & raw.keys())
    if old:
        issues.append(
            {
                "code": "obsolete_mechanism_fields",
                "rejected_fields": old,
                "action": "Use mechanism_realizations with requirement_id, "
                "components and memory_states. Choose the actual realization "
                "components explicitly; memory states alone are not a complete "
                "record. Use remove_realizations for deletion by requirement_id. "
                "No fields were migrated automatically and no edits were committed.",
            }
        )
    return issues


def _realization_issue(
    issue: dict, brief: PublicScientificBrief, draft: realizations.Draft
) -> dict:
    """Attach complete replacement destinations without assigning scientific roles."""
    row = deepcopy(issue)
    name = row.get("requirement", row.get("name"))
    record = next(
        (r for r in draft.mechanism_realizations if r.requirement_id == name), None
    )
    requirement = next((r for r in brief.requirements if r.id == name), None)
    row.pop("binding_feedback", None)
    row.pop("realization_feedback", None)
    row.update(
        field_to_edit="mechanism_realizations",
        retained_entry=record.model_dump(mode="json") if record else None,
        action="Return this requirement's complete realization record with the "
        "intended components and memory_states. Use supplied quantities or "
        "declare intended generated variables; shared processes may be added at "
        "their stage. A local contribution is identified by the variable it "
        "defines. Do not add a memory state merely to mark an algebraic effect "
        "represented. Preserve other intended components when replacing a record.",
    )
    if requirement is None:
        row["action"] = (
            "This ID is not a public requirement. Remove this record with "
            "remove_realizations, or supply a record for the intended public "
            "requirement. No scientific assignment is inferred."
        )
        row["removal"] = {"remove_realizations": [name]}
    elif record is not None:
        row["action"] = (
            "If these components are intended, keep this record and correct the "
            "displayed declaration or pathway issue through variables, equations "
            "and processes. If the assignment was mistaken, replace this complete "
            "record with the intended components and memory_states. An unchanged "
            "record needs no edit; a description alone adds no dependency."
        )
        if record.memory_states and not requirement.requires_dynamic_memory:
            row["if_optional_history_was_unintended"] = {
                "edit": {
                    "mechanism_realizations": [
                        {**record.model_dump(mode="json"), "memory_states": []}
                    ]
                },
                "meaning": "Withdraw only the optional history claim. Preserve "
                "or explicitly revise components to match the intended mechanism; "
                "its public requirements and applicable paths still apply.",
            }
    if requirement is not None:
        row["memory_rule"] = (
            "This explicit public memory requirement needs differential components "
            "in memory_states, distinct from its listed drivers and targets."
            if requirement.requires_dynamic_memory
            else "memory_states may be empty. Any selected memory states must be "
            "differential components that carry this mechanism's history."
        )
    return row


def payload(
    stage: str,
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    enriched_brief: dict,
    draft: realizations.Draft,
    contract: PublicGraphContract | None,
    last: dict | None = None,
    continuation: dict | None = None,
) -> dict:
    """Retain public evidence and established diagnostics under the new schema."""
    packet = ownership.payload(
        stage,
        brief,
        context,
        target_definitions,
        enriched_brief,
        realizations.project(draft),
        contract,
        last,
        continuation,
    )
    packet.update(
        stage_instructions=stage_text(stage),
        editing_rules=EDITING,
        mechanism_realization_rules=REALIZATION_RULES,
        model_view=model_view(draft),
        response_template=response_model(stage)(stage_complete=False).model_dump(
            mode="json"
        ),
    )
    packet.pop("memory_assignment_rules", None)
    packet.pop("memory_binding_options", None)
    receipt = packet["previous_edit"]
    if receipt and "mechanism_bindings" in receipt["changed"]:
        receipt["changed"]["mechanism_realizations"] = receipt["changed"].pop(
            "mechanism_bindings"
        )
    declarations = realizations.variable_checklist(
        brief, draft, target_definitions, contract
    )
    check = realizations.assess(
        brief,
        context,
        target_definitions,
        draft,
        graph_contract=contract,
        clarify_overlaps=False,
    )
    topology_stage = stage in {"equations", "repair"}
    checks = (check if topology_stage else declarations)["mechanism_realization_checks"]
    packet["requirements_checklist"] = [
        row
        for row in packet["requirements_checklist"]
        if not row["id"].startswith(("memory_assignment:", "mechanism_topology:"))
        and row["id"] != "structural_validity"
    ]
    for row in checks:
        declaration = row["declaration_status"]
        packet["requirements_checklist"].append(
            {
                **deepcopy(row),
                "id": f"mechanism_realization:{row['requirement']}",
                "status": row["structural_status"]
                if topology_stage and declaration == "satisfied"
                else declaration,
                "checked_at": "topology" if topology_stage else "variables",
            }
        )
    packet["requirements_checklist"].append(
        {
            "id": "structural_validity",
            "status": "deferred"
            if not topology_stage
            else "passed"
            if check["eligible"]
            else "failed",
            "checked_at": "topology",
        }
    )
    feedback = packet["feedback"]
    feedback.pop("memory_assignment_decisions", None)
    issues = []
    for issue in feedback["issues"]:
        if issue["code"] in _OLD_MEMORY_ISSUES:
            continue
        issue.pop("binding_context", None)
        issue.pop("assignment_decision", None)
        if issue["code"] == "state_process_ownership":
            issue["if_changing_scientific_type"] = (
                "A scientific type change requires an explicit variables edit "
                "and coordinated topology, realization records, feedback "
                "coordinates and references. Do not change a state type merely "
                "to resolve a naming collision."
            )
        elif issue["code"] == "duplicate_variable_declarations":
            issue["action"] = (
                "Declare each quantity once with its intended type. A quantity "
                "can contribute to several mechanisms without duplicate variable "
                "declarations; identify those roles in mechanism_realizations."
            )
        elif issue["code"] == "supplied_declarations_in_rejected_reply":
            issue["action"] = issue["action"].replace(
                "any bindings", "realization records and feedback coordinates"
            )
            retry = issue["conditional_variables_retry"]
            retry["if"] = retry["if"].replace(
                "errors and bindings",
                "errors, realization records and feedback coordinates",
            )
        if issue["code"] == "public_path":
            issue["repair_options"] = [
                "Implement the displayed public pathway using intended local "
                "or shared processes. Inspect the receiving equation and any "
                "intermediate quantities; no specific edge or sign is inferred.",
                "Keep mechanism_realizations consistent with the chosen "
                "components. A record alone creates no path; preserve other "
                "intended contributions in each replacement entry.",
            ]
        issues.append(issue)
    issues.extend(
        _realization_issue(row, brief, draft)
        for row in declarations["items"]
        if row["code"] == "mechanism_realization"
        and row["status"] in {"missing", "inconsistent"}
    )
    if topology_stage:
        issues.extend(
            _realization_issue(row, brief, draft)
            for row in check["errors"]
            if "realization" in row["code"] or row["code"] in _OLD_MEMORY_ISSUES
        )
    issues.extend(_control_issues(draft, last, brief))
    feedback["issues"] = list({str(row): row for row in issues}.values())
    public_names = {row.name for row in brief.public_variables}
    accidental_controls = _CONTROL_NAMES - public_names
    if "symbol_role_questions" in feedback:
        feedback["symbol_role_questions"] = [
            row
            for row in feedback["symbol_role_questions"]
            if row["name"] not in accidental_controls
        ]
    for issue in feedback["issues"]:
        if issue["code"] == "undeclared_lhs_entries":
            for row in issue["entries"]:
                if row["exact_name"] in accidental_controls:
                    row["choices"] = [
                        {
                            "action": "Follow reply_control_is_not_quantity "
                            "for this name."
                        }
                    ]
    if "scientific_review" in feedback:
        feedback["scientific_review"]["response"] = (
            "Use the reply fields to clarify scientific_meaning or scientific_role "
            "and make consistent topology and realization edits. If already "
            "consistent, leave entries unchanged. Do not return this question table."
        )
    if "repair_agenda" in feedback:
        feedback["repair_agenda"]["representation_issues"] = list(
            dict.fromkeys(row["code"] for row in feedback["issues"])
        )
        feedback["repair_agenda"]["failed_requirements"] = [
            row["id"]
            for row in packet["requirements_checklist"]
            if row["status"] in {"failed", "missing", "inconsistent"}
        ]
    return packet
