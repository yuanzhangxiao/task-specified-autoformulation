"""Specific current-prompt feedback; no scientific assignments or graph edits."""

from copy import deepcopy

from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
)
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import target_feedback_evidence

VARIABLE_COMPLETION = """Before finishing this variable stage, explicitly declare
every public target as differential or algebraic. Listing it in the public source
catalog does not declare it. Valid partial declarations are retained, but a missing
target keeps this stage open for bounded correction. No topology or function is
needed to finish the variable inventory; choose the types yourself."""

FEEDBACK_REPAIR = """For target-feedback failures, read the separate binding,
state-type, readout-path and cycle results. A binding names your chosen coordinate;
it does not create any dependency edge. If the binding is already present but its
cycle is missing, repeating the binding cannot repair it. Choose scientifically
justified topology edits under the public requirement; do not add generic decay
or flip a type merely to satisfy a graph check. Address remaining_topology_failures
as well as annotation/overlap questions."""


def variable_completion(brief: PublicScientificBrief, draft: ledger.Draft) -> dict:
    """Check only mechanically required public-target declarations, not science."""
    declared = {v.name for v in draft.variables}
    missing = sorted(
        v.name
        for v in brief.public_variables
        if v.data_role == "target" and v.name not in declared
    )
    return {
        "code": "variable_stage_completion",
        "status": "incomplete" if missing else "ready",
        "missing_public_targets": missing,
        "instruction": (
            "Your valid declarations were retained. Declare each missing public "
            "target in variables and choose its differential/algebraic definition. "
            "Do not redeclare public inputs or supply topology/functions here. "
            "The inventory has not been frozen."
            if missing
            else "All public targets have generated-variable declarations."
        ),
        "scope": "Declaration completeness only; scientific sufficiency is unassessed.",
    }


def explain_feedback(check: dict, draft: ledger.Draft) -> dict:
    """Explain existing target-feedback predicates using the same graph evaluator."""
    result = deepcopy(check)
    equations = (
        tuple(EquationDefinition.model_validate(e) for e in check["equations"])
        if check["graph_check_status"] == "assessed"
        else None
    )
    coordinates = {b.target: b.states for b in draft.feedback_bindings}
    details = {}
    for row in result["reviewed_public_graph_checks"]:
        if row["kind"] != "target_feedback":
            continue
        evidence = target_feedback_evidence(equations, row["target"], coordinates)
        row["feedback_evidence"] = evidence
        if row["passed"] is False:
            row["reason"] = feedback_reason(evidence)
            row["repair_actions"] = feedback_actions(evidence)
            row["annotation_limit"] = (
                "Repeating or adding a binding creates no graph edges. "
                "A missing cycle/readout path needs explicit topology changes "
                "unless the coordinate assignment itself was mistaken."
            )
        details[row["id"]] = row
    for issue in result["errors"]:
        if issue["code"] == "reviewed_public_graph" and issue["id"] in details:
            issue.update(deepcopy(details[issue["id"]]))
    return result


def feedback_reason(evidence: dict) -> str:
    """Replace generic missing-binding advice with the actual predicate results."""
    binding = evidence["binding_status"]
    lines = [
        "The differential target is its own coordinate; no binding is needed."
        if binding == "implicit_target_coordinate"
        else "Coordinate binding is present."
        if binding == "present"
        else "The algebraic target has no coordinate binding."
    ]
    for row in evidence["coordinate_checks"]:
        facts = [
            f"{label}: {'yes' if row[key] else 'no'}"
            for key, label in (
                ("declared", "declared"),
                ("differential", "differential"),
                ("algebraic_readout_path_exists", "algebraic readout path"),
                ("feedback_cycle_exists", "feedback cycle"),
            )
        ]
        lines.append(f"{row['state']}: " + "; ".join(facts) + ".")
    return " ".join(lines)


def feedback_actions(evidence: dict) -> list[dict]:
    """Name each failed structural predicate without choosing a scientific law."""
    actions = []
    if evidence["binding_status"] == "missing":
        actions.append(
            {
                "missing": "coordinate_binding",
                "action": "Choose the actual dynamic coordinate(s) for this algebraic "
                "target in feedback_bindings. This annotation alone does not "
                "establish their cycles or readout paths.",
            }
        )
    for row in evidence["coordinate_checks"]:
        state = row["state"]
        if not row["differential"]:
            actions.append(
                {
                    "state": state,
                    "missing": "differential_coordinate",
                    "action": "Check your coordinate assignment and explicit variable "
                    "definition. Choose a scientifically valid state/representation; "
                    "the runtime will not change its type.",
                }
            )
            continue
        if not row["feedback_cycle_exists"]:
            actions.append(
                {
                    "state": state,
                    "missing": "feedback_cycle",
                    "action": f"No dependency path returns to {state}. Inspect its "
                    "ordinary contributions and inserted process uses. Choose the "
                    "dependencies needed by the quoted public mechanism, directly "
                    "or through other generated variables. Do not repeat the binding "
                    "or add an arbitrary decay term.",
                }
            )
        if not row["algebraic_readout_path_exists"]:
            actions.append(
                {
                    "state": state,
                    "missing": "algebraic_readout_path",
                    "action": "No algebraic-only readout path connects this coordinate "
                    "to the target. Revise its topology or correct the scientific "
                    "coordinate assignment; another annotation does not add a path.",
                }
            )
    return actions
