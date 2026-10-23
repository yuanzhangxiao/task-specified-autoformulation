"""V4 presentation and patch bookkeeping; never infer a scientific correction."""

from __future__ import annotations

import ast
from copy import deepcopy

from autoformalism.research.construction_revision_edits import (
    _bounded_ast,
    _canonical_rhs,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "directional-equation-repair-feedback-1"


def function_view(report: dict) -> dict:
    """Distinguish a model-implied bound from the observed required change."""
    from autoformalism.research.construction_function_consistency import prompt_view

    result = deepcopy(prompt_view(report))
    for check in result["conflicts"]:
        for witness in [check["witness"]]:
            bound = witness.pop("required_change")
            witness["model_implied_change_bound"] = bound
            upper = bound["comparison"] == "at_most"
            observed = witness["observed_change_interval"]
            relation = "at most" if upper else "at least"
            witness["directional_explanation"] = (
                f"This candidate forces the change in {check['target']} to be "
                f"{relation} {bound['value']}, but its observed change interval is "
                f"[{observed[0]}, {observed[1]}]. "
                + (
                    "The model's upper bound is too low: it cannot increase enough "
                    "(or it forces too much decrease). Making that upper bound "
                    "smaller cannot resolve this witness."
                    if upper
                    else "The model's lower bound is too high: it cannot decrease "
                    "enough (or it forces too much increase). Making that lower "
                    "bound larger cannot resolve this witness."
                )
            )
        check["repair_feedback"] = (
            "The bound belongs to the displayed candidate, not to the data. "
            "Revise a scientifically justified complete RHS so this implication "
            "no longer contradicts the observations. This does not prescribe a "
            "particular sign or new dependence. Preserve required pathways and "
            "unaffected terms. Types, existing parameter roles and initializers "
            "remain fixed; explain if a wider change is needed."
        )
    return result


def normalize_citations(reply: object) -> tuple[object, list[dict]]:
    """Resolve only displayed, unambiguous nested-section citation aliases."""
    if not isinstance(reply, dict):
        raise ValueError("revision response must be one JSON object")
    result = deepcopy(reply)
    aliases = {
        "function_consistency",
        "initial_family_consistency",
        "affine_family_consistency",
    }
    changes = []
    ids = result.get("evidence_ids")
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        return result, changes
    normalized = []
    for item in ids:
        target = "training_consistency" if item in aliases else item
        if target != item:
            changes.append({"from": item, "to": target})
        if target not in normalized:
            normalized.append(target)
    result["evidence_ids"] = normalized
    return result, changes


def patch_signature(reply: object, time_symbol: str = "t") -> str | None:
    """Identify repeated edits independently of prose and advisory source lists.

    This is syntactic bookkeeping, not algebraic equivalence or acceptance. Bounded
    AST parsing never executes text. Invalid/unbounded patches have no signature.
    """
    if not isinstance(reply, dict):
        return None
    edits, parameters = reply.get("edits", []), reply.get("new_parameters", [])
    if not isinstance(edits, list) or not isinstance(parameters, list):
        return None
    if len(edits) > 32 or len(parameters) > 64:
        return None
    normalized = []
    try:
        for item in edits:
            component, kind = item["component"], item["kind"]
            source = _canonical_rhs(item["expression"], component, kind, time_symbol)
            _bounded_ast(source, "repeat bookkeeping")
            tree = ast.parse(source, mode="eval")
            normalized.append(
                {
                    "component": component,
                    "kind": kind,
                    "expression": ast.dump(tree, include_attributes=False),
                }
            )
        return content_hash(
            {
                "edits": sorted(normalized, key=content_hash),
                "new_parameters": sorted(parameters, key=content_hash),
            }
        )
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def targeted_guidance(diagnostic: dict, parent_problem: dict) -> dict:
    """Keep type/initializer clarifications beside the relevant failed edit."""
    result = deepcopy(diagnostic)
    code = result.get("code")
    if code in {"COMPONENT_IDENTITY", "PARAMETER_IDENTITY"}:
        candidate = parent_problem["base_candidate"]
        result["permitted_edit_locations"] = {
            "dynamic": [x["state"] for x in candidate["state_equations"]],
            "algebraic": [x["name"] for x in candidate["processes"]],
        }
        result["scope_clarification"] = (
            "Keep each component's displayed kind. A dynamic RHS may contain "
            "its own state. An existing parameter is not an editable equation. "
            "To alter the model's dependence, edit the consuming equation or "
            "an existing algebraic definition; do not redeclare the parameter. "
            "Changing dZ/dt to a supplied signal u makes Z integrate u; it does "
            "not make Z equal u. dZ/dt=0 preserves Z(0), which need not be zero."
        )
    return result
