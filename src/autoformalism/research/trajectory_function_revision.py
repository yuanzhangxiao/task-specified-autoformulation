"""Review explicit RHS source changes against the public task and training evidence."""

from __future__ import annotations

import json

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY, RestrictedParser
from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.research.preproposal_topology import assess_topology
from autoformalism.research.trajectory_basin_balance import check_transfer_balance
from autoformalism.research.trajectory_function_fitting import Formula, validate_formula
from autoformalism.staged_topology import content_hash


def permitted_sources(graph: dict, frozen: dict) -> list[str]:
    """Return names that can acquire new RHS edges, excluding initial-only readings."""
    context = frozen["evidence"]["context"]
    initial_only = set(frozen.get("measured_initial_bindings", {}).values())
    return sorted(
        set(graph["variable_types"])
        | set(frozen["histories"])
        | (set(context["fixed_covariates"]) - initial_only)
        | {context["time_symbol"]}
    )


def _revised_sources(parent: dict, frozen: dict, formula: Formula) -> dict:
    """Resolve each expression name as an allowed source or a fitted coefficient."""
    kinds = parent["variable_types"]
    if set(formula.equations) != set(kinds):
        raise ValueError(f"Provide exactly the parent equations {sorted(kinds)}")
    context = frozen["evidence"]["context"]
    allowed = set(permitted_sources(parent, frozen))
    reserved = (
        allowed
        | {entry["name"] for entry in frozen["public_brief"]["public_variables"]}
        | set(context["fixed_covariates"])
        | set(context["unavailable_observed_channels"])
        | set(APPROVED_FUNCTION_ARITY)
    )
    coefficients = set(formula.parameters)
    collisions = (reserved & coefficients) | {
        name for name in coefficients if name.startswith("state0_")
    }
    if collisions:
        raise ValueError(
            f"Parameter names shadow reserved symbols: {sorted(collisions)}"
        )
    parser = RestrictedParser()
    sources = {}
    for target, expression in formula.equations.items():
        symbols = parser.parse(expression, location=f"equations.{target}").symbols
        rhs = symbols - coefficients
        # A covariate already admitted on this exact parent edge may be retained.
        # An initial measurement carries no new ongoing edge by default.
        unavailable = rhs - allowed - set(parent["sources"][target])
        if unavailable:
            raise ValueError(
                f"{target} has unavailable RHS sources: {sorted(unavailable)}"
            )
        sources[target] = sorted(rhs)
    return sources


def _check_graph(child: dict, frozen: dict, reason: str) -> dict:
    """Run the complete-graph gate with every public requirement on the same cohort."""
    case = SelectionCase.model_validate(
        {
            "case_id": frozen["case_id"],
            "target": frozen["target"],
            "inventory": [
                {"name": name, "kind": kind}
                for name, kind in sorted(child["variable_types"].items())
            ],
            "evidence_sha256": frozen["evidence"]["sha256"],
            "evidence_summary": "Replay all measured training variables.",
            "public_task": frozen["public_brief"]["scientific_context"],
            "required_options": [
                "initial:" + frozen["target"],
                *("history:" + name for name in frozen["required_histories"]),
            ],
            "candidates": [],
        }
    )
    proposal = {
        "hypothesis": reason,
        "sources": child["sources"],
        "latent_initializers": [
            {
                "state": state,
                "rule": rule,
                "justification": "Retain the admitted initialization.",
            }
            for state, rule in child["initializers"].items()
            if rule["mode"] != "observed_initial"
        ],
    }
    assessment = assess_topology(
        case, None, proposal, {**frozen, "check_all_measured": True}
    )
    # The topology gate returns tuples for some dependencies. A second read of
    # a JSON receipt must reproduce exactly the same list-valued evidence.
    return json.loads(json.dumps(assessment, allow_nan=False))


def _feedback(assessment: dict) -> str:
    if reason := assessment.get("reason"):
        return reason
    details = []
    if missing := assessment.get("missing_selected_or_required"):
        details.append(f"Missing required routes: {', '.join(missing)}")
    for name, check in assessment.get("training_checks", {}).items():
        if check["status"] == "screened":
            continue
        message = f"{name}: {check['status']}"
        if pair := check.get("witness"):
            observations = [
                f"{w['trajectory']} at t={w['time']:g} has {name}={w['value']:g}"
                for w in pair
            ]
            message += " (" + ", ".join(observations) + ")"
        details.append(message)
    return "; ".join(details) or assessment["status"]


def _check_balance(frozen: dict, graph: dict, formula: dict) -> dict:
    balance = check_transfer_balance(frozen, graph, formula)
    if balance["status"] == "fail":
        return {
            "status": "balance_conflict",
            "reason": balance["reason"],
            "balance_check": balance,
        }
    return {
        "status": "eligible",
        "graph": graph,
        "formula": formula,
        "balance_check": balance,
    }


def assess_fixed(graph: dict, frozen: dict, formula: dict) -> dict:
    """Check expressions on an already admitted graph, including physical evidence."""
    try:
        value = Formula.model_validate(formula)
        _revised_sources(graph, frozen, value)
        verified = validate_formula(graph, value.model_dump(mode="json"))
    except (ModelValidationError, ValueError) as exc:
        return {"status": "invalid_formula", "reason": str(exc)}
    return _check_balance(frozen, graph, verified)


def assess_revision(
    parent_graph: dict, frozen: dict, formula: dict, reason: str
) -> dict:
    """Return a fit-ready child only for a justified, reviewed change to RHS sources."""
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1500:
        return {
            "status": "invalid_topology",
            "reason": "Give a scientific reason of 1 to 1500 characters.",
            "patch": {},
        }
    patch = {}
    try:
        value = Formula.model_validate(formula)
        sources = _revised_sources(parent_graph, frozen, value)
        for name, row in sources.items():
            before, after = set(parent_graph["sources"][name]), set(row)
            if before != after:
                patch[name] = {
                    "added": sorted(after - before),
                    "removed": sorted(before - after),
                }
        child = {**parent_graph, "sources": sources}
        verified = validate_formula(child, value.model_dump(mode="json"))
    except (ModelValidationError, ValueError) as exc:
        return {"status": "invalid_formula", "reason": str(exc), "patch": patch}

    if not patch:
        return {
            "status": "no_topology_change",
            "reason": "RHS sources are unchanged. Use the fixed-graph route.",
            "patch": {},
        }
    assessment = _check_graph(child, frozen, reason.strip())
    outcome = {
        "patch": patch,
        "scientific_reason": reason.strip(),
        "parent_graph_sha256": content_hash(parent_graph),
        "child_graph_sha256": content_hash(child),
        "assessment": assessment,
    }
    if assessment["status"] != "topology_screen_passed":
        return outcome | {
            "status": assessment["status"],
            "reason": _feedback(assessment),
        }
    return outcome | _check_balance(frozen, child, verified)
