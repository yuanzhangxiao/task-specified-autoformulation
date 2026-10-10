"""Opt-in topology feedback that defers functional ambiguity to interaction work."""

from __future__ import annotations

from copy import deepcopy

from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_ledger as ledger

POLICY = "minimal-deferred-interaction-1"
SIGN_QUESTION = """Does this generated quantity denote a positive outward rate or a
signed flux? Review its defining law and every consuming contribution together.
Choose consistent function forms and assembly signs; do not flip a sign merely
because two negative topology signs appear along a path."""


def interaction_handoff(brief, draft: ledger.Draft) -> dict:
    """Retain exact review candidates, without changing or rejecting topology."""
    overlaps = []
    for item in ledger.contribution_overlaps(draft, brief):
        overlaps.append(
            {
                **{k: v for k, v in item.items() if k not in {"question", "status"}},
                "status": "deferred_to_interaction",
                "reason": "Matching dependencies do not establish equal functions.",
            }
        )
    definitions = {v.name: v.definition for v in draft.variables}
    signed = []
    for definition in draft.equations:
        if definitions.get(definition.name) != "algebraic":
            continue
        negative = [
            t.model_dump(mode="json")
            for t in definition.terms
            if t.outer_weight_sign.value == "negative"
        ]
        if not negative:
            continue
        consumers = [
            {"target": e.name, "term": t.model_dump(mode="json")}
            for e in draft.equations
            for t in e.terms
            if definition.name in t.sources and t.outer_weight_sign.value == "negative"
        ]
        consumers.extend(
            {"process": p.name, "use": u.model_dump(mode="json")}
            for p in draft.processes
            if definition.name in p.depends_on
            for u in p.uses
            if u.sign == "negative"
        )
        if consumers:
            signed.append(
                {
                    "quantity": definition.name,
                    "definition": definition.model_dump(mode="json"),
                    "negative_consumers": consumers,
                    "status": "deferred_to_interaction",
                    "question_for_interaction": SIGN_QUESTION,
                    "scope": "A syntactic review candidate, not a sign-error verdict.",
                }
            )
    return {
        "status": "pending_interaction_review",
        "topology_blocking": False,
        "same_dependency_candidates": overlaps,
        "signed_quantity_candidates": signed,
        "interaction_instruction": (
            "After proposing functions, compare candidates within each receiving "
            "equation using their complete assembled terms, parameters, conversions "
            "and shared consumers. Clarify ambiguous repeated effects or signed-rate "
            "interpretations with the proposer. Equal drivers, or equal formula "
            "text with different parameters, alone do not prove duplicate effects. "
            "Preserve distinct intended mechanisms and cross-equation sharing."
        ),
        "execution_scope": (
            "Saved handoff only. This construction runner stops before interaction; "
            "no function comparison or scientific clarification has been executed."
        ),
    }


def requirement_status(brief, draft, target_definitions, graph_contract, stage, check):
    """Expose existing predicates and coverage limits; never interpret free prose."""
    declarations = checklist.variable_checklist(
        brief, draft, target_definitions, graph_contract
    )
    rows = [
        {**r, "checked_at": "variables", "scope": "Declaration consistency only."}
        for r in declarations["items"]
        if r["code"] not in {"mechanism_topology", "public_graph"}
    ]
    topology_stage = stage in {"equations", "repair", "final"}
    available = topology_stage and check["graph_check_status"] == "assessed"
    memory = {r["requirement"]: r for r in check["memory_binding_checks"]}
    for requirement in brief.requirements:
        evidence = [
            r
            for r in check["public_structure_checks"]
            if r.get("requirement") == requirement.id
        ]
        bound = memory.get(requirement.id)
        endpoints = bool(requirement.drivers and requirement.targets)
        passed = bool(evidence) and all(r["passed"] for r in evidence)
        if requirement.requires_dynamic_memory:
            passed &= bound is not None and bound["status"] == "passed"
        rows.append(
            {
                "id": f"mechanism_topology:{requirement.id}",
                "public_requirement": requirement.public_requirement,
                "status": "deferred"
                if not topology_stage
                else "unavailable"
                if not available
                else "unresolved"
                if not endpoints
                else "passed"
                if passed
                else "failed",
                "checked_at": "topology",
                "evidence": evidence if available else [],
                "memory_evidence": bound if available else None,
                "scope": "Declared paths through selected states only; function "
                "forms, signs of the physical effect and scientific sufficiency "
                "are unassessed.",
            }
        )
    for row in check["reviewed_public_graph_checks"]:
        rows.append(
            {
                **row,
                "id": f"public_graph:{row['id']}",
                "status": "deferred"
                if not topology_stage
                else "unavailable"
                if row["passed"] is None
                else "passed"
                if row["passed"]
                else "failed",
                "checked_at": "topology",
                "passed": row["passed"] if topology_stage else None,
                "witness": row.get("witness") if topology_stage else None,
            }
        )
    composition = [
        r for r in check["public_structure_checks"] if r["kind"] == "composition_path"
    ]
    for index, dependency in enumerate(brief.target_dependencies):
        evidence = (
            composition[index] if available and index < len(composition) else None
        )
        rows.append(
            {
                "id": f"target_composition:{index}",
                "public_requirement": dependency.public_requirement,
                "target": dependency.target,
                "acceptable_sources": list(dependency.acceptable_sources),
                "status": "deferred"
                if not topology_stage
                else "unavailable"
                if evidence is None
                else "passed"
                if evidence["passed"]
                else "failed",
                "checked_at": "topology",
                "evidence": evidence,
            }
        )
    rows.append(
        {
            "id": "structural_validity",
            "status": "deferred"
            if not topology_stage
            else "failed"
            if check["errors"]
            else "passed",
            "checked_at": "topology",
            "evidence": check["errors"] if topology_stage else [],
            "scope": "Existing compiler, reference, process-use, explicit public "
            "sign and graph checks; not unrestricted scientific validity.",
        }
    )
    for index, text in enumerate(check["deferred_scientific_checks"]):
        rows.append(
            {
                "id": f"public_scientific_scope:{index}",
                "public_contract_note": text,
                "status": "unassessed",
                "checked_at": "function/initialization/behavior review as applicable",
                "scope": "Public contract scope note, not an executed predicate.",
            }
        )
    rows.append(
        {
            "id": "remaining_public_prose",
            "status": "unassessed",
            "checked_at": "coverage review",
            "scope": "Only the listed typed predicates are checked. Other public "
            "prose is not automatically translated into rules; missing applicable "
            "checks remain coverage gaps, never implicit passes.",
        }
    )
    return {"items": rows, "scientific_completeness": "not_established"}


def assessment(brief, context, target_definitions, draft, graph_contract):
    """Keep every existing structural rule; defer only ambiguous source overlap."""
    result = ledger.assess(
        brief,
        context,
        target_definitions,
        draft,
        graph_contract=graph_contract,
        clarify_overlaps=False,
    )
    handoff = interaction_handoff(brief, draft)
    result["contribution_overlaps"] = handoff["same_dependency_candidates"]
    result["interaction_review_handoff"] = handoff
    return result


def update_payload(payload, brief, context, target_definitions, draft, graph_contract):
    """Use this same adapter in live requests and the human-readable prompt review."""
    result = deepcopy(payload)
    stage = result["stage"]
    check = assessment(brief, context, target_definitions, draft, graph_contract)
    result["bookkeeping_policy"] = POLICY
    result["requirement_status"] = requirement_status(
        brief, draft, target_definitions, graph_contract, stage, check
    )
    # The new table includes those declaration rows; do not repeat them twice.
    result.pop("variable_checklist", None)
    if stage != "variables":
        result["interaction_review_handoff"] = check["interaction_review_handoff"]
        result["current_draft"].pop("potential_contribution_overlaps", None)
    if stage == "repair":
        # Do not leave the old blocking questions in a counterfactual preview.
        diagnostic = result.get("runtime_diagnostics") or {}
        diagnostic["clarification_requests"] = []
        result["runtime_diagnostics"] = diagnostic
    return result
