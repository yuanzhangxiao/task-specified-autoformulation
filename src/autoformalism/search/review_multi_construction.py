"""Bounded public-contract repair before any joint-output parameter fitting."""

from __future__ import annotations

import json

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    RevisionContractError,
)
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import review_revision_multi as edits
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import TrainingEvidence
from autoformalism.staged_topology import content_hash


def public_contract(cell: dict, task: dict) -> dict:
    """Expose exactly the existing public gate, including its role interpretation."""
    contract = cell["target_contract"]
    return {
        "source": "frozen_public_target_contract",
        "public_prompt_sha256": contract["public_prompt_sha256"],
        "requirements": contract["targets"]
        if task["arm"] != "no_spec"
        else [
            {
                "target_channel": t["target_channel"],
                "public_requirement": "Generate this target",
            }
            for t in contract["targets"]
        ],
        "gate_interpretation": (
            "Every target requires an explicit generated mapping. Specified roles "
            "are enforced by this frozen contract: dynamic_state requires a state "
            "mapping; instantaneous_process requires an algebraic output. The role "
            "interpretation is runtime policy based on the quoted public description, "
            "not a claim that every model of a physical rate must be algebraic."
        ),
    }


def construction_brief(cell: dict, task: dict) -> PublicScientificBrief:
    """Make enforced constraints visible before both inventory and RHS generation."""
    brief = pipeline._brief(cell, task)
    return brief.model_copy(
        update={
            "scientific_context": brief.scientific_context
            + "\nRuntime public target contract (enforced before fitting):\n"
            + json.dumps(public_contract(cell, task), sort_keys=True)
        }
    )


def fresh(cell: dict, task: dict, client, directory) -> dict:
    """Construct with original staged budgets; preserve an executable failing draft."""
    brief = construction_brief(cell, task)
    context = ValidationContext.model_validate(cell["context"])
    evidence = (
        None
        if task["arm"] == "brief_only"
        else TrainingEvidence.model_validate(cell["evidence"])
    )
    topology = run_staged_topology(
        brief,
        context,
        client,
        directory / "topology",
        hybrid_variable_construction=True,
        audit_public_polarity_policy=True,
        proposer_owns_unfixed_signs=True,
        training_evidence=evidence,
    )
    functions = None
    if topology["complete_topology"] and topology["public_structure_checks_passed"]:
        functions = run_staged_functions(
            brief,
            context,
            topology,
            client,
            directory / "functions",
            generation_granularity="equation_batch_atomic_repair",
            function_repair_policy="certified_outer_gain",
            initialization_policy="causal_training",
            training_evidence=evidence,
        )
    value = {
        "topology": topology,
        "functions": functions,
        "status": "construction_failed",
        "bundle": None,
    }
    if functions and functions["complete_model"]:
        try:
            value["bundle"] = pipeline._bundle(cell, task, brief, topology, functions)
        except (ValueError, KeyError, ModelValidationError) as error:
            value.update(status="contract_failed", error=str(error))
    return value


def propose(
    plan: dict,
    task: dict,
    parent: dict,
    client,
    directory,
    *,
    fresh_builder=None,
    strict_parameters=False,
    shared_relationships=False,
) -> dict:
    """Repair a saved unfitted draft, or construct if none exists."""
    cell = plan["cells"][task["cell"]]
    bundle = parent.get("construction_draft")
    value = {} if bundle else (fresh_builder or fresh)(cell, task, client, directory)
    bundle = bundle or value.get("bundle") or value.get("construction_draft")
    if bundle is None:
        return {**value, "construction_draft": None}
    certificate = pipeline.certificates(bundle, cell, task)
    if certificate["eligible_for_development_selection"]:
        return {
            **value,
            "status": "constructed",
            "bundle": bundle,
            "certificate": certificate,
            "construction_draft": bundle,
            "attempts": [],
        }
    attempts, retry, draft = (
        [],
        pipeline._certificate_feedback(certificate, task),
        bundle,
    )
    for attempt in range(3):
        raw, record = None, None
        user = edits.payload(bundle, None, None, retry)
        user.update(
            public_target_contract=public_contract(cell, task),
            public_requirement_findings=pipeline._certificate_feedback(
                certificate, task
            ),
            descriptive_training_evidence=None
            if task["arm"] == "brief_only"
            else cell["evidence"],
        )
        if shared_relationships:
            from autoformalism.search.fresh_shared import add_relationships

            add_relationships(user)
        if strict_parameters:
            user["parameter_declaration_policy"] = (
                "Reuse existing canonical names/aliases without redeclaration. "
                "Their declared roles are immutable. To change a role, introduce "
                "a new name and update the intended expressions explicitly."
            )
        try:
            record = client.call(
                system=edits.SYSTEM_PROMPT,
                user=json.dumps(user, sort_keys=True, separators=(",", ":")),
                response_model=edits.ScientificRevision,
                step="repair_public_construction",
                attempt=attempt,
            )
            raw = visible_response(record)
            decision = edits.apply_edits(
                bundle, None, raw, reject_existing_role_conflicts=strict_parameters
            )
            if shared_relationships:
                from autoformalism.search.fresh_shared import propagation

                propagation(bundle, decision)
            draft = decision["bundle"] or bundle
            checked = pipeline.certificates(draft, cell, task)
            if not checked["eligible_for_development_selection"]:
                raise RevisionContractError(
                    "PUBLIC_MODEL_REQUIREMENTS",
                    "The draft fails public checks; correct the named definitions.",
                    **pipeline._certificate_feedback(checked, task),
                )
            attempts.append(
                {"request_hash": record["request_hash"], "accepted": True, "raw": raw}
            )
            return {
                **value,
                "status": "constructed",
                "bundle": draft,
                "certificate": checked,
                "construction_draft": draft,
                "attempts": attempts,
                "decision": decision,
            }
        except RepairBudgetExceeded as error:
            return {
                **value,
                "status": "construction_repair_failed",
                "bundle": None,
                "construction_draft": draft,
                "attempts": attempts,
                "error": str(error),
            }
        except (ValueError, KeyError, TypeError, ModelValidationError) as error:
            if record is None:
                raise
            retry = edits.feedback(bundle, None, None, raw, error)
            attempts.append(
                {
                    "request_hash": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "accepted": False,
                    "raw": raw,
                    "feedback": retry,
                }
            )
    return {
        **value,
        "status": "construction_repair_failed",
        "bundle": None,
        "construction_draft": draft,
        "attempts": attempts,
    }
