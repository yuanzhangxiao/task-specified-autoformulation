"""Domain-independent optional process construction using the existing stages."""

from __future__ import annotations

from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.process_gains import compile_bundle
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import TrainingEvidence

POLICY = "general-shared-construction-1"


def _stage(path: Path, build):
    """Seal completed stage output; interrupted provider work resumes its cache."""
    if path.exists():
        return sealed_read(path)["result"]
    return sealed_write(path, {"result": build()})["result"]


def construct(
    cell,
    task,
    client,
    directory,
    *,
    build_bundle,
    certificate_for,
    retain_failed_draft=False,
):
    """Preserve one variable inventory and one budget across process fallback.

    Callbacks are the campaign's existing canonical reconstruction and public
    requirement checks. No domain names, reference equations or physics probes
    occur here. Allocation interruptions deliberately propagate to the worker.
    """
    brief = PublicScientificBrief.model_validate(cell["brief"])
    context = ValidationContext.model_validate(cell["context"])
    evidence = (
        TrainingEvidence.model_validate(cell["evidence"])
        if task["arm"] == "full"
        else None
    )
    options = {
        "hybrid_variable_construction": True,
        "audit_public_polarity_policy": True,
        "proposer_owns_unfixed_signs": True,
        "training_evidence": evidence,
        "shared_process_guidance": True,
    }
    variables = _stage(
        directory / "variables.json",
        lambda: run_staged_topology(
            brief,
            context,
            client,
            directory / "variables",
            stop_after_inventory=True,
            **options,
        ),
    )
    if variables.get("status") != "variables_complete":
        return {
            "status": "construction_failed",
            "variables": variables,
            "construction_policy": POLICY,
            "attempts": [],
            "fallback_used": False,
        }
    inventory = tuple(
        ScientificVariable.model_validate(v) for v in variables["inventory"]
    )
    attempts = []
    last_draft = None
    for route in ("process", "ordinary_fallback"):
        output = directory / route
        topology = functions = bundle = certificate = None
        error = None
        try:
            topology = _stage(
                output / "topology_stage.json",
                lambda output=output, route=route: run_staged_topology(
                    brief,
                    context,
                    client,
                    output / "topology",
                    initial_inventory=inventory,
                    initial_memory_candidates=variables["memory_candidates"],
                    optional_process_review=route == "process",
                    bind_shared_processes=route == "process",
                    signed_shared_processes=route == "process",
                    **options,
                ),
            )
            if (
                topology["complete_topology"]
                and topology["public_structure_checks_passed"]
            ):
                functions = _stage(
                    output / "function_stage.json",
                    lambda output=output, topology=topology: run_staged_functions(
                        brief,
                        context,
                        topology,
                        client,
                        output / "functions",
                        generation_granularity="equation_batch_atomic_repair",
                        function_repair_policy="certified_outer_gain",
                        initialization_policy="causal_training",
                        training_evidence=evidence,
                        shared_process_guidance=True,
                        dependency_policy="local-function-dependencies-1",
                        assembly_policy="process-assembly-contract-1",
                        function_delivery_policy="identified-function-delivery-1",
                    ),
                )
            if functions and functions["complete_model"]:
                bundle = build_bundle(cell, task, brief, topology, functions)
                bundle = compile_bundle(
                    bundle,
                    functions.get("shared_process_contract"),
                    "declared_or_independent",
                    cell["training"],
                )
                certificate = certificate_for(bundle, cell, task)
                last_draft = bundle
                if not certificate["eligible_for_development_selection"]:
                    error, bundle = "public requirement check failed", None
            else:
                error = (
                    (functions or {}).get("error")
                    or topology.get("error")
                    or "incomplete construction"
                )
        except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
            error = str(exc)[:6000]
        attempts.append(
            {
                "route": route,
                "error": error,
                "process_review": (topology or {}).get("process_review"),
                "certificate": certificate,
            }
        )
        if bundle is not None:
            return {
                "status": "constructed",
                "bundle": bundle,
                "certificate": certificate,
                "construction_policy": POLICY,
                "attempts": attempts,
                "fallback_used": route == "ordinary_fallback",
                "shared_process_contract": functions.get("shared_process_contract"),
            }
        if not ((topology or {}).get("process_review") or {}).get("bindings"):
            break  # An empty/invalid optional suggestion adds no retry allowance.
    return {
        "status": "construction_failed",
        "bundle": None,
        "attempts": attempts,
        "construction_policy": POLICY,
        "fallback_used": len(attempts) > 1,
        **({"construction_draft": last_draft} if retain_failed_draft else {}),
    }
