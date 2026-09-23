"""Compose general shared-law construction with joint-output revision safeguards."""

from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.search import review_multi_construction as construction
from autoformalism.search.shared_construction import construct
from autoformalism.search.shared_model_revision import relationships

SHARED_INSTRUCTION = (
    "Shared processes are named algebraic definitions with multiple consumers. "
    "Edit a shared law once; the runtime recompiles every consumer. Consumer "
    "signs and conversions are already present in the assembled equations. "
    "Splitting/removing a law requires coordinated changes to its consumers. "
    "Shared parameter names denote one globally fitted value; neither naming "
    "nor opposite signs certify conservation."
)


def fresh(cell: dict, task: dict, client, directory) -> dict:
    """Expose enforced public requirements before inventory and function proposals."""
    visible = {
        **cell,
        "brief": construction.construction_brief(cell, task).model_dump(mode="json"),
    }
    return construct(
        visible,
        task,
        client,
        directory,
        build_bundle=pipeline._bundle,
        certificate_for=pipeline.certificates,
        retain_failed_draft=True,
    )


def propose(plan: dict, task: dict, parent: dict, client, directory) -> dict:
    """Reuse bounded public-contract repairs, including unsuccessful unfitted drafts."""
    return construction.propose(
        plan,
        task,
        parent,
        client,
        directory,
        fresh_builder=fresh,
        strict_parameters=True,
        shared_relationships=True,
    )


def add_relationships(value: dict) -> None:
    """Describe actual dependencies using the same parameter aliases as the prompt."""
    equations = value["model"]["equations"]
    candidate = {
        "state_equations": [
            {"state": e["component"], "rhs": e["expression"]}
            for e in equations
            if e["kind"] == "dynamic"
        ],
        "processes": [
            {"name": e["component"], "expression": e["expression"]}
            for e in equations
            if e["kind"] == "algebraic"
        ],
    }
    value["shared_law_relationships"] = relationships(candidate)
    value["shared_law_instruction"] = SHARED_INSTRUCTION


def propagation(bundle: dict, decision: dict) -> None:
    """Record complete before/after dependencies after a single compiled patch."""
    child = decision["bundle"]
    decision["provenance"]["shared_law_revision"] = {
        "policy": "general-shared-revision-1",
        "before": relationships(bundle["candidate"]),
        "after": relationships((child or bundle)["candidate"]),
        "conservation_certified": False,
    }
    if child:
        child["revision_provenance"] = decision["provenance"]
