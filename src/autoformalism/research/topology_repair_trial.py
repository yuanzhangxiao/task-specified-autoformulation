"""Matched topology-only comparison on a frozen, named-target construction."""

from __future__ import annotations

import json
from copy import deepcopy

from pydantic import Field

from autoformalism.research import t2_breadth as construction
from autoformalism.research import topology_repair_case as intake
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.search.topology_repair_generator import (
    EditAssessment,
    TopologyEdit,
    assess_edit,
    generate_repairs,
)
from autoformalism.staged_topology import content_hash


class RepairReply(StrictSchema):
    """One local topology modification and its scientific rationale."""

    edit: TopologyEdit
    reason: str = Field(min_length=1, max_length=1600)


def verify_case(case: dict) -> None:
    """Verify the frozen parent, training evidence and its public projection."""
    intake.validate_case(case)


def messages(case: dict) -> tuple[str, str]:
    """Return the exact system and user strings for every independent draw."""
    verify_case(case)
    system = (
        "Repair the displayed topology using the public training evidence. "
        "Choose one edit to the indicated target that can account for the observed "
        "difference while preserving the stated mechanisms. Use the literal "
        "quantity names from the packet. add_dependency adds your source to the "
        "target's existing law. via_new_memory introduces one differential memory "
        "state driven by your source and its own value, initialized with one shared "
        "constant, and feeds that state into the target. The runtime names the new "
        "state and applies the entire edit. Generated quantities use model values "
        "during prediction. Give a short scientific reason for your choice. "
        "Return the structured reply."
    )
    user = json.dumps(intake.public_packet(case), sort_keys=True, ensure_ascii=False)
    return system, user


def _edited_parent(parent: dict, edit: TopologyEdit, memory: str | None) -> dict:
    """Materialize a dependency edit in the saved constructor's native format."""
    graph = deepcopy(parent)
    target = graph["targets"][edit.target]
    if edit.kind == "via_new_memory":
        if not memory:
            raise ValueError("Missing generated memory identifier")
        graph["additional_components"].append(
            {
                "name": memory,
                "kind": "dynamic",
                "sources": [edit.source, memory],
                "role": f"Carries the history of {edit.source} into {edit.target}.",
                "represents_auxiliary": None,
            }
        )
    target["sources"].append(memory or edit.source)
    return graph


def _record(case: dict, evidence, assessment: EditAssessment) -> dict:
    row = assessment.model_dump(mode="json")
    row["edit_sha256"] = content_hash(row["edit"])
    row["public_graph_check"] = "not_run"
    if assessment.status == "invalid":
        return row
    graph = _edited_parent(case["parent"], assessment.edit, assessment.memory_state)
    graph_type, _ = construction.named_topology_forms(evidence.context.targets)
    try:
        topology = graph_type.model_validate(graph).as_topology()
        construction.check_topology(
            topology,
            evidence.context,
            PublicScientificBrief.model_validate(case["public_brief"]),
        )
    except ValueError as error:
        row.update(status="invalid", public_graph_check="failed", reason=str(error))
        return row
    row.update(public_graph_check="passed", revised_topology=graph)
    if assessment.memory_state:
        row["added_initial_rule"] = {
            "state": assessment.memory_state,
            "mode": "shared_parameter",
        }
    return row


def _scope(case: dict) -> dict:
    return {
        "allowed_sources": tuple(case["allowed_sources"]),
        "pair": tuple(case["pair"]),
        "time": case["witness_time"],
    }


def assess_reply(case: dict, reply: RepairReply) -> dict:
    """Score a 20B edit under the same graph and data checks as enumeration."""
    verify_case(case)
    if reply.edit.target != case["target"]:
        edit = reply.edit.model_dump(mode="json")
        return {
            "edit": edit,
            "edit_sha256": content_hash(edit),
            "status": "invalid",
            "public_graph_check": "not_run",
            "reason": "The edit changes a different target from the requested one.",
        }
    evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    assessment = assess_edit(
        intake.draft_from_parent(case["parent"]), evidence, reply.edit, **_scope(case)
    )
    return _record(case, evidence, assessment)


def offline_portfolio(case: dict) -> list[dict]:
    """Enumerate the bounded repair grammar and retain every assessment."""
    verify_case(case)
    evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    alternatives = generate_repairs(
        intake.draft_from_parent(case["parent"]),
        evidence,
        case["target"],
        **_scope(case),
    )
    return [_record(case, evidence, alt) for alt in alternatives]
