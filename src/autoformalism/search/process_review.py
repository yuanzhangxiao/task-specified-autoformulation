"""Optional scientific inventory review, before any equation topology is frozen."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import ScientificVariable
from autoformalism.staged_topology import content_hash, freeze_inventory

POLICY = "optional-process-review-1"
SYSTEM = """Review the complete variable inventory for scientifically useful shared
algebraic processes before equation topology is constructed. Return only the
requested JSON. A process is one instantaneous law reused by several equations,
not another memory state. It has no independent initial condition.

Propose a process only when it has a clear scientific meaning in the supplied
task. It may represent a shared transfer, conversion or response. Explain that
meaning in scientific_role; identify its drivers and at least two plausible
consumers from the current active inventory. Reuse an existing algebraic name
where appropriate. Do not rename, delete or change any existing variable.
Do not provide equations, parameter values, or initial conditions here.

For a transfer, distinguish the shared amount flow from the conversions needed
at consumers (for example, different areas or volumes). Independent response
gains can be necessary. Opposite signs alone do not prove conservation. These
suggestions inform later scientific decisions; they do not prescribe equations.
Do not invent coupling between physically disconnected subsystems. Similar
formula shapes alone do not establish a shared physical process.

Return processes: [] when none is justified. That is a successful review,
not a failure. More process names is not the objective.
"""


class ProcessSuggestion(StrictSchema):
    """A proposed algebraic identity with advisory scientific dependencies."""

    name: Identifier
    scientific_role: str = Field(min_length=1, max_length=650)
    drivers: tuple[Identifier, ...] = Field(max_length=64)
    consumers: tuple[Identifier, ...] = Field(min_length=2, max_length=64)


class ProcessReviewReply(StrictSchema):
    """An empty response is a valid no-addition decision."""

    processes: tuple[ProcessSuggestion, ...] = Field(max_length=64)


def apply_suggestions(brief, inventory, reply):
    """Atomically add algebraic identities, preserving every original declaration."""
    active = {v.name: v for v in inventory if v.definition != "unused"}
    all_names = {v.name: v for v in inventory}
    names = [p.name for p in reply.processes]
    if len(names) != len(set(names)):
        raise ValueError("duplicate process names")
    additions = []
    for p in reply.processes:
        if len(p.drivers) != len(set(p.drivers)) or len(p.consumers) != len(
            set(p.consumers)
        ):
            raise ValueError("duplicate driver or consumer")
        if not {*p.drivers, *p.consumers} <= active.keys():
            raise ValueError("drivers and consumers must be existing active variables")
        if p.name in (*p.drivers, *p.consumers):
            raise ValueError("process cannot drive or consume itself")
        if any(
            active[n].definition not in {"differential", "algebraic"}
            for n in p.consumers
        ):
            raise ValueError("supplied or unused variables cannot consume a process")
        if p.name in all_names:
            if all_names[p.name].definition != "algebraic":
                raise ValueError("cannot repurpose an existing non-algebraic variable")
            continue
        role = (
            p.scientific_role
            + " Suggested drivers: "
            + ", ".join(p.drivers)
            + "; suggested consumers: "
            + ", ".join(p.consumers)
            + ". Advisory relationships; determine equations scientifically."
        )
        additions.append(
            ScientificVariable(
                name=p.name, definition="algebraic", scientific_role=role
            )
        )
    return freeze_inventory(brief, (*inventory, *additions))


def review(brief, enriched_brief: dict, inventory, client, output: Path):
    """One cached optional call. Invalid delivery/content never erases the inventory.

    Allocation-drain signals still propagate; resume uses the same cached call.
    Budget exhaustion is recorded as a skipped review, not repaired repeatedly.
    """
    identity = content_hash(
        [POLICY, enriched_brief, [v.model_dump(mode="json") for v in inventory]]
    )
    path = output / "process_review.json"
    if path.exists():
        saved = json.loads(path.read_text())
        digest = saved.pop("sha256")
        if content_hash(saved) != digest or saved["identity"] != identity:
            raise ValueError("process review checkpoint differs")
        saved["sha256"] = digest
        return tuple(
            ScientificVariable.model_validate(v) for v in saved["inventory"]
        ), saved
    result = {
        "protocol": POLICY,
        "identity": identity,
        "status": "skipped_invalid",
        "request_hash": None,
        "suggestions": [],
        "added_names": [],
        "error": None,
    }
    accepted = inventory
    try:
        record = client.call(
            system=SYSTEM,
            user=json.dumps(
                {
                    "protocol": POLICY,
                    "public_brief": enriched_brief,
                    "inventory": [v.model_dump(mode="json") for v in inventory],
                },
                sort_keys=True,
            ),
            response_model=ProcessReviewReply,
            step="optional_process_review",
            attempt=0,
        )
        result["request_hash"] = record["request_hash"]
        reply = ProcessReviewReply.model_validate(visible_response(record))
        proposed = apply_suggestions(brief, inventory, reply)
        result.update(
            status="accepted" if reply.processes else "empty",
            suggestions=[p.model_dump(mode="json") for p in reply.processes],
            added_names=[
                v.name for v in proposed if v.name not in {x.name for x in inventory}
            ],
        )
        accepted = proposed
    except (ValueError, TypeError, KeyError, OSError) as error:
        result["error"] = str(error)[:6000]
    result["inventory"] = [v.model_dump(mode="json") for v in accepted]
    result["sha256"] = content_hash(result)
    atomic_json(path, result)
    return accepted, result
