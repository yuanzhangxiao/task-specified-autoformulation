"""Opt-in initializer repair with deterministic model/data acceptance checks.

Historical graph construction remains unchanged. Every proposed dependency graph
is assessed before acceptance; unavailable checks never become scientific passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.construction_graph_feedback import acceptance_feedback
from autoformalism.search.construction_graph_runner import SYSTEM as DEPENDENCY_SYSTEM
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    GraphDependencyRecord,
    validate_contract,
    validate_handoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.construction_readable import model_view
from autoformalism.search.construction_scope_feedback import scoped_feedback
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "initialization-dependency-repair-1"
LEGACY_FEEDBACK = "legacy-1"
SCOPE_FEEDBACK = "scope-aware-2"
SYSTEM = (
    DEPENDENCY_SYSTEM
    + """
Acceptance includes the displayed deterministic model/data checks. A demonstrated
contradiction blocks acceptance even when the dependency graph is well formed.
Read the current equations, paired observations and assumptions before revising.
Changing prose, or changing a shared constant to a map of identical readings,
does not repair a demonstrated equal-input/different-output conflict.

Return decision=revise with a complete records list and a short explanation of
how the change addresses the evidence. Choose scientifically justified dependencies
or an authorized trajectory root; do not add arbitrary dependencies to evade a check.
A trajectory root defers the equality check to numerical prefix inference; it is
not evidence of recovery. If fixing the conflict requires a dynamics or output
change outside this initializer-only scope, return decision=needs_topology_revision,
records=[], and explain the needed change. This stops unresolved, not accepted.
No retain-approximation or acknowledgment option clears a concrete contradiction.
"""
)

SCOPE_SYSTEM = SYSTEM.replace(
    "Initial dependencies must be acyclic. Do not use future\n"
    "measurements, trajectory identifiers, algebraic variables or process names.",
    "Initial dependencies must be acyclic. Only differential states may appear in\n"
    "state_initials or receive initializer records. A recorded reading listed in\n"
    "permitted_public_initials may be used in public_initials, including a reading\n"
    "of an algebraic output; this does not give that output an independent "
    "initializer.\n"
    "Do not use unavailable future measurements, trajectory identifiers or "
    "process names.",
).replace(
    "On repair return the complete replacement record list, retaining valid records.",
    "On repair return the complete replacement record list, retaining unaffected\n"
    "records. A schema-valid initializer that causes a model-data conflict must\n"
    "still be revised. Removing a separate invalid record does not repair that\n"
    "conflict, and the model cannot be accepted while the conflict remains.",
)


def system_prompt(feedback_policy: str = LEGACY_FEEDBACK) -> str:
    """Keep historical wording fixed and reject an unknown presentation policy."""
    if feedback_policy == LEGACY_FEEDBACK:
        return SYSTEM
    if feedback_policy == SCOPE_FEEDBACK:
        return SCOPE_SYSTEM
    raise ValueError(f"unknown initializer feedback policy: {feedback_policy}")


class GraphDependencyReply(StrictSchema):
    """A complete replacement or an explicit unresolved request for wider edits."""

    decision: Literal["revise", "needs_topology_revision"]
    records: tuple[GraphDependencyRecord, ...] = Field(default=(), max_length=64)
    explanation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def coherent_decision(self) -> GraphDependencyReply:
        """Do not combine an out-of-scope disposition with silent model edits."""
        if self.decision == "needs_topology_revision" and self.records:
            raise ValueError("needs_topology_revision requires records=[]")
        return self


def _check_inputs(draft, context, scientific_context, contract, evidence, initial):
    """Validate sealed evidence and fixed model identities before any request."""
    validate_contract(
        contract, draft=draft, context=context, scientific_context=scientific_context
    )
    if evidence is None:
        raise ValueError("deterministic acceptance requires training evidence")
    validate_evidence(evidence)
    if evidence.context != context:
        raise ValueError("initialization evidence context differs")
    if initial is not None:
        validate_handoff(
            initial,
            contract,
            draft=draft,
            context=context,
            scientific_context=scientific_context,
        )


def dependency_construction_identity(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    client: StagedTopologyClient,
    *,
    evidence: TrainingCompatibilityEvidence,
    previous_questions: tuple[dict, ...] = (),
    initial_handoff: GraphDependencyHandoff | None = None,
    feedback_policy: str = LEGACY_FEEDBACK,
) -> str:
    """Bind preview, execution, historical evidence and the acceptance policy."""
    from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

    return content_hash(
        {
            "protocol": PROTOCOL,
            "draft": draft.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "scientific_context": scientific_context,
            "contract": contract.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "previous_questions": previous_questions,
            "initial_handoff": initial_handoff.model_dump(mode="json")
            if initial_handoff
            else None,
            "system": system_prompt(feedback_policy),
            **(
                {"feedback_policy": feedback_policy}
                if feedback_policy != LEGACY_FEEDBACK
                else {}
            ),
            "schema": GraphDependencyReply.model_json_schema(),
            "runtime": runtime_source_hash(),
            "namespace": client.namespace,
            "settings": client.settings.model_dump(mode="json"),
            "seed": client.seed,
            "request_policy": client.request_policy(),
        }
    )


def _feedback_view(gate: dict | None) -> dict | None:
    """Omit full arrays and duplicated assessments from the proposer payload."""
    if gate is None:
        return None
    fields = {
        "protocol",
        "blocked",
        "status",
        "conflict_count",
        "witness_count",
        "conflicts",
        "repair_options",
        "interpretation",
        "scientific_repair_established",
    }
    return {
        **{key: value for key, value in gate.items() if key in fields},
        "deferred_check_count": len(gate["deferred_checks"]),
        "deferred_interpretation": "Deferred checks are not passes. Full evidence "
        "and all additional witnesses remain in the saved assessment.",
    }


def dependency_request(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    *,
    evidence: TrainingCompatibilityEvidence,
    previous: dict | None = None,
    previous_questions: tuple[dict, ...] = (),
    initial_handoff: GraphDependencyHandoff | None = None,
    construction_identity: str | None = None,
    feedback_policy: str = LEGACY_FEEDBACK,
) -> str:
    """Lead with current, rechecked conflicts and retain one model/context view."""
    system_prompt(feedback_policy)
    if previous is None:
        gate = (
            acceptance_feedback(
                draft,
                evidence,
                contract,
                initial_handoff,
                previous_questions=previous_questions,
            )
            if initial_handoff is not None
            else None
        )
        feedback = {"current_checks": _feedback_view(gate)}
    else:
        gate = previous.get("gate")
        feedback = {
            key: value
            for key, value in previous.items()
            if key not in {"assessment", "gate", "candidate", "last_valid_candidate"}
        }
        feedback["current_checks"] = _feedback_view(gate)
    current = (
        previous.get("candidate", previous.get("last_valid_candidate"))
        if previous
        and (previous.get("candidate") or previous.get("last_valid_candidate"))
        else initial_handoff.model_dump(mode="json")
        if initial_handoff
        else None
    )
    if feedback_policy == SCOPE_FEEDBACK:
        feedback["scope_guidance"] = scoped_feedback(
            draft,
            context,
            contract,
            GraphDependencyHandoff.model_validate(current) if current else None,
            gate,
            previous,
        )
    payload = {
        "construction_identity": construction_identity,
        "feedback": feedback,
        "model_view": model_view(draft),
        "current_dependency_records": current,
        "latent_states_requiring_records": sorted(
            v.name
            for v in draft.variables
            if v.definition == "differential"
            and v.name not in contract.observed_initial_channels
        ),
        "runtime_initialization_contract": contract.model_dump(mode="json"),
        "public_context": scientific_context,
        "history": {
            "retained_question_count": len(previous_questions),
            "interpretation": "Current checks are recomputed from sealed training "
            "evidence. Historical questions remain in the artifact; stale wording "
            "does not override the current model or initializer declarations.",
        },
    }
    return "Initialization dependency repair\n" + json.dumps(
        payload, ensure_ascii=False
    )


def _semantic(handoff: dict) -> str:
    """Ignore descriptions when identifying repeated dependency choices."""
    return content_hash(
        sorted(
            (
                {"state": record["state"], "rule": record["rule"]}
                for record in handoff["records"]
            ),
            key=lambda record: record["state"],
        )
    )


def _evaluate(
    record,
    draft,
    context,
    scientific_context,
    contract,
    evidence,
    questions,
    *,
    feedback_policy: str = LEGACY_FEEDBACK,
):
    """Record atomic validity and deterministic conflict checks separately."""
    event = {"attempt": record["attempt"], "request_hash": record["request_hash"]}
    try:
        raw = visible_response(record)
    except (ValueError, TypeError, KeyError) as exc:
        return {
            **event,
            "accepted": False,
            "structural_valid": False,
            "unavailable": True,
            "error": str(exc)[:2000],
        }
    try:
        reply = GraphDependencyReply.model_validate(raw)
        if reply.decision == "needs_topology_revision":
            return {
                **event,
                "accepted": False,
                "structural_valid": False,
                "decision": reply.decision,
                "explanation": reply.explanation,
            }
        handoff = GraphDependencyHandoff(records=reply.records)
        validate_handoff(
            handoff,
            contract,
            draft=draft,
            context=context,
            scientific_context=scientific_context,
        )
    except (ValueError, TypeError, KeyError) as exc:
        return {
            **event,
            "accepted": False,
            "structural_valid": False,
            "attempted_records": raw,
            "error": str(exc)[:2000],
            "repair": (
                "Correct the reported declaration error; scope_guidance explains "
                "invalid targets when present. Also revise any initializer still "
                "causing the displayed conflict. "
                "Return the complete replacement, retaining unaffected records. "
                "Removing an invalid record alone does not establish acceptance."
                if feedback_policy == SCOPE_FEEDBACK
                else "Return decision=revise with the complete corrected "
                "record list, preserving valid siblings; or report "
                "needs_topology_revision if initialization alone cannot fix it."
            ),
        }
    gate = acceptance_feedback(
        draft,
        evidence,
        contract,
        handoff,
        previous_questions=questions,
    )
    return {
        **event,
        "accepted": not gate["blocked"],
        "structural_valid": True,
        "candidate": handoff.model_dump(mode="json"),
        "gate": gate,
        "explanation": reply.explanation,
    }


def _advance(state: dict, event: dict, initial: GraphDependencyHandoff | None):
    """Keep rejected candidates inspectable without exporting an accepted model."""
    if event.get("candidate"):
        old = [x["candidate"] for x in state["attempts"] if x.get("candidate")]
        if initial is not None:
            old.insert(0, initial.model_dump(mode="json"))
        repeated = sum(_semantic(x) == _semantic(event["candidate"]) for x in old)
        if repeated and not event["accepted"]:
            event["repeated_dependency_choices"] = repeated
            event["repair"] = (
                "This repeats a previously submitted dependency choice. The "
                "displayed contradiction remains; changing its explanation does "
                "not change the model. Revise the relevant rule or explicitly "
                "report that the fixed topology must change."
            )
        state["latest_candidate"] = event["candidate"]
        state["gate"] = event["gate"]
    elif state.get("gate") and not event.get("unavailable"):
        # Invalid syntax cannot erase the last verified conflict from feedback.
        event["gate"] = state["gate"]
        if state.get("latest_candidate"):
            event["last_valid_candidate"] = state["latest_candidate"]
    state["attempts"].append(event)
    if event.get("unavailable"):
        state["status"] = "unavailable"
    elif event.get("decision") == "needs_topology_revision":
        state["status"] = "needs_topology_revision"
    elif event["accepted"]:
        state["status"] = "complete"
        state["handoff"] = event["candidate"]


def _result(state, contract, client, feedback_policy=LEGACY_FEEDBACK):
    """No unresolved conflict can coexist with an accepted exported handoff."""
    gate = state.get("gate")
    accepted = state["status"] == "complete"
    compatibility = (
        "conflict"
        if gate and gate["blocked"]
        else "deferred"
        if gate and gate["status"] == "eligible_with_deferred_evidence"
        else "no_detected_conflict"
        if gate
        else "unassessed"
    )
    result = {
        "protocol": PROTOCOL,
        **(
            {"feedback_policy": feedback_policy}
            if feedback_policy != LEGACY_FEEDBACK
            else {}
        ),
        "identity": state["identity"],
        "status": state["status"],
        "contract": contract.model_dump(mode="json"),
        "structural_valid": bool(
            state["attempts"] and state["attempts"][-1]["structural_valid"]
        ),
        "accepted": accepted,
        "compatibility_status": compatibility,
        "conflict_count": gate["conflict_count"] if gate else 0,
        "deferred_count": len(gate["deferred_checks"]) if gate else 0,
        "handoff": state.get("handoff") if accepted else None,
        "latest_candidate": state.get("latest_candidate"),
        "assessment": gate["assessment"] if gate else None,
        "acceptance": gate,
        "attempts": state["attempts"],
        "physical_requests": len(client.records),
        "budget_charge": sum(r.get("budget_charge", 0) for r in client.records),
        "observed_total_tokens": sum(
            r.get("observed_total_tokens") or 0 for r in client.records
        ),
        "unmeasured_requests": sum(
            r.get("observed_total_tokens") is None for r in client.records
        ),
        "optimizer_calls": 0,
        "test_data_opened": False,
        "scientific_correctness_established": False,
        "automatic_followup": False,
    }
    return {**result, "result_sha256": content_hash(result)}


def _restore_cache(client: StagedTopologyClient) -> set[str]:
    """Retain all started requests, including one interrupted before event saving."""
    records = {}
    for path in sorted(client.directory.glob("*.json")):
        record = json.loads(path.read_text())
        request = record["request"]
        key = content_hash(request)
        if (
            path.stem != key
            or record.get("request_hash") != key
            or request.get("namespace") != client.namespace
            or request.get("settings") != client.settings.model_dump(mode="json")
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("initialization repair cached request provenance differs")
        records[key] = record
    if len(records) > client.settings.maximum_requests:
        raise ValueError("initialization repair cache exceeds request budget")
    client.records = list(records.values())
    return set(records)


def construct_graph_dependencies(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    client: StagedTopologyClient,
    directory: Path,
    *,
    evidence: TrainingCompatibilityEvidence,
    previous_questions: tuple[dict, ...] = (),
    initial_handoff: GraphDependencyHandoff | None = None,
    feedback_policy: str = LEGACY_FEEDBACK,
) -> dict:
    """Recheck each reply before acceptance, replaying cached work without refunds."""
    _check_inputs(
        draft, context, scientific_context, contract, evidence, initial_handoff
    )
    cached_keys = _restore_cache(client)
    seen_keys: set[str] = set()
    if any(not isinstance(question, dict) for question in previous_questions):
        raise ValueError("previous initialization questions must be records")
    identity = dependency_construction_identity(
        draft,
        context,
        scientific_context,
        contract,
        client,
        evidence=evidence,
        previous_questions=previous_questions,
        initial_handoff=initial_handoff,
        feedback_policy=feedback_policy,
    )
    path = directory / "state.json"
    saved = None
    if path.exists():
        saved = json.loads(path.read_text())
        digest = saved.pop("state_sha256", None)
        if digest != content_hash(saved) or saved.get("identity") != identity:
            raise ValueError("initialization repair checkpoint differs")
        if len(saved.get("attempts", [])) > client.settings.attempts_per_step:
            raise ValueError("initialization repair checkpoint exceeds budget")
    state = {"identity": identity, "status": "running", "attempts": []}
    if initial_handoff is not None:
        state["gate"] = acceptance_feedback(
            draft,
            evidence,
            contract,
            initial_handoff,
            previous_questions=previous_questions,
        )

    def save():
        atomic_json(path, {**state, "state_sha256": content_hash(state)})

    def call(attempt):
        record = client.call(
            system=system_prompt(feedback_policy),
            user=dependency_request(
                draft,
                context,
                scientific_context,
                contract,
                evidence=evidence,
                previous=state["attempts"][-1] if state["attempts"] else None,
                previous_questions=previous_questions,
                initial_handoff=initial_handoff,
                construction_identity=identity,
                feedback_policy=feedback_policy,
            ),
            response_model=GraphDependencyReply,
            step="initialization_dependency_repair",
            attempt=attempt,
        )
        # ConstructionClient already deduplicates; the base synthetic/replay
        # client need not. Keep the same physical accounting for both.
        client.records = list({r["request_hash"]: r for r in client.records}.values())
        seen_keys.add(record["request_hash"])
        return record

    # Reconstruct all outcomes and feedback from the exact expected cached calls.
    # A missing/mismatched cache must not cause a fresh network request on replay.
    for index, recorded in enumerate(saved.get("attempts", []) if saved else []):
        if state["status"] != "running":
            raise ValueError("initialization repair checkpoint continues after stop")
        can_start = client.can_start
        client.can_start = lambda: False
        try:
            record = call(index)
        except DeferredCall as exc:
            raise ValueError("initialization repair cached request differs") from exc
        finally:
            client.can_start = can_start
        event = _evaluate(
            record,
            draft,
            context,
            scientific_context,
            contract,
            evidence,
            previous_questions,
            feedback_policy=feedback_policy,
        )
        _advance(state, event, initial_handoff)
        if event != recorded:
            raise ValueError("initialization repair checkpoint event differs")
    if saved:
        if (
            state["status"] == "running"
            and len(state["attempts"]) == client.settings.attempts_per_step
            and saved.get("status") == "repair_exhausted"
        ):
            state["status"] = "repair_exhausted"
        comparison = {key: value for key, value in saved.items() if key != "result"}
        if comparison != state:
            raise ValueError("initialization repair reconstructed checkpoint differs")
        if "result" in saved:
            if cached_keys - seen_keys:
                raise ValueError(
                    "initialization repair checkpoint has unbound cached requests"
                )
            expected = _result(state, contract, client, feedback_policy)
            if saved["result"] != expected:
                raise ValueError("initialization repair result differs")
            atomic_json(directory / "result.json", expected)
            return expected
    save()
    for attempt in range(len(state["attempts"]), client.settings.attempts_per_step):
        if state["status"] != "running":
            break
        if cached_keys - seen_keys:
            # A physical response may precede its event checkpoint. Require the
            # exact expected cache before allowing a further generation request.
            can_start = client.can_start
            client.can_start = lambda: False
            try:
                record = call(attempt)
            except DeferredCall as exc:
                raise ValueError(
                    "initialization repair unbound cached request"
                ) from exc
            finally:
                client.can_start = can_start
        else:
            record = call(attempt)
        event = _evaluate(
            record,
            draft,
            context,
            scientific_context,
            contract,
            evidence,
            previous_questions,
            feedback_policy=feedback_policy,
        )
        _advance(state, event, initial_handoff)
        save()
    if state["status"] == "running":
        state["status"] = "repair_exhausted"
    if cached_keys - seen_keys:
        raise ValueError("initialization repair checkpoint has unbound cached requests")
    result = _result(state, contract, client, feedback_policy)
    state["result"] = result
    save()
    atomic_json(directory / "result.json", result)
    return result
