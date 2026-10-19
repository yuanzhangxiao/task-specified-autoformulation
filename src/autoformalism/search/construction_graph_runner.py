"""Bounded initialization-dependency construction at the topology boundary.

The runtime fixes information access; the proposer chooses independent roots and
acyclic dependencies. Analytic formulas are constructed in a later stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.search.causal_initialization import _restore_attempt_costs
from autoformalism.search.construction_graph_evidence import (
    inspect_graph_initialization,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    validate_handoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.construction_readable import model_view
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "initialization-dependency-construction-1"
SYSTEM = """Declare how the latent differential states will be initialized.
This is the initialization part of topology construction. Choose dependencies
now; analytic map formulas and coefficients belong to the later interaction stage.
The displayed dynamics are context, not a prescription to copy their arguments
into the initializer. Return exactly one record per listed latent state.

Use value/shared_training for one unknown initial constant learned on training
and reused across trajectories. Use value/trajectory only when the runtime
authorizes observation_prefix: each trajectory has its own independent root,
inferred on new trajectories using only the permitted early observations while
shared parameters stay frozen. The fitting experiments choose and freeze the
window policy; you do not choose its duration. This option does not establish
that the window identifies the root or that the dynamics are scientifically correct.

Use map for a shared function of the listed state_initials and public_initials.
A state_initial is the value of a differential state at t0; a public_initial is
only the permitted first reading, initial input, covariate or time. These are
different namespaces. Dependent initial values are calculated, never independently
fitted. For example z1(0)=alpha and z2(0)=g(z1(0),y(0)) require a value record for
z1 and a map record for z2 with state_initials=[z1], public_initials=[y]. All map
coefficients are shared. Initial dependencies must be acyclic. Do not use future
measurements, trajectory identifiers, algebraic variables or process names.

Observed differential states keep their own initial measurements; do not return
records for them. Known values are runtime-owned and may only be retained as known.
Explain the scientific preparation hypothesis briefly. Do not provide formulas,
numerical values, optimizer guesses, new variables, dynamics edits or stage flags.
On repair return the complete replacement record list, retaining valid records.
"""


def dependency_construction_identity(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    client: StagedTopologyClient,
    *,
    evidence: TrainingCompatibilityEvidence | None = None,
    previous_questions: tuple[dict, ...] = (),
) -> str:
    """Bind the full construction inputs identically for preview and execution."""
    from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

    return content_hash(
        {
            "protocol": PROTOCOL,
            "draft": draft.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "scientific_context": scientific_context,
            "contract": contract.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json") if evidence else None,
            "previous_questions": previous_questions,
            "system": SYSTEM,
            "schema": GraphDependencyHandoff.model_json_schema(),
            "runtime": runtime_source_hash(),
            "namespace": client.namespace,
            # The launcher pins serving provenance separately. Its loopback URL
            # is operational: a new allocation may assign another local port.
            "settings": client.settings.model_dump(mode="json"),
            "seed": client.seed,
            "request_policy": client.request_policy(),
        }
    )


def _latest_question_views(previous_questions: tuple[dict, ...]) -> list[dict]:
    """Show the latest diagnostic per target without mutating retained history."""
    latest = {
        content_hash([question.get("property"), question.get("target")]): question
        for question in previous_questions
    }
    fields = {
        "question_id",
        "property",
        "target",
        "scope",
        "witnesses",
        "assumptions",
        "unavailable",
        "reason",
        "repair_status",
        "evidence_sha256",
        "status",
        "conditioning",
        "question",
        "model_relation",
        "paired_recorded_values",
    }
    return [
        {key: value for key, value in question.items() if key in fields}
        for question in latest.values()
    ]


def dependency_request(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    *,
    previous: dict | None = None,
    previous_questions: tuple[dict, ...] = (),
    construction_identity: str | None = None,
) -> str:
    """Render one compact stateless request, without repeating the full dataset."""
    observed = set(contract.observed_initial_channels)
    latent = sorted(
        v.name
        for v in draft.variables
        if v.definition == "differential" and v.name not in observed
    )
    questions = _latest_question_views(previous_questions)
    return "Initialization dependencies\n" + json.dumps(
        {
            "construction_identity": construction_identity,
            "public_context": scientific_context,
            "model_view": model_view(draft),
            "latent_states_requiring_records": latent,
            "runtime_initialization_contract": contract.model_dump(mode="json"),
            "retained_conditional_evidence": {
                "interpretation": "These witnesses were recorded under earlier "
                "initializer assumptions. They have not yet been rechecked under "
                "your proposed graph. Retain their uncertainty and assumptions; "
                "a changed dependency declaration alone is not a scientific repair.",
                "retained_history_count": len(previous_questions),
                "display_count": len(questions),
                "questions": questions,
            },
            "feedback": previous,
        },
        ensure_ascii=False,
    )


def _validate_checkpoint(state, client, contract, draft, context, scientific_context):
    """Bind terminal choices to cached accepted replies, not just editable hashes."""
    attempts = state["attempts"]
    if (
        state["status"]
        not in {"running", "complete", "unavailable", "repair_exhausted"}
        or len(attempts) > client.settings.attempts_per_step
        or [event["attempt"] for event in attempts] != list(range(len(attempts)))
        or len({event["request_hash"] for event in attempts}) != len(attempts)
    ):
        raise ValueError("initialization dependency checkpoint attempts differ")
    accepted = [event for event in attempts if event.get("accepted")]
    if bool(accepted) != (state.get("handoff") is not None) or len(accepted) > 1:
        raise ValueError("initialization dependency checkpoint handoff differs")
    if accepted:
        records = {record["request_hash"]: record for record in client.records}
        handoff = GraphDependencyHandoff.model_validate(
            visible_response(records[accepted[0]["request_hash"]])
        )
        validate_handoff(
            handoff,
            contract,
            draft=draft,
            context=context,
            scientific_context=scientific_context,
        )
        if (
            handoff.model_dump(mode="json") != state["handoff"]
            or state["status"] != "complete"
        ):
            raise ValueError(
                "initialization dependency checkpoint accepted reply differs"
            )
    elif state["status"] == "complete":
        raise ValueError(
            "initialization dependency complete checkpoint has no accepted reply"
        )
    if "result" in state:
        result = state["result"]
        unsigned = {
            key: value for key, value in result.items() if key != "result_sha256"
        }
        expected = {
            "protocol": PROTOCOL,
            "identity": state["identity"],
            "status": state["status"],
            "contract": contract.model_dump(mode="json"),
            "handoff": state.get("handoff"),
            "attempts": attempts,
        }
        if result.get("result_sha256") != content_hash(unsigned) or any(
            result.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("initialization dependency result differs from checkpoint")


def construct_graph_dependencies(
    draft: Draft,
    context: ValidationContext,
    scientific_context: str,
    contract: GraphConstructionContract,
    client: StagedTopologyClient,
    directory: Path,
    *,
    evidence: TrainingCompatibilityEvidence | None = None,
    previous_questions: tuple[dict, ...] = (),
) -> dict:
    """Validate whole replies atomically and retain costs across deterministic resume.

    Structural completion does not clear conditional scientific conflicts. Those
    remain in the exported evidence for the subsequent model review and fitting.
    """
    from autoformalism.search.construction_initialization_graph import validate_contract

    validate_contract(
        contract, draft=draft, context=context, scientific_context=scientific_context
    )
    if evidence is not None:
        validate_evidence(evidence)
        if evidence.context != context:
            raise ValueError("initialization evidence context differs")
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
    )
    path = directory / "state.json"
    state = {"identity": identity, "status": "running", "attempts": []}
    if path.exists():
        saved = json.loads(path.read_text())
        digest = saved.pop("state_sha256", None)
        if digest != content_hash(saved) or saved.get("identity") != identity:
            raise ValueError("initialization dependency checkpoint differs")
        state = saved

    def save() -> None:
        atomic_json(path, {**state, "state_sha256": content_hash(state)})

    save()
    _restore_attempt_costs(client, state["attempts"])
    _validate_checkpoint(state, client, contract, draft, context, scientific_context)
    if "result" in state:
        atomic_json(directory / "result.json", state["result"])
        return state["result"]
    attempts = range(len(state["attempts"]), client.settings.attempts_per_step)
    for attempt in attempts if state["status"] == "running" else ():
        previous = state["attempts"][-1] if state["attempts"] else None
        record = client.call(
            system=SYSTEM,
            user=dependency_request(
                draft,
                context,
                scientific_context,
                contract,
                previous=previous,
                previous_questions=previous_questions,
                construction_identity=identity,
            ),
            response_model=GraphDependencyHandoff,
            step="initialization_dependencies",
            attempt=attempt,
        )
        event = {"attempt": attempt, "request_hash": record["request_hash"]}
        try:
            raw = visible_response(record)
        except (ValueError, TypeError, KeyError) as exc:
            event.update(accepted=False, error=str(exc)[:2000], unavailable=True)
            state["attempts"].append(event)
            state["status"] = "unavailable"
            save()
            break
        try:
            handoff = GraphDependencyHandoff.model_validate(raw)
            validate_handoff(
                handoff,
                contract,
                draft=draft,
                context=context,
                scientific_context=scientific_context,
            )
        except (ValueError, TypeError, KeyError) as exc:
            event.update(
                accepted=False,
                error=str(exc)[:2000],
                attempted_records=raw,
                repair=(
                    "Return the complete corrected record list; "
                    "preserve valid siblings."
                ),
            )
        else:
            event["accepted"] = True
            state["handoff"] = handoff.model_dump(mode="json")
            state["status"] = "complete"
        state["attempts"].append(event)
        save()
        if event["accepted"]:
            break
    if state["status"] == "running":
        state["status"] = "repair_exhausted"
    assessment = None
    if state.get("handoff") is not None and evidence is not None:
        assessment = inspect_graph_initialization(
            draft,
            evidence,
            contract,
            GraphDependencyHandoff.model_validate(state["handoff"]),
            previous_questions=previous_questions,
        )
    result = {
        "protocol": PROTOCOL,
        "identity": identity,
        "status": state["status"],
        "contract": contract.model_dump(mode="json"),
        "handoff": state.get("handoff"),
        "assessment": assessment,
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
    result["result_sha256"] = content_hash(result)
    state["result"] = result
    save()
    atomic_json(directory / "result.json", result)
    return result
