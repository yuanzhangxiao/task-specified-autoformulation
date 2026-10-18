"""Topology-boundary initializer choices, kept separate from interaction formulas."""

from __future__ import annotations

from copy import deepcopy

from pydantic import Field, create_model

from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.candidate import StateKind
from autoformalism.schemas.staged_topology import EquationDefinition
from autoformalism.search import construction_actionable_repair as previous
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import signed_processes
from autoformalism.search import topology_initialization as initial
from autoformalism.search.causal_initialization import validate_dependency_handoff
from autoformalism.staged_topology import content_hash, lower_topology

POLICY = "topology-initialization-repair-1"
POLICY_V2 = "topology-initialization-repair-2"
POLICY_V3 = "topology-initialization-repair-3"
POLICY_V4 = "topology-initialization-repair-4"
SYSTEM = previous.SYSTEM
Draft = previous.Draft


def system_for(packet: dict) -> str:
    """Keep historical system text stable; opt in through the frozen policy."""
    if packet.get("prompt_policy") in {POLICY_V3, POLICY_V4}:
        from autoformalism.search.construction_initialization_focus import (
            SYSTEM as focused,
        )

        return focused
    return SYSTEM


INITIALIZATION = """Complete the topology of initial conditions for differential
states. For each unresolved state choose shared_parameter (one shared unknown
initial value), or common_map with the public quantities whose initial readings
determine it. unknown keeps a choice unresolved. Formulas and coefficients are
chosen later, in interaction construction. Algebraic quantities are evaluated
from the initial state and have no independent initial value.

Choose this information from the experiment setup and scientific reasoning.
ODE arguments do not determine initializer arguments: z'=u may start at z(0)=0,
whereas a justified equilibrium assumption may imply a map of initial inputs.
Do not copy RHS arguments automatically. Use only permitted recorded initial
quantities; no future observations, trajectory IDs or unlisted prehistory.
Inspect training observations and the matched-trajectory feedback when choosing.

Return a complete initialization record, retaining intended earlier choices.
Known runtime initial rules are fixed. stage_complete requests rechecking;
unknown or omitted latent-state choices remain unfinished. This call chooses
initial dependencies only. Later repair can revise topology and initialization
together; neither names nor formulas are inferred by the runtime."""


class InitializationReply(StrictSchema):
    """A complete dependency handoff; no formulas or topology edits in this call."""

    stage_complete: bool
    initialization: initial.InitializerDependencyHandoff


class DraftPatch(previous.DraftPatch):
    """Null retains the handoff; a record replaces it as part of the transaction."""

    initialization: initial.InitializerDependencyHandoff | None = None


def response_model(stage: str):
    """Keep old schemas unchanged and extend only this policy's repair contract."""
    if stage == "initialization":
        return InitializationReply
    if stage != "repair":
        raise ValueError("initializer handoff follows completed topology declarations")
    return create_model(
        "InitializedTopologyRepairReply",
        __base__=previous.response_model("repair"),
        initialization=(
            initial.InitializerDependencyHandoff | None,
            Field(default=None),
        ),
    )


def _plain(patch):
    return previous.DraftPatch.model_validate(
        patch.model_dump(mode="json", exclude={"initialization"})
    )


def normalize_reply(brief, raw, *, stage="repair", **kwargs):
    """Separate declaration records from the existing topology edit schema."""
    if stage == "initialization":
        reply = InitializationReply.model_validate(raw)
        patch = DraftPatch(
            stage_complete=reply.stage_complete, initialization=reply.initialization
        )
        return patch.model_dump(mode="json"), []
    reply = response_model(stage).model_validate(raw)
    legacy = reply.model_dump(mode="json", exclude={"initialization"})
    if reply.initialization is not None:
        # Handle normalization may simulate the topology-only transaction. Permit
        # an unchanged topology when the same reply changes initialization.
        legacy["stage_complete"] = True
    value, log = previous.normalize_reply(brief, legacy, **kwargs)
    value["stage_complete"] = reply.stage_complete
    value["initialization"] = (
        reply.initialization.model_dump(mode="json") if reply.initialization else None
    )
    return value, log


def apply_patch(brief, draft, patch):
    plain = _plain(patch)
    if patch.initialization is not None:
        # The legacy topology view cannot see an initializer-only edit. This flag
        # bypasses only its no-op rejection, never the actual completion gate.
        plain = plain.model_copy(update={"stage_complete": True})
    return previous.apply_patch(brief, draft, plain)


def normalize_consumers(draft, patch, previous_draft):
    return previous.normalize_consumers(draft, _plain(patch), previous_draft)


def handoff_status(draft, evidence, contract, handoff):
    """Validate records without selecting missing scientific choices."""
    request = initial.request(draft, evidence, contract, handoff)
    return {
        "complete": not request["unresolved_states"],
        "unresolved_states": request["unresolved_states"],
        "fixed_runtime_rules": request["fixed_runtime_rules"],
        "retained_handoff": handoff.model_dump(mode="json"),
    }


def changed_handoff(draft, evidence, contract, retained, patch):
    """Validate against the proposed topology before either edit is committed."""
    candidate = retained if patch.initialization is None else patch.initialization
    handoff_status(draft, evidence, contract, candidate)
    return candidate


def qualify_completion(state, status):
    """Initializer readiness is separate from conditional scientific assessments."""
    return {
        **state,
        "initialization_complete": status["complete"],
        "initialization_unresolved_states": status["unresolved_states"],
        "completion_allowed": state["completion_allowed"] and status["complete"],
    }


def payload(stage, *args, evidence, initial_contract, handoff, policy=POLICY, **kwargs):
    """Present equations, initial conditions and existing conditional witnesses once."""
    if policy not in {POLICY, POLICY_V2, POLICY_V3, POLICY_V4}:
        raise ValueError("unknown initialization repair policy")
    kwargs["policy"] = previous.POLICY_V2
    packet = previous.payload("repair", *args, **kwargs)
    draft = args[4]
    info = initial.request(draft, evidence, initial_contract, handoff)
    status = handoff_status(draft, evidence, initial_contract, handoff)
    info.pop("response_schema")
    info.pop("response_template")
    info["diagnostic_conditioning"] = {
        "unresolved_states": status["unresolved_states"],
        "interpretation": "For an unresolved state, a diagnostic witness may "
        "temporarily use a common map of ALL runtime-permitted initial readings. "
        "This is maximal-family conditioning for a conditional comparison, not "
        "a retained or proposed initializer and not a request to copy all these "
        "arguments. Actual choices appear only in fixed_runtime_rules and "
        "retained_handoff. Once a choice is declared, checks use its own "
        "initial information set.",
    }
    packet["initialization"] = info
    packet["stage"] = stage
    packet["feedback"]["initialization_status"] = status
    packet["stage_instructions"] = (
        INITIALIZATION
        if stage == "initialization"
        else previous.stage_text("repair")
        + "\nRevise initialization dependencies when scientifically needed. "
        "initialization=null retains existing records; a non-null record replaces "
        "the whole handoff. Keep every intended state record. When removing a state "
        "or changing it to algebraic, remove its initializer record in the same "
        "reply. Newly introduced latent states need an initializer choice before "
        "completion. Initialization and topology edits commit together or neither "
        "commits. Do not copy ODE arguments to initializer arguments automatically."
    )
    if stage == "initialization":
        packet["response_template"] = {
            "stage_complete": False,
            "initialization": handoff.model_dump(mode="json"),
        }
        packet["editing_rules"] = "Return only stage_complete and initialization."
        packet["compatibility_questions_scope"] = (
            "The observations and compatibility checks inform initial dependencies. "
            "Topology edits and compatibility decisions are requested in the next "
            "repair stage; do not answer them with extra fields in this reply."
        )
    else:
        packet["response_template"] = response_model(stage)(
            stage_complete=False
        ).model_dump(mode="json")
    work = packet["feedback"]["repair_work"]
    work["initialization_dependencies"] = status
    if not status["complete"]:
        work["mode"] = "repair_required"
        work["initialization_action"] = (
            "Choose a permitted shared initial value or initial-measurement map for "
            "each listed unresolved state. An unknown initializer is not a repair."
        )
    if stage == "initialization":
        # Do not inherit action instructions for fields absent from this schema.
        # Keep the numerical witnesses and assumptions as decision context.
        packet = {
            key: packet[key]
            for key in (
                "stage",
                "stage_instructions",
                "public_brief",
                "model_view",
                "training_highlights",
                "initialization",
                "previous_edit",
                "editing_rules",
                "response_template",
            )
            if key in packet
        } | {
            "feedback": {
                "initialization_status": status,
                "repair_work": {
                    "mode": "repair_required"
                    if not status["complete"]
                    else "recheck_completion",
                    "initialization_dependencies": status,
                    "instruction": "Choose initial dependencies only now. Topology "
                    "edits and scientific compatibility responses are revisited "
                    "in the following joint repair stage. Return only the schema "
                    "fields; do not change topology through an initializer record.",
                },
                "matched_training_observations": [
                    {
                        key: deepcopy(q[key])
                        for key in (
                            "question_id",
                            "target",
                            "property",
                            "scope",
                            "verified",
                            "assumptions",
                            "witnesses",
                            "conditioning",
                            "repair_status",
                        )
                        if key in q
                    }
                    for q in kwargs.get("compatibility_questions", ())
                ],
                "interpretation": "These conditional comparisons inform your "
                "choice of initial dependencies. Matching all permitted initial "
                "readings cannot be distinguished by choosing another shared map "
                "of those same readings. Missing choices still require an explicit "
                "record. Other model conflicts are handled in joint repair.",
            }
        }
    if policy in {POLICY_V2, POLICY_V3, POLICY_V4}:
        from autoformalism.search.construction_feedback_projection import project

        packet["stage_instructions"] += (
            "\nOnly editable_states need proposer initializer records. Fixed observed "
            "and measured initial rules are already supplied by the runtime; do not "
            "replace them. If a reply is rejected, no edits commit. Retain valid "
            "intended editable records when resubmitting the corrected transaction."
        )
        last = args[6] if len(args) > 6 else None
        attempted = (last or {}).get("rejected_reply")
        if isinstance(attempted, dict) and attempted.get("initialization"):
            try:
                parsed = initial.InitializerDependencyHandoff.model_validate(
                    attempted["initialization"]
                )
                receipt = initial.rejected_handoff_feedback(
                    draft, evidence, initial_contract, parsed
                )
            except (ValueError, TypeError) as exc:
                receipt = {
                    "transaction_committed": False,
                    "schema_error": str(exc),
                    "instruction": "Correct the initialization record using the "
                    "displayed schema. No rejected edits were committed.",
                }
            packet["feedback"]["rejected_initialization"] = receipt
        return project(packet, draft)
    return packet


def interaction_source(brief, draft, check, evidence, contract, handoff):
    """Export a checked source carrying the exact initializer dependency contract."""
    status = handoff_status(draft, evidence, contract, handoff)
    if not check["eligible"] or not status["complete"]:
        return None
    records = {record.state: record for record in handoff.records}
    origins = {name: {"origin": "proposer"} for name in records}
    for name, rule in status["fixed_runtime_rules"].items():
        if rule["mode"] in {"shared_parameter", "common_map"}:
            records[name] = initial.InitializerDependencyRecord.model_validate(
                {"state": name, "rule": rule}
            )
        if name in records:
            origins[name] = {"origin": "runtime", "evidence_sha256": evidence.sha256}
    effective = initial.InitializerDependencyHandoff(records=tuple(records.values()))
    inventory = tuple(ledger.inventory(brief, draft))
    equations = tuple(
        EquationDefinition.model_validate(row) for row in check["equations"]
    )
    topology, aliases = lower_topology(brief, inventory, equations, evidence.context)
    try:
        validate_dependency_handoff(
            evidence.context,
            tuple(state.name for state in topology.states),
            {s.name for s in topology.states if s.kind is StateKind.OBSERVED},
            contract,
            effective,
            state_aliases=aliases,
        )
    except ValueError:
        # A diagnostic initial rule may lack an executable consumer (for example
        # an observed-initial auxiliary). Do not invent a replacement policy.
        return None
    return {
        "complete_topology": True,
        "inventory": [v.model_dump(mode="json") for v in inventory],
        "equations": deepcopy(check["equations"]),
        "topology": deepcopy(check["topology"]),
        "shared_process_contract": {
            "protocol": signed_processes.POLICY,
            "bindings": deepcopy(check["shared_process_bindings"]),
        },
        "initializer_handoff": effective.model_dump(mode="json"),
        "proposer_initializer_handoff": handoff.model_dump(mode="json"),
        "initializer_handoff_origins": origins,
        "initial_information_contract": contract.model_dump(mode="json"),
        "source_draft_sha256": content_hash(draft.model_dump(mode="json")),
        "training_evidence_sha256": evidence.sha256,
        "scientific_correctness": "unverified",
    }
