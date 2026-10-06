"""Compare construction schedules on one ledger, public context and budget policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from autoformalism.expressions import ValidationContext
from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash

Policy = Literal["separate", "joint_fixed", "joint_adaptive"]
POLICIES = ("separate", "joint_fixed", "joint_adaptive")

SYSTEM = """Construct a scientific model's VARIABLES AND TOPOLOGY, not functions.
The unchanged public scientific task is authoritative. Work on the displayed
stage using the structured edit schema. The runtime maintains the current draft.

PUBLIC SOURCES AND GENERATED VARIABLES
Public non-target input/observation channels, covariates and time are available
RHS sources; no supplied/unused declarations are needed. Actual use is determined
by your equation skeletons. Their physical meanings and permitted availability
remain those in the public task. Initial readings are boundary information, not
ongoing inputs; diagnostic thresholds are not physical forcing merely because
they are in the catalog. Covariates can parameterize ongoing laws when justified.
Every public target must be generated, never read from its future measurements.
Choose differential (state with derivative equation) or algebraic (instantaneous
readout/process) for each generated variable. Supplied observations can optionally
be modeled; then RHS references denote your generated quantity, not the supplied
trajectory. Do not invent observed channels. Fitted coefficients are not variables.
For example, a later law k*x has sources=["x"], NOT ["k","x"]. Declare k in
the later function stage, not as an algebraic variable needing its own equation.
Do not create chains of parameter aliases or invent an observed constant channel.

MEMORY AND RETURN TO BASELINE
A differential state already carries history; self-dependence is NOT mandatory.
dX/dt=u represents accumulated input, whereas dX/dt=a*u-b*X can represent fading
memory. If a mechanism should relax after forcing ends, include the dependencies
needed for its restoring dynamics, directly on X or through coupled states.
Choose accumulation, relaxation or another memory mechanism from the public task;
do not add a decay term mechanically. Explain the intended behavior. These are
scientific choices, not a new runtime requirement that every state depends on itself.

RELATIONSHIPS BEFORE INDIVIDUAL EQUATIONS
Consider all targets together, their necessary states and shared mechanisms.
Bind each displayed dynamic-memory requirement to your chosen differential
mediator(s), distinct from that requirement's driver and target. Optional memory
bindings may also refer to other EXISTING public requirements: not requiring
memory does not forbid it. Such a binding may include a target state itself.
Use mechanism_bindings=[
  {"requirement_id":"<listed ID>","memory_states":["<chosen state>"]}].
The structured binding_context gives exact IDs, formats and type-eligible choices
on every request. Choose the scientific assignment; the runtime never chooses it.
Use only allowed_memory_bindings IDs, never public graph check IDs. The runtime
logs and discards memory entries that name automatic graph checks; it does not
choose a replacement requirement or state. feedback_bindings is a separate
field ONLY for targets listed in binding_context.feedback_bindings.applicable_targets:
{"target":"y","states":["volume"]}. If that list is empty, leave this field empty.
It is NOT a substitute for mechanism_bindings, even when the same state is involved.
It declares the actual storage/energy realization of that readout, not an arbitrary
upstream cause. Differential targets are already their own coordinate and need
no such binding. Bindings express
your intended scientific assignment; complete equations must establish the paths.
For each optional named process, choose its drivers, scientific meaning and signed
consumers together. The runtime defines it once and inserts every declared use.
A SHARED process has at least two DISTINCT generated-equation consumers. A process
with one consumer is LOCAL, even if its name or formula resembles another process.
Prefer an ordinary equation term for a local contribution. An explicitly named
local law remains allowed, but is not counted as shared. Do not invent consumers
to make a process shared, and do not duplicate one consumer with multiple signs.
An INTERNAL pairwise transfer uses kind=transfer with two opposite signed consumers.
An EXTERNAL source or sink has only its modeled consumer; use kind=influence.
Never add a second consumer solely to satisfy the transfer format. An influence
may be local or shared. When repairing same-sign transfer consumers, consider
whether the kind should be influence; do not change scientifically intended signs
merely to keep the transfer label. Empty processes is legitimate.
One process is one law,
not a list of unrelated contributions. Conversion is null if unknown, or a positive
fixed factor such as 1/area using public covariates and multiplication/division.
For example: {"target":"x","sign":"positive","conversion":null}; null is
the unquoted JSON value, not a string and not an unknown coefficient name.
Never put an unknown kinetic coefficient (k_abs, 1/tau, etc.) in conversion.
Use null for an unknown conversion; the runtime handles fitted magnitudes later.
The consumer sign, conversion and fitted magnitude are outside the one shared law.
Opposite signs or shared syntax alone do not prove conservation.

EQUATION SKELETONS
Group jointly interacting variables in each term's sources. Select the outer
weight sign: positive, negative or unrestricted. A fixed outer sign does not
assert global monotonicity/nonnegativity of the later function. Self-dependence
can represent decay, relaxation or feedback when scientifically justified.
Prefer ordinary contributions in equations. A singleton sources=["P"] with an
explicit positive/negative outer sign declares use of that SAME named law. Runtime
records a new consumer with conversion=null (unknown), or includes an existing
matching use exactly once, preserving its conversion. Do not delete a physical
effect to fix a representation error. For changed signs/conversions, mixed sources,
or conflicting declarations, explicitly revise the process and affected equations.
Runtime never infers a conversion, changes kind, or chooses a sign.
The displayed process uses are
ALREADY INCLUDED; do not repeat them or expand their drivers into another term
for the same effect. Return terms=[] when process uses provide the whole equation.
equation_views
shows those uses even when ordinary_rhs_declared=false. A missing ordinary-RHS
declaration is NOT an empty equation. If runtime flags possible overlapping
contributions, clarify whether they are the same physical effect. Remove the
ordinary repetition with an equation replacement if appropriate. If they are
distinct effects, give overlap_confirmations=[{"overlap_id":"<displayed hash>",
"scientific_distinction":"<why both effects are needed>"}]. This is your scientific
decision; identical drivers do not prove duplication. Confirmation applies only
to the displayed declarations; edits can require a new clarification.
Do not emit functions, coefficients, initializers or fitted values at this stage.

EDITS AND PENDING WORK
Each variables/processes/bindings entry replaces that named declaration; omitted
entries survive. Each equations entry REPLACES the complete ordinary RHS of its
named LHS, not an addition to its old terms. Remove entries only with explicit
remove_* lists. Empty lists do not clear prior declarations. Binding removal uses
requirement_id for remove_bindings and target for remove_feedback_bindings;
binding_context.existing_binding_removals shows exact edits. Changes to a process
propagate automatically to all its uses;
the next request displays the rebuilt draft. You need not repeat unchanged items.
Never put the same entry in both a replacement list and its remove_* list. To
revise x, supply its replacement and omit x from remove_*; to delete x, list its
name in remove_* and omit its replacement. Coordinate affected equation edits.
Forward references are allowed: an undeclared generated RHS name or an undefined
equation is pending work, not an instruction for the runtime to guess its meaning.
Resolve all such names and equations before finishing. All generated variables
need equations; algebraic loops are unsupported, differential feedback is allowed.
Explanations help communicate scientific meaning; they are not machine proofs.

The schedule shown in stage_instructions controls the current editing scope.
stage_complete=true ends the CURRENT STAGE, not merely the selected equation.
The full brief, current declarations, assembled contributions and pending work
are repeated on every request. last_edit_result labels accepted edits separately.
A rejected_reply is a failed attempt, NOT a model
to copy. Use its separately labeled error to correct the draft. Global structural
feedback is supplied only after the initial construction ends. Do not claim
scientific correctness merely because those finite structural checks pass.
An unavailable graph check requires fixing compilation first; it is not a missing
path verdict. Unresolved public predicates are not scientific passes. Topology
specifies dependencies; do not defer a known missing dependency to function writing.
When public_graph_contract is supplied, its reviewed public interpretations are
part of this prospective construction contract and appear on EVERY request.
target_feedback requires a cycle through the differential target itself, or
through EACH explicitly declared dynamic coordinate of its algebraic readout.
The readout path cannot pass through another differential state. This stops an
unrelated upstream loop from standing in for local storage feedback. Physical
identity still rests on your declaration, not name matching.
Legacy dynamic_feedback means a genuine dependency cycle through a differential state
that affects the target, directly or through generated readouts. It does NOT
require the target's literal name on a RHS, nor a self-loop on every memory state.
forbidden_path excludes indirect as well as direct influence. These checks run
after the whole graph compiles, before functions. Resolve their concrete failures
within overall repair; do not invent scientific bindings merely to satisfy a check.
Deferred scientific checks are not passes and do not ask you to claim proof.

RESPONSE DELIVERY
Return ONE complete JSON object matching response_template. Keep every listed key,
using [] for lists with no edits and a boolean stage_complete. End the JSON object
after stage_complete; do not stop after equations or emit trailing whitespace.
Prefer omitting unchanged declarations by using empty edit lists. Exact repeats
are harmless; actual edits outside this stage's scope are rejected explicitly.
"""


def validate_scope(
    policy: Policy,
    stage: str,
    focus: str | None,
    patch: ledger.DraftPatch,
    draft: ledger.Draft | None = None,
) -> None:
    """Check effective scientific edits, not harmless repeated declarations."""
    if stage == "repair":
        return
    current = draft or ledger.Draft()
    variables = {v.name: v.definition for v in current.variables}
    changed_variables = [
        v.name for v in patch.variables if variables.get(v.name) != v.definition
    ]
    equations = {e.name: e for e in current.equations}
    changed_equations = [e.name for e in patch.equations if equations.get(e.name) != e]
    if stage == "variables" and (
        patch.equations
        or patch.processes
        or patch.remove_equations
        or patch.remove_processes
    ):
        raise ValueError(
            "variable stage permits variable and memory-binding edits only"
        )
    if stage == "relationships" and (patch.equations or patch.remove_equations):
        raise ValueError("relationship planning precedes equation skeletons")
    if (
        policy == "separate"
        and stage != "variables"
        and (changed_variables or patch.remove_variables)
    ):
        raise ValueError(
            "separate policy fixes the variable inventory until overall repair; "
            f"new/type-changed={changed_variables}, "
            f"removed={list(patch.remove_variables)}. "
            "Unchanged variables may be omitted or repeated."
        )
    if (
        stage == "equations"
        and policy != "joint_adaptive"
        and (patch.remove_equations or any(n != focus for n in changed_equations))
    ):
        raise ValueError(
            f"fixed schedule requests only equation {focus}; "
            f"other changed equations={[n for n in changed_equations if n != focus]}. "
            "Previously accepted unchanged equations may be omitted or repeated."
        )


def instructions(policy: Policy, stage: str, focus: str | None) -> str:
    """State the selected work unit while preserving the same scientific context."""
    if stage == "variables":
        return (
            "Declare necessary generated variables, public targets first, and memory "
            "bindings. You may declare several variables per reply. No topology yet. "
            "Finish this stage with stage_complete=true when the inventory is ready."
        )
    if stage == "relationships":
        return (
            "Plan relationships across all targets: memory bindings and optional "
            "shared processes. Do not define ordinary equations yet. "
            + (
                "The generated-variable inventory is fixed. "
                if policy == "separate"
                else "Declare the preliminary generated variables needed by this plan. "
            )
            + "Finish relationship planning with stage_complete=true."
        )
    if stage == "repair":
        return (
            "Repair structural failures and answer contribution clarification requests "
            "with explicit coordinated edits. "
            "All variable, equation, process and binding edits are allowed. Preserve "
            "unaffected declarations. Inspect the rebuilt model before further edits. "
            "Set stage_complete=true when the whole topology is ready for checking."
        )
    if policy == "joint_adaptive":
        return (
            "Choose the order and number of related variables/equations to address "
            "in this reply. Add or amend declarations together as scientifically "
            "useful. Use pending work to finish all targets and dependencies. "
            "stage_complete=true means the WHOLE topology draft is ready."
        )
    return (
        f"Construct the ordinary RHS for selected LHS {focus}; only this equation "
        "may be defined/replaced in this response. Shared-process and binding edits "
        "remain possible. "
        + (
            "You may declare its newly needed variables alongside it. "
            if policy == "joint_fixed"
            else "Use the previously selected generated-variable inventory. "
        )
        + "The runtime then visits remaining targets/dependencies. Do not set "
        "stage_complete=true just because this ONE equation is finished. It ends "
        "the WHOLE topology stage, including any still-pending work."
    )


def delivery_feedback(record: dict) -> dict | None:
    """Describe truncated delivery without salvaging or inventing missing JSON."""
    raw = record.get("raw_response")
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "length"
    ):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return None
    return {
        "finish_reason": "length",
        "visible_characters": len(content),
        "trailing_whitespace_characters": len(content) - len(content.rstrip()),
        "instruction": (
            "The previous response did not finish. Return the entire response_template "
            "with all keys, empty lists for no edits, stage_complete and a closing }. "
            "Do not emit whitespace to fill the output allowance. No part of that "
            "truncated reply was committed."
        ),
    }


def run(
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    enriched_brief: dict,
    client: ConstructionClient,
    directory: Path,
    policy: Policy,
    *,
    repair_requests: int = 3,
    repair_tokens: int = 131072,
    graph_contract: PublicGraphContract | None = None,
) -> dict:
    """Replay cached transactions, then spend only remaining budget."""
    draft, events, records = ledger.Draft(), [], []
    original = client.settings
    if policy not in POLICIES:
        raise ValueError("unknown construction policy")
    if graph_contract is not None:
        graph_contract.validate_public(brief)
    if (
        original.maximum_requests <= repair_requests
        or original.maximum_total_tokens <= repair_tokens + 256
    ):
        raise ValueError("construction and repair both need a positive frozen budget")
    client.settings = original.model_copy(
        update={
            "maximum_requests": original.maximum_requests - repair_requests,
            "maximum_total_tokens": original.maximum_total_tokens - repair_tokens,
        }
    )

    def request(stage: str, focus: str | None, diagnostic: dict | None, attempt: int):
        nonlocal draft
        payload = {
            "policy": policy,
            "stage": stage,
            "stage_instructions": instructions(policy, stage, focus),
            "selected_lhs": focus,
            "public_brief": enriched_brief,
            "public_source_catalog": [
                p.model_dump(mode="json") for p in brief.public_variables
            ],
            "required_target_definitions": target_definitions,
            "public_graph_contract": graph_contract.model_dump(mode="json")
            if graph_contract is not None
            else None,
            "allowed_memory_bindings": [
                {
                    "requirement_id": r.id,
                    "required": r.requires_dynamic_memory,
                    "public_requirement": r.public_requirement,
                }
                for r in brief.requirements
            ],
            "memory_requirements": [
                r.model_dump(mode="json")
                for r in brief.requirements
                if r.requires_dynamic_memory
            ],
            "binding_context": handoff.binding_context(brief, draft, graph_contract),
            "response_template": ledger.DraftPatch(stage_complete=False).model_dump(
                mode="json"
            ),
            "current_draft": ledger.snapshot(brief, draft),
            "runtime_diagnostics": diagnostic,
        }
        record = client.call(
            system=SYSTEM,
            user=json.dumps(payload, sort_keys=True),
            response_model=ledger.DraftPatch,
            step=f"{stage}_{len(events):03d}",
            attempt=attempt,
        )
        records.append(record)
        before = draft
        raw, accepted, complete, error = None, False, False, None
        normalizations = []
        try:
            raw = visible_response(record)
            normalized, normalizations = ledger.normalize_reply(
                brief, raw, graph_contract=graph_contract, draft=draft
            )
            patch = ledger.DraftPatch.model_validate(normalized)
            validate_scope(policy, stage, focus, patch, draft)
            candidate = ledger.apply_patch(brief, draft, patch)
            candidate, consumer_log = handoff.normalize_consumers(
                candidate, patch, draft
            )
            normalizations.extend(consumer_log)
            draft = candidate
            accepted, complete = True, patch.stage_complete
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:6000]
        event = {
            "index": len(events),
            "stage": stage,
            "selected_lhs": focus,
            "attempt": attempt,
            "request_hash": record["request_hash"],
            "record_sha256": content_hash(record),
            "accepted": accepted,
            "error": error,
            "normalizations": normalizations,
            "stage_complete": complete,
            "before": before.model_dump(mode="json"),
            "after": draft.model_dump(mode="json"),
            "pending_after": ledger.pending(brief, draft),
        }
        sealed_write(directory / "events" / f"{len(events):03d}.json", event)
        events.append(event)
        return (
            accepted,
            complete,
            {
                "status": "accepted" if accepted else "rejected",
                "rejected_reply": None if accepted else raw,
                "normalizations": normalizations,
                "error": error,
                "edit_effects": handoff.edit_effects(before, draft)
                if accepted
                else None,
                "delivery": delivery_feedback(record),
            },
        )

    def stage(name: str) -> tuple[bool, str | None]:
        while True:
            work = ledger.pending(brief, draft)
            focus = None
            if name == "equations" and policy != "joint_adaptive":
                if not work["equations_to_define"]:
                    return True, None
                focus = work["equations_to_define"][0]
            diagnostic = None
            for attempt in range(original.attempts_per_step):
                accepted, complete, diagnostic = request(
                    name, focus, diagnostic, attempt
                )
                if accepted:
                    if complete:
                        return True, None
                    break
            else:
                return False, f"local response repair exhausted in {name}"

    try:
        ready, stop_reason = False, None
        stage_outcomes = []
        stages = (
            ("variables", "relationships", "equations")
            if policy == "separate"
            else ("relationships", "equations")
        )
        try:
            for name in stages:
                ready, stop_reason = stage(name)
                stage_outcomes.append(
                    {
                        "stage": name,
                        "completed": ready,
                        "error": stop_reason,
                        "continued_to_equations": name == "relationships" and not ready,
                    }
                )
                # Optional relationship delivery must not prevent construction of
                # ordinary equations. Retain all accepted declarations; required
                # bindings and unresolved references still face the final checks.
                if name == "relationships" and not ready:
                    continue
                if not ready:
                    break
        except (RepairBudgetExceeded, PromptPreflightError) as exc:
            ready, stop_reason = False, str(exc)
        initial_check = ledger.assess(
            brief,
            context,
            target_definitions,
            draft,
            graph_contract=graph_contract,
            clarify_overlaps=True,
        )
        initial = sealed_write(
            directory / "before_repair.json",
            {
                "draft": draft.model_dump(mode="json"),
                "assessment": initial_check,
                "ready_requested": ready,
                "stop_reason": stop_reason,
                "cost": _cost(sorted(records, key=lambda r: r["request_hash"])),
                "event_count": len(events),
                "stage_outcomes": stage_outcomes,
            },
        )
        # Reserve the SAME additional allowance for every arm. Unspent initial
        # allowance is not silently converted into more scientific repair calls.
        client.settings = original.model_copy(
            update={
                "maximum_requests": len(records) + repair_requests,
                "maximum_total_tokens": sum(r["budget_charge"] for r in records)
                + repair_tokens,
            }
        )
        check = initial_check
        diagnostic = None
        for attempt in range(repair_requests):
            if check["eligible"] and ready:
                break
            feedback = {
                "structural_failures": check["errors"],
                "graph_check_status": check["graph_check_status"],
                "graph_check_reason": check["graph_check_reason"],
                "unresolved_public_predicates": check["unresolved_public_predicates"],
                "reviewed_public_graph_checks": check["reviewed_public_graph_checks"],
                "deferred_scientific_checks": check["deferred_scientific_checks"],
                "ready_requested": ready,
                "last_edit_result": diagnostic,
                "clarification_requests": check["clarification_requests"],
            }
            try:
                accepted, complete, diagnostic = request(
                    "repair", None, feedback, attempt
                )
            except (RepairBudgetExceeded, PromptPreflightError) as exc:
                stop_reason = str(exc)
                break
            if accepted:
                check = ledger.assess(
                    brief,
                    context,
                    target_definitions,
                    draft,
                    graph_contract=graph_contract,
                    clarify_overlaps=True,
                )
                ready = complete
        return {
            "status": "topology_complete"
            if check["eligible"] and ready
            else "topology_incomplete",
            "policy": policy,
            "draft": draft.model_dump(mode="json"),
            "assessment": check,
            "ready_requested": ready,
            "before_repair": initial,
            "stop_reason": stop_reason,
            "event_count": len(events),
            "cost": _cost(sorted(records, key=lambda r: r["request_hash"])),
        }
    finally:
        client.settings = original
