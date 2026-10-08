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
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_feedback
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_repair_fidelity as fidelity
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash

Policy = Literal["separate", "joint_guided", "joint_adaptive"]
POLICIES = ("separate", "joint_guided", "joint_adaptive")
ProcessQuestion = Literal["integrated", "dedicated"]
PROCESS_QUESTIONS = ("integrated", "dedicated")

SHARED_LAW_QUESTION = """Is there ONE physical rate or quantity whose value is reused
as a contribution to two or more generated variables? Consider all targets together.
For example, a single internal flow P leaves A and enters B: A uses -P and B uses +P.
P is computed once from its drivers; consumer conversions and magnitudes are applied
outside it. Two different flows with similar formulas are NOT the same shared process.
For each justified shared process, return one processes entry with name, depends_on,
scientific_meaning, kind and signed uses. Each use names a DIFFERENT receiving
variable; uses are not the separate terms inside one variable's balance.
This is topology: give drivers and uses, not a formula. Separate ordinary terms
with identical drivers do not establish reuse of the same later interaction.
Keep local, single-consumer contributions for the remaining topology work; do not
create named local processes merely to answer this question. If no quantity is
reused, return no process additions. Never invent a consumer or physical coupling.
Empty processes is a valid answer. Unknown conversions may remain null. Runtime
inserts each declared use once; do not repeat the effect as an ordinary term."""

SYSTEM = """Construct a scientific model's VARIABLES AND TOPOLOGY, not functions.
The unchanged public scientific task is authoritative. Work on the displayed
stage using the structured edit schema. The runtime maintains the current draft.
Here equations means equation TOPOLOGY: the LHS, grouped dependencies and signs.
The later interaction stage supplies mathematical expressions and parameters.
A sources list such as ["x"] does not specify a linear function of x.

PUBLIC SOURCES AND GENERATED VARIABLES
Public non-target input/observation channels, covariates and time are available
RHS sources; no supplied/unused declarations are needed. Actual use is determined
by your equation skeletons. Their physical meanings and permitted availability
remain those in the public task. Initial readings are boundary information, not
ongoing inputs; diagnostic thresholds are not physical forcing merely because
they are in the catalog. Covariates can parameterize ongoing laws when justified.
Every public target must be generated, never read from its future measurements.
Being listed as a public target does NOT declare its generated variable: include
its differential/algebraic declaration as well as its eventual equation topology.
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

RELATIONSHIPS BEFORE INDIVIDUAL EQUATION TOPOLOGIES
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
For each scientifically justified named process, choose its drivers, meaning and signed
consumers together. The runtime defines it once and inserts every declared use.
Consumers are generated variables with their own equations, not named-process
definitions. Never list a process itself as its own consumer. For a composite
law, declare its process dependencies in depends_on; runtime will not silently
drop uses or reinterpret them as dependencies. Edit a named law via processes,
not via a second equation with the same name.
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

SIGNED DEPENDENCIES, NOT INTERACTION FUNCTIONS
Group jointly interacting variables in each term's sources. Select the outer
weight sign: positive, negative or unrestricted. A fixed outer sign does not
assert global monotonicity/nonnegativity of the later function. Self-dependence
can represent decay, relaxation or feedback when scientifically justified.
Use ordinary terms for contributions not linked by a shared-law declaration.
You MAY add or revise processes while declaring topology.
Such a process edit can affect OTHER equations: runtime propagates all its uses.
The selected LHS is a suggested starting point, never an editing restriction.
You may add, replace or remove several ordinary topologies together.
A singleton sources=["P"] with an
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
contributions with the same dynamic drivers but different fixed covariates,
consider whether a conversion is being repeated; no duplicate is inferred.
For all potentially overlapping
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
to copy. Its uncommitted_edits list shows the entries that were NOT applied and
their actual retained values. Redeclare anything still needed; no part of a rejected
transaction survives. Use its separately labeled error to correct the draft.
Global structural feedback is supplied only after the initial construction ends.
Do not claim scientific correctness merely because those finite checks pass.
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
    if stage == "variables" and (
        patch.equations
        or patch.processes
        or patch.remove_equations
        or patch.remove_processes
    ):
        raise ValueError(
            "variable stage permits variable and memory-binding edits only"
        )
    if stage in {"relationships", "shared_laws"} and (
        patch.equations or patch.remove_equations
    ):
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


def instructions(
    policy: Policy,
    stage: str,
    focus: str | None,
    process_question: ProcessQuestion = "integrated",
) -> str:
    """State the selected work unit while preserving the same scientific context."""
    if stage == "variables":
        return (
            "Declare necessary generated variables, public targets first, and memory "
            "bindings. You may declare several variables per reply. No topology yet. "
            "Finish this stage with stage_complete=true when the inventory is ready."
        )
    if stage == "relationships":
        return (
            "Plan relationships across all targets and choose memory/readout bindings. "
            + (
                "Answer shared_law_question in this same reply. "
                if process_question == "integrated"
                else "A dedicated shared-law question follows this stage. You may "
                "already record a law if identified; no declaration is discarded. "
            )
            + "Do not define ordinary equation topologies yet. "
            + (
                "The generated-variable inventory is fixed. "
                if policy == "separate"
                else "Declare the preliminary generated variables needed by this plan. "
            )
            + "Finish relationship planning with stage_complete=true."
        )
    if stage == "shared_laws":
        return (
            "Answer shared_law_question in this focused call, using the displayed "
            "public task and current draft. Preserve earlier variables and bindings "
            "unless an explicit correction is needed. Do not write ordinary equation "
            "topologies or function expressions. "
            + (
                "The generated-variable inventory remains fixed. "
                if policy == "separate"
                else "You may declare generated variables needed by the relationship. "
            )
            + "Return stage_complete=true when this decision is finished, including "
            "when no shared law is justified."
        )
    if stage == "repair":
        return (
            "Repair structural failures and answer contribution clarification requests "
            "with explicit coordinated edits. "
            "All variable, equation-topology, process and binding edits are allowed. "
            "This is still topology; do not supply function expressions. Preserve "
            "unaffected declarations. Inspect the rebuilt model before further edits. "
            "Address remaining_topology_failures as well as binding_failures; changing "
            "an annotation does not itself restore a missing input-to-target path. "
            "Set stage_complete=true when the whole topology is ready for checking."
        )
    if policy == "joint_adaptive":
        return (
            "Declare signed dependencies: choose the order and number of related "
            "variables/equation topologies to address "
            "in this reply. Add or amend declarations together as scientifically "
            "useful. Use pending work to finish all targets and dependencies. "
            "stage_complete=true means the WHOLE topology draft is ready."
        )
    return (
        "You MAY add or revise a shared process and its consumers in this reply; "
        "runtime propagates that declaration to ALL affected equations. "
        f"Suggested next LHS: {focus}. Declare signed dependencies, "
        "not function expressions. "
        "You may handle this and any related LHS variables together, or choose a "
        "different order. Multiple ordinary topology replacements/removals are "
        "allowed in one reply. Coordinate processes and bindings as needed. "
        + (
            "You may declare its newly needed variables alongside it. "
            if policy == "joint_guided"
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


def uncommitted_edits(raw: object, draft: ledger.Draft) -> list[dict]:
    """Show attempted edits against retained truth, without salvaging a rejection."""
    if not isinstance(raw, dict):
        return []
    fields = (
        ("variables", "name", "remove_variables"),
        ("equations", "name", "remove_equations"),
        ("processes", "name", "remove_processes"),
        ("mechanism_bindings", "requirement_id", "remove_bindings"),
        ("feedback_bindings", "target", "remove_feedback_bindings"),
        ("overlap_confirmations", "overlap_id", "remove_overlap_confirmations"),
    )
    result = []
    for field, key, removal in fields:
        current = {
            getattr(entry, key): entry.model_dump(mode="json")
            for entry in getattr(draft, field)
        }
        for edit_field, operation in ((field, "replace"), (removal, "remove")):
            entries = raw.get(edit_field, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                name = entry.get(key) if isinstance(entry, dict) else entry
                if isinstance(name, str):
                    result.append(
                        {
                            "field": field,
                            "key": name,
                            "attempted_operation": operation,
                            "applied": False,
                            "retained_entry": current.get(name),
                        }
                    )
    return result


def repair_issue_groups(errors: list[dict]) -> dict:
    """Keep model failures visible separately from annotation failures."""
    binding_codes = {
        "required_memory_bindings",
        "unknown_memory_requirement",
        "unknown_feedback_target",
        "differential_target_is_own_coordinate",
        "memory_type",
    }
    return {
        "remaining_topology_failures": [
            error for error in errors if error["code"] not in binding_codes
        ],
        "binding_failures": [
            error for error in errors if error["code"] in binding_codes
        ],
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
    process_question: ProcessQuestion = "integrated",
    prompt_family: str | None = None,
    bookkeeping_policy: bookkeeping.Policy = "legacy",
    starting_draft: ledger.Draft | None = None,
) -> dict:
    """Replay cached transactions, then spend only remaining budget."""
    draft, events, records = starting_draft or ledger.Draft(), [], []
    last_edit_result = None
    original = client.settings
    from autoformalism.search import construction_prompts as prompts

    bookkeeping.validate_policy(bookkeeping_policy, prompt_family)
    variable_checks = bookkeeping_policy == checklist.POLICY
    minimal_clarity = bookkeeping_policy in {
        bookkeeping.MINIMAL_POLICY,
        checklist.POLICY,
    }
    improved_bookkeeping = bookkeeping_policy not in {
        "legacy",
        bookkeeping.MINIMAL_POLICY,
        checklist.POLICY,
    }
    specific_feedback = bookkeeping_policy in {
        bookkeeping.FEEDBACK_POLICY,
        bookkeeping.FIDELITY_POLICY,
    }
    fidelity_feedback = bookkeeping_policy == bookkeeping.FIDELITY_POLICY
    completion_check = specific_feedback or minimal_clarity
    if starting_draft is not None and not fidelity_feedback:
        raise ValueError("saved-draft repair requires the fidelity policy")
    if prompt_family is not None and (
        prompt_family not in prompts.FAMILIES
        or policy != "joint_adaptive"
        or process_question != "dedicated"
    ):
        raise ValueError(
            "prompt comparison requires the common adaptive staged schedule"
        )
    if policy not in POLICIES:
        raise ValueError("unknown construction policy")
    if process_question not in PROCESS_QUESTIONS:
        raise ValueError("unknown shared-law question placement")
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
        nonlocal draft, last_edit_result
        payload = {
            "policy": policy,
            "stage": stage,
            "process_question": process_question,
            "stage_instructions": instructions(policy, stage, focus, process_question),
            "shared_law_question": SHARED_LAW_QUESTION
            if stage == "shared_laws"
            or (stage == "relationships" and process_question == "integrated")
            else None,
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
            "last_edit_result": last_edit_result,
        }
        response_model = ledger.DraftPatch
        system = SYSTEM
        if improved_bookkeeping:
            payload["bookkeeping_policy"] = bookkeeping_policy
            payload["current_draft"] = bookkeeping.snapshot(brief, draft)
            system = system.replace("equation_views", "read_only_balances")
        if fidelity_feedback and stage == "repair":
            payload["stage_instructions"] += "\n" + fidelity.INSTRUCTION
            payload["explicit_removal_options"] = fidelity.removal_options(draft)
        if specific_feedback and stage in {"variables", "repair"}:
            payload["stage_instructions"] += "\n" + (
                construction_feedback.VARIABLE_COMPLETION
                if stage == "variables"
                else construction_feedback.FEEDBACK_REPAIR
            )
        if prompt_family is not None:
            response_model = prompts.response_model(stage)
            payload.update(
                prompt_family=prompt_family,
                stage_schedule=prompts.SCHEDULE,
                response_template=response_model(stage_complete=False).model_dump(
                    mode="json"
                ),
                binding_context=prompts.binding_questions(brief, draft, graph_contract),
            )
            if prompt_family == "minimal":
                system = prompts.SYSTEM
                payload["stage_instructions"] = prompts.stage_text(stage)
                # The question already appears once in the stage instructions.
                payload["shared_law_question"] = None
                payload["editing_rules"] = prompts.EDITING
                if minimal_clarity:
                    payload["bookkeeping_policy"] = bookkeeping_policy
                    payload["stage_instructions"] = prompts.clarity_stage_text(stage)
                    payload["current_draft"] = bookkeeping.snapshot(brief, draft)
                    payload["current_draft"]["editing_contract"] = (
                        prompts.CLARITY_EDITING
                    )
                    payload["explicit_removal_options"] = fidelity.removal_options(
                        draft
                    )
        if variable_checks:
            payload["variable_checklist"] = checklist.variable_checklist(
                brief, draft, target_definitions, graph_contract
            )
            if stage == "variables":
                payload["stage_instructions"] += "\n" + checklist.INSTRUCTION
                payload["current_draft"] = checklist.variable_snapshot(draft)
                payload["editing_rules"] = payload["current_draft"]["editing_contract"]
                payload.pop("explicit_removal_options", None)
        record = client.call(
            system=system,
            user=json.dumps(payload, sort_keys=True),
            response_model=response_model,
            step=f"{stage}_{len(events):03d}",
            attempt=attempt,
        )
        records.append(record)
        before = draft
        raw, accepted, complete, error = None, False, False, None
        normalizations, conflicts = [], []
        completion = None
        try:
            raw = visible_response(record)
            normalized, normalizations = ledger.normalize_reply(
                brief,
                raw,
                graph_contract=graph_contract,
                draft=draft,
                ignore_definition_description=improved_bookkeeping,
            )
            patch = ledger.DraftPatch.model_validate(normalized)
            validate_scope(policy, stage, focus, patch, draft)
            if improved_bookkeeping or minimal_clarity:
                conflicts = bookkeeping.definition_conflicts(draft, patch)
            candidate = ledger.apply_patch(brief, draft, patch)
            candidate, consumer_log = handoff.normalize_consumers(
                candidate, patch, draft
            )
            normalizations.extend(consumer_log)
            draft = candidate
            accepted, complete = True, patch.stage_complete
            if completion_check and stage == "variables" and complete:
                completion = (
                    checklist.variable_checklist(
                        brief, draft, target_definitions, graph_contract
                    )
                    if variable_checks
                    else construction_feedback.variable_completion(brief, draft)
                )
                complete = completion["status"] == "ready"
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:6000]
            if conflicts and "runtime-generated definition" in error:
                error += ": " + ", ".join(c["process"] for c in conflicts)
        event = {
            "index": len(events),
            "stage": stage,
            "process_question": process_question,
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
            **(
                {
                    "variable_checklist_after": checklist.variable_checklist(
                        brief, draft, target_definitions, graph_contract
                    )
                }
                if variable_checks
                else {}
            ),
            **({"stage_completion": completion} if completion_check else {}),
            **(
                {
                    "repair_receipt": fidelity.receipt(before, draft)
                    if accepted
                    else None
                }
                if fidelity_feedback or minimal_clarity
                else {}
            ),
            **(
                {
                    "bookkeeping_policy": bookkeeping_policy,
                    "definition_conflicts": conflicts,
                }
                if improved_bookkeeping or minimal_clarity
                else {}
            ),
        }
        sealed_write(directory / "events" / f"{len(events):03d}.json", event)
        events.append(event)
        last_edit_result = {
            "status": "accepted" if accepted else "rejected",
            "rejected_reply": None if accepted else raw,
            "uncommitted_edits": [] if accepted else uncommitted_edits(raw, before),
            "transaction_applied": accepted,
            "normalizations": normalizations,
            "error": error,
            "edit_effects": handoff.edit_effects(before, draft) if accepted else None,
            "delivery": delivery_feedback(record),
            **(
                {"variable_checklist": event["variable_checklist_after"]}
                if variable_checks
                else {}
            ),
            **({"stage_completion": completion} if completion_check else {}),
            **(
                {"definition_conflicts": conflicts}
                if improved_bookkeeping or minimal_clarity
                else {}
            ),
            **(
                {"repair_receipt": event["repair_receipt"]}
                if fidelity_feedback or minimal_clarity
                else {}
            ),
        }
        return accepted, complete, last_edit_result

    def assess_draft() -> dict:
        """Recheck declarations after later edits as well as on stage completion."""
        check = ledger.assess(
            brief,
            context,
            target_definitions,
            draft,
            graph_contract=graph_contract,
            clarify_overlaps=True,
        )
        if variable_checks:
            values = checklist.variable_checklist(
                brief, draft, target_definitions, graph_contract
            )
            check["variable_checklist"] = values
            check["errors"].extend(
                {"code": "variable_declaration", "item": row}
                for row in values["items"]
                if row["status"] in {"missing", "inconsistent"}
            )
            check["eligible"] &= values["status"] == "ready"
        if improved_bookkeeping:
            check = bookkeeping.assessment_context(
                check, draft, policy=bookkeeping_policy
            )
        return check

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
                    if (
                        completion_check
                        and name == "variables"
                        and (diagnostic.get("stage_completion") or {}).get("status")
                        == "incomplete"
                    ):
                        # Keep the accepted partial inventory editable. Repeated
                        # premature completion consumes the existing local attempts,
                        # not an unbounded loop or the reserved global repair budget.
                        continue
                    if complete or name == "shared_laws":
                        # One focused decision, with existing local delivery retries;
                        # no open-ended self-review or extra successful calls.
                        return complete, None if complete else "decision left open"
                    break
            else:
                return False, f"local response repair exhausted in {name}"

    try:
        ready, stop_reason = False, None
        stage_outcomes = []
        stages = (
            prompts.STAGES
            if prompt_family is not None
            else (
                (("variables",) if policy == "separate" else ())
                + ("relationships",)
                + (("shared_laws",) if process_question == "dedicated" else ())
                + ("equations",)
            )
        )
        try:
            for name in () if starting_draft is not None else stages:
                ready, stop_reason = stage(name)
                stage_outcomes.append(
                    {
                        "stage": name,
                        "completed": ready,
                        "error": stop_reason,
                        "continued_to_equations": name
                        in {"relationships", "shared_laws"}
                        and not ready,
                    }
                )
                # Optional relationship delivery must not prevent construction of
                # ordinary equations. Retain all accepted declarations; required
                # bindings and unresolved references still face the final checks.
                if name in {"relationships", "shared_laws"} and not ready:
                    continue
                if not ready:
                    break
        except (RepairBudgetExceeded, PromptPreflightError) as exc:
            ready, stop_reason = False, str(exc)
        initial_check = assess_draft()
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
                **repair_issue_groups(check["errors"]),
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
                check = assess_draft()
                ready = complete
        return {
            "status": "topology_complete"
            if check["eligible"] and ready
            else "topology_incomplete",
            "policy": policy,
            "process_question": process_question,
            "draft": draft.model_dump(mode="json"),
            "assessment": check,
            "ready_requested": ready,
            "before_repair": initial,
            "stop_reason": stop_reason,
            "event_count": len(events),
            "cost": _cost(sorted(records, key=lambda r: r["request_hash"])),
            **(
                {"bookkeeping_policy": bookkeeping_policy}
                if improved_bookkeeping
                else {}
            ),
        }
    finally:
        client.settings = original
