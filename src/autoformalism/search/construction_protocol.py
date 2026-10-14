"""Opt-in concise terminology, control-field separation and substantive repair."""

from __future__ import annotations

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_control_fields as controls
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_ownership_feedback as ownership
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-protocol-clarity-1"
REPAIR_REQUESTS = 10
REPAIR_TOKENS = 262144
REPEATED_REPAIRS = 3
model_view = ownership.model_view


def repair_budget(
    packet: dict,
    maximum: int,
    attempt: int,
    token_limit: int,
    charged: int,
    repair_tokens: int,
) -> None:
    """Expose this policy's allowance and semantic repetition stopping rule."""
    ownership.repair_budget(
        packet, maximum, attempt, token_limit, charged, repair_tokens
    )
    packet["feedback"]["repair_budget"]["early_stop"] = (
        "Success, budget exhaustion, or three consecutive semantically repeated "
        "repairs with no topology progress. A different structured alternative "
        "may still be tried; rewording explanations does not reset the counter."
    )


DICTIONARY = """Differential state: a generated quantity governed by an ODE.
Algebraic variable: a generated quantity defined by an algebraic equation.
Process: a function of its listed quantities that contributes to an equation.
A shared process uses the same computed value in several equations.
sources/depends_on list its arguments; uses.target names
the generated quantity receiving a contribution.
Pathway: a chain of declared dependencies, possibly through several quantities.
Memory: history carried by differential states; an algebraic readout can depend
on that history. A required mechanism can involve several states, processes and
pathways; it is not itself a variable type."""

SYSTEM = ownership.SYSTEM + "\n\nTerms:\n" + DICTIONARY

MEMORY_RULE = """The JSON field mechanism_bindings records only memory-state
choices: requirement_id -> memory_states (differential states). Algebraic readouts
may lie on the resulting pathway but are not independent memory states. Record
the choices listed as required below; pathways are checked after topology.
Every public mechanism remains required, including those needing no memory-state
choice. Their coverage is tracked in requirements_checklist, not by adding a
binding. Omitted bindings stay unchanged; remove_bindings withdraws a binding,
not a public requirement."""

EDITING = (
    ownership.EDITING
    + """
stage_complete and remove_* are reply-control fields, never scientific quantities.
Set stage_complete at the top level of the reply, outside all model lists."""
)

VARIABLES = """Declare every prediction target, then other generated quantities
needed for the public mechanisms. Use supplied observations where sufficient;
otherwise introduce differential states or algebraic variables. Give each name,
type and meaning. Supplied quantities need no generated declaration.
Complete any required memory-state choices listed below. Coefficients and
initialization formulas belong to interaction. Resolve the variable checklist,
then set stage_complete=true at the reply's top level. No topology is needed yet."""

SHARED = """Identify quantities reused across generated equations. Give each
shared process a distinct name, depends_on, scientific_meaning and signed uses.
This defines its algebraic quantity once; the runtime inserts every use. For a
function of state X use a different name, e.g. P with depends_on=[X].

For a pairwise internal transfer, use kind=transfer: two distinct generated
balances with opposite signs. For example, positive P moving from x to z gives
-P in d(x)/dt and +P in d(z)/dt, with the appropriate conversions. Other effects
use kind=influence, without asserting a transfer. An exchange with outside the
modeled system needs no invented second modeled balance. A one-equation effect
can be a local process next, or a one-use influence entry.

State what a positive process value means; choose signs consistent with it.
Give known fixed conversions, otherwise null. Functions and fitted coefficients
come later; opposite signs alone do not certify conservation. Return no additions
if no sharing is needed. Set stage_complete=true when the local checklist and
this decision are complete; remaining local processes come next."""

TOPOLOGY = """Complete every generated variable's topology. Shared contributions
are already included in model_view; return only additional local processes in
equations. Each term gives sources, outer sign and meaning. sources=[x, z] means
one joint function, not two additive functions. Give no formulas or coefficients.
You may edit related variables together or add a scientifically needed variable.

Locate every public mechanism in the processes and pathways to its targets.
Keep related quantities' definitions and uses consistent across balances. Shared
and equivalent local forms are allowed. Use the public science and training
observations; no particular textbook names are required. The checklist checks
only its stated predicates; function and behavior checks remain later work.
Set stage_complete=true when the whole topology is ready for checking."""

REPAIR = """Repair the accepted model using the issues and public requirements.
For a rejected deletion, check which retained category actually owns the entry;
an uncommitted addition needs no deletion. Keep intended mechanisms when changing
their representation. Fix signs, types and related uses consistently with their
scientific meanings. You may coordinate several edits in one reply.
Return complete replacement lists for edited entries, preserving other intended
terms. Explanations alone do not change the topology. Coefficients belong to the
later interaction stage. Set stage_complete=true when ready for the full check."""


def stage_text(stage: str) -> str:
    """Return one minimal instruction, shared by previews and live calls."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def payload(
    stage: str,
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    enriched_brief: dict,
    draft: ledger.Draft,
    contract: PublicGraphContract | None,
    last: dict | None = None,
    continuation: dict | None = None,
) -> dict:
    """Use the existing checks, with short definitions and conditional binding help."""
    packet = ownership.payload(
        stage,
        brief,
        context,
        target_definitions,
        enriched_brief,
        draft,
        contract,
        last,
        continuation,
    )
    packet["stage_instructions"] = stage_text(stage)
    packet["editing_rules"] = EDITING
    packet.pop("memory_assignment_rules", None)
    options = packet["memory_binding_options"]
    for row in options:
        required = row.pop("memory_assignment_required")
        row["choice"] = "required" if required else "existing optional history claim"
    # The full public brief is unchanged. Avoid an unexplained false flag in the
    # summary: it refers to a memory annotation, not the public mechanism.
    for row in packet["requirements_checklist"]:
        if "memory_assignment_required" in row:
            row["memory_state_choice"] = (
                "required" if row.pop("memory_assignment_required") else "not required"
            )
    issues = packet["feedback"]["issues"]
    binding_problem = any(
        "memory" in row.get("code", "") or "binding" in row.get("code", "")
        for row in issues
    ) or bool(packet["feedback"].get("memory_assignment_decisions"))
    if (stage == "variables" and options) or binding_problem:
        packet["memory_state_assignments"] = MEMORY_RULE
    return controls.augment_payload(packet, draft, last, brief=brief)


_PROSE = frozenset({"scientific_role", "scientific_meaning", "stage_complete"})


def substantive(value: object) -> object:
    """Canonicalize unordered declarations while retaining multiplicity and math."""
    import json

    if isinstance(value, dict):
        return {
            key: substantive(item) for key, item in value.items() if key not in _PROSE
        }
    if isinstance(value, (list, tuple)):
        items = [substantive(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def repair_signature(draft: ledger.Draft, last: dict | None, errors: list) -> dict:
    """Track repeated semantic edits, including distinct rejected alternatives."""
    last = last or {}
    return {
        "draft": substantive(draft.model_dump(mode="json")),
        "errors": errors,
        "rejected_alternative": substantive(last.get("rejected_reply")),
    }
