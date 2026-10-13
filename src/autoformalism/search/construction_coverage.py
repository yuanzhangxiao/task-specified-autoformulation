"""Public-coverage prompting; no new scientific predicates or inferred edits."""

from __future__ import annotations

from collections import Counter

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_completion as completion
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-public-coverage-1"
SYSTEM = completion.SYSTEM
EDITING = completion.EDITING
model_view = completion.model_view
repair_budget = completion.repair_budget

SHARED = """Identify physical quantities reused across generated equations. Give each
shared process one distinct name, depends_on, meaning and signed uses. This
defines an algebraic quantity once; the runtime inserts all uses. For a function
of X use another name, e.g. P with depends_on=[X]; no second definition is needed.

For each flow, decide what loses and gains the quantity. A pairwise internal
transfer has kind=transfer and opposite uses in two distinct generated balances.
For example, flow P from x to z contributes -P to d(x)/dt and +P to d(z)/dt.
A discharge leaving the modeled system does not return to another modeled
balance; do not invent a second use. Other effects use kind=influence, which
does not change their physical meaning or fix incorrect signs. A one-equation
effect can be a local process at the next stage or a one-use influence entry.

Choose signs consistently with the declared rate meaning. Give known fixed
conversions, otherwise null; functions and fitted coefficients come later.
Opposite signs alone do not certify conservation. Return no additions if no
sharing is needed. Set stage_complete=true when this decision and its local
checklist are complete; remaining local processes come next."""

TOPOLOGY = """Complete every generated variable's topology. Shared contributions are
already included in model_view; return only additional local processes in
equations. Each term gives sources, outer sign and meaning. sources=[x, z]
means one joint function, not two additive functions. Give no formulas or
coefficients. You may edit related variables together or add a needed variable.

Before finishing, read each public requirement against the assembled model:
where is each required mechanism represented, and how does it affect its
target(s)? A declaration or mechanism name alone is insufficient. When targets
describe related physical quantities, keep their definitions and balances
consistent: a source, sink or internal transfer must have its intended effect
wherever it is used. Shared and equivalent local representations are allowed;
no textbook variable names or particular intermediate edge are prescribed.
Use the public science and supplied training summary. Checklist passes cover
only their stated predicates; function/behavior checks remain deferred.
Set stage_complete=true when the whole topology is ready for checking."""

REPAIR = """Repair the accepted model using the issues and public requirements.
Resolve representation errors and the exact missing pathways together. Keep
intended mechanisms when changing their representation; deleting one does not
satisfy its requirement. Changing a flow's kind alone does not fix its physical
direction. Review all uses against its meaning and the affected balances.

Return each equations entry once with its complete local-process list; preserve
unaffected intended terms. You may coordinate several edits in one reply.
For an unfamiliar symbol, decide whether it is a generated quantity or a later
coefficient; do not invent an equation for a coefficient. Check the assembled
model against every public requirement before declaring completion, including
those whose scientific interpretation is not decided by a graph predicate.
Set stage_complete=true when ready for the complete check."""


def stage_text(stage: str) -> str:
    """Keep the successful variable stage; focus later stages on public coverage."""
    return {
        "variables": completion.stage_text("variables"),
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def _path_destination(issue: dict) -> dict:
    """Name endpoints of an existing failed positive path check, not a new edge."""
    code = issue["code"]
    if code == "public_path" and issue.get("kind") == "driver_path":
        source, target = issue["driver"], issue["target"]
    elif code == "memory_driver_path":
        source, target = issue["driver"], issue["memory"]
    elif code == "memory_target_path":
        source, target = issue["memories"], issue["target"]
    else:
        return issue
    return {
        **issue,
        "missing_path": {
            "from_any" if isinstance(source, list) else "from": source,
            "to": target,
        },
        "repair_options": [
            "Provide your intended dependency pathway from "
            + (f"at least one of {source}" if isinstance(source, list) else source)
            + f" to {target}. "
            f"Inspect the assembled RHS of {target} and its intermediate quantities. "
            "A direct dependency is only one option; the runtime chooses no edge "
            "or sign. Edit equations[].terms[].sources or processes[].depends_on "
            "and uses as scientifically appropriate.",
            "A description or binding alone adds no dependency. Keep the other "
            "intended contributions when returning a complete replacement entry. "
            "If a memory assignment was mistaken, use the separate assignment "
            "decision; a public source-to-target obligation still applies.",
        ],
    }


def _duplicate_equations(last: dict | None) -> list[dict]:
    """Identify duplicate rejected keys without choosing or merging their terms."""
    raw = (last or {}).get("rejected_reply")
    entries = raw.get("equations") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    counts = Counter(
        e["name"]
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("name"), str)
    )
    names = sorted(n for n, count in counts.items() if count > 1)
    if not names:
        return []
    return [
        {
            "code": "duplicate_equation_entries",
            "names": names,
            "field_to_edit": "equations",
            "action": "Return each name once, with the complete intended list of "
            "local processes for that variable. Multiple local processes belong in "
            "one terms list, not several equations entries with the same name. "
            "Keep distinct intended effects; omit contributions already inserted "
            "through shared uses. The entire previous transaction was rejected; "
            "resubmit other intended edits too. The runtime does not merge "
            "these lists.",
        }
    ]


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
    """Present existing diagnostics with scoped scientific decisions and endpoints."""
    packet = completion.payload(
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
    issues = packet["feedback"]["issues"]
    packet["feedback"]["issues"] = [_path_destination(i) for i in issues]
    packet["feedback"]["issues"].extend(_duplicate_equations(last))
    for issue in packet["feedback"]["issues"]:
        if issue["code"] == "supplied_receiver":
            previous = issue["preserve_valid_effects"]
            conditional = previous.get("conditional_process_edit")
            issue["preserve_valid_effects"] = {
                "currently_valid_uses": conditional["uses"] if conditional else [],
                "decision": "Should this quantity be generated, or remain supplied? "
                "If you intend to model it, declare its permitted type explicitly "
                "and complete its topology; consult receiver_choices. External "
                "inputs, covariates and time must remain supplied. If it remains "
                "supplied, remove only its use and preserve intended effects in "
                "generated equations. Review the physical classification: an "
                "exchange with an unmodeled reservoir can be a boundary effect, "
                "but when both balances are generated, internal transfer requires "
                "opposite contributions. Do not add or delete a modeled balance "
                "solely to satisfy the format.",
            }
        if issue["code"] == "flow_classification_decision":
            # Replace the classification-only example: it is not a scientific fix.
            issue.pop("if_boundary_flow_or_other_nontransfer_effect")
            issue["if_boundary_flow_or_other_nontransfer_effect"] = (
                "Use kind=influence only if the physical meaning is not an "
                "internal pairwise transfer. Choose the actual uses and signs "
                "explicitly. An outlet leaving the system has no receiving balance "
                "inside it. Do not relabel an internal exchange just to bypass "
                "opposite-sign checks. If a sign was mistaken, correct it instead."
            )
            issue["preservation"] = (
                "Keep the intended physical effects. Do not invent a receiver, "
                "delete one, or retain a wrong sign just to satisfy the format. "
                "Review the retained_entry and return the complete intended uses."
            )
    # There is deliberately no scientific pass inferred from these instructions.
    return packet
