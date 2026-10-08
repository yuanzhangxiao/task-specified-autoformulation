"""Reviewed minimal wording and a common stage contract for a matched prompt pilot."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import create_model

from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract

PromptFamily = Literal["current", "minimal"]
FAMILIES = ("current", "minimal")
SCHEDULE = "variables-shared-topology-1"
STAGES = ("variables", "shared_laws", "equations")

SYSTEM = """
Construct a system of ODEs and algebraic equations for the supplied scientific
task. Work in three stages: variables, topology, then interaction functions.
Follow the current stage's instructions. The runtime stores accepted declarations
and shows the current model. Return JSON in the supplied format."""

VARIABLES = """Declare the variables the model will generate.

1. Start with every prediction target. Choose differential for a state governed
   by an ODE, or algebraic for an instantaneous quantity.
2. Review the supplied inputs and observed quantities, including when they are
   available during prediction. Supplied sources need no generated declaration.
3. Consider what is needed to generate the targets and represent the required
   mechanisms. Use permitted observations where sufficient; otherwise introduce
   the necessary latent states or algebraic quantities.

For each generated variable, give its name, type and scientific meaning. You may
declare several together. Leave dependencies and function formulas for later stages."""

SHARED = """
Consider all generated variables together. Identify any single physical rate or
quantity whose value contributes to at least two different generated variables.
For each such shared process, give its name, drivers, scientific meaning, receiving
variables and contribution signs. Return an empty list if none is needed.

For each shared process, use transfer for a pairwise internal transfer with two
opposite signs; otherwise use influence and choose the scientifically appropriate
signs. Give any known fixed consumer conversions; leave unknown conversions null.

Give dependencies, not function formulas. This declaration defines the process
once; the runtime inserts its uses in the receiving variables' topologies."""

TOPOLOGY = """
Complete the topology for each generated variable: which contributions enter its
derivative (differential) or its value (algebraic)? The current draft already
includes declared shared-process uses.

For each remaining contribution, give the variables that jointly determine it
and its outer sign: positive, negative or unrestricted. Each contribution becomes
one additive term: a coefficient times a function of those variables. The function
may be nonlinear; its formula will be chosen in the interaction stage.

Use the scientific specification and, when supplied, the training-trajectory
summary to choose dependencies. You may address several related variables together
in any order and explicitly revise earlier declarations. Finish when all necessary
variables and dependencies are declared."""

EDITING = """Return changes only. Omitted entries remain unchanged.

For a variable whose topology you revise, return its COMPLETE new list of ordinary
(non-shared) contributions, including those you wish to keep. Its old ordinary list
is replaced. Shared-process uses are maintained separately by the runtime.
Suppose a variable's ordinary contribution list is [A, B]. To change only B
into C, return [A, C]. Returning [C] removes both old entries and installs only C.
A variable with no topology entry in this reply keeps its previous list.

Other named entries replace the corresponding declaration. Use the explicit
removal lists to delete entries. Never replace and remove the same entry in one
reply. Transaction-valid edits are saved together; an invalid transaction saves
none of its edits. Incomplete topology may remain pending until the final check.
stage_complete means the entire current stage is ready for checking."""

REPAIR = """The complete topology has the displayed failures and clarification requests.
Revise the accepted draft to resolve them, using the same public specification
and source availability. You may make coordinated changes across variables and
contributions. Make each scientific choice explicitly in your structured reply."""

# Versioned additions are independent of the current-prompt repair experiment.
COMPLETION = {
    "variables": "Set stage_complete=true when the generated-variable inventory and "
    "applicable state assignments are complete. No topology is needed yet. "
    "If nothing more is needed, return empty edit lists and true; do not repeat "
    "unchanged declarations with false. Fitted coefficients are parameters for "
    "the later interaction stage, not generated variables needing equations.",
    "shared_laws": "Set stage_complete=true after this shared-process decision, "
    "including when no shared process is needed.",
    "equations": "Set stage_complete=true when the whole topology is ready to check.",
    "repair": "Set stage_complete=true when the revised topology is ready to check.",
}
CLARITY_EDITING = """Only declarations are editable; read_only views are rebuilt.
Edit a named process in processes, not in equations: its drivers already define
its one topology. Empty processes means no edits, not deletion. To delete P,
return remove_processes=["P"]; runtime removes its generated definition and uses.
Preserve or replace its physical contributions explicitly if they are still needed.
Explanations do not perform edits. After a reply, inspect the recorded changes.
Fitted coefficients belong in the later interaction stage, not in variables."""


def clarity_stage_text(stage: str) -> str:
    """Minimal additions justified by the first saved run, not a new science gate."""
    return stage_text(stage) + "\n" + COMPLETION[stage]


def stage_text(stage: str) -> str:
    """Only instructions for the stage being performed; no interaction tutorial."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def response_model(stage: str) -> type[StrictSchema]:
    """Expose existing typed ledger fields for this stage, without an alias layer."""
    fields = set(ledger.DraftPatch.model_fields)
    if stage in {"variables", "shared_laws"}:
        fields -= {
            "equations",
            "remove_equations",
            "overlap_confirmations",
            "remove_overlap_confirmations",
        }
    if stage == "variables":
        fields -= {"processes", "remove_processes"}
    return create_model(
        "Construction" + stage.title().replace("_", "") + "Reply",
        __base__=StrictSchema,
        **{
            name: (field.annotation, deepcopy(field))
            for name, field in ledger.DraftPatch.model_fields.items()
            if name in fields
        },
    )


def binding_questions(
    brief: PublicScientificBrief,
    draft: ledger.Draft,
    graph_contract: PublicGraphContract | None,
) -> dict:
    """Ask only applicable assignments; retain IDs and public requirements verbatim."""
    questions = [
        {
            "requirement_id": r.id,
            "public_requirement": r.public_requirement,
            "instruction": (
                "Identify the declared differential state(s) carrying this history "
                "in mechanism_bindings. Topology will establish the connections."
            ),
        }
        for r in brief.requirements
        if r.requires_dynamic_memory
    ]
    algebraic = {v.name for v in draft.variables if v.definition == "algebraic"}
    targets = sorted(
        {o.target for o in graph_contract.obligations if o.kind == "target_feedback"}
        if graph_contract
        else set()
    )
    return {
        "mechanism_bindings": questions,
        "feedback_bindings": {
            "applicable_targets": targets,
            "needs_proposer_coordinates": [t for t in targets if t in algebraic],
            "assignments": [
                {
                    "target": target,
                    "instruction": (
                        "In feedback_bindings, identify the differential "
                        "storage/energy states represented by this algebraic target."
                    ),
                }
                for target in targets
                if target in algebraic
            ],
        },
    }
