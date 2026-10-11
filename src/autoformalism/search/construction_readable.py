"""Opt-in topology display and repair guidance without changing scientific checks."""

from __future__ import annotations

from copy import deepcopy

from autoformalism.search import construction_focus as focused
from autoformalism.search import construction_ledger as ledger

POLICY = "minimal-readable-construction-1"
SYSTEM = focused.SYSTEM
EDITING = """Return JSON changes only; omitted entries remain unchanged.
variables declares a generated quantity as differential (ODE) or algebraic
(algebraic equation). equations.name is that quantity's name, not its derivative.
Each term.sources lists the variables ONE local process depends on: [x, z]
means one joint function of x and z, not two additive functions.

processes defines a reusable algebraic quantity once: name, depends_on, kind,
meaning and signed uses. The runtime creates its defining algebraic equation
and inserts every use automatically. Do not also define it in equations or
repeat a use as a local process. uses.target is the generated variable whose
equation receives that contribution, not the process being defined.

An equations entry replaces ALL local processes for that variable. To change
B into C in [A, B], return [A, C]; returning [C] removes A too. terms=[] means
no local processes; it is valid if shared contributions supply the equation.
A processes entry replaces its complete declaration, including ALL uses.
Preserve other intended dependencies and contributions. Use remove_* for
deletions; never replace and remove the same entry. An invalid transaction
commits none of its edits. Explanations alone do not change declarations.

Function names and fitted coefficients are assigned later; do not declare them
as variables or sources. conversion is a known fixed factor using public
covariates, or null if unknown; it is not a slot for a fitted coefficient."""
VARIABLES = """Declare every prediction target, then other generated quantities
needed for the required mechanisms. Use supplied observations where sufficient;
otherwise introduce differential states or algebraic variables. Give each name,
type and meaning. Supplied quantities need no generated declaration.
A required mechanism can involve several variables, processes and their pathway.
Record required memory-state choices in mechanism_bindings; paths are checked
after topology. Coefficients and initialization formulas belong to interaction.
Resolve the variable checklist, then set stage_complete=true; no topology yet."""
SHARED = """Identify any single physical quantity reused in at least two generated
equations. Give each shared process one distinct name, depends_on (the variables
it depends on), meaning and signed uses. A process is an algebraic quantity;
this declaration already defines it, so no second variable/equation is needed.
For a function of state X, use another name such as P, with depends_on=[X].

Example: process P depends on [x, z], with a negative use in x and positive use
in z. The runtime shows d(x)/dt = ... - P and d(z)/dt = ... + P. Its formula
and any fitted coefficients are later work. A physical pairwise internal
transfer uses kind=transfer and two distinct variables with opposite signs.
Other shared processes use kind=influence, without a balance claim. Opposite
signs alone do not guarantee balance; conversions and coefficients also matter.
Known fixed conversions go in conversion; otherwise use null. A one-equation
effect is a local process: add it at the next stage, or keep a one-use influence
entry without inventing a second use. Return no additions if none is needed.
Set stage_complete=true when this decision and its local checklist are complete."""
TOPOLOGY = """Complete every generated variable's topology. The model view includes
all shared contributions already; return only additional local processes in
equations. Each term gives its sources, outer sign and scientific meaning.
sources=[x, z] is one joint function; separate terms mean an additive sum.
Select dependencies from the public science and supplied training summary.
Give topology only, not formulas or coefficients. You may edit related variables
together and declare additional variables if needed. Set stage_complete=true
when the whole topology is ready for checking."""
REPAIR = """Repair the accepted model using the issue list and public requirements.
First resolve rejected edits and inconsistent representations, while keeping
the outstanding required pathways visible. You may fix several related issues
in one reply. Priority is an order of work, not permission to omit requirements.
Preserve intended effects when changing their representation. If an unfamiliar
symbol is a coefficient, confirm that choice by removing it from
topology sources/declarations; its formula belongs to interaction. If it is a
modeled quantity, declare its type and topology instead. Provisional paths are
observations of this draft, not completed verification. Set stage_complete=true
when ready for the complete check."""


def stage_text(stage: str) -> str:
    """Use the same wording for live calls and offline review."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def model_view(draft: ledger.Draft) -> dict:
    """Show actual quantity names; hide runtime function/weight/occurrence labels."""
    view = focused.model_view(draft)
    view.pop("local_term_meanings")
    for row, process in zip(view["processes"], draft.processes, strict=True):
        row["topology"] = f"{process.name} depends on ({', '.join(process.depends_on)})"
    equations = {e.name: e for e in draft.equations}
    process_names = {p.name for p in draft.processes}
    for row in view["balances"]:
        name, terms, meanings = row["variable"], [], []
        for term in equations[name].terms if name in equations else ():
            sign = {"positive": "+", "negative": "-", "unrestricted": "±"}[
                term.outer_weight_sign.value
            ]
            terms.append(f"{sign} f({', '.join(term.sources)})")
            meanings.append(term.scientific_role)
        for p in draft.processes:
            for use in p.uses:
                if use.target == name:
                    factor = (
                        f"({use.conversion}) * " if use.conversion is not None else ""
                    )
                    terms.append(
                        f"{'-' if use.sign == 'negative' else '+'} {factor}{p.name}"
                    )
        if name not in equations and name not in process_names:
            terms.append("+ [local topology not yet declared]")
        lhs = f"d({name})/dt" if row["type"] == "differential" else name
        row["topology"] = (
            lhs + " = " + (" ".join(terms).removeprefix("+ ") or "[empty RHS]")
        )
        row["local_process_meanings_in_order"] = meanings
    view["notation"] = (
        "Each occurrence of f denotes a separate local process with the displayed "
        "joint dependencies; f is notation, not a quantity name. ± means an "
        "unrestricted outer sign. Empty arguments allow a constant. Shared "
        "process names denote one reusable algebraic value. The runtime assigns "
        "function/parameter identifiers later. Fitted magnitudes are suppressed, "
        "not fixed to one; displayed factors are exactly the declared fixed "
        "conversions. An absent conversion is unknown, not a unity assertion."
    )
    return view


def _role_questions(brief, draft: ledger.Draft) -> list[dict]:
    """Ask about unknown quantities or self-described parameters; never guess."""
    names = set(ledger.pending(brief, draft)["missing_declarations"])
    claimed = {
        v.name
        for v in draft.variables
        if v.scientific_role.strip().lower()
        in {"parameter", "coefficient", "rate constant"}
    }
    public = {v.name for v in brief.public_variables}
    equations = {e.name for e in draft.equations}
    destinations = {u.target for p in draft.processes for u in p.uses}
    bound = {s for b in draft.mechanism_bindings for s in b.memory_states}
    bound.update(s for b in draft.feedback_bindings for s in b.states)
    rows = []
    for name in sorted((names | claimed) - public):
        locations = [
            f"equations[{e.name}].terms[{i}].sources"
            for e in draft.equations
            for i, t in enumerate(e.terms)
            if name in t.sources
        ] + [
            f"processes[{p.name}].depends_on"
            for p in draft.processes
            if name in p.depends_on
        ]
        rows.append(
            {
                "name": name,
                "locations": locations,
                "used_as_generated_quantity": name in equations | destinations | bound,
                "question": "Is this a modeled quantity or a coefficient for a "
                "later function?",
                "if_coefficient": "Confirm through your edits: remove this name from "
                "every topology "
                "dependency and any accidental variable/equation declaration. "
                "Preserve the "
                "intended process and its real dependencies. Do not remove a "
                "required state.",
                "if_quantity": "Declare its permitted type and topology; retain "
                "dependencies. "
                "A quantity used as an equation LHS, memory state or uses.target "
                "needs a "
                "generated representation, not just a coefficient.",
            }
        )
    return rows


def _specific_issue(
    issue: dict, draft: ledger.Draft, target_definitions: dict[str, str]
) -> dict:
    """Attach conditional field-level actions to existing deterministic findings."""
    row = deepcopy(issue)
    code = row["code"]
    if code == "algebraic_dependency_cycle":
        cycle = set(row["cycle"])
        row["interpretation"] = (
            "The cycle is a property of these algebraic equations together; "
            "it does not mean every individual dependency is wrong."
        )
        row["conditional_type_edits"] = [
            {
                "if": f"If {v.name} is scientifically a dynamic state, "
                "choose its ODE type.",
                "edit": {
                    "variables": [
                        {**v.model_dump(mode="json"), "definition": "differential"}
                    ]
                },
                "scope": "Keeps process dependencies and uses unchanged; review "
                "whether "
                "the retained topology is appropriate for the ODE. This resolves no "
                "other cycle or requirement automatically.",
            }
            for v in draft.variables
            if v.name in cycle
            and v.definition == "algebraic"
            and v.name not in {p.name for p in draft.processes}
            and target_definitions.get(v.name) != "algebraic"
        ]
        row["alternative"] = (
            "If these quantities are instantaneous, revise the dependencies to a "
            "scientifically justified explicit algebraic ordering."
        )
    if code in {
        "memory_target_path",
        "driver_memory_path",
        "memory_driver_path",
        "public_path",
    }:
        row["binding_context"] = [
            b.model_dump(mode="json")
            for b in draft.mechanism_bindings
            if b.requirement_id == row.get("requirement")
        ]
        row["repair_options"] = [
            "If the assigned state is intended, connect it through the appropriate "
            "local/shared processes to the required target; changing a description "
            "adds no path.",
            "If the scientific assignment was mistaken, explicitly replace that "
            "mechanism_bindings entry. Keep the public mechanism; do not just "
            "delete its state.",
        ]
    if code == "process_consumer_decision":
        p = next(p for p in draft.processes if p.name == row["process"])
        row["existing_contribution"] = [
            {"process": p.name, **u.model_dump(mode="json")}
            for u in p.uses
            if u.target == row["target"]
        ]
        row["decision"] = (
            f"The RHS already includes {p.name} through uses. Does the local term "
            f"mean that same additive contribution plus other effects, or a distinct "
            f"function of {p.name}? If additive, keep uses and declare only the other "
            "effects as local processes. If a composite function is intended, revise "
            "the representation explicitly, including whether the direct use is still "
            "intended. The runtime cannot split a joint function into a sum for you."
        )
    if code == "process_definition_consumer":
        row["field_to_edit"] = f"processes[{row['process']}].uses"
        row["action"] = (
            "uses.target names a generated variable's ODE/algebraic equation, not "
            "a process definition. A process depending on another process lists "
            "it in depends_on. Preserve all other intended signed contributions."
        )
    if code in {"missing_declarations", "equations_to_define"}:
        row["action"] = (
            "Resolve symbol_role_questions before inventing equations for unknown "
            "names. Generated quantities need equations; coefficients do not."
        )
    if code == "compiler" and "every declared consumer" in row.get("reason", ""):
        missing = {u.target for p in draft.processes for u in p.uses} - {
            e.name for e in draft.equations
        }
        if missing:
            row["pending_topology_entries"] = sorted(missing)
            row["action"] = (
                "Declare equations entries for these generated variables. "
                "Use terms=[] if no local process is intended; the runtime inserts "
                "all shared contributions. Do not repeat them as local processes."
            )
    return row


def _conversion_feedback(last: dict | None) -> list[dict]:
    """Explain a rejected conversion without interpreting or executing provider text."""
    if not last or "invalid fixed conversion" not in (last.get("error") or ""):
        return []
    raw = last.get("rejected_reply") or {}
    processes = raw.get("processes", []) if isinstance(raw, dict) else []
    return [
        {
            "code": "conversion_role_decision",
            "attempted": [
                {"process": p.get("name"), "uses": p.get("uses")}
                for p in processes
                if isinstance(p, dict)
            ],
            "action": "conversion accepts only a known fixed positive factor using "
            "public covariates. If the rejected value is an unknown coefficient or "
            "function placeholder, confirm that choice by returning null for "
            "that conversion. Preserve the process dependencies and signed uses. "
            "If a fixed conversion is known, give that explicit allowed expression.",
        }
    ]


def _rejected_declarations(brief, last: dict | None) -> list[dict]:
    """Show all forbidden public LHSs in a rejected reply, not only the first one."""
    raw = (last or {}).get("rejected_reply")
    if not isinstance(raw, dict) or not isinstance(raw.get("variables"), list):
        return []
    supplied = {
        v.name
        for v in brief.public_variables
        if v.data_role in {"external_input", "covariate", "time"}
    }
    rows = [
        v
        for v in raw["variables"]
        if isinstance(v, dict) and isinstance(v.get("name"), str)
    ]
    forbidden = sorted(
        {
            v["name"]
            for v in rows
            if isinstance(v.get("name"), str) and v["name"] in supplied
        }
    )
    issues = []
    if forbidden:
        issues.append(
            {
                "code": "supplied_declarations_in_rejected_reply",
                "names": forbidden,
                "action": "These quantities are already supplied; keep them available "
                "in process dependencies without generated declarations. The entire "
                "reply was rejected, including valid target declarations. Resubmit the "
                "intended permitted declarations and any bindings together.",
                "conditional_variables_retry": {
                    "if": "If the other variable declarations are still intended, "
                    "this list removes only forbidden supplied declarations. Review "
                    "other reported errors and bindings before returning the reply.",
                    "variables": [v for v in rows if v.get("name") not in supplied],
                },
            }
        )
    names = [v.get("name") for v in rows if isinstance(v.get("name"), str)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        issues.append(
            {
                "code": "duplicate_variable_declarations",
                "names": duplicates,
                "action": "Declare each quantity once with its intended type. A "
                "quantity can have multiple scientific roles without being declared "
                "twice. Put state assignments in the binding fields separately.",
            }
        )
    return issues


def payload(
    stage,
    brief,
    context,
    target_definitions,
    enriched_brief,
    draft,
    contract,
    last=None,
    continuation=None,
) -> dict:
    """Project the same draft/checks with simpler notation and a repair agenda."""
    packet = focused.payload(
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
    packet.update(
        stage_instructions=stage_text(stage),
        editing_rules=EDITING,
        model_view=model_view(draft),
    )
    packet["receiver_choices"]["rule"] = (
        "uses.target must be in currently_generated, or explicitly declared as a "
        "new generated quantity. Supplied observations may be used in dependencies "
        "without modeling them; to generate one, declare its permitted type and "
        "topology first. External inputs, fixed covariates and time stay supplied."
    )
    feedback = packet["feedback"]
    feedback["issues"] = [
        _specific_issue(i, draft, target_definitions) for i in feedback["issues"]
    ]
    feedback["issues"].extend(_conversion_feedback(last))
    feedback["issues"].extend(_rejected_declarations(brief, last))
    if stage != "variables":
        feedback["symbol_role_questions"] = _role_questions(brief, draft)
    if stage == "repair":
        feedback["repair_agenda"] = {
            "representation_issues": [
                i["code"]
                for i in feedback["issues"]
                if i["code"]
                not in {"memory_target_path", "public_path", "memory_driver_path"}
            ],
            "required_paths": [
                r
                for r in feedback["provisional_paths"]
                if (r["status"] == "missing_in_draft")
                != (r["kind"] == "forbidden_path")
            ],
            "failed_requirements": [
                r["id"]
                for r in packet["requirements_checklist"]
                if r["status"] in {"failed", "missing", "inconsistent"}
            ],
            "instruction": "Resolve representation errors and the applicable required "
            "pathways together where possible. Unavailable checks remain obligations; "
            "a description or deletion is not a repair of a missing required path.",
        }
    changes = (last or {}).get("edit_effects") or {}
    feedback["recently_changed_processes"] = [
        {
            "process": r["key"],
            "dependencies_before": r["before"]["depends_on"],
            "dependencies_after": r["after"]["depends_on"],
            "uses_before": r["before"]["uses"],
            "uses_after": r["after"]["uses"],
        }
        for r in changes.get("processes", [])
        if r.get("before")
        and r.get("after")
        and any(r["before"][k] != r["after"][k] for k in ("depends_on", "uses"))
    ]
    return packet


def repair_budget(packet: dict, maximum: int, attempt: int) -> None:
    """Expose the unchanged global allowance in live and preview requests."""
    packet["feedback"]["repair_budget"] = {
        "maximum_global_calls": maximum,
        "current_call": attempt + 1,
        "remaining_after_this_call": maximum - attempt - 1,
        "early_stop": "Success, budget exhaustion, or two consecutive identical "
        "no-progress repairs.",
    }
