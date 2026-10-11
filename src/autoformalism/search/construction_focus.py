"""Focused proposer view; full ledgers and scientific acceptance stay runtime-owned."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_prompts as prompts
from autoformalism.search import construction_stage_checks as stages
from autoformalism.search.public_graph_obligations import PublicGraphContract

POLICY = "minimal-focused-construction-1"
REPAIR_REQUESTS = 5
LOCAL_RECOVERY_REQUESTS = 2
SYSTEM = prompts.SYSTEM
EDITING = """Return JSON changes only; omitted entries remain unchanged. Variables
chooses differential or algebraic. In equations, name is the generated variable
(e.g. Gp, not dGp_dt); its type determines the LHS. Each sources list gives all
arguments of ONE function: [Gp, X] means phi(Gp, X), not phi(Gp)+phi(X).

An equations entry replaces that variable's complete LOCAL contribution list.
To change B into C in [A, B], return [A, C]; returning [C] removes A as well.
Process uses are inserted separately. terms=[] explicitly adds no local terms;
use it only if process uses supply the RHS. Omission means no edit.

A processes entry replaces that process's drivers, kind and complete uses list.
Its name defines a new quantity; a function of state X needs another name, e.g.
P_X. Use variables to change a variable's type. Use the remove_* lists for deletions;
empty lists delete nothing. Preserve other intended effects when editing a receiver.
Valid edits are committed together; an invalid transaction commits none.
Display symbols/term labels are not new variables or fitted parameter declarations."""
VARIABLES = """Declare every prediction target, then the other generated quantities
needed for the public mechanisms. Use supplied observations where sufficient;
otherwise introduce states or algebraic quantities. Give each name, type and meaning.
Supplied sources need no declaration. Coefficients and initialization formulas
belong to later interaction work. Record required state assignments explicitly in
mechanism_bindings: declaring a state or describing its role does not bind it.
Resolve variable-stage checklist items, then set stage_complete=true. No topology
is needed yet; do not repeat unchanged declarations just to signal completion."""
SHARED = """Identify a single physical quantity reused in at least two generated
variables. For each shared process return name, depends_on, scientific_meaning,
kind and signed uses; return no additions if none is needed. Drivers may be supplied
sources. Receivers must be generated; see receiver_choices.
Use transfer only for a pairwise internal transfer with two distinct receivers
and opposite signs; otherwise use influence with the scientifically intended signs.
Give known fixed conversions, or null if unknown. Give dependencies, not formulas.
A one-receiver effect is a local process. It may remain as a one-use influence
entry or be added as a local term in the next stage; do not invent another receiver.
Set stage_complete=true when this decision and its local checklist are complete."""
TOPOLOGY = """Complete every generated variable's topology. Process uses in the model
view are already inserted. Return only additional local terms in equations.
For each term give the variables that jointly determine it, its outer sign
(positive, negative or unrestricted), and meaning. For example, sources=[Gp, X]
allows a later interaction phi(Gp, X); separate terms impose additive separation.
Choose dependencies from the public science and supplied training summary.
You may edit related variables together. Declare additional variables only when
needed. Set stage_complete=true when the whole topology is ready for checking."""
REPAIR = """Resolve the displayed issues in the accepted model. You may coordinate
changes across variables, local terms and shared processes. Preserve scientifically
intended effects and make assignments/conversions explicit in the JSON edits.
Provisional path observations describe the current draft, not a final verdict;
consider them alongside blocking errors. Set stage_complete=true when ready."""


def stage_text(stage: str) -> str:
    """Stage-specific wording shared by the live runner and offline preview."""
    return {
        "variables": VARIABLES,
        "shared_laws": SHARED,
        "equations": TOPOLOGY,
        "repair": REPAIR,
    }[stage]


def model_view(draft: ledger.Draft) -> dict:
    """Render exact joint arguments, signs and conversions without a second ledger."""
    definitions = {v.name: v.definition for v in draft.variables}
    process_names = {p.name for p in draft.processes}
    equations = {e.name: e for e in draft.equations}
    laws, balances, meanings = [], [], {}
    for p in draft.processes:
        laws.append(
            {
                "name": p.name,
                "scope": "shared process"
                if len({u.target for u in p.uses}) > 1
                else "local process",
                "kind": p.kind,
                "topology": f"{p.name} = Phi_{p.name}({', '.join(p.depends_on)})",
                "meaning": p.scientific_meaning,
            }
        )
    names = list(
        dict.fromkeys(
            [v.name for v in draft.variables if v.name not in process_names]
            + [e.name for e in draft.equations if e.name not in process_names]
            + [u.target for p in draft.processes for u in p.uses]
        )
    )
    for name in names:
        terms = []
        for i, t in enumerate(equations[name].terms if name in equations else ()):
            label = f"{name}_local_{i}"
            sign = t.outer_weight_sign.value
            gain = "c" if sign == "unrestricted" else "g"
            terms.append(
                f"{'-' if sign == 'negative' else '+'} {gain}_{label} * "
                f"phi_{label}({', '.join(t.sources)})"
            )
            meanings[label] = t.scientific_role
        for p in draft.processes:
            for i, use in enumerate(p.uses):
                if use.target == name:
                    conversion = (
                        f"({use.conversion}) * " if use.conversion is not None else ""
                    )
                    terms.append(
                        f"{'-' if use.sign == 'negative' else '+'} "
                        f"g_{p.name}_use_{i} * {conversion}{p.name}"
                    )
        if name not in equations and name not in process_names:
            terms.append("+ [local topology not yet declared]")
        definition = definitions.get(name)
        lhs = f"d({name})/dt" if definition == "differential" else name
        balances.append(
            {
                "variable": name,
                "type": definition
                or (
                    "algebraic process"
                    if name in process_names
                    else "undeclared receiver"
                ),
                "topology": lhs
                + " = "
                + (" ".join(terms).removeprefix("+ ") or "[empty RHS]"),
            }
        )
    return {
        "variables": [v.model_dump(mode="json") for v in draft.variables],
        "processes": laws,
        "balances": balances,
        "local_term_meanings": meanings,
        "memory_assignments": [
            b.model_dump(mode="json") for b in draft.mechanism_bindings
        ],
        "readout_coordinates": [
            b.model_dump(mode="json") for b in draft.feedback_bindings
        ],
        "notation": "phi/Phi are unspecified joint functions; g is nonnegative "
        "and c unrestricted. Symbols are display placeholders only. Empty "
        "arguments allow a constant, not necessarily zero. Process-use factors "
        "include exactly the declared fixed conversion; an absent conversion "
        "is unknown, not a physical assertion of unity. Process kinds and type "
        "come from structured declarations.",
    }


def receiver_choices(brief: PublicScientificBrief, draft: ledger.Draft) -> dict:
    """Distinguish observed quantities that may be modeled from external forcing."""
    generated = {v.name for v in draft.variables} - {p.name for p in draft.processes}
    return {
        "currently_generated": sorted(generated),
        "observations_needing_generated_declaration_first": sorted(
            v.name
            for v in brief.public_variables
            if v.data_role in {"auxiliary", "target"} and v.name not in generated
        ),
        "supplied_only": sorted(
            v.name
            for v in brief.public_variables
            if v.data_role in {"external_input", "covariate", "time"}
        ),
        "rule": "A receiver has a model-generated equation. Supplied observations "
        "can be drivers; to model one instead, explicitly declare its type and "
        "topology, subject to public type rules. External inputs, covariates "
        "and time remain supplied. New latent receivers can be declared explicitly.",
    }


def _path(graph: dict[str, set[str]], source: str, target: str) -> list[str] | None:
    queue = deque([(source, [source])])
    seen = {source}
    while queue:
        node, route = queue.popleft()
        for nxt in sorted(graph.get(node, ())):
            if nxt == target:
                return [*route, nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, [*route, nxt]))
    return None


def provisional_paths(
    brief: PublicScientificBrief,
    draft: ledger.Draft,
    contract: PublicGraphContract | None,
) -> list[dict]:
    """Advisory declared-edge evidence even when another RHS cannot compile.

    No placeholder equations, inferred bindings or acceptance changes are made.
    Unknown references and conflicting definitions can change these observations.
    """
    graph: dict[str, set[str]] = {}
    for e in draft.equations:
        for t in e.terms:
            for s in t.sources:
                graph.setdefault(s, set()).add(e.name)
    for p in draft.processes:
        for s in p.depends_on:
            graph.setdefault(s, set()).add(p.name)
        for u in p.uses:
            graph.setdefault(p.name, set()).add(u.target)
    rows = []

    def observe(
        rule: str, source: str, target: str, kind: str = "required_path"
    ) -> None:
        route = _path(graph, source, target)
        rows.append(
            {
                "requirement": rule,
                "kind": kind,
                "source": source,
                "target": target,
                "status": "present_in_draft" if route else "missing_in_draft",
                "witness": route,
            }
        )

    for r in brief.requirements:
        for source in r.drivers:
            for target in r.targets:
                observe(r.id, source, target)
        for b in draft.mechanism_bindings:
            if b.requirement_id == r.id:
                for state in b.memory_states:
                    for source in r.drivers:
                        observe(r.id, source, state, "driver_to_assigned_memory")
                    for target in r.targets:
                        observe(r.id, state, target, "assigned_memory_to_target")
    for i, dependency in enumerate(brief.target_dependencies):
        routes = [
            _path(graph, s, dependency.target) for s in dependency.acceptable_sources
        ]
        rows.append(
            {
                "requirement": f"target_composition:{i}",
                "kind": "any_acceptable_source",
                "sources": list(dependency.acceptable_sources),
                "target": dependency.target,
                "status": "present_in_draft" if any(routes) else "missing_in_draft",
                "witnesses": [r for r in routes if r],
            }
        )
    states = {v.name for v in draft.variables if v.definition == "differential"}
    bindings = {b.target: b.states for b in draft.feedback_bindings}
    readout = {s: {t for t in ends if t not in states} for s, ends in graph.items()}
    for o in contract.obligations if contract else ():
        if o.kind == "forbidden_path":
            observe(o.id, o.source, o.target, "forbidden_path")
            continue
        coordinates = (o.target,) if o.target in states else bindings.get(o.target, ())
        if o.kind == "dynamic_feedback":
            coordinates = tuple(
                sorted(s for s in states if s == o.target or _path(graph, s, o.target))
            )
        witnesses = []
        for s in coordinates:
            cycle = _path(graph, s, s) if s in states else None
            route = (
                [s]
                if s == o.target
                else _path(
                    readout if o.kind == "target_feedback" else graph, s, o.target
                )
            )
            if cycle and route:
                witnesses.append({"state": s, "cycle": cycle, "target_path": route})
        present = bool(witnesses) and (
            o.kind == "dynamic_feedback" or len(witnesses) == len(coordinates)
        )
        rows.append(
            {
                "requirement": o.id,
                "kind": o.kind,
                "target": o.target,
                "status": "present_in_draft" if present else "missing_in_draft",
                "witnesses": witnesses,
            }
        )
    return rows


def edit_receipt(last: dict | None) -> dict | None:
    """A small receipt, not a second copy of accepted/rejected model declarations."""
    if last is None:
        return None
    changes = last.get("edit_effects") or {}
    return {
        "status": last.get("status"),
        "applied": last.get("transaction_applied"),
        "changed": {
            field: [r["key"] for r in rows] for field, rows in changes.items() if rows
        },
        "error": last.get("error"),
        "normalizations": last.get("normalizations", []),
        "note": "The model view is the accepted draft. "
        "A rejected reply changed nothing.",
    }


def _small_issue(issue: dict) -> dict:
    """Retain diagnostic identity/actions without copying another full draft."""
    result = {
        k: deepcopy(v)
        for k, v in issue.items()
        if k
        not in {
            "current_balance",
            "current_process",
            "runtime_definition",
            "variable_declarations",
            "binding_feedback",
            "declaration",
            "scope",
            "policy",
        }
    }
    if "ordinary_terms" in result:
        result["local_terms"] = result.pop("ordinary_terms")
    for key in ("action", "repair_options"):
        value = result.get(key)
        if value is None:
            continue
        texts = value if isinstance(value, list) else [value]
        texts = [
            text.replace("ordinary contributions", "local processes")
            .replace("ordinary reference", "local reference")
            for text in texts
        ]
        result[key] = texts if isinstance(value, list) else texts[0]
    return result


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
    """Build the sole live/preview proposer projection from an accepted draft."""
    check = deferred.assessment(brief, context, target_definitions, draft, contract)
    declarations = checklist.variable_checklist(
        brief, draft, target_definitions, contract
    )
    shared = stages.shared_checklist(brief, draft)
    status = deferred.requirement_status(
        brief, draft, target_definitions, contract, stage, check
    )
    issues = []
    for item in declarations["items"]:
        if item["status"] not in {"missing", "inconsistent"}:
            continue
        issue = _small_issue(item)
        if item["code"] == "feedback_coordinates" and item["definition"] is None:
            issue["action"] = (
                "Declare this target and choose its permitted type first. "
                "A differential target is its own feedback coordinate. For an "
                "algebraic target, identify the intended differential coordinates "
                "in feedback_bindings; paths and cycles are checked later."
            )
        if item["code"] == "memory_assignment":
            r = next((r for r in brief.requirements if r.id == item["name"]), None)
            excluded = (
                set(r.drivers) | set(r.targets)
                if r and r.requires_dynamic_memory
                else set()
            )
            eligible = [
                v.name
                for v in draft.variables
                if v.definition == "differential" and v.name not in excluded
            ]
            issue["type_eligible_states"] = eligible
            if "required" in issue:
                issue["memory_assignment_required"] = issue.pop("required")
            issue["example_scope"] = (
                "These are syntax examples for each type-eligible state, "
                "not scientific endorsements. Choose the intended state(s), "
                "or declare suitable new states. The runtime assigns nothing."
            )
            issue["conditional_examples"] = [
                {
                    "if": f"If {state} carries this mechanism, return this binding.",
                    "edit": {
                        "mechanism_bindings": [
                            {"requirement_id": item["name"], "memory_states": [state]}
                        ]
                    },
                }
                for state in eligible
                if r
            ]
        issues.append(issue)
    if stage != "variables":
        choices = receiver_choices(brief, draft)
        for item in shared["items"]:
            issue = _small_issue(item)
            if item["code"] == "supplied_receiver":
                p = next(p for p in draft.processes if p.name == item["process"])
                uses = [
                    u.model_dump(mode="json")
                    for u in p.uses
                    if u.target in choices["currently_generated"]
                ]
                issue["preserve_valid_effects"] = {
                    "instruction": "Correct receiver entries without deleting "
                    "intended effects. Drivers need not be receivers. If the "
                    "remaining receivers are intended, this example retains "
                    "all drivers and valid uses. Review kind/signs scientifically; "
                    "a one-receiver local law uses influence.",
                    "conditional_process_edit": {
                        **p.model_dump(mode="json"),
                        "uses": uses,
                        "kind": "influence" if len(uses) == 1 else p.kind,
                    }
                    if uses
                    else None,
                    "alternative": "Move the effect into local terms in the "
                    "receiving variable at the topology stage. A removed process "
                    "is shown in the next request's recently_removed_effects.",
                }
            issues.append(issue)
    if stage in {"equations", "repair"}:
        issues.extend(
            _small_issue(e)
            for e in check["errors"]
            if e["code"] != "variable_declaration"
        )
    if last:
        for k in ("name_conflicts", "definition_conflicts"):
            for item in last.get(k, []):
                issue = _small_issue(item)
                if issue.get("code") == "process_name_collision":
                    issue["repair_options"] = [
                        text.replace(
                            "current_draft.declarations.processes",
                            "model_view.processes",
                        )
                        .replace("ordinary contribution", "local process")
                        .replace("ordinary topology", "local topology")
                        .replace("new named law", "new process")
                        for text in issue["repair_options"]
                    ]
                issues.append(issue)
        if last.get("error"):
            issues.append(
                {
                    "code": "last_reply_rejected",
                    "message": last["error"],
                    "action": "Revise the rejected edit; "
                    "the model view remains unchanged.",
                }
            )
    # Identical errors may appear in both declaration and final checks.
    issues = list({str(x): x for x in issues}.values())
    changes = (last or {}).get("edit_effects") or {}
    removed = [
        r["before"]
        for r in changes.get("processes", [])
        if r.get("before") and r.get("after") is None
    ]
    # Public science/data are retained verbatim once. Only redundant typed rules
    # and catalogs outside that brief are removed from the request.
    rows = []
    for r in status["items"]:
        row = {
            k: deepcopy(v)
            for k, v in r.items()
            if k
            not in {
                "scope",
                "action",
                "state_checks",
                "evidence",
                "memory_evidence",
                "reason",
            }
        }
        if "required" in row:
            row["memory_assignment_required"] = row.pop("required")
        rows.append(row)
    return {
        "stage": stage,
        "stage_instructions": stage_text(stage),
        "public_brief": deepcopy(enriched_brief),
        "model_view": model_view(draft),
        "requirements_checklist": rows,
        "memory_binding_options": [
            {
                "requirement_id": r.id,
                "memory_assignment_required": r.requires_dynamic_memory,
            }
            for r in brief.requirements
        ],
        "receiver_choices": receiver_choices(brief, draft),
        "previous_edit": edit_receipt(last),
        "feedback": {
            "issues": issues,
            "provisional_paths": provisional_paths(brief, draft, contract)
            if stage in {"equations", "repair"}
            and check["graph_check_status"] != "assessed"
            else [],
            "provisional_scope": "Declared dependencies only; missing/present "
            "paths are advisory until the complete topology compiles. "
            "These observations never change acceptance.",
            "recently_removed_effects": removed,
            "removal_note": "Keep each scientifically intended effect as a "
            "local or shared process; deletion is not a substitute for fixing "
            "its receiver. No effect is restored automatically."
            if removed
            else None,
            "continuation": continuation,
        },
        "editing_rules": EDITING,
        "response_template": prompts.response_model(stage)(
            stage_complete=False
        ).model_dump(mode="json"),
    }
