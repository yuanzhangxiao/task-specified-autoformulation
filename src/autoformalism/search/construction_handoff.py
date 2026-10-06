"""Explicit binding questions and lossless named-law delivery for draft construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoformalism.schemas.staged_topology import PublicScientificBrief
    from autoformalism.search.construction_ledger import Draft, DraftPatch
    from autoformalism.search.public_graph_obligations import PublicGraphContract


def binding_context(
    brief: PublicScientificBrief, draft: Draft, contract: PublicGraphContract | None
) -> dict:
    """Show field formats and choices without assigning a scientific mediator."""
    states = [v.name for v in draft.variables if v.definition == "differential"]
    definitions = {v.name: v.definition for v in draft.variables}
    targets = sorted(
        {o.target for o in contract.obligations if o.kind == "target_feedback"}
        if contract
        else set()
    )
    assigned = {b.requirement_id for b in draft.mechanism_bindings}
    return {
        "mechanism_bindings": [
            {
                "requirement_id": r.id,
                "required": r.requires_dynamic_memory,
                "assignment_missing": r.requires_dynamic_memory
                and r.id not in assigned,
                "question": "Which differential states realize this public mechanism?",
                "public_requirement": r.public_requirement,
                "eligible_declared_states_by_type_only": [
                    s
                    for s in states
                    if not r.requires_dynamic_memory
                    or s not in {*r.drivers, *r.targets}
                ],
                "reply_example_replace_placeholder": {
                    "mechanism_bindings": [
                        {"requirement_id": r.id, "memory_states": ["CHOSEN_STATE"]}
                    ]
                },
            }
            for r in brief.requirements
        ],
        "feedback_bindings": {
            "applicable_targets": targets,
            "needs_proposer_coordinates": [
                t for t in targets if definitions.get(t) == "algebraic"
            ],
            "rule": (
                "Only listed algebraic targets need storage/readout coordinates. "
                "Differential targets are already their own coordinates. This field "
                "does not satisfy a mechanism_bindings requirement. An empty "
                "applicable_targets list means no feedback binding is requested."
            ),
            "reply_examples_replace_placeholder": [
                {"feedback_bindings": [{"target": t, "states": ["CHOSEN_STATE"]}]}
                for t in targets
                if definitions.get(t) != "differential"
            ],
        },
        "placeholder_rule": (
            "CHOSEN_STATE is a format placeholder, not a suggested variable. "
            "Choose scientifically; declare another state if needed. Type eligibility "
            "is not scientific assignment. Empty edit lists preserve existing entries."
        ),
        "existing_binding_removals": [
            {
                "entry": b.model_dump(mode="json"),
                "edit": {"remove_bindings": [b.requirement_id]},
            }
            for b in draft.mechanism_bindings
        ]
        + [
            {
                "entry": b.model_dump(mode="json"),
                "edit": {"remove_feedback_bindings": [b.target]},
            }
            for b in draft.feedback_bindings
        ],
    }


def normalize_consumers(
    draft: Draft, patch: DraftPatch, previous: Draft
) -> tuple[Draft, list[dict]]:
    """Canonicalize only unambiguous signed identity references in this reply.

    No process kind, sign, driver or known conversion is inferred or changed.
    Conflicting/repeated/composite references remain for explicit proposer repair.
    """
    from autoformalism.search.construction_ledger import Draft

    value = draft.model_dump(mode="json")
    processes = {p["name"]: p for p in value["processes"]}
    replaced = {p.name for p in patch.processes}
    old = {p.name: p for p in previous.processes}
    edited = {e.name for e in patch.equations}
    generated = {v.name for v in draft.variables}
    log = []
    for equation in value["equations"]:
        target = equation["name"]
        if target not in edited or target not in generated:
            continue  # Pending declarations cannot silently become consumers.
        retained = []
        for index, term in enumerate(equation["terms"]):
            sources = term["sources"]
            process = processes.get(sources[0]) if len(sources) == 1 else None
            sign = term["outer_weight_sign"]
            if process is None or sign not in {"positive", "negative"}:
                retained.append(term)
                continue
            name = process["name"]
            mentions = sum(name in t["sources"] for t in equation["terms"])
            uses = [u for u in process["uses"] if u["target"] == target]
            removed_here = (
                name in replaced
                and not uses
                and name in old
                and any(u.target == target for u in old[name].uses)
            )
            if (
                mentions != 1
                or len(uses) > 1
                or removed_here
                or (uses and uses[0]["sign"] != sign)
            ):
                retained.append(term)
                continue
            if not uses:
                use = {"target": target, "sign": sign, "conversion": None}
                process["uses"].append(use)
            else:
                use = uses[0]
            log.append(
                {
                    "code": "explicit_process_consumer"
                    if not uses
                    else "repeated_process_use",
                    "process": name,
                    "target": target,
                    "ordinary_term_index": index,
                    "original_term": term,
                    "recorded_use": dict(use),
                    "conversion_inferred": False,
                    "scientific_type_changed": False,
                }
            )
        equation["terms"] = retained
    return Draft.model_validate(value), log


def process_reference_issues(draft: Draft) -> list[dict]:
    """Describe actual consumer conflicts, rather than ordering term deletion."""
    issues = []
    for e in draft.equations:
        for p in draft.processes:
            terms = [t for t in e.terms if p.name in t.sources]
            if not terms:
                continue
            uses = [u for u in p.uses if u.target == e.name]
            if len(uses) == 1 and all(
                t.sources == (p.name,) and t.outer_weight_sign.value == uses[0].sign
                for t in terms
            ):
                continue
            issues.append(
                {
                    "code": "process_consumer_decision",
                    "process": p.name,
                    "target": e.name,
                    "ordinary_terms": [t.model_dump(mode="json") for t in terms],
                    "declared_uses_for_target": [
                        u.model_dump(mode="json") for u in uses
                    ],
                    "current_process": p.model_dump(mode="json"),
                    "repair_options": [
                        "For a use of this same law, replace the process declaration "
                        "with "
                        "the intended signed consumer and conversion; preserve other "
                        "uses. "
                        "Remove the corresponding ordinary reference, "
                        "not the physical effect.",
                        "For a distinct composite law, explicitly choose another "
                        "representation "
                        "and coordinate its equations; runtime will not infer "
                        "its meaning.",
                        "Withdraw the contribution only if scientifically unintended.",
                    ],
                    "warning": (
                        "No automatic sign flip, type change or conversion inference."
                    ),
                }
            )
    return issues


def consumer_destination_issues(draft: Draft) -> list[dict]:
    """Never silently drop a use at a runtime-generated process definition."""
    names = {p.name for p in draft.processes}
    return [
        {
            "code": "process_definition_consumer",
            "process": p.name,
            "use": use.model_dump(mode="json"),
            "reason": (
                "A process definition is its one law, not an equation receiving "
                "additional signed uses. It cannot consume itself or another law "
                "through uses. No consumer was removed or reassigned."
            ),
            "repair_options": [
                "Replace this process's uses with its intended generated-equation "
                "consumers; preserve their signs and conversions. One actual "
                "consumer is a valid local influence, not a shared process.",
                "If a composite process law is intended, declare that dependency "
                "in depends_on and remove this use; choose its scientific meaning "
                "explicitly. Algebraic cycles remain invalid.",
            ],
        }
        for p in draft.processes
        for use in p.uses
        if use.target in names
    ]


def normalize_definition_repeats(
    raw: object, draft: Draft
) -> tuple[object, list[dict]]:
    """Drop only an exact repeat of an automatically generated law definition.

    Validate the entire edit first. Conflicting operations and different laws
    remain unchanged for the ordinary transactional validator to reject.
    """
    from autoformalism.search import shared_process_contract, signed_processes
    from autoformalism.search.construction_ledger import DraftPatch

    try:
        patch = DraftPatch.model_validate(raw)
    except ValueError:
        return raw, []
    if set(patch.remove_processes) & {p.name for p in patch.processes} or set(
        patch.remove_equations
    ) & {e.name for e in patch.equations}:
        return raw, []
    processes = {
        p.name: p for p in draft.processes if p.name not in patch.remove_processes
    }
    processes.update((p.name, p) for p in patch.processes)
    removed, log = set(), []
    for e in patch.equations:
        if e.name not in processes:
            continue
        expected = shared_process_contract.definition(
            signed_processes.binding_for(processes[e.name])
        )
        if e.terms == expected.terms:
            removed.add(e.name)
            log.append(
                {
                    "code": "repeated_process_definition",
                    "equation": e.model_dump(mode="json"),
                    "canonical_definition": expected.model_dump(mode="json"),
                    "scientific_content_changed": False,
                }
            )
    if not removed:
        return raw, []
    return {
        **raw,
        "equations": [e for e in raw["equations"] if e["name"] not in removed],
    }, log


def algebraic_cycle_issues(equations: tuple) -> list[dict]:
    """Give a concrete dependency cycle; differential feedback remains allowed."""
    algebraic = {e.name: e for e in equations if e.definition == "algebraic"}
    edges = {
        n: sorted({s for t in e.terms for s in t.sources if s in algebraic})
        for n, e in algebraic.items()
    }
    done: set[str] = set()

    def visit(name: str, path: list[str]) -> list[str] | None:
        if name in path:
            return [*path[path.index(name) :], name]
        if name in done:
            return None
        for child in edges[name]:
            cycle = visit(child, [*path, name])
            if cycle:
                return cycle
        done.add(name)
        return None

    for name in sorted(edges):
        cycle = visit(name, [])
        if cycle:
            return [
                {
                    "code": "algebraic_dependency_cycle",
                    "cycle": cycle,
                    "edge_direction": "equation depends on source",
                    "equations": [
                        algebraic[n].model_dump(mode="json") for n in cycle[:-1]
                    ],
                    "repair_options": [
                        "Reconsider the dependencies and consumers on this cycle. "
                        "Replace the intended equations or process declarations "
                        "together; runtime cannot choose which link is wrong.",
                        "If a quantity is scientifically a state, explicitly revise "
                        "its type and derivative dependencies; do not change type "
                        "merely to suppress the cycle check.",
                    ],
                    "process_definition_rule": (
                        "Edit depends_on to revise a named law. Runtime rebuilds its "
                        "definition; do not provide a second equation for it."
                    ),
                }
            ]
    return []


def edit_effects(before: Draft, after: Draft) -> dict:
    """Expose replaced RHSs and changed uses so the next call sees consequences."""
    changes = {}
    for field, key in (
        ("variables", "name"),
        ("equations", "name"),
        ("processes", "name"),
        ("mechanism_bindings", "requirement_id"),
        ("feedback_bindings", "target"),
    ):
        old = {
            getattr(v, key): v.model_dump(mode="json") for v in getattr(before, field)
        }
        new = {
            getattr(v, key): v.model_dump(mode="json") for v in getattr(after, field)
        }
        changes[field] = [
            {"key": k, "before": old.get(k), "after": new.get(k)}
            for k in sorted(old.keys() | new.keys())
            if old.get(k) != new.get(k)
        ]
    return changes
