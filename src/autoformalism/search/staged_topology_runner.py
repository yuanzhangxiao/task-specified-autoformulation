"""Fixed scientific agenda and bounded local repairs for topology construction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
    VariableAgendaItem,
    VariableReply,
    equation_reply_model,
)
from autoformalism.search.staged_topology_prompts import (
    render_equation_topology_system_prompt,
    render_equation_topology_user_prompt,
    render_variable_identification_system_prompt,
    render_variable_identification_user_prompt,
)
from autoformalism.staged_topology import (
    audit_equation_polarity_policy,
    audit_explicit_equation_polarity,
    compile_equation_polarity_policy,
    content_hash,
    freeze_inventory,
    lower_topology,
    merge_variable_reply,
    merge_variable_reply_partially,
    public_structure_checks,
    validate_equation,
    validate_inventory_revision,
)


def scientific_agenda(brief: PublicScientificBrief) -> tuple[VariableAgendaItem, ...]:
    """Visit shared mechanisms together, then complete each public target."""
    return tuple(
        VariableAgendaItem(
            agenda_id=f"mechanism_{item.id}",
            purpose="mechanism_variables",
            requirement_ids=(item.id,),
            targets=item.targets,
            drivers=item.drivers,
            requires_dynamic_memory=item.requires_dynamic_memory,
        )
        for item in brief.requirements
    ) + tuple(
        VariableAgendaItem(
            agenda_id=f"target_{item.name}",
            purpose="target_completion",
            targets=(item.name,),
        )
        for item in brief.public_variables
        if item.data_role == "target"
    )


def runtime_seeded_inventory(
    brief: PublicScientificBrief,
) -> tuple[ScientificVariable, ...]:
    """Register routine public forcing needed by explicit public obligations."""
    public_roles = {item.name: item.data_role for item in brief.public_variables}
    target_names = {
        item.name for item in brief.public_variables if item.data_role == "target"
    }
    required_names = {name for item in brief.requirements for name in item.drivers} | {
        name for item in brief.target_dependencies for name in item.acceptable_sources
    }
    required_names |= {
        item.name
        for item in brief.public_variables
        if item.data_role in {"external_input", "covariate", "time"}
    }
    return tuple(
        ScientificVariable(
            name=name,
            definition="supplied",
            scientific_role=f"runtime-registered public {public_roles[name]}",
        )
        for name in sorted(required_names - target_names)
    )


def _agenda_gaps(
    brief: PublicScientificBrief,
    item: VariableAgendaItem,
    inventory: tuple[ScientificVariable, ...],
    memory_candidates: dict[str, set[str]],
) -> tuple[str, ...]:
    """Return only unresolved typed obligations for one variable agenda item."""
    active = {
        variable.name for variable in inventory if variable.definition != "unused"
    }
    generated = {
        variable.name
        for variable in inventory
        if variable.definition in {"differential", "algebraic"}
    }
    gaps = [
        f"generated_target:{name}" for name in item.targets if name not in generated
    ]
    gaps.extend(f"active_driver:{name}" for name in item.drivers if name not in active)
    if item.requires_dynamic_memory:
        gaps.extend(
            f"dynamic_memory_mediator:{identifier}"
            for identifier in item.requirement_ids
            if not memory_candidates.get(identifier)
        )
    return tuple(gaps)


def _record_memory_candidates(
    brief: PublicScientificBrief,
    item: VariableAgendaItem,
    reply: VariableReply,
    decisions: tuple[dict[str, object], ...],
    inventory: tuple[ScientificVariable, ...],
    memory_candidates: dict[str, set[str]],
) -> None:
    """Associate explicitly returned differential mediators with this mechanism."""
    if not item.requires_dynamic_memory:
        return
    accepted = {
        str(decision["name"]) for decision in decisions if bool(decision["accepted"])
    }
    public_targets = {
        variable.name
        for variable in brief.public_variables
        if variable.data_role == "target"
    }
    excluded = set(item.targets) | set(item.drivers) | public_targets
    returned = {variable.name for variable in reply.variables} & accepted
    candidates = [
        variable.name
        for variable in reply.variables
        if variable.name in returned
        and variable.name not in excluded
        and next(
            current.definition for current in inventory if current.name == variable.name
        )
        == "differential"
    ]
    for identifier in item.requirement_ids:
        if candidates and not memory_candidates.get(identifier):
            memory_candidates[identifier] = {candidates[0]}


def _equation_agenda(
    brief: PublicScientificBrief,
    selected: ScientificVariable,
    memory_candidates: dict[str, set[str]],
) -> dict[str, object]:
    """Route only public obligations relevant to the selected left-hand side."""
    obligations: list[dict[str, object]] = []
    for requirement in brief.requirements:
        if selected.name in requirement.targets:
            obligations.append(
                {
                    "kind": "required_target_path",
                    "requirement_id": requirement.id,
                    "drivers": list(requirement.drivers),
                    "requires_dynamic_memory": requirement.requires_dynamic_memory,
                    "candidate_memory_mediators": sorted(
                        memory_candidates.get(requirement.id, set())
                    ),
                }
            )
        if selected.name in memory_candidates.get(requirement.id, set()):
            obligations.append(
                {
                    "kind": "required_memory_input",
                    "requirement_id": requirement.id,
                    "drivers": list(requirement.drivers),
                    "targets": list(requirement.targets),
                }
            )
    for dependency in brief.target_dependencies:
        if dependency.target == selected.name:
            obligations.append(
                {
                    "kind": "required_target_composition",
                    "acceptable_sources": list(dependency.acceptable_sources),
                    "public_requirement": dependency.public_requirement,
                }
            )
    return {
        "purpose": "define the selected variable",
        "selected_lhs": selected.name,
        "public_obligations": obligations,
    }


def _validate_memory_equation_obligations(
    brief: PublicScientificBrief,
    selected: ScientificVariable,
    equations: tuple[EquationDefinition, ...],
    definition: EquationDefinition,
    memory_candidates: dict[str, set[str]],
) -> None:
    """Enforce the typed driver-to-memory-to-target path locally."""
    dependencies = {
        equation.name: {source for term in equation.terms for source in term.sources}
        for equation in (*equations, definition)
    }
    ancestors = _topology_ancestors(selected.name, dependencies)
    for requirement in brief.requirements:
        candidates = memory_candidates.get(requirement.id, set())
        if not requirement.requires_dynamic_memory or not candidates:
            continue
        if selected.name in candidates:
            missing = set(requirement.drivers) - ancestors
            if missing:
                raise ValueError(
                    "dynamic-memory mediator must receive the required driver path: "
                    f"requirement={requirement.id}, missing={sorted(missing)}"
                )
        if selected.name in requirement.targets and not candidates.intersection(
            ancestors
        ):
            raise ValueError(
                "dynamic-memory target must receive a path through the selected "
                f"mediator: requirement={requirement.id}, "
                f"mediators={sorted(candidates)}"
            )


def _topology_ancestors(name: str, dependencies: dict[str, set[str]]) -> set[str]:
    """Compute dependency ancestors without importing a private helper."""
    pending = list(dependencies.get(name, set()))
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend(dependencies.get(node, set()) - visited)
    return visited


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _variables(inventory: tuple[ScientificVariable, ...]) -> str:
    return _json([item.model_dump(mode="json") for item in inventory])


def run_staged_topology(
    brief: PublicScientificBrief,
    context: ValidationContext,
    client: StagedTopologyClient,
    output: Path,
    *,
    initial_inventory: tuple[ScientificVariable, ...] | None = None,
    audit_public_polarity_policy: bool = False,
    hybrid_variable_construction: bool = False,
    proposer_owns_unfixed_signs: bool = False,
) -> dict[str, Any]:
    """Build one topology without functions, numerical data, or a scientific judge."""
    inventory: tuple[ScientificVariable, ...] = initial_inventory or (
        runtime_seeded_inventory(brief) if hybrid_variable_construction else ()
    )
    equations: tuple[EquationDefinition, ...] = ()
    memory_candidates: dict[str, set[str]] = {}
    polarity_policies: list[dict[str, Any]] = []
    polarity_audits: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    brief_json = brief.model_dump_json()
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        atomic_json(
            output / "progress.json",
            {
                "inventory": [item.model_dump(mode="json") for item in inventory],
                "equations": [item.model_dump(mode="json") for item in equations],
                "memory_candidates": {
                    key: sorted(value) for key, value in memory_candidates.items()
                },
                "events": events,
            },
        )

    def request(
        stage: str,
        system: str,
        renderer: Callable[[str | None], str],
        model: type[StrictSchema],
        validate: Callable[[Any], Any],
    ) -> Any:
        diagnostic: str | None = None
        for attempt in range(client.settings.attempts_per_step):
            rejected: object = None
            record = client.call(
                system=system,
                user=renderer(diagnostic),
                response_model=model,
                step=stage,
                attempt=attempt,
            )
            try:
                rejected = visible_response(record)
                reply = model.model_validate(rejected)
                accepted = validate(reply)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)[:6000]
                if rejected is None:
                    choices = record.get("raw_response", {}).get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        visible = choices[0].get("message", {}).get("content")
                        if isinstance(visible, str):
                            rejected = visible
                diagnostic = _json({"rejected_response": rejected, "error": error})
                events.append(
                    {
                        "step": stage,
                        "attempt": attempt,
                        "accepted": False,
                        "request_hash": record["request_hash"],
                        "error": error,
                    }
                )
                checkpoint()
                continue
            events.append(
                {
                    "step": stage,
                    "attempt": attempt,
                    "accepted": True,
                    "request_hash": record["request_hash"],
                }
            )
            return accepted
        raise ValueError(f"bounded local repair exhausted for {stage}")

    status = "failed"
    failure: str | None = None
    revision: object = None
    topology = None
    aliases: dict[str, str] = {}
    agenda = scientific_agenda(brief) if initial_inventory is None else ()
    try:
        for index, item in enumerate(agenda):
            if hybrid_variable_construction:
                gaps = _agenda_gaps(brief, item, inventory, memory_candidates)
                if not gaps:
                    events.append(
                        {
                            "step": f"variables_{index}",
                            "agenda_id": item.agenda_id,
                            "accepted": True,
                            "skipped_as_resolved": True,
                        }
                    )
                    checkpoint()
                    continue
                diagnostic: str | None = None
                for attempt in range(client.settings.attempts_per_step):
                    rejected: object = None
                    record = client.call(
                        system=render_variable_identification_system_prompt(),
                        user=render_variable_identification_user_prompt(
                            public_brief_json=brief_json,
                            agenda_json=item.model_dump_json(),
                            inventory_json=_variables(inventory),
                            diagnostics_json=diagnostic,
                        ),
                        response_model=VariableReply,
                        step=f"variables_{index}",
                        attempt=attempt,
                    )
                    try:
                        rejected = visible_response(record)
                        reply = VariableReply.model_validate(rejected)
                    except (ValueError, TypeError, KeyError) as exc:
                        error = str(exc)[:6000]
                        diagnostic = _json(
                            {
                                "rejected_response": rejected,
                                "error": error,
                                "unresolved_obligations": list(gaps),
                            }
                        )
                        events.append(
                            {
                                "step": f"variables_{index}",
                                "agenda_id": item.agenda_id,
                                "attempt": attempt,
                                "accepted": False,
                                "partial_acceptance": False,
                                "request_hash": record["request_hash"],
                                "error": error,
                            }
                        )
                        checkpoint()
                        continue
                    inventory, decisions = merge_variable_reply_partially(
                        brief, inventory, reply
                    )
                    _record_memory_candidates(
                        brief,
                        item,
                        reply,
                        decisions,
                        inventory,
                        memory_candidates,
                    )
                    gaps = _agenda_gaps(brief, item, inventory, memory_candidates)
                    accepted_names = [
                        str(decision["name"])
                        for decision in decisions
                        if bool(decision["accepted"])
                    ]
                    rejected_decisions = [
                        decision for decision in decisions if not decision["accepted"]
                    ]
                    complete = not gaps
                    events.append(
                        {
                            "step": f"variables_{index}",
                            "agenda_id": item.agenda_id,
                            "attempt": attempt,
                            "accepted": complete,
                            "partial_acceptance": bool(accepted_names) and not complete,
                            "accepted_variable_names": accepted_names,
                            "rejected_variables": rejected_decisions,
                            "unresolved_obligations": list(gaps),
                            "request_hash": record["request_hash"],
                            "error": None
                            if complete
                            else f"unresolved variable obligations: {list(gaps)}",
                        }
                    )
                    checkpoint()
                    if complete:
                        break
                    diagnostic = _json(
                        {
                            "retained_valid_variables": accepted_names,
                            "rejected_variables": rejected_decisions,
                            "unresolved_obligations": list(gaps),
                            "instruction": (
                                "Return only variables needed to resolve these "
                                "obligations; retained variables must not be repeated."
                            ),
                        }
                    )
                else:
                    raise ValueError(
                        "bounded hybrid repair exhausted for "
                        f"variables_{index}: {list(gaps)}"
                    )
            else:
                is_final_item = index == len(agenda) - 1

                def accept_variables(
                    reply: VariableReply,
                    parent=inventory,
                    final_item=is_final_item,
                ) -> tuple[ScientificVariable, ...]:
                    merged = merge_variable_reply(brief, parent, reply)
                    return freeze_inventory(brief, merged) if final_item else merged

                inventory = request(
                    f"variables_{index}",
                    render_variable_identification_system_prompt(),
                    lambda diagnostic,
                    item=item,
                    inventory=inventory: render_variable_identification_user_prompt(
                        public_brief_json=brief_json,
                        agenda_json=item.model_dump_json(),
                        inventory_json=_variables(inventory),
                        diagnostics_json=diagnostic,
                    ),
                    VariableReply,
                    accept_variables,
                )
            checkpoint()
        inventory = freeze_inventory(brief, inventory)
        allowed = tuple(item.name for item in inventory if item.definition != "unused")
        model = equation_reply_model(
            allowed, maximum_terms=brief.limits.terms_per_equation
        )
        memory_names = {name for names in memory_candidates.values() for name in names}
        public_targets = {
            item.name for item in brief.public_variables if item.data_role == "target"
        }
        equation_order = sorted(
            inventory,
            key=lambda item: (
                0
                if item.name in memory_names
                else 2
                if item.name in public_targets
                else 1,
                item.name,
            ),
        )
        for selected in equation_order:
            if selected.definition not in {"differential", "algebraic"}:
                continue
            polarity_policy = compile_equation_polarity_policy(
                brief,
                selected.name,
                proposer_owns_unfixed_sources=proposer_owns_unfixed_signs,
            )

            def accept_equation(
                reply: Any, selected=selected, equations=equations
            ) -> Any:
                if reply.inventory_revision is not None:
                    validate_inventory_revision(
                        brief, inventory, reply.inventory_revision
                    )
                    return reply
                definition = EquationDefinition(
                    name=selected.name,
                    definition=selected.definition,
                    terms=reply.terms,
                )
                validate_equation(inventory, equations, definition, brief.limits)
                if hybrid_variable_construction:
                    _validate_memory_equation_obligations(
                        brief,
                        selected,
                        equations,
                        definition,
                        memory_candidates,
                    )
                return definition

            accepted = request(
                f"equation_{selected.name}",
                render_equation_topology_system_prompt(),
                lambda diagnostic,
                selected=selected,
                equations=equations,
                polarity_policy=polarity_policy: render_equation_topology_user_prompt(
                    public_brief_json=brief_json,
                    agenda_json=_json(
                        _equation_agenda(brief, selected, memory_candidates)
                    ),
                    inventory_json=_variables(inventory),
                    selected_lhs_json=_json(
                        {"name": selected.name, "definition": selected.definition}
                    ),
                    equation_sketch_json=_json(
                        [item.model_dump(mode="json") for item in equations]
                    ),
                    allowed_sources_json=_json(allowed),
                    polarity_policy_json=(
                        polarity_policy.model_dump_json()
                        if audit_public_polarity_policy
                        else None
                    ),
                    diagnostics_json=diagnostic,
                ),
                model,
                accept_equation,
            )
            if not isinstance(accepted, EquationDefinition):
                revision = accepted.inventory_revision.model_dump(mode="json")
                status = "inventory_revision_requested"
                break
            equations += (accepted,)
            if audit_public_polarity_policy:
                polarity_policies.append(polarity_policy.model_dump(mode="json"))
                polarity_audits.append(
                    audit_explicit_equation_polarity(accepted, polarity_policy)
                    if proposer_owns_unfixed_signs
                    else audit_equation_polarity_policy(accepted, polarity_policy)
                )
            checkpoint()
        else:
            topology, aliases = lower_topology(brief, inventory, equations, context)
            status = "complete"
    except ValueError as exc:
        failure = str(exc)[:6000]
    checks = public_structure_checks(brief, equations)
    source_checks = [
        item for item in checks if item["kind"] in {"driver_path", "composition_path"}
    ]
    active_sources = {item.name for item in inventory if item.definition != "unused"}
    source_closure_passed = all(
        set(term.sources) <= active_sources
        for equation in equations
        for term in equation.terms
    )
    result = {
        "protocol": "scientific-staged-topology-1",
        "status": status,
        "error": failure,
        "complete_topology": topology is not None,
        "public_structure_checks_passed": topology is not None
        and all(item["passed"] for item in checks),
        "public_structure_checks": checks,
        "public_source_coverage_passed": topology is not None
        and source_closure_passed
        and all(bool(item["passed"]) for item in source_checks),
        "public_mechanism_coverage_passed": topology is not None
        and all(bool(item["passed"]) for item in checks),
        "public_target_coverage_passed": topology is not None
        and {item.name for item in brief.public_variables if item.data_role == "target"}
        <= {item.name for item in equations},
        "public_polarity_policy_enabled": audit_public_polarity_policy,
        "equation_polarity_policies": polarity_policies,
        "polarity_policy_audits": polarity_audits,
        "public_polarity_consistency_passed": (
            topology is not None
            and len(polarity_audits) == len(equations)
            and all(bool(item["passed"]) for item in polarity_audits)
            if audit_public_polarity_policy
            else None
        ),
        "inventory": [item.model_dump(mode="json") for item in inventory],
        "inventory_sha256": content_hash(
            [item.model_dump(mode="json") for item in inventory]
        ),
        "equations": [item.model_dump(mode="json") for item in equations],
        "topology": topology.model_dump(mode="json") if topology else None,
        "generated_auxiliary_aliases": aliases,
        "memory_candidates": {
            key: sorted(value) for key, value in memory_candidates.items()
        },
        "inventory_revision": revision,
        "events": events,
        "physical_requests": len(client.records),
        "budget_charge": sum(item.get("budget_charge", 0) for item in client.records),
        "observed_total_tokens": sum(
            item.get("observed_total_tokens") or 0 for item in client.records
        ),
        "unmeasured_requests": sum(
            item.get("observed_total_tokens") is None for item in client.records
        ),
        "provider_seconds": sum(
            item.get("latency_seconds", 0) for item in client.records
        ),
        "test_data_opened": False,
        "private_reference_opened": False,
        "function_generation_performed": False,
        "diagnostic_inventory_supplied": initial_inventory is not None,
        "hybrid_variable_construction": hybrid_variable_construction,
        "proposer_owns_unfixed_signs": proposer_owns_unfixed_signs,
        "parameter_fitting_performed": False,
    }
    checkpoint()
    atomic_json(output / "result.json", result)
    return result
