"""Realize a selected information source set in a complete, named topology."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, create_model

from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.research.t2_breadth import Component, Topology, check_topology
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    GraphMapDependency,
    build_contract,
    dependency_summary,
)
from autoformalism.search.preproposal_screen import assess_realized_roots
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    _closure,
)
from autoformalism.staged_topology import content_hash

Sources = Annotated[tuple[Identifier, ...], Field(max_length=32)]


class InventoryTopology(Topology):
    """A declared inventory may contain a single public state."""

    components: tuple[Component, ...] = Field(min_length=1, max_length=64)


class SharedInitial(StrictSchema):
    mode: Literal["value"]
    scope: Literal["shared_training"]


class LatentInitial(StrictSchema):
    state: Identifier
    rule: Annotated[SharedInitial | GraphMapDependency, Field(discriminator="mode")]
    justification: str = Field(min_length=1, max_length=2000)


def topology_schema(case: SelectionCase) -> type[StrictSchema]:
    """Pin each receiver and its type to the completed variable stage."""
    # Aliases allow legal scientific identifiers that shadow Pydantic attributes.
    equations = create_model(
        "InventorySources",
        __base__=StrictSchema,
        **{
            f"row_{i}": (Sources, Field(alias=v.name))
            for i, v in enumerate(case.inventory)
        },
    )
    return create_model(
        "SelectedTopology",
        __base__=StrictSchema,
        hypothesis=(str, Field(min_length=1, max_length=1500)),
        sources=(equations, ...),
        latent_initializers=(tuple[LatentInitial, ...], Field(max_length=64)),
    )


def _choice(case, choice):
    candidate_id = (
        choice["candidate_id"] if isinstance(choice, dict) else choice.candidate_id
    )
    rationale = choice["rationale"] if isinstance(choice, dict) else choice.rationale
    row = next((r for r in case.candidates if r.candidate_id == candidate_id), None)
    if row is None:
        raise ValueError("Selected ID is outside this case's screened catalogue")
    return row, rationale


def _inputs(case, frozen):
    evidence = TrainingCompatibilityEvidence.model_validate(frozen["evidence"])
    if case.case_id != frozen["case_id"] or case.evidence_sha256 != evidence.sha256:
        raise ValueError("Selection and topology evidence differ")
    brief = PublicScientificBrief.model_validate(frozen["public_brief"])
    return evidence, brief


def _state_names(case):
    kinds = {"dynamic", "differential", "algebraic"}
    if any(row.kind not in kinds for row in case.inventory):
        raise ValueError("Unsupported variable definition in stage inventory")
    return {v.name for v in case.inventory if v.kind != "algebraic"}


def _initial_channels(states, evidence, frozen):
    permitted = set(frozen["initials"]) | {frozen["target"]}
    observed = {
        s: s
        for s in sorted(states)
        if s in permitted
        and s in evidence.contract.initial_rules
        and evidence.contract.initial_rules[s].mode == "observed_initial"
    }
    return observed, sorted(permitted)


def topology_messages(case, choice, frozen_case):
    """Expose the choice, permitted channels and public graph obligations."""
    row, rationale = _choice(case, choice)
    evidence, brief = _inputs(case, frozen_case)
    observed, permitted = _initial_channels(_state_names(case), evidence, frozen_case)
    packet = {
        "target": case.target,
        "selected_information": row.sources,
        "rationale": rationale,
        "public_task": case.public_task,
        "training_evidence": case.evidence_summary,
        "inventory": [v.model_dump() for v in case.inventory],
        "supplied_histories": frozen_case["histories"],
        "fixed_covariates": sorted(
            set(evidence.context.fixed_covariates).intersection(permitted)
        ),
        "time": evidence.context.time_symbol,
        "observed_initials": observed,
        "permitted_baseline_readings": permitted,
        "required_paths": [v.model_dump() for v in brief.requirements],
        "required_target_dependencies": [
            v.model_dump() for v in brief.target_dependencies
        ],
    }
    return (
        "Propose a complete topology for the declared variables. List sources for each "
        "derivative or algebraic value. Use the generated values of modeled variables. "
        "Give the selected information a scientifically justified path to the target. "
        "Include the other public pathways. For each latent differential state give "
        "an initializer: a shared constant (mode value, scope shared_training) or a "
        "map of state_initials and public_initials at time zero. The runtime supplies "
        "the observed initials. Give the topology and its scientific hypothesis.",
        json.dumps(packet, ensure_ascii=False),
    )


def _actual_assessment(target, actual, evidence, minimum_target_gap=0.0):
    histories = tuple(sorted(v[8:] for v in actual if v.startswith("history:")))
    initials = tuple(sorted(v[8:] for v in actual if v.startswith("initial:")))
    screened, measured = assess_realized_roots(
        evidence,
        target,
        histories=histories,
        initials=initials,
        minimum_target_gap=minimum_target_gap,
    )
    return {
        **measured.model_dump(),
        "actual_options": sorted(actual),
        "unavailable": list(screened.unavailable),
        "pair_count": screened.pair_count,
        "minimum_target_gap": minimum_target_gap,
    }


def assess_topology(case, choice, value, frozen_case):
    """Check public paths and the information upstream of each measured state."""
    row, _ = _choice(case, choice)
    evidence, brief = _inputs(case, frozen_case)
    context = evidence.context
    states = _state_names(case)
    generated = {v.name for v in case.inventory}
    observed, permitted = _initial_channels(states, evidence, frozen_case)
    result = {
        "candidate_id": row.candidate_id,
        "scientific_correctness_assessed": False,
    }
    try:
        reply = topology_schema(case).model_validate(value).model_dump(by_alias=True)
        graph = reply["sources"]
        if any(len(src) != len(set(src)) for src in graph.values()):
            raise ValueError("Repeated source in a receiver's list")
        if len(graph) > brief.limits.generated_variables:
            raise ValueError("Generated inventory exceeds the public variable limit")
        if context.forbid_latent_states and states - observed.keys():
            raise ValueError("The public task forbids latent differential states")
        actual_topology = InventoryTopology(
            hypothesis=reply["hypothesis"],
            components=tuple(
                Component(
                    name=v.name,
                    kind="dynamic" if v.name in states else "algebraic",
                    role=v.name,
                    sources=graph[v.name],
                )
                for v in case.inventory
            ),
        )
        handoff = GraphDependencyHandoff.model_validate(
            {"records": reply["latent_initializers"]}
        )
        contract = build_contract(
            state_names=sorted(states),
            context=context,
            permitted_public_initials=permitted,
            observed_initial_channels=observed,
            scientific_context=brief.scientific_context,
            provenance=case.case_id,
        )
        initialization = dependency_summary(handoff, contract)
        # Freeze which channels are available throughout the forecast.
        allowed_rhs = (
            generated
            | set(frozen_case["histories"])
            | (set(context.fixed_covariates) & set(permitted))
            | {context.time_symbol}
        )
        unknown = set().union(*graph.values()) - allowed_rhs
        if unknown:
            raise ValueError(
                f"Unavailable or undeclared ongoing sources: {sorted(unknown)}"
            )
        # A modeled auxiliary owns its equation and cannot also supply a history.
        supplied = context.forcing_channels - generated
        reduced = context.model_copy(
            update={
                "auxiliaries": tuple(
                    v for v in context.auxiliaries if v not in generated
                ),
                "forcing_bounds": {
                    k: v for k, v in context.forcing_bounds.items() if k in supplied
                },
            }
        )
        check_topology(
            actual_topology,
            reduced,
            brief,
            initial_state_dependencies={
                state: sources["state_initials"]
                for state, sources in initialization.items()
            },
        )
    except ValueError as exc:
        return {
            **result,
            "status": "invalid_topology",
            "reason": str(exc),
            "public_graph_check": "failed",
        }

    checks = {}
    exact_target = None
    for target in context.targets:
        if target not in states:
            continue
        upstream = _closure(target, graph)
        histories = (
            upstream & set(context.external_inputs + context.auxiliaries) - generated
        )
        initials = upstream & set(context.fixed_covariates)
        for state in upstream & states:
            initials.update(initialization[state]["public_initials"])
        actual = {f"history:{v}" for v in histories} | {
            f"initial:{v}" for v in initials
        }
        if target in observed:
            gap = (
                frozen_case.get("minimum_target_gap", 0.0)
                if target == case.target
                else 0.0
            )
            checks[target] = _actual_assessment(target, actual, evidence, gap)
            if gap:
                exact_target = _actual_assessment(target, actual, evidence)
        else:
            checks[target] = {
                "status": "unresolved",
                "actual_options": sorted(actual),
                "unavailable": ["No permitted observed initial for this target."],
            }

    missing = sorted(
        (set(row.sources) | set(case.required_options))
        - set(checks[case.target]["actual_options"])
    )
    verdicts = {item["status"] for item in checks.values()}
    if "contradicted" in verdicts:
        status = "training_conflict"
    elif missing:
        status = "selected_route_missing"
    elif "unresolved" in verdicts:
        status = "unresolved"
    else:
        status = "topology_screen_passed"
    initials_by_state = {}
    for record in handoff.records:
        rule = record.rule.model_dump(mode="json")
        for key in ("state_initials", "public_initials"):
            if key in rule:
                rule[key] = sorted(rule[key])
        initials_by_state[record.state] = rule
    return {
        **result,
        "status": status,
        "public_graph_check": "passed",
        "missing_selected_or_required": missing,
        "training_checks": checks,
        "exact_selected_target_check": exact_target or checks[case.target],
        "sources": graph,
        "structure_sha256": content_hash(
            {
                "states": sorted(states),
                "sources": {k: sorted(v) for k, v in graph.items()},
                "initializers": initials_by_state,
            }
        ),
        "initializers": reply["latent_initializers"],
        "hypothesis": reply["hypothesis"],
    }
