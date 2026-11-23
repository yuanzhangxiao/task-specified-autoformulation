"""Realize a selected information source set in a complete, named topology."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, create_model

from autoformalism.research.preproposal_observed_gate import (
    ObservedGraphGate,
    check_bound_memories,
)
from autoformalism.research.preproposal_selection import DeclaredVariable, SelectionCase
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


class CommonLatentInitial(LatentInitial):
    rule: SharedInitial


class NewLatent(DeclaredVariable):
    """Declare a model-owned variable together with the sources of its equation."""

    sources: Sources


def topology_schema(
    case: SelectionCase,
    *,
    shared_hidden_initials: bool = False,
    allow_topology_latents: bool = False,
) -> type[StrictSchema]:
    """Require an equation for each existing variable; optionally extend the graph."""
    # Aliases allow legal scientific identifiers that shadow Pydantic attributes.
    equations = create_model(
        "InventorySources",
        __base__=StrictSchema,
        **{
            f"row_{i}": (Sources, Field(alias=v.name))
            for i, v in enumerate(case.inventory)
        },
    )
    initial = CommonLatentInitial if shared_hidden_initials else LatentInitial
    return create_model(
        "SelectedTopology",
        __base__=StrictSchema,
        hypothesis=(str, Field(min_length=1, max_length=1500)),
        sources=(equations, ...),
        latent_initializers=(tuple[initial, ...], Field(max_length=64)),
        **(
            {"new_latents": (tuple[NewLatent, ...], Field(default=(), max_length=64))}
            if allow_topology_latents
            else {}
        ),
    )


def reply_schema(case: SelectionCase, frozen: dict) -> type[StrictSchema]:
    """Return the proposer schema for the experiment settings."""
    return topology_schema(
        case,
        shared_hidden_initials=frozen.get("shared_hidden_initials", False),
        allow_topology_latents=frozen.get("allow_topology_latents", False),
    )


def graph_sources(reply: dict) -> dict:
    """View fixed and newly declared equations together, preserving source lists."""
    return reply["sources"] | {
        entry["name"]: entry["sources"] for entry in reply.get("new_latents", ())
    }


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


def topology_messages(case, choice, frozen_case, *, source_checklist=None):
    """Expose the choice, permitted channels and public graph obligations."""
    row, rationale = _choice(case, choice) if choice is not None else (None, None)
    evidence, brief = _inputs(case, frozen_case)
    observed, permitted = _initial_channels(_state_names(case), evidence, frozen_case)
    packet = {
        "target": case.target,
        **({"selected_information": row.sources} if row is not None else {}),
        **(
            {"rationale": rationale}
            if row is not None and source_checklist is None
            else {}
        ),
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
    if source_checklist is not None:
        for key in (
            "training_evidence",
            "required_paths",
            "required_target_dependencies",
        ):
            del packet[key]
        packet["source_checklist"] = source_checklist
        packet["permitted_initial_values"] = packet.pop("permitted_baseline_readings")
    selected_instruction = (
        "Give the selected information a scientifically justified path to the target. "
        if row is not None
        else ""
    )
    extension_instruction = ""
    if frozen_case.get("allow_topology_latents"):
        packet["inventory_roles"] = frozen_case.get("inventory_roles", {})
        packet["assigned_memory_states"] = frozen_case.get("mechanism_bindings", [])
        packet["available_rhs_names"] = sorted(
            {v.name for v in case.inventory}
            | set(frozen_case["histories"])
            | set(packet["fixed_covariates"])
            | {evidence.context.time_symbol}
        )
        packet["initializer_owners"] = sorted(_state_names(case) - observed.keys())
        extension_instruction = (
            "Choose RHS names from available_rhs_names or a variable you explicitly "
            "declare in new_latents with its name, kind, and sources. Use [] when no "
            "new variable is needed. A history or initial-value label is not an RHS "
            "name. A source can reach the target through another variable. Route "
            "each assigned memory state along its public requirement's pathway. Give "
            "latent_initializers only for initializer_owners and new differential "
            "states. "
        )
    latent_instruction = (
        "give a shared constant (mode value, scope shared_training). Every run "
        "starts with the same hidden initial values. "
        if frozen_case.get("shared_hidden_initials")
        else "give an initializer: a shared constant (mode value, scope "
        "shared_training) or a map of state_initials and public_initials at time zero. "
    )
    return (
        "Propose a complete topology for the declared variables. List sources for each "
        "derivative or algebraic value. Use the generated values of modeled variables. "
        + selected_instruction
        + extension_instruction
        + "Include the other public pathways. For each latent differential state "
        + latent_instruction
        + "The runtime supplies "
        "the observed initials. Give the topology and its scientific hypothesis.",
        json.dumps(packet, ensure_ascii=False, sort_keys=source_checklist is not None),
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


def make_topology_assessor(case, frozen_case):
    """Prepare one sealed training packet for many whole-graph evaluations."""
    prepared = _inputs(case, frozen_case)
    gate = (
        ObservedGraphGate(prepared[0])
        if frozen_case.get("check_all_measured")
        else None
    )

    def assess(choice, value):
        return assess_topology(
            case, choice, value, frozen_case, _prepared=prepared, _gate=gate
        )

    return assess


def assess_topology(
    case,
    choice,
    value,
    frozen_case,
    *,
    root_assessor=_actual_assessment,
    _prepared=None,
    _gate=None,
):
    """Check public paths and the information upstream of each measured state."""
    row = _choice(case, choice)[0] if choice is not None else None
    evidence, brief = _prepared or _inputs(case, frozen_case)
    context = evidence.context
    states = _state_names(case)
    generated = {v.name for v in case.inventory}
    observed, permitted = _initial_channels(states, evidence, frozen_case)
    result = {
        "candidate_id": row.candidate_id if row is not None else None,
        "scientific_correctness_assessed": False,
    }
    try:
        reply = (
            topology_schema(
                case,
                allow_topology_latents=frozen_case.get("allow_topology_latents", False),
            )
            .model_validate(value)
            .model_dump(by_alias=True)
        )
        additions = reply.get("new_latents", ())
        new_names = [entry["name"] for entry in additions]
        reserved = (
            generated
            | context.forcing_channels
            | set(context.targets)
            | set(context.unavailable_observed_channels)
            | {context.time_symbol}
            | {variable.name for variable in brief.public_variables}
        )
        if len(new_names) != len(set(new_names)) or reserved.intersection(new_names):
            raise ValueError(
                "Each new latent needs a unique name distinct from existing "
                "variables and channels"
            )
        known = reserved | set(new_names)
        for name in new_names:
            prefix, _, suffix = name.partition("_")
            if prefix in {"initial", "history"} and suffix in known:
                raise ValueError(
                    f"{name} is an information label. Give the new latent its own name."
                )
        generated.update(new_names)
        states.update(
            entry["name"] for entry in additions if entry["kind"] == "differential"
        )
        graph = graph_sources(reply)
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
                    name=name,
                    kind="dynamic" if name in states else "algebraic",
                    role=name,
                    sources=sources,
                )
                for name, sources in graph.items()
            ),
        )
        handoff = GraphDependencyHandoff.model_validate(
            {"records": reply["latent_initializers"]}
        )
        if frozen_case.get("shared_hidden_initials") and any(
            record.rule.mode != "value" for record in handoff.records
        ):
            raise ValueError(
                "This task requires one shared initial constant per latent state."
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
        check_bound_memories(
            graph, states, brief, frozen_case.get("mechanism_bindings", ())
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
    use_full_gate = frozen_case.get("check_all_measured", False)
    gate = (_gate or ObservedGraphGate(evidence)) if use_full_gate else None
    measured = sorted(generated & gate.measured) if gate else context.targets
    for target in measured:
        if gate is None and target not in states:
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
        gap = (
            frozen_case.get("minimum_target_gap", 0.0) if target == case.target else 0.0
        )
        if gate:
            checks[target] = gate.assess(target, graph, states, initialization, gap) | {
                "actual_options": sorted(actual)
            }
            if gap:
                exact_target = gate.assess(
                    target, graph, states, initialization, 0.0
                ) | {"actual_options": sorted(actual)}
        elif target in observed:
            checks[target] = root_assessor(target, actual, evidence, gap)
            if gap:
                exact_target = root_assessor(target, actual, evidence, 0.0)
        else:
            checks[target] = {
                "status": "unresolved",
                "actual_options": sorted(actual),
                "unavailable": ["No permitted observed initial for this target."],
            }

    missing = sorted(
        (set(row.sources if row is not None else ()) | set(case.required_options))
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
        **(
            {
                "training_check_scope": "Necessary conditions for joint reproduction "
                "on training trajectories under the stated observation bounds. "
                "Conditioning on an exact measured model variable tests its observed "
                "value only; it supplies no rollout input. A selected target gap is "
                "a contrast threshold and does not widen any conditioning bound."
            }
            if gate
            else {}
        ),
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
