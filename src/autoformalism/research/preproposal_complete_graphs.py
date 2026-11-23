"""Build bounded pools of complete graphs and select from the verified graphs by ID."""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Literal

from pydantic import Field, create_model

from autoformalism.research import preproposal_topology as topology
from autoformalism.research.preproposal_cases import source_checklist
from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


def _graph_proposal(
    case,
    frozen,
    checks,
    states,
    observed,
    permitted,
    brief,
    index,
    rng,
    max_initial_arguments,
):
    """Sample one complete wiring of the public routes and measured-source choices."""
    context = frozen["evidence"]["context"]
    nodes = sorted(v.name for v in case.inventory)
    targets = sorted(context["targets"])
    latent = sorted(states - observed.keys())
    maps = {name: set() for name in latent}
    histories = sorted(set(frozen["histories"]) - set(nodes))
    fixed = sorted(context.get("fixed_covariates", ()))
    graph = {name: set() for name in nodes}
    available_sources = set(nodes) | set(histories) | set(fixed)
    common = frozen.get("shared_hidden_initials", False)

    def route(source, target, *, memory=False, exclude=()):
        if source == target or target not in graph or source not in available_sources:
            return
        possible = sorted(
            (states if memory else set(nodes)) - {source, target} - set(exclude)
        )
        # A differential receiver already supplies memory for its own trajectory.
        direct = not memory or (target in states and source != target)
        if possible and (not direct or rng.randrange(3) == 0):
            middle = rng.choice(possible)
            graph[middle].add(source)
            graph[target].add(middle)
            return middle
        else:
            graph[target].add(source)

    assignments = {
        record["requirement_id"]: record["memory_states"]
        for record in frozen.get("mechanism_bindings", ())
    }
    for requirement in brief.requirements:
        for target in requirement.targets:
            for source in requirement.drivers:
                if assigned := assignments.get(requirement.id):
                    for state in assigned:
                        # Keep both legs on one path through the assigned state.
                        prefix_node = route(source, state, exclude=(target,))
                        excluded = set(requirement.drivers)
                        if prefix_node is not None:
                            excluded.add(prefix_node)
                        route(state, target, exclude=excluded)
                else:
                    route(source, target, memory=requirement.requires_dynamic_memory)
    for dependency in brief.target_dependencies:
        route(rng.choice(sorted(dependency.acceptable_sources)), dependency.target)

    for target, item in sorted(checks["variables"].items()):
        if item.get("status") != "screened":
            continue
        sets = item["minimal_additional_source_sets"]
        options = set(item["fixed_sources"])
        if sets:
            # Cycle each target through *all* minimal choices instead of always
            # starting with the first. Independent draws vary their combinations.
            options.update(
                sets[index % len(sets)] if target == case.target else rng.choice(sets)
            )
        for option in sorted(options):
            kind, source = option.split(":", 1)
            if kind == "history" or source in fixed or source in observed:
                route(source, target)
            elif latent and not common:
                state = rng.choice(latent)
                maps[state].add(source)
                route(state, target)

    for name in latent:
        # Sample shared constants as well as maps with one or two public arguments.
        if permitted and not common:
            width = rng.randrange(min(max_initial_arguments, len(permitted)) + 1)
            if len(maps[name]) < width:
                available = sorted(set(permitted) - maps[name])
                maps[name].update(rng.sample(available, width - len(maps[name])))
    for name in nodes:
        if name not in targets and not any(
            name in sources for sources in graph.values()
        ):
            graph[rng.choice(targets)].add(name)
        if not graph[name]:
            allowed = sorted(available_sources - {name})
            if name in states:
                allowed.append(name)
            if allowed:
                graph[name].add(rng.choice(allowed))

    # Permit additional causal terms outside the minimum route choices.
    for _ in range(index % 3):
        target = rng.choice(nodes)
        allowed = available_sources - graph[target]
        if target not in states:
            allowed.discard(target)
        if allowed:
            graph[target].add(rng.choice(sorted(allowed)))

    return {
        "hypothesis": "Algorithm-generated dependencies for the declared variables.",
        "sources": {name: sorted(deps) for name, deps in graph.items()},
        "latent_initializers": [
            {
                "state": name,
                "rule": {"mode": "map", "public_initials": sorted(args)}
                if args
                else {"mode": "value", "scope": "shared_training"},
                "justification": "Candidate shared initialization rule.",
            }
            for name, args in maps.items()
        ],
    }


def generate_pool(
    case: SelectionCase,
    frozen: dict,
    *,
    max_checks: int = 64,
    max_candidates: int = 8,
    seed: int = 0,
    checklist: dict | None = None,
    max_rhs_sources: int = 6,
    max_initial_arguments: int = 2,
) -> dict:
    """Sample complete graphs directly from the inventory and verify before admission.

    The grammar samples direct and one-intermediate paths, optional extra edges
    and shared constants or maps of permitted observed initial values. Search is
    reproducible and bounded. Exhausting a budget is not a proof of infeasibility.
    """
    if not frozen.get("check_all_measured"):
        raise ValueError("Complete graph pools require the full measured-variable gate")
    if (
        min(max_checks, max_candidates, max_rhs_sources) < 1
        or max_initial_arguments < 0
    ):
        raise ValueError("Invalid complete-graph search limits")
    evidence, brief = topology._inputs(case, frozen)
    states = topology._state_names(case)
    observed, permitted = topology._initial_channels(states, evidence, frozen)
    checks = checklist if checklist is not None else source_checklist(frozen)
    assess = topology.make_topology_assessor(case, frozen)
    draw_seed = content_hash(
        [
            "complete-graph-draws-v1",
            seed,
            case.case_id,
            [v.model_dump() for v in case.inventory],
        ]
    )
    rng = random.Random(draw_seed)
    max_draws = 4 * max_checks
    checked = 0
    counts = Counter()
    skips = Counter()
    seen = set()
    candidates = {}
    for index in range(max_draws):
        graph = _graph_proposal(
            case,
            frozen,
            checks,
            states,
            observed,
            permitted,
            brief,
            index,
            rng,
            max_initial_arguments,
        )
        graph_key = content_hash(graph)
        if graph_key in seen:
            skips["repeated_graph"] += 1
            continue
        seen.add(graph_key)
        if any(len(v) > max_rhs_sources for v in graph["sources"].values()) or any(
            len(v["rule"].get("public_initials", ())) > max_initial_arguments
            for v in graph["latent_initializers"]
        ):
            skips["complexity_limit"] += 1
            continue
        checked += 1
        outcome = assess(None, graph)
        status = outcome["status"]
        counts[status] += 1
        if status == "topology_screen_passed":
            digest = outcome["structure_sha256"]
            candidate_id = "g_" + digest[:16]
            if (
                candidate_id in candidates
                and candidates[candidate_id]["structure_sha256"] != digest
            ):
                raise ValueError("Complete graph ID collision")
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "structure_sha256": digest,
                "graph": graph,
            }
        if len(candidates) == max_candidates or checked == max_checks:
            break
    ordered = [candidates[key] for key in sorted(candidates)]
    result = {
        "case_id": case.case_id,
        "evidence_sha256": case.evidence_sha256,
        "seed": seed,
        "inventory_roles": frozen.get("inventory_roles", {}),
        "mechanism_bindings": frozen.get("mechanism_bindings", []),
        **(
            {"measured_initial_bindings": frozen["measured_initial_bindings"]}
            if frozen.get("measured_initial_bindings") else {}
        ),
        "limits": {
            "max_checks": max_checks,
            "max_draws": max_draws,
            "max_candidates": max_candidates,
            "max_rhs_sources": max_rhs_sources,
            "max_initial_arguments": max_initial_arguments,
        },
        "draws": index + 1,
        "checked": checked,
        "status_counts": dict(counts),
        "skipped_counts": dict(skips),
        "stop_reason": "pool_filled"
        if len(ordered) == max_candidates
        else "check_limit"
        if checked == max_checks
        else "draw_limit",
        "candidates": ordered,
    }
    return {**result, "pool_sha256": content_hash(result)}


def _menu(pool: dict, count: int) -> dict:
    expected = content_hash({k: v for k, v in pool.items() if k != "pool_sha256"})
    if expected != pool["pool_sha256"]:
        raise ValueError("Complete graph pool differs from its recorded identity")
    available = {row["candidate_id"]: row for row in pool["candidates"]}
    if not 0 <= count <= len(available):
        raise ValueError("Selection count exceeds the complete graph pool")
    return available


def resolve_selection(pool: dict, ids: list[str], count: int) -> list[dict]:
    """Return whole previously admitted graphs in requested order."""
    available = _menu(pool, count)
    if len(ids) != count or len(set(ids)) != count or set(ids) - available.keys():
        raise ValueError(
            "Select exactly the requested number of distinct listed graph IDs"
        )
    return [available[identifier] for identifier in ids]


def random_selection(pool: dict, count: int, seed: int) -> list[dict]:
    """Matched uniform selection without replacement from the identical pool."""
    available = _menu(pool, count)
    rng = random.Random(
        content_hash(["uniform-graph-ids-v1", pool["pool_sha256"], seed])
    )
    return resolve_selection(pool, rng.sample(sorted(available), count), count)


def graph_selection_schema(pool: dict, count: int) -> type[StrictSchema]:
    available = _menu(pool, count)
    if not count:
        raise ValueError("An empty selection needs no proposer call")
    return create_model(
        "CompleteGraphSelection",
        __base__=StrictSchema,
        graph_ids=(
            tuple[Literal[tuple(available)], ...],
            Field(min_length=count, max_length=count),
        ),
    )


def graph_selection_messages(
    case: SelectionCase, pool: dict, count: int
) -> tuple[str, str]:
    """Present each complete graph for selection by ID."""
    available = _menu(pool, count)
    if (
        not count
        or pool["case_id"] != case.case_id
        or pool["evidence_sha256"] != case.evidence_sha256
    ):
        raise ValueError("Need a nonempty decision for the matching case")
    return (
        f"Choose exactly {count} distinct graph IDs in order of scientific "
        "plausibility for the public task. Every listed graph passed the same public "
        "graph and training checks. Consider pathways, scientific roles and initial "
        "values. Return only the chosen IDs.",
        json.dumps(
            {
                "public_task": case.public_task,
                "inventory": [v.model_dump() for v in case.inventory],
                "inventory_roles": pool["inventory_roles"],
                "mechanism_bindings": pool["mechanism_bindings"],
                **(
                    {"measured_initial_bindings": pool["measured_initial_bindings"]}
                    if pool.get("measured_initial_bindings") else {}
                ),
                "graphs": [
                    {
                        "graph_id": row["candidate_id"],
                        "sources": row["graph"]["sources"],
                        "latent_initializers": {
                            r["state"]: r["rule"]
                            for r in row["graph"]["latent_initializers"]
                        },
                    }
                    for row in available.values()
                ],
            },
            ensure_ascii=False,
        ),
    )
