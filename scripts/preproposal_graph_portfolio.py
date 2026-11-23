"""Build the graph-selection experiment from verified public training inputs."""

from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import preproposal_cases as cases
from autoformalism.research import preproposal_selection as selection
from autoformalism.staged_topology import content_hash

# Three structurally different variable-stage exits in the same T2-easy run.
EXIT_RECEIPTS = {
    0: ("000", "9b10f686d512e657a86d65a338200150a3f0eaa1915377d1ad370bc041aaf297"),
    1: ("001", "5430f7cc595c579f9209fa233ea837653ca9a7cc10d791d56ff42d8fb309ffb6"),
    3: ("000", "c0dd84e656d432825c3ba48e28208e859e5a51c5716b55f539c9b17b529fa0a4"),
}
NAME = "phase-c-complete-graphs-20b-v1"
LAUNCHER = "scripts/hpc/start_complete_graphs_aces.sh"
SEEDS = (100, 101, 102)
MAX_CHECKS = 64
MAX_GRAPHS = 8


def read_variable_exits(project: Path, frozen: dict) -> dict:
    """Read only completed variable-stage events from the archived training run."""
    result = {}
    base = project / "output/jetstream-completion-v2-inspection-20261012/results"
    for seed, (index, digest) in EXIT_RECEIPTS.items():
        event = sealed_read(
            base
            / f"cell02_seed{seed}_full_minimal_completion"
            / "construction/events"
            / f"{index}.json"
        )
        if event["artifact_sha256"] != digest:
            raise ValueError(f"Unexpected variable receipt for seed {seed}")
        result[str(seed)] = event
    for key in ("t2_seed102_insulin_gap1", "basin_seed0_downstream"):
        reference = frozen[key]["provenance"]
        item = sealed_read(project / reference["receipt_path"])
        if item["artifact_sha256"] != reference["receipt_sha256"]:
            raise ValueError(f"Original inventory receipt differs for {key}")
        result[key] = item
    return result


def portfolio_cases(verified: dict, exits: dict) -> dict:
    """Pair the same evidence, tolerance and forecast permissions across inventories."""
    reference = verified["t2_seed102_insulin_gap1"]
    frozen = {reference["case_id"]: dict(reference)}
    if set(exits) != {str(s) for s in EXIT_RECEIPTS} | {
        reference["case_id"],
        "basin_seed0_downstream",
    }:
        raise ValueError("The three registered variable-stage receipts are required")
    for seed, (_, digest) in EXIT_RECEIPTS.items():
        event = exits[str(seed)]
        if event["artifact_sha256"] != digest:
            raise ValueError(f"Wrong variable-stage event for seed {seed}")
        item = cases.from_variable_exit(
            case_id=f"t2_inventory_seed{seed}",
            event=event,
            target=reference["target"],
            public_brief=reference["public_brief"],
            evidence=reference["evidence"],
            histories=reference["histories"],
            initials=[*reference["initials"], reference["target"]],
            required_histories=reference["required_histories"],
            minimum_target_gap=reference["minimum_target_gap"],
        )
        item["mechanism_bindings"] = event["after"]["mechanism_bindings"]
        item["inventory_roles"] = {
            v["name"]: v["scientific_role"] for v in event["after"]["variables"]
        }
        frozen[item["case_id"]] = item
    basin = verified["basin_seed0_downstream"]
    frozen[basin["case_id"]] = dict(basin)
    for key in (reference["case_id"], basin["case_id"]):
        event = exits[key]
        digest = content_hash(
            {k: v for k, v in event.items() if k != "artifact_sha256"}
        )
        if digest != frozen[key]["provenance"]["receipt_sha256"]:
            raise ValueError(f"Unverified inventory receipt for {key}")
        if key == basin["case_id"]:
            declarations = event["after"]["variables"]
            frozen[key]["inventory_roles"] = {
                v["name"]: v["scientific_role"] for v in declarations
            }
            frozen[key]["mechanism_bindings"] = event["after"]["mechanism_bindings"]
        else:
            frozen[key]["inventory_roles"] = {
                c["name"]: c["role"] for c in event["value"]["additional_components"]
            }
    for item in frozen.values():
        item.update(check_all_measured=True, allow_topology_latents=True)
    return frozen


def prepare_payload(frozen: dict) -> dict:
    """Generate the CPU pools once, before any proposal, and retain the exact menus."""
    from autoformalism.research.preproposal_complete_graphs import generate_pool

    checklists, pools, menus = {}, {}, []
    for key, item in frozen.items():
        case = selection.SelectionCase.model_validate(cases.selection_case(item))
        menus.append(case.model_dump(mode="json"))
        checklist = cases.source_checklist(item)
        checklists[key] = checklist
        pools[key] = generate_pool(
            case,
            item,
            max_checks=MAX_CHECKS,
            max_candidates=MAX_GRAPHS,
            seed=0,
            checklist=checklist,
        )
    return {
        "selection": {"cases": menus},
        "frozen_cases": frozen,
        "source_checklists": checklists,
        "graph_pools": pools,
    }


def design(registered, payload):
    """Freeze candidate pools, call limits and source sampling for the matched draws."""
    from scripts.t2_breadth import COMPLETE_GRAPHS_20B

    pools = payload["graph_pools"]
    source_rankings = graph_rankings = graph_calls = 0
    uniform, counts, menus = {}, {}, {}
    for case in registered:
        key = case.case_id
        active = selection.needs_selection(case)
        pool = pools[key]
        count = min(selection.matched_count(case), len(case.candidates))
        counts[key] = count
        uniform[key] = {
            str(seed): selection.uniform_ids(case, seed)
            if active
            else [row.candidate_id for row in case.candidates]
            for seed in SEEDS
        }
        source_rankings += len(SEEDS) * active
        graph_rankings += len(SEEDS) * bool(pool["candidates"])
        graph_calls += len(SEEDS) * min(2 * count, len(case.candidates))
        menus[key] = pool["pool_sha256"]
    parents = len(SEEDS) * len(registered)
    return {
        "protocol": COMPLETE_GRAPHS_20B,
        "sampling_design": "shared_complete_graph_pool_and_matched_source_draws",
        "seeds": list(SEEDS),
        "graph_pool_hashes": menus,
        "uniform_selections": uniform,
        "choices_per_arm": counts,
        "maximum_graph_first_requests": parents,
        "maximum_graph_selection_requests": graph_rankings,
        "maximum_selection_requests": source_rankings,
        "maximum_selected_topology_requests": graph_calls,
        "maximum_topology_requests": parents + graph_calls,
        "maximum_generation_requests": parents
        + graph_calls
        + graph_rankings
        + source_rankings,
    }


def selection_count(pool: dict) -> int:
    """Allocate matching scientific and uniform choices, including an empty pool."""
    return min(3, len(pool["candidates"]))


def source_ballot(case, checklist):
    """Use IDs alone when the caller hands source membership back to the runtime."""
    from typing import Literal

    from pydantic import Field, create_model

    from autoformalism.schemas.base import StrictSchema

    count = selection.matched_count(case)
    ids = tuple(v.candidate_id for v in case.candidates)
    schema = create_model(
        "SourceBallot",
        __base__=StrictSchema,
        source_ids=(
            tuple[Literal[ids], ...],
            Field(min_length=count, max_length=count),
        ),
    )
    _, packet = selection.messages(case, source_checklist=checklist)
    system = (
        f"Choose exactly {count} distinct listed source IDs for {case.target}, "
        "in order of scientific plausibility. Each source must reach the target "
        "through the graph. Return only the IDs in source_ids."
    )

    def assess(value):
        chosen = value["source_ids"]
        if len(set(chosen)) != len(chosen):
            return {"status": "invalid_selection", "reason": "Repeated source ID"}
        rows = {
            row.candidate_id: row.model_dump(mode="json") for row in case.candidates
        }
        return {"status": "selected", "selected": [rows[v] for v in chosen]}

    return (system, packet), schema, assess


def graph_ballot(case, pool):
    from autoformalism.research import preproposal_complete_graphs as graphs

    count = selection_count(pool)
    schema = graphs.graph_selection_schema(pool, count)

    def assess(value):
        try:
            selected = graphs.resolve_selection(pool, value["graph_ids"], count)
        except ValueError as exc:
            return {"status": "invalid_selection", "reason": str(exc)}
        return {
            "status": "selected",
            "pool_sha256": pool["pool_sha256"],
            "graph_ids": [row["candidate_id"] for row in selected],
        }

    return graphs.graph_selection_messages(case, pool, count), schema, assess


def pool_status(case, pool, seeds, read_ballot) -> dict:
    from autoformalism.research.preproposal_complete_graphs import random_selection

    count = selection_count(pool)
    draws = []
    for seed in seeds:
        scientific = (
            read_ballot(seed) or {"status": "pending"}
            if count
            else {"status": "empty_pool", "graph_ids": []}
        )
        uniform = random_selection(pool, count, seed)
        draws.append(
            {
                "seed": seed,
                "scientific": scientific,
                "uniform_graph_ids": [r["candidate_id"] for r in uniform],
            }
        )
    return {
        "case_id": case.case_id,
        "pool_sha256": pool["pool_sha256"],
        "eligible_count": len(pool["candidates"]),
        "candidates": pool["candidates"],
        "generation": {
            k: v for k, v in pool.items() if k not in ("candidates", "inventory_roles")
        },
        "draws": draws,
    }
