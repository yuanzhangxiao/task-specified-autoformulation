"""Compare matched historical T2 topology ballots on training-only path criteria."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_complete_graphs import random_selection
from autoformalism.research.preproposal_observed_gate import _path_via_memory
from autoformalism.research.preproposal_selection import SelectionIntake
from autoformalism.search.mechanism_realizations import _memory_target_witness
from scripts import preproposal_graph_portfolio as portfolio
from scripts import replay_topology_admission as admission

OUT = Path("output/revision-round-20261124/topology-paired-paths-v1.json")


def indicators(graph: dict, memories: set[str]) -> dict[str, bool]:
    """Measure directed RHS routes and a shared insulin-driven latent fork."""
    deps = graph["sources"]
    outgoing: dict[str, set[str]] = {}
    for receiver, sources in deps.items():
        for source in sources:
            outgoing.setdefault(source, set()).add(receiver)
    latent = set(graph["variable_types"]) - {"Gp", "I", "U"}
    states = {k for k in latent if graph["variable_types"][k] == "differential"}
    if memories - states:
        raise ValueError(
            "The assigned insulin memory must be a latent differential state"
        )
    has_disposal_route = bool(_memory_target_witness(outgoing, "U", "Gp"))
    shared_latents = latent.intersection(deps["U"], deps["Gp"])
    fork = any(
        _memory_target_witness(outgoing, "I", s, frozenset({"Gp", "U"}))
        for s in shared_latents
    )
    return {
        "meal_to_Gp": bool(
            _memory_target_witness(outgoing, "meal_rate_g_per_min", "Gp")
        ),
        "I_through_latent_memory_to_U": any(
            _path_via_memory(outgoing, "I", s, "U", frozenset({"I"}))
            for s in (memories or states)
        ),
        "U_direct_to_Gp": "U" in deps["Gp"],
        "U_any_path_to_Gp": has_disposal_route,
        "shared_insulin_driven_latent_direct_to_U_and_Gp": fork,
        "U_path_or_shared_latent": has_disposal_route or fork,
    }


def summarize(ballots: list[dict], metrics: list[str]) -> dict:
    """Compare the two selectors within each inventory and seed, counting repeats."""
    result = {}
    for name in metrics:
        totals = {"20B": 0, "uniform": 0}
        pairs = Counter()
        for ballot in ballots:
            count = {
                arm: sum(graph[name] for graph in ballot[arm].values())
                for arm in totals
            }
            for arm, n in count.items():
                totals[arm] += n
            if count["20B"] == count["uniform"]:
                pairs["equal"] += 1
            else:
                pairs[max(count, key=count.get)] += 1
        result[name] = {"picks_with_feature": totals, "draws_with_more": dict(pairs)}
    return result


def audit(root: Path) -> dict:
    """Recheck the original menus and replay the authenticated selections."""
    intake, _, _ = admission._archived_inputs(root)
    plan = sealed_read(root / "plan.json")
    seeds = plan["seeds"]
    ballots, inventories = [], {}
    total_tokens = 0
    for case in SelectionIntake.model_validate(intake["selection"]).cases:
        if not case.case_id.startswith("t2_"):
            continue
        key = case.case_id
        frozen = portfolio.with_reviewed_initial_bindings(intake["frozen_cases"][key])
        assess = admission.make_topology_assessor(case, frozen)
        schema = admission.reply_schema(case, frozen)
        bound = {
            name
            for b in frozen.get("mechanism_bindings", ())
            if b["requirement_id"] == "delayed_insulin_action"
            for name in b["memory_states"]
        }
        features = {}
        pool = intake["graph_pools"][key]
        for c in pool["candidates"]:
            verdict = assess(None, c["graph"])
            if verdict["status"] != admission.PASS:
                raise ValueError(
                    "Expected all original T2 candidates to pass the corrected gate"
                )
            graph = admission._canonical(
                case, frozen, schema, c["graph"], verdict["structure_sha256"]
            )
            features[c["candidate_id"]] = indicators(graph, bound)
        count = portfolio.selection_count(pool)
        inventories[key] = {
            "pool_sha256": pool["pool_sha256"],
            "size": len(features),
            "feature_counts": {
                k: sum(f[k] for f in features.values())
                for k in next(iter(features.values()))
            },
        }
        for seed in seeds:
            slot = f"{key}--seed-{seed}--graph-rank"
            saved = sealed_read(root / "results" / f"{slot}.json")
            calls = list((root / "steps" / slot / "calls").glob("*.json"))
            if len(calls) != 1:
                raise ValueError(f"Expected one recorded call for {slot}")
            total_tokens += json.loads(calls[0].read_text())["observed_total_tokens"]
            choice = saved["assessment"]
            chosen = choice["graph_ids"]
            if (
                saved["plan_sha256"] != plan["artifact_sha256"]
                or choice["status"] != "selected"
                or choice["pool_sha256"] != pool["pool_sha256"]
                or len(chosen) != count
                or len(set(chosen)) != count
            ):
                raise ValueError("Ballot differs from expected pool or pick count")
            uniform = [c["candidate_id"] for c in random_selection(pool, count, seed)]
            ballots.append(
                {
                    "case": key,
                    "seed": seed,
                    "20B": {k: features[k] for k in chosen},
                    "uniform": {k: features[k] for k in uniform},
                }
            )
    if not ballots:
        raise ValueError("No T2 ballots found")
    metrics = list(next(iter(ballots[0]["20B"].values())))
    return {
        "protocol": "paired-t2-topology-paths-1",
        "training_only": True,
        "new_llm_calls": 0,
        "new_fits": 0,
        "archived_rank_calls": len(ballots),
        "archived_rank_tokens": total_tokens,
        "source_plan": plan["artifact_sha256"],
        "source_receipts": admission.ASSESSMENTS,
        "inventories": inventories,
        "ballots": ballots,
        "summary": summarize(ballots, metrics),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=admission.SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    saved = sealed_write(args.out, audit(args.source))
    print(saved["summary"])
