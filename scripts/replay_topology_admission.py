"""Replay the completed 20B topology experiment (ACES job 2403807) on CPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.preproposal_complete_graphs import generate_pool
from autoformalism.research.preproposal_graph_repair import PASS, assess_and_repair
from autoformalism.research.preproposal_selection import SelectionIntake
from autoformalism.research.preproposal_topology import (
    make_topology_assessor,
    reply_schema,
)
from autoformalism.staged_topology import content_hash
from scripts import preproposal_graph_portfolio as portfolio
from scripts.preproposal_graph_exports import _canonical
from scripts.replay_complete_graph_architectures import _receipt

SOURCE = Path("output/complete-graphs-results-20261123/returned")
OUTPUT = Path("output/topology-admission-replay-20261123/replay.json")
PLAN = "727a6264c5ab1a3fe600a25162ada3acb3f379fd3e0dd0bd450b7dcf03356ea9"
ASSESSMENTS = "ff882d57740c0bfdd1c2baa658f2ae03570b14d3fdc731a7f4bdfb2f623eaeab"
EXPORT = "3f86d3a61ecf67156a675206d9214ad13a393c340fff645f46dec9db4d9f6951"
# Digests of the sorted successful parent slots and (case, structure) pairs
# from the earlier manual normalization diagnostic, independent of new code.
PREVIOUS_PARENTS = "d2f6fd5bef77a13b815132476479e2e6b7fe919b80108d0d9cc6cb3262a01b0b"
PREVIOUS_CHILDREN = "d071c8d70bde566fa7025094e72c407f150b54b192cf6891ecd170db7c05b516"


def _archived_inputs(root: Path) -> tuple[dict, set, dict]:
    """Authenticate the input, selected structures, and all physical replies."""
    plan = sealed_read(root / "plan.json")
    intake = sealed_read(root / "intake.json")
    if (
        plan["artifact_sha256"] != PLAN
        or intake["artifact_sha256"] != plan["intake_sha256"]
    ):
        raise ValueError("Expected the sealed ACES job 2403807 plan and intake")
    export = json.loads((root / "admitted-graphs.json").read_text())
    if content_hash(export) != EXPORT:
        raise ValueError("Original ACES topology export differs")
    selected = {(r["case_id"], r["structure_sha256"]) for r in export["graphs"]}
    receipts, seals = {}, {}
    for path in sorted((root / "results").glob("*.json")):
        saved = sealed_read(path)
        old, reply = _receipt(root, path, PLAN)
        seals[path.stem] = saved["artifact_sha256"]
        if "--source-" in path.stem or path.stem.endswith("--graph-first"):
            if reply["status"] != "visible":
                raise ValueError(f"Expected a visible graph reply for {path.stem}")
            receipts[path.stem] = (old, reply["value"])
    if content_hash(seals) != ASSESSMENTS:
        raise ValueError("Original ACES result receipt set differs")
    return intake, selected, receipts


def _decision(result: dict) -> dict:
    """Keep the rejection reason and the individual contradicted targets."""
    return {
        k: result[k]
        for k in (
            "status",
            "structure_sha256",
            "reason",
            "merged_inventory_redeclarations",
        )
        if k in result
    } | {
        "contradicted": sorted(
            k
            for k, v in result.get("training_checks", {}).items()
            if v["status"] == "contradicted"
        )
    }


def replay(root: Path) -> dict:
    """Replay the original inputs and regenerate the seed-zero pools."""
    intake, selected, receipts = _archived_inputs(root)
    pools, replies, new_pools = [], [], {}
    successful_parents, children = set(), set()
    for case in SelectionIntake.model_validate(intake["selection"]).cases:
        key = case.case_id
        frozen = portfolio.with_reviewed_initial_bindings(intake["frozen_cases"][key])
        assess = make_topology_assessor(case, frozen)
        schema = reply_schema(case, frozen)
        for row in intake["graph_pools"][key]["candidates"]:
            digest = row["structure_sha256"]
            outcome = assess(None, row["graph"])
            checked = {
                "case_id": key,
                "original_hash": digest,
                "selected": (key, digest) in selected,
            } | _decision(outcome)
            if outcome["status"] == PASS:
                checked["graph"] = _canonical(
                    case, frozen, schema, row["graph"], outcome["structure_sha256"]
                )
            pools.append(checked)
        for slot, (old, graph) in receipts.items():
            if slot.split("--", 1)[0] != key:
                continue
            entry = {"slot": slot, "case_id": key, "original_status": old["status"]}
            if slot.endswith("--graph-first"):
                result = assess_and_repair(case, graph, frozen)
                if result["repairs"]:
                    successful_parents.add(slot)
                for child in result["repairs"]:
                    digest = child["structure_sha256"]
                    child["canonical_graph"] = _canonical(
                        case, frozen, schema, child["graph"], digest
                    )
                    children.add((key, digest))
                entry.update(
                    arm="graph_first",
                    original_patch_checks=old["checked"],
                    **{k: result[k] for k in ("repair_status", "checked", "repairs")},
                )
            else:
                choice = {
                    "candidate_id": old["candidate_id"],
                    "rationale": "Saved source choice.",
                }
                result = assess(choice, graph)
                entry.update(arm="selected_source")
            replies.append(entry | _decision(result))
        archived = intake["graph_pools"][key]
        limits = archived["limits"]
        new_pools[key] = generate_pool(
            case,
            frozen,
            seed=archived["seed"],
            max_checks=limits["max_checks"],
            max_candidates=limits["max_candidates"],
            max_rhs_sources=limits["max_rhs_sources"],
            max_initial_arguments=limits["max_initial_arguments"],
        )
    return {
        "source_plan_sha256": PLAN,
        "original_receipts_sha256": ASSESSMENTS,
        "provider_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
        "summary": {
            "saved_graphs_per_arm": dict(Counter(r["arm"] for r in replies)),
            "corrected_parent_status_per_arm": {
                arm: dict(Counter(r["status"] for r in replies if r["arm"] == arm))
                for arm in ("graph_first", "selected_source")
            },
            "rescued_parents": len(successful_parents),
            "original_patch_checks": sum(
                r.get("original_patch_checks", 0) for r in replies
            ),
            "corrected_patch_checks": sum(r.get("checked", 0) for r in replies),
            "same_eight_parent_slots_as_manual_replay": (
                content_hash(sorted(successful_parents)) == PREVIOUS_PARENTS
            ),
            "distinct_repaired_children": len(children),
            "same_113_child_structures_as_manual_replay": (
                content_hash(sorted(children)) == PREVIOUS_CHILDREN
            ),
            "original_pool_recheck": dict(Counter(r["status"] for r in pools)),
            "original_export_recheck": dict(
                Counter(r["status"] for r in pools if r["selected"])
            ),
            "regenerated_pool_sizes": {
                k: len(v["candidates"]) for k, v in new_pools.items()
            },
        },
        "saved_replies": replies,
        "original_pool_recheck": pools,
        "regenerated_pools": new_pools,
    }


def save(root: Path, output: Path) -> dict:
    """Write the derived analysis separately from the historical inspection."""
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Replay output must be outside the archived inspection")
    result = replay(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result["summary"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=SOURCE)
    cli.add_argument("--output", type=Path, default=OUTPUT)
    args = cli.parse_args()
    print(json.dumps(save(args.source, args.output), indent=2))
