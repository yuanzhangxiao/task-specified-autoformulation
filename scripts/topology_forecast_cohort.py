"""Select equal numbers of admitted graphs with identical variable inventories."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN, T2
from autoformalism.research.preproposal_graph_repair import PASS
from autoformalism.research.preproposal_selection import SelectionIntake
from autoformalism.research.preproposal_topology import reply_schema
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash
from scripts.preproposal_graph_exports import _canonical
from scripts.preproposal_graph_portfolio import with_reviewed_initial_bindings
from scripts.trajectory_function_intake import (
    INSPECTION,
    ORIGINAL_INTAKE,
    REPLAY,
    REPLAY_FILE,
)

ARMS = ("algorithm", "20b_repair")
SELECTION_SALT = "matched-topology-forecast-uniform-v1"


def match_inventory(
    inventory: str, variable_types: dict, candidates: dict, cap: int
) -> tuple[list[dict], dict]:
    """Deduplicate within each arm and select a balanced, hash-shuffled sample."""
    if cap < 1 or set(candidates) != set(ARMS):
        raise ValueError("Matching requires both arms and a positive cap")
    pools, counts = {}, {}
    for arm in ARMS:
        distinct = {}
        for row in candidates[arm]:
            digest = row["graph_id"]
            if digest in distinct:
                if distinct[digest]["graph"] != row["graph"]:
                    raise ValueError("Duplicate graph ID has conflicting structure")
                distinct[digest]["origins"].append(row["origin"])
            else:
                distinct[digest] = {
                    "graph_id": digest,
                    "structure_sha256": digest,
                    "graph": row["graph"],
                    "origins": [row["origin"]],
                }
        same = [
            item
            for item in distinct.values()
            if item["graph"]["variable_types"] == variable_types
        ]
        pools[arm] = sorted(
            same,
            key=lambda item: (
                content_hash([SELECTION_SALT, inventory, item["graph_id"]]),
                item["graph_id"],
            ),
        )
        counts[arm] = {
            "admitted_occurrences": len(candidates[arm]),
            "distinct_admitted": len(distinct),
            "matching_inventory": len(same),
            "different_inventory": sorted(
                digest
                for digest, item in distinct.items()
                if item["graph"]["variable_types"] != variable_types
            ),
        }
    take = min(cap, *(len(pool) for pool in pools.values()))
    rows = []
    for arm in ARMS:
        for item in pools[arm][:take]:
            rows.append(
                {
                    **item,
                    "origins": sorted(set(item["origins"])),
                    "case": inventory,
                    "inventory": inventory,
                    "arm": arm,
                }
            )
    return rows, {
        "variable_types": variable_types,
        "arms": counts,
        "selected_per_arm": take,
        "selected_in_both_arms": sorted(
            {r["graph_id"] for r in rows if r["arm"] == ARMS[0]}
            & {r["graph_id"] for r in rows if r["arm"] == ARMS[1]}
        ),
    }


def build_cohort(cap: int = 8) -> dict:
    """Match the two generation arms using authenticated training artifacts."""
    if file_hash(REPLAY) != REPLAY_FILE:
        raise ValueError("Corrected replay file identity differs")
    intake = sealed_read(INSPECTION)
    if intake["artifact_sha256"] != ORIGINAL_INTAKE:
        raise ValueError("Source intake identity differs")
    replay = json.loads(REPLAY.read_text())
    rows, counts, cases, training_hashes = [], {}, {}, {}
    for case in SelectionIntake.model_validate(intake["selection"]).cases:
        key = case.case_id
        frozen = with_reviewed_initial_bindings(intake["frozen_cases"][key])
        dataset = BASIN if key.startswith("basin_") else T2
        observations = frozen["evidence"]["trajectories"]
        evidence_hash = content_hash(observations)
        if (
            frozen["evidence"]["split"] != "train"
            or training_hashes.setdefault(dataset, evidence_hash) != evidence_hash
        ):
            raise ValueError(f"Unexpected training data for inventory {key}")
        cases[key] = {
            **frozen,
            "evidence": {
                k: v for k, v in frozen["evidence"].items() if k != "trajectories"
            },
        }
        pools = {arm: [] for arm in ARMS}
        schema = reply_schema(case, frozen)
        for item in replay["regenerated_pools"][key]["candidates"]:
            digest = item["structure_sha256"]
            graph = _canonical(case, frozen, schema, item["graph"], digest)
            pools["algorithm"].append(
                {
                    "graph": graph,
                    "graph_id": digest,
                    "origin": item["candidate_id"],
                }
            )
        for reply in replay["saved_replies"]:
            if reply["case_id"] != key or reply["arm"] != "graph_first":
                continue
            if reply["status"] == PASS:
                raise ValueError(
                    "Passing parent requires inclusion in the matching pool"
                )
            for child in reply["repairs"]:
                if child["status"] != PASS:
                    continue
                digest = child["structure_sha256"]
                graph = _canonical(case, frozen, schema, child["graph"], digest)
                if graph != child["canonical_graph"]:
                    raise ValueError("Saved repair and canonical graph differ")
                pools["20b_repair"].append(
                    {
                        "graph": graph,
                        "graph_id": digest,
                        "origin": reply["slot"],
                    }
                )
        variable_types = {v.name: v.kind for v in case.inventory}
        matched, counts[key] = match_inventory(key, variable_types, pools, cap)
        rows.extend({**row, "dataset": dataset} for row in matched)
    return {
        "protocol": SELECTION_SALT,
        "cap_per_inventory_arm": cap,
        "source_intake_sha256": ORIGINAL_INTAKE,
        "source_replay_file_sha256": REPLAY_FILE,
        "training_trajectory_sha256": training_hashes,
        "inventory_counts": counts,
        "cases": cases,
        "rows": rows,
        "unmatched_inventories": sorted(
            key for key, receipt in counts.items() if receipt["selected_per_arm"] == 0
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cohort = sealed_write(args.output, build_cohort())
    print(
        json.dumps(
            {
                "selected": len(cohort["rows"]),
                "sha256": cohort["artifact_sha256"],
                "unmatched_inventories": cohort["unmatched_inventories"],
            },
            indent=2,
        )
    )
