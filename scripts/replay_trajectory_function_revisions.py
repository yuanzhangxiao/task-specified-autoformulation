"""Replay saved basin functions as source changes without a new proposer call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

ARCHIVED_PLAN = "22f97e0d5ba0adb9ef0552addb0a4d1a3608ba9b96a95388ac508589c9a92cc1"
ARCHIVED_GENERATION = "d58bc8dabd174fbaa3b1069109463a9bdd11b2dec66be8d0ecdf5a475cb232b1"
# A legacy reply did not choose a route or give a scientific reason. The audit
# supplies this neutral request; the proposed mathematical content stays intact.
AUDIT_REASON = (
    "Check whether the sources used in these saved basin equations can represent "
    "the public upstream transfer and downstream discharge."
)


def replay(archive: Path, root: Path) -> dict:
    """Verify the original receipts, reconstruct each slot and freeze the new cohort."""
    if root.resolve().is_relative_to(archive.resolve()):
        raise ValueError("Use a separate replay directory outside the original results")
    original = sealed_read(archive / "plan.json")
    generation = trial._frozen_candidates(archive, original)
    intake = sealed_read(archive / "intake.json")
    if (
        original["artifact_sha256"] != ARCHIVED_PLAN
        or generation["artifact_sha256"] != ARCHIVED_GENERATION
        or original["intake_sha256"] != intake["artifact_sha256"]
        or intake["source"] != trial.TRAIN
    ):
        raise ValueError("Expected the frozen ACES 2404388 generation and its intake")
    parents = [i for i, g in enumerate(intake["graphs"]) if g["dataset"] == BASIN]
    graphs = [intake["graphs"][g] for g in parents]
    with _lock(root / "prepare-lock"):
        data = sealed_write(
            root / "intake.json",
            {
                "profile": trial.BASIN_REPLAY,
                "source": intake["source"],
                "source_generation": ARCHIVED_GENERATION,
                "source_plan": ARCHIVED_PLAN,
                "replay_code_sha256": file_hash(Path(__file__)),
                "graphs": graphs,
                "menus": [intake["menus"][i] for i in parents],
                "cases": {r["case"]: intake["cases"][r["case"]] for r in graphs},
                "train": {BASIN: intake["train"][BASIN]},
                "observations": {BASIN: intake["observations"][BASIN]},
            },
        )
        plan = sealed_write(root / "plan.json", trial._plan(data))
    old_indices = {slot: i for i, slot in enumerate(trial.slots(intake))}
    records = []
    with _lock(root / "generation-lock"):
        for i, (g, arm, rank) in enumerate(trial.slots(data)):
            old = old_indices[parents[g], arm, rank]
            # Includes verification of the model reply and underlying physical call.
            before = trial._read_candidate(archive, original, old)
            graph = graphs[g]
            if (
                before["artifact_sha256"] != generation["candidates"][old]
                or before["graph"] != graph["structure_sha256"]
                or (before["arm"], before["rank"], before["seed"])
                != (arm, rank, trial.SEEDS[rank])
            ):
                raise ValueError("Saved candidate differs from its registered slot")
            record = {
                "index": i,
                "arm": arm,
                "rank": rank,
                "seed": before["seed"],
                "graph": before["graph"],
                "plan": plan["artifact_sha256"],
                "archived_index": old,
                "archived_status": before["status"],
                "archived_candidate_sha256": before["artifact_sha256"],
            }
            if arm == "algorithm":
                if before["formula"] != data["menus"][g]["candidates"][rank]:
                    raise ValueError(
                        "Algorithm receipt differs from the frozen shortlist"
                    )
                record.update(trial.algorithm_candidate(data, g, rank))
            else:
                reply = sealed_read(archive / "steps" / f"slot-{old:03}" / "reply.json")
                if reply["status"] != "visible":
                    raise ValueError("Expected an archived visible basin reply")
                candidate = reply["value"]["candidate"]
                if before["status"] == "eligible":
                    fields = {"route": "fixed_graph"}
                elif before["status"] == "invalid_formula":
                    fields = {
                        "route": "request_topology_change",
                        "scientific_reason": AUDIT_REASON,
                    }
                    record["reason_source"] = "retrospective_audit"
                else:
                    raise ValueError("Unrecognized outcome in the archived study")
                record["archived_reply_sha256"] = reply["artifact_sha256"]
                record.update(
                    trial.assess_reply(data, g, arm, {"candidate": candidate | fields})
                )
            records.append(sealed_write(root / "candidates" / f"{i:03}.json", record))
        sealed_write(
            root / "generation.json",
            {
                "plan": plan["artifact_sha256"],
                "candidates": [r["artifact_sha256"] for r in records],
            },
        )
    return trial.status(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.archive, args.root)
    for i in range(result["candidate_slots"]):
        trial.fit(args.root, i)
    plan, data = trial.load(args.root)
    trial.finalists(args.root, plan, data)
    print(json.dumps(trial.status(args.root), indent=2))


if __name__ == "__main__":
    main()
