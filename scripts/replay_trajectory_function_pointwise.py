"""Recheck frozen ACES function candidates under an explicit pointwise contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash
from scripts.replay_trajectory_function_revisions import (
    ARCHIVED_GENERATION,
    ARCHIVED_PLAN,
)


def replay(archive: Path, root: Path, policy: dict) -> dict:
    """Replay admission and exercise the fit path for newly blocked slots."""
    archive, root = archive.resolve(), root.resolve()
    if root.is_relative_to(archive) or archive.is_relative_to(root):
        raise ValueError("Use a separate output directory for retrospective admission")
    prior = sealed_read(archive / "plan.json")
    source = sealed_read(archive / "intake.json")
    freeze = trial._frozen_candidates(archive, prior)
    if (
        prior["artifact_sha256"] != ARCHIVED_PLAN
        or freeze["artifact_sha256"] != ARCHIVED_GENERATION
        or prior["intake_sha256"] != source["artifact_sha256"]
        or source["source"] != trial.TRAIN
        or source.get("profile", trial.FIXED) != trial.FIXED
    ):
        raise ValueError("Expected ACES 2404388 and its original fixed-graph intake")
    policy = trial.checked_tolerances(source, source["graphs"], policy)
    with _lock(root / "prepare-lock"):
        data = sealed_write(
            root / "intake.json",
            {
                **{k: v for k, v in source.items() if k != "artifact_sha256"},
                "profile": trial.POINTWISE_REPLAY,
                "source_profile": trial.FIXED,
                "source_plan": ARCHIVED_PLAN,
                "source_generation": ARCHIVED_GENERATION,
                "replay_code_sha256": file_hash(Path(__file__)),
                "pointwise_tolerances": policy,
            },
        )
        plan = sealed_write(root / "plan.json", trial._plan(data))
    comparisons = []
    proofs = []
    changed = []
    with _lock(root / "generation-lock"):
        for i, (g, arm, rank) in enumerate(trial.slots(source)):
            row = source["graphs"][g]
            before = trial._read_candidate(archive, prior, i)
            if (
                before["artifact_sha256"] != freeze["candidates"][i]
                or before["graph"] != row["structure_sha256"]
                or (before["arm"], before["rank"], before["seed"])
                != (arm, rank, trial.SEEDS[rank])
            ):
                raise ValueError("Archived candidate differs from its frozen slot")
            if trial.pointwise_admission(source, row, before) is not before:
                raise ValueError("Default admission changed the frozen candidate")
            after = trial.pointwise_admission(data, row, before)
            record = {
                k: v
                for k, v in after.items()
                if k not in {"artifact_sha256", "plan", "reply_sha256"}
            }
            if before.get("reply_sha256"):
                record["archived_reply_sha256"] = before["reply_sha256"]
            record.update(
                plan=plan["artifact_sha256"],
                archived_candidate_sha256=before["artifact_sha256"],
                archived_status=before["status"],
            )
            saved = sealed_write(root / "candidates" / f"{i:03}.json", record)
            if before["status"] != after["status"]:
                if (before["status"], after["status"]) != (
                    "eligible",
                    "pointwise_conflict",
                ):
                    raise ValueError("Unexpected admission transition")
                changed.append(i)
                proofs.append(
                    {"index": i, "arm": arm, "feedback": after["repair_feedback"]}
                )
            comparisons.append(
                {
                    "index": i,
                    "arm": arm,
                    "dataset": row["dataset"],
                    "before": before["status"],
                    "after": after["status"],
                    "checks": {
                        k: v["status"]
                        for k, v in after.get("pointwise_checks", {}).items()
                    },
                    "candidate_sha256": saved["artifact_sha256"],
                }
            )
        generation = sealed_write(
            root / "generation.json",
            {
                "plan": plan["artifact_sha256"],
                "candidates": [r["candidate_sha256"] for r in comparisons],
            },
        )
    for i in changed:
        result = trial.fit(root, i)["result"]
        if result != {"status": "ineligible", "reason": "pointwise_conflict"}:
            raise ValueError("A confirmed conflict reached the optimizer")
    if list(root.glob("fits/*/reservation.json")) or list(
        root.glob("steps/*/calls/*.json")
    ):
        raise ValueError(
            "Admission-only replay consumed an optimization or generation allowance"
        )
    if proofs:
        sealed_write(root / "example-feedback.json", proofs[0])
    return sealed_write(
        root / "admission.json",
        {
            "retrospective": True,
            "plan": plan["artifact_sha256"],
            "source_generation": ARCHIVED_GENERATION,
            "generation": generation["artifact_sha256"],
            "policy": policy,
            "before": dict(Counter(r["before"] for r in comparisons)),
            "after": dict(Counter(r["after"] for r in comparisons)),
            "newly_blocked_fit_indices": changed,
            "rows": comparisons,
            "new_generation_requests": 0,
            "fit_reservations": 0,
            "validation_data_opened": False,
            "test_data_opened": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pointwise-tolerances", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.archive, args.root, trial.read_tolerances(args.pointwise_tolerances)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
