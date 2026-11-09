#!/usr/bin/env python3
"""Repeat the four signed T2 fits with a nonnegative direct infusion gain."""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from autoformalism.research import t2_breadth_fitting as campaign
from scripts import run_t2_breadth_multistart as starts

GAIN = "k_insulin_term_7_0_92f124a75e"
OUTER = "af_process_gain_7_0"
SIGNED_PLAN = "31a962352ba61c98299fe7b589f8dfb7a9f800f9f55491c6e66b04070b640b14"


def bound_infusion(data: dict) -> dict:
    """Change just the inner coefficient domain in a separate numerical control."""
    control = {**data, "models": dict(data["models"])}
    for branch in ("c15-3", "c15-2"):
        entry = deepcopy(data["models"][branch])
        params = {
            p["name"]: p for p in entry["problem"]["base_candidate"]["parameters"]
        }
        if params[GAIN]["role"] != "coefficient" or params[OUTER]["role"] != (
            "nonnegative_coefficient"
        ):
            raise ValueError("unexpected direct insulin parameter domains")
        params[GAIN].update(role="nonnegative_coefficient", domain="nonnegative")
        entry["model_sha256"] = campaign.public.content_sha256(entry["problem"])
        control["models"][branch] = entry
    return control


def run(archive: Path, signed: Path, root: Path, workers: int) -> dict:
    data = campaign.intake.load_archive(archive)
    reference = campaign.read_seal(signed / "plan.json")
    if campaign.public.content_sha256(reference) != SIGNED_PLAN:
        raise ValueError("unrecognized signed comparison")
    bounded = bound_infusion(data)
    for branch in ("c15-3", "c15-2"):
        original, model = (campaign._model(branch, d) for d in (data, bounded))
        idx = model.names.index(GAIN)
        if original.names != model.names or model.lower[idx] != 0:
            raise ValueError("unexpected constrained optimization coordinates")
        if (
            any(
                model.lower[i] != original.lower[i]
                for i in range(len(model.names))
                if i != idx
            )
            or not (model.upper == original.upper).all()
        ):
            raise ValueError("other coordinate bounds changed")
    plan = starts.prepare(root, bounded)
    if any(
        plan[k] != reference[k]
        for k in (
            "archive_sha256",
            "training_sha256",
            "source_sha256",
            "runtime",
            "starts",
            "blocks",
            "seconds_per_search",
            "seconds_per_original_replay",
        )
    ):
        raise ValueError("matched policy or starts differ")
    comparison = {
        "protocol": "t2-nonnegative-infusion-control-1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "signed_plan_sha256": SIGNED_PLAN,
        "bounded_plan_sha256": campaign.public.content_sha256(plan),
        "original_model_sha256": reference["models"],
        "bounded_model_sha256": plan["models"],
        "constraint": f"{GAIN} >= 0",
        "training_only": True,
    }
    with campaign.public._lock(root):
        campaign.seal(root / "comparison-plan.json", comparison)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(starts.fit_start, root, t, bounded, plan) for t in starts.TASKS
        ]
        rows = [job.result() for job in jobs]
    counts = {}
    for row in rows:
        key = row["identity"]["task"]
        if row["status"] != "complete" or not row["best_admissible"]:
            raise ValueError(f"{key} has no completed, domain-passing fit")
        for category in ("best_numerical", "best_admissible"):
            p = row[category]["parameters"]
            if p[GAIN] < 0 or p[OUTER] < 0:
                raise ValueError("nonnegative direct infusion gain was violated")
        counts[key] = campaign.public.content_sha256(row)
    result = {"comparison": comparison, "status": "complete", "result_sha256": counts}
    with campaign.public._lock(root):
        campaign.seal(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--signed", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    print(run(args.archive, args.signed, args.root.resolve(), args.workers), flush=True)
