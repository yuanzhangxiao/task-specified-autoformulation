#!/usr/bin/env python3
"""Describe T2-hard search history and match candidates using development evidence."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from itertools import combinations, groupby
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity

if __package__:
    from .prepare_t2_hard_subjects import CELL, ROUND, verify_seal
else:
    from prepare_t2_hard_subjects import CELL, ROUND, verify_seal

LINEAGE = "cell08_seed0_full_c1v1s1"


def model_summary(row: dict) -> dict:
    """Use name-invariant functional fingerprints and count all fitted parameters."""
    summary = {
        "source_id": row["source_id"],
        "method": row["method"],
        "seed": row["repetition"],
        "prompt": row["task"]["arm"],
        "status": row["status"],
    }
    if row["status"] == "ready":
        candidate = CandidateModel.model_validate(row["candidate"])
        summary.update(
            states=len(candidate.states),
            parameters=len(candidate.parameters),
            validation_nmse=row["validation_nmse"],
            source_model_sha256=row["source_model_sha256"],
            functional_sha256=candidate_identity(candidate).functional_sha256,
        )
    return summary


def match_pair(left: dict, right: dict) -> dict:
    """Apply every prespecified caliper and keep reasons for ineligibility."""
    result = {"left": left["source_id"], "right": right["source_id"]}
    if left["status"] != "ready" or right["status"] != "ready":
        return {**result, "eligible": False, "reasons": ["model_unavailable"]}
    a, b = left["validation_nmse"], right["validation_nmse"]
    if not all(
        isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in (a, b)
    ):
        return {**result, "eligible": False, "reasons": ["validation_unavailable"]}
    ratio = max(1e-8, a, b) / max(1e-8, min(a, b))
    p, q = left["parameters"], right["parameters"]
    reasons = []
    if ratio > 1.2:
        reasons.append("validation_caliper")
    if left["states"] != right["states"]:
        reasons.append("state_count")
    if abs(p - q) > max(2, 0.20 * min(p, q)):
        reasons.append("parameter_count")
    if left["functional_sha256"] == right["functional_sha256"]:
        reasons.append("same_functional_fingerprint")
    return {
        **result,
        "eligible": not reasons,
        "reasons": reasons,
        "validation_ratio": ratio,
        "state_gap": abs(left["states"] - right["states"]),
        "parameter_gap": abs(p - q),
    }


def matched_pairs(bundle: dict) -> dict:
    """Enumerate the six within-block pairs in each of the four matched blocks."""
    summaries = sorted(
        (model_summary(r) for r in bundle["rows"] if "task" in r),
        key=lambda m: (m["prompt"], m["seed"], m["method"]),
    )
    pairs = []
    for _, block in groupby(summaries, lambda m: (m["prompt"], m["seed"])):
        members = list(block)
        if len(members) != 4:
            raise ValueError("expected four arms in each matched block")
        pairs.extend(
            match_pair(left, right) for left, right in combinations(members, 2)
        )
    if len(pairs) != 24:
        raise ValueError("require the complete four-block pairing roster")
    return {
        "protocol": "t2-hard-validation-pairs-1",
        "subject_bundle_sha256": bundle["artifact_sha256"],
        "selection": "All arm pairs within the same six-visit prompt/seed block. "
        "Equal total dynamic state count; parameter-count gap <= max(2, 0.20 * "
        "smaller count); "
        "validation-NMSE ratio <= 1.20 (floor 1e-8); distinct alpha-invariant "
        "functional hashes. "
        "Hashes compare declared executable forms, without certifying "
        "physical inequivalence.",
        "models": summaries,
        "pairs": pairs,
        "eligible_pairs": sum(r["eligible"] for r in pairs),
        "planned_pairs": len(pairs),
        "probe_outcomes_read": False,
    }


def lineage(inventory: dict) -> dict:
    """Save the first full-method lineage, including trials and exact equations."""
    stages = {
        v["round"]: {"round": v["round"], "visit_status": v["status"]}
        for v in inventory["visits"]
        if v["task"]["task_id"] == LINEAGE and v["round"] <= ROUND
    }
    if set(stages) != set(range(ROUND + 1)):
        raise ValueError("expected all six planned lineage visits")
    models = {}
    for r in inventory["occurrences"]:
        if (
            r["task"]["task_id"] != LINEAGE
            or r["role"] not in ("trial", "selected")
            or r["round"] > ROUND
        ):
            continue
        identifier = r["model_id"]
        model = inventory["models"][identifier]
        if r["role"] in stages[r["round"]]:
            raise ValueError("duplicate role in lineage history")
        models[identifier] = {
            "candidate": model["candidate"],
            "initialization_plan": model["initialization_plan"],
            "fitted_parameters": model["parameters"],
            "parameter_status": model["parameter_status"],
        }
        stages[r["round"]][r["role"]] = {
            "model_id": identifier,
            "fit_status": r["fit"]["status"],
            "origin_round": r["provenance"].get("origin_round"),
            "training_nmse": r["fit"]["training"].get("normalized_mse"),
            "validation_nmse": r["fit"]["validation"].get("normalized_mse"),
            "source_result_sha256": r["source_sha256"],
        }
    return {
        "protocol": "t2-hard-development-lineage-1",
        "lineage": LINEAGE,
        "selection": "First numeric seed, Full prompt, critic and verifier on. "
        "Each visit preserves its trial and retained model if available. "
        "The inspection archive does not contain the complete critique messages.",
        "source_inventory_sha256": inventory["artifact_sha256"],
        "cell": CELL,
        "stages": [stages[k] for k in sorted(stages)],
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = sealed_read(args.models)
    with zipfile.ZipFile(args.inventory) as archive:
        inventory = json.loads(archive.read("inventory.json"))
    if (
        bundle.get("snapshot_protocol") != "iclr-t2-hard-six-visits-1"
        or verify_seal(inventory) != bundle["source_inventory_sha256"]
    ):
        raise ValueError("fixed roster and inspection source differ")
    matches = sealed_write(args.output / "matched-pairs.json", matched_pairs(bundle))
    sealed_write(args.output / "lineage.json", lineage(inventory))
    print(
        json.dumps(
            {
                k: matches[k]
                for k in ("planned_pairs", "eligible_pairs", "artifact_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
