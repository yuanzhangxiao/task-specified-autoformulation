"""Freeze and score training winners that meet every registered admission condition."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import matched_topology_forecast as experiment

RULE = {
    "protocol": "matched-eligible-nonnegative-forecast-v1",
    "scope": "All original formula slots per inventory and generator",
    "prefit_status": "eligible",
    "fit_status": "fitted",
    "maximum_scaled_training_grid_difference": 0.01,
    "nonnegative": "Zero training samples below the fitter threshold (-1e-10) "
    "for every observed target",
    "ranking": ["finite_training_nmse", "graph_id", "formula_rank"],
    "empty_cell": "No model advanced",
}


def register(root: Path) -> dict:
    """Record the eligibility and ranking rule using the frozen plan alone."""
    plan = sealed_read(root / "plan.json")
    return sealed_write(
        root / "eligibility-rule.json", {"plan": plan["artifact_sha256"], "rule": RULE}
    )


def qualifies(prefit: dict | None, result: dict) -> bool:
    """Require proof that every registered training condition passed."""
    if (prefit or {}).get("status") != "eligible" or result["status"] != "fitted":
        return False
    nmse = result.get("training_nmse")
    grid = result.get("maximum_scaled_grid_difference")
    negative = result.get("negative_samples", {})
    targets = set(result.get("scales", {}))
    return (
        result.get("parameters") is not None
        and nmse is not None
        and math.isfinite(nmse)
        and grid is not None
        and 0 <= grid <= RULE["maximum_scaled_training_grid_difference"]
        and bool(targets)
        and targets == set(negative)
        and all(negative[name] == 0 for name in targets)
    )


def freeze(root: Path) -> dict:
    """Authenticate every training fit, select the winners and seal the yields."""
    plan = sealed_read(root / "plan.json")
    rule = sealed_read(root / "eligibility-rule.json")
    if rule["rule"] != RULE or rule["plan"] != plan["artifact_sha256"]:
        raise ValueError("Eligibility rule differs from registration")
    pools = defaultdict(list)
    hashes = {}
    for slot in plan["slots"]:
        row = plan["cohort"]["rows"][slot["graph_index"]]
        record = experiment._receipt(root, slot, plan["artifact_sha256"])
        key = slot["fit_key"]
        if (
            key in hashes
            or key != record["fit_key"]
            or record["formula"] != slot["formula"]
        ):
            raise ValueError("Duplicate or mismatched formula slot")
        hashes[key] = record["artifact_sha256"]
        result = record["result"]
        pools[row["inventory"], row["arm"]].append(
            {
                **{k: row[k] for k in ("inventory", "arm", "graph_id")},
                "rank": slot["rank"],
                "fit_key": key,
                "train_nmse": result.get("training_nmse"),
                "prefit": slot.get("prefit"),
                "fit_status": result["status"],
                "physical_train": qualifies(slot.get("prefit"), result),
            }
        )
    groups = []
    for (inventory, arm), records in sorted(pools.items()):
        eligible = [r for r in records if r["physical_train"]]
        groups.append(
            {
                "inventory": inventory,
                "arm": arm,
                "advanced_graphs": len({r["graph_id"] for r in records}),
                "formula_slots": len(records),
                "eligible_graphs": len({r["graph_id"] for r in eligible}),
                "eligible_slots": len(eligible),
                "winner": experiment._lowest(eligible),
            }
        )
    return sealed_write(
        root / "eligibility-freeze.json",
        {
            "plan": plan["artifact_sha256"],
            "rule_sha256": rule["artifact_sha256"],
            "fit_receipts": hashes,
            "groups": groups,
        },
    )


def development(root: Path) -> dict:
    """Replay only registered winners on the ordinary development adapter."""
    plan = sealed_read(root / "plan.json")
    frozen = sealed_read(root / "eligibility-freeze.json")
    if (
        frozen["plan"] != plan["artifact_sha256"]
        or register(root)["artifact_sha256"] != frozen["rule_sha256"]
    ):
        raise ValueError("Development and eligibility freeze do not match")
    path = root / "eligibility-development.json"
    if path.exists():
        cached = sealed_read(path)
        if cached["freeze"] != frozen["artifact_sha256"]:
            raise ValueError("Eligibility scores belong to another freeze")
        return cached
    slots = {slot["fit_key"]: slot for slot in plan["slots"]}
    rows = []
    for group in frozen["groups"]:
        item = group["winner"]
        if item is None:
            continue
        key = item["fit_key"]
        slot = slots[key]
        graph = plan["cohort"]["rows"][slot["graph_index"]]
        receipt = experiment._receipt(root, slot, plan["artifact_sha256"])
        if receipt["artifact_sha256"] != frozen["fit_receipts"][key]:
            raise ValueError("Selected parameters differ from the eligibility freeze")
        marker = {
            "freeze": frozen["artifact_sha256"],
            "fit": receipt["artifact_sha256"],
        }
        checkpoint = root / "eligibility-scores" / (key + ".json")
        if checkpoint.exists():
            cached = sealed_read(checkpoint)
            if any(cached[k] != v for k, v in marker.items()):
                raise ValueError("Eligibility score checkpoint differs")
        else:
            score = experiment.adapter.validate(
                graph, slot["formula"], receipt["result"]
            )
            cached = sealed_write(checkpoint, {**marker, "score": score})
        rows.append({**item, "score": cached["score"]})
    return sealed_write(path, {"freeze": frozen["artifact_sha256"], "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "freeze", "score"))
    parser.add_argument("--root", type=Path, default=experiment.DEFAULT)
    args = parser.parse_args()
    result = {"register": register, "freeze": freeze, "score": development}[
        args.action
    ](args.root)
    print(json.dumps({"file_sha256": result["artifact_sha256"]}))
