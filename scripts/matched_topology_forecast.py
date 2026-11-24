"""Compare topology generators with common inventories, equations and fit budgets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.staged_topology import content_hash
from scripts import topology_forecast_fit as adapter
from scripts.construction_functions import file_hash
from scripts.topology_forecast_cohort import build_cohort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/matched-topology-forecast-20261124"
PROTOCOL = "inventory-matched-topology-to-forecast-1"
CODE = (
    "scripts/matched_topology_forecast.py",
    "scripts/topology_forecast_fit.py",
    "scripts/topology_forecast_cohort.py",
    "scripts/trajectory_functions.py",
    "scripts/preproposal_graph_portfolio.py",
    "scripts/preproposal_graph_exports.py",
    "scripts/trajectory_function_intake.py",
    "scripts/pointwise_repair_trial.py",
)


def implementation() -> dict:
    """Bind the campaign to the code and numerical libraries used."""
    return {
        "source_package": runtime_source_hash(),
        "sources": {name: file_hash(ROOT / name) for name in CODE},
        "versions": {name: version(name) for name in ("numpy", "scipy", "casadi")},
    }


def prepare(root: Path) -> dict:
    """Freeze graph selection and both formula slots before optimization."""
    existing = root / "plan.json"
    identity = implementation()
    if existing.exists():
        plan = sealed_read(existing)
        if plan["implementation"] != identity or plan["protocol"] != PROTOCOL:
            raise ValueError("Experiment implementation differs from the frozen plan")
        return plan
    cohort = build_cohort()
    slots = []
    for index, row in enumerate(cohort["rows"]):
        formulas = adapter.construct(row)
        if len(formulas) > len(adapter.SEEDS):
            raise ValueError("Constructor returned more portfolios than registered")
        for rank, seed in enumerate(adapter.SEEDS):
            formula = formulas[rank] if rank < len(formulas) else None
            slots.append(
                {
                    "graph_index": index,
                    "rank": rank,
                    "seed": seed,
                    "fit_key": f"{index:03d}-{rank}",
                    "formula": formula,
                    "prefit": adapter.prefit(row, formula)
                    if formula is not None
                    else None,
                }
            )
    return sealed_write(
        existing,
        {
            "protocol": PROTOCOL,
            "cohort": cohort,
            "fit_budget": adapter.FIT_BUDGET,
            "implementation": identity,
            "slots": slots,
            "validation_release": adapter.VALIDATION,
            "selection": (
                "minimum finite grid-verified training NMSE within graph, "
                "then within inventory and arm"
            ),
            "secondary_selection": (
                "same rule among formulas with zero negative "
                "training samples across observed targets"
            ),
        },
    )


def _receipt(root: Path, slot: dict, plan_id: str) -> dict:
    record = sealed_read(root / "fits" / (slot["fit_key"] + ".json"))
    if (record["plan"], record["slot_sha256"]) != (plan_id, content_hash(slot)):
        raise ValueError("Fit receipt belongs to a different plan or slot")
    return record


def _run_slot(args: tuple) -> str:
    root, plan_id, row, slot = args
    key = slot["fit_key"]
    path = root / "fits" / (key + ".json")
    if path.exists():
        _receipt(root, slot, plan_id)
        return key
    # A terminated worker may have spent its allowance. Preserve its reservation
    # and count that slot as unavailable instead of restarting its optimization.
    reservation = root / "reservations" / (key + ".json")
    marker = {"plan": plan_id, "slot_sha256": content_hash(slot)}
    if reservation.exists():
        sealed_write(reservation, marker)
        result = {"status": "interrupted", "training_nmse": None}
    elif slot["formula"] is None:
        result = {"status": "constructor_unavailable", "training_nmse": None}
    else:
        sealed_write(reservation, marker)
        result = adapter.fit(row, slot["formula"], slot["rank"])
    sealed_write(
        path, {**marker, "fit_key": key, "formula": slot["formula"], "result": result}
    )
    return key


def run_fits(root: Path, plan: dict, workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = plan["cohort"]["rows"]
    buckets = defaultdict(list)
    for slot in plan["slots"]:
        row = rows[slot["graph_index"]]
        buckets[row["inventory"], row["arm"]].append(
            (root, plan["artifact_sha256"], row, slot)
        )
    ordered = (
        item
        for group in zip_longest(*buckets.values())
        for item in group
        if item is not None
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, key in enumerate(pool.map(_run_slot, ordered, chunksize=1), 1):
            print(f"Fit {i}/{len(plan['slots'])}  {key}", flush=True)


def _lowest(rows: list) -> dict | None:
    return min(
        rows, key=lambda r: (r["train_nmse"], r["graph_id"], r["rank"]), default=None
    )


def freeze(root: Path, plan: dict) -> dict:
    """Select and seal using training scores exclusively, including failed slots."""
    selected, receipts, physical = [], {}, defaultdict(list)
    by_graph = defaultdict(list)
    for slot in plan["slots"]:
        receipt = _receipt(root, slot, plan["artifact_sha256"])
        receipts[slot["fit_key"]] = receipt["artifact_sha256"]
        result = receipt["result"]
        graph_idx = slot["graph_index"]
        row = plan["cohort"]["rows"][graph_idx]
        counts = result.get("negative_samples") or {}
        checked = (
            result["status"] == "fitted"
            and result.get("parameters") is not None
            and result.get("training_nmse") is not None
            and math.isfinite(result["training_nmse"])
        )
        item = {
            **{k: row[k] for k in ("inventory", "arm", "graph_id")},
            "rank": slot["rank"],
            "fit_key": slot["fit_key"],
            "prefit": slot["prefit"],
            "fit_status": result["status"],
            "train_nmse": result.get("training_nmse") if checked else None,
            "physical_train": bool(
                checked and counts and all(n == 0 for n in counts.values())
            ),
        }
        by_graph[graph_idx].append(item)
        if item["physical_train"]:
            physical[row["inventory"], row["arm"]].append(item)

    winners, secondary = {}, {}
    for idx in range(len(plan["cohort"]["rows"])):
        records = by_graph[idx]
        if len(records) != len(adapter.SEEDS):
            raise ValueError("Graph does not have both registered slots")
        eligible = [r for r in records if r["train_nmse"] is not None]
        best = _lowest(eligible)
        if best is None:
            best = {
                **records[0],
                "rank": None,
                "fit_key": None,
                "prefit": None,
                "fit_status": "no_verified_training_fit",
                "physical_train": False,
            }
        selected.append(best)
    for inv in sorted({r["inventory"] for r in selected}):
        winners[inv], secondary[inv] = {}, {}
        for arm in ("algorithm", "20b_repair"):
            eligible = [
                r
                for r in selected
                if r["inventory"] == inv
                and r["arm"] == arm
                and r["train_nmse"] is not None
            ]
            best = _lowest(eligible)
            winners[inv][arm] = best["graph_id"] if best else None
            secondary[inv][arm] = _lowest(physical[inv, arm])
    return sealed_write(
        root / "frozen.json",
        {
            "plan": plan["artifact_sha256"],
            "fit_receipts": receipts,
            "selections": selected,
            "winners": winners,
            "nonnegative_winners": secondary,
        },
    )


def development(root: Path, plan: dict, frozen: dict) -> dict:
    """Score sealed winners, sharing validation replays for identical selections."""
    registered = sealed_read(root / "frozen.json")
    if registered != frozen or frozen["plan"] != plan["artifact_sha256"]:
        raise ValueError("Development requires the matching training freeze")
    slots = {s["fit_key"]: s for s in plan["slots"]}

    def score(item: dict) -> dict:
        key = item["fit_key"]
        if key is None:
            return {**item, "score": None}
        slot = slots[key]
        receipt = _receipt(root, slot, plan["artifact_sha256"])
        if receipt["artifact_sha256"] != frozen["fit_receipts"][key]:
            raise ValueError("Selected parameters differ from frozen receipt")
        graph = plan["cohort"]["rows"][slot["graph_index"]]
        marker = {
            "freeze": frozen["artifact_sha256"],
            "fit": receipt["artifact_sha256"],
        }
        path = root / "scores" / (key + ".json")
        if path.exists():
            cached = sealed_read(path)
            if any(cached[k] != v for k, v in marker.items()):
                raise ValueError("Score checkpoint belongs to different fit")
        else:
            value = adapter.validate(graph, slot["formula"], receipt["result"])
            cached = sealed_write(path, {**marker, "score": value})
        return {**item, "score": cached["score"]}

    return sealed_write(
        root / "development.json",
        {
            "freeze": frozen["artifact_sha256"],
            "rows": [score(r) for r in frozen["selections"]],
            "nonnegative_rows": [
                score(row)
                for arms in frozen["nonnegative_winners"].values()
                for row in arms.values()
                if row is not None
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "fit", "freeze", "score", "status")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "status":
        results = [
            sealed_read(p)["result"]["status"]
            for p in sorted((root / "fits").glob("*.json"))
        ]
        print(
            json.dumps(
                {
                    "fit_count": len(results),
                    "statuses": Counter(results),
                    "frozen": (root / "frozen.json").exists(),
                    "scored": (root / "development.json").exists(),
                }
            )
        )
        return
    with _lock(root):
        plan = prepare(root)
        print(f"Plan {plan['artifact_sha256']}  {len(plan['slots'])} slots", flush=True)
        if args.action == "fit":
            run_fits(root, plan, args.workers)
        if args.action == "freeze":
            freeze(root, plan)
        if args.action == "score":
            development(root, plan, sealed_read(root / "frozen.json"))


if __name__ == "__main__":
    main()
