"""Refit the previously frozen formulas under sampled nonnegativity constraints."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import zip_longest
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import matched_topology_forecast as original
from scripts import topology_forecast_eligible as eligible
from scripts import topology_forecast_fit as adapter
from scripts.construction_functions import file_hash

PARENT = "62e9bd0fda7c92f0981671278006c81d6ab9d3917a01220881acd3fae83939d5"
DEFAULT = original.ROOT / "output/matched-nonnegative-forecast-20261124"
TARGETS = ("Gp", "I", "U")
PROTOCOL = "matched-sampled-nonnegative-refit-1"


def prepare(root: Path, parent: Path) -> dict:
    """Authenticate the original plan without opening historical fit endpoints."""
    if root.resolve() == parent.resolve():
        raise ValueError("The refit requires a separate results directory")
    previous = sealed_read(parent / "plan.json")
    if previous["artifact_sha256"] != PARENT:
        raise ValueError("Expected the registered 96-formula forecast experiment")
    identity = original.implementation()
    if (
        identity["versions"] != previous["implementation"]["versions"]
        or identity["sources"] != previous["implementation"]["sources"]
    ):
        raise ValueError("Common fitting adapter, graph pipeline or packages differ")
    identity["refit_runner"] = file_hash(Path(__file__))
    identity["eligibility"] = file_hash(Path(eligible.__file__))
    plan = sealed_write(
        root / "plan.json",
        {
            **{
                key: value
                for key, value in previous.items()
                if key != "artifact_sha256"
            },
            "protocol": PROTOCOL,
            "parent_plan": PARENT,
            "implementation": identity,
            "constraint": {
                "targets": list(TARGETS),
                "split": "train",
                "locations": "all observation times in all training events",
                "starting_parameters": "frozen constructor and original rank seed",
                "objective": "original training-scaled pooled squared error",
            },
        },
    )
    eligible.register(root)
    return plan


def _fit(row: dict, slot: dict, budget: dict) -> dict:
    from autoformalism.research.trajectory_nonnegative_fitting import fit_nonnegative

    train = adapter._training(row["case"], row["dataset"])["evidence"]["trajectories"]
    result = fit_nonnegative(
        row["graph"],
        slot["formula"],
        train,
        seed=slot["seed"],
        targets=TARGETS,
        **budget,
    )
    if result["budget"] != {**budget, "seed": slot["seed"]}:
        raise ValueError("Optimizer used a different registered budget")
    if not 0 <= result["evaluations"] <= budget["max_evaluations"]:
        raise ValueError("Optimizer exceeded its residual evaluation cap")
    if result["status"] == "fitted":
        counts = result["negative_samples"]
        if set(counts) != set(TARGETS) or any(counts.values()):
            raise ValueError("Constrained fit has negative or unchecked outputs")
    return result


def _slot(args: tuple, *, root: Path, digest: str, budget: dict) -> str:
    row, slot = args
    key = slot["fit_key"]
    path = root / "fits" / f"{key}.json"
    marker = {"plan": digest, "slot_sha256": content_hash(slot)}
    # Per-slot lock protects a surviving subprocess if its coordinator exits.
    with _lock(root / "reservations" / key):
        if path.exists():
            original._receipt(root, slot, digest)
            return key
        reservation = root / "reservations" / key / "start.json"
        if reservation.exists():
            sealed_write(reservation, marker)
            result = {"status": "interrupted", "training_nmse": None}
        else:
            sealed_write(reservation, marker)
            result = _fit(row, slot, budget)
        sealed_write(
            path,
            {
                **marker,
                "fit_key": key,
                "formula": slot["formula"],
                "result": result,
            },
        )
    return key


def fit_all(root: Path, plan: dict, workers: int) -> None:
    """Interleave inventory and generator while giving every formula its own slot."""
    groups = defaultdict(list)
    for slot in plan["slots"]:
        row = plan["cohort"]["rows"][slot["graph_index"]]
        groups[row["inventory"], row["arm"]].append((row, slot))
    ordered = (
        pair
        for group in zip_longest(*groups.values())
        for pair in group
        if pair is not None
    )
    run = partial(
        _slot, root=root, digest=plan["artifact_sha256"], budget=plan["fit_budget"]
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, key in enumerate(pool.map(run, ordered), 1):
            print(f"Fit {completed}/{len(plan['slots'])}  {key}", flush=True)


def forecast(root: Path, plan: dict) -> None:
    """Open development only after both complete training selections are frozen."""
    frozen = sealed_read(root / "frozen.json")
    qualification = sealed_read(root / "eligibility-freeze.json")
    if (
        qualification["plan"] != plan["artifact_sha256"]
        or qualification["fit_receipts"] != frozen["fit_receipts"]
    ):
        raise ValueError(
            "Eligibility and forecast selections must share all fit receipts"
        )
    original.development(root, plan, frozen)
    eligible.development(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "fit", "freeze", "score", "run", "status")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--parent", type=Path, default=original.DEFAULT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "status":
        stats = Counter(
            sealed_read(p)["result"]["status"] for p in (root / "fits").glob("*.json")
        )
        print(
            json.dumps(
                {"fit_status": stats, "scored": (root / "development.json").exists()}
            )
        )
        return
    with _lock(root):
        plan = prepare(root, args.parent)
        print(
            f"Plan {plan['artifact_sha256']}  {len(plan['slots'])} formulas", flush=True
        )
        if args.stage in ("fit", "run"):
            fit_all(root, plan, args.workers)
        if args.stage in ("freeze", "run"):
            original.freeze(root, plan)
            eligible.freeze(root)
        if args.stage in ("score", "run"):
            forecast(root, plan)


if __name__ == "__main__":
    main()
