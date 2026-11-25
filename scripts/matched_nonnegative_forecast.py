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
from scripts import public_target_bounds as public_bounds
from scripts import topology_forecast_eligible as eligible
from scripts import topology_forecast_fit as adapter
from scripts.construction_functions import file_hash

PARENT = "62e9bd0fda7c92f0981671278006c81d6ab9d3917a01220881acd3fae83939d5"
DEFAULT = original.ROOT / "output/matched-nonnegative-forecast-v2"
PROTOCOL = "matched-sampled-nonnegative-refit-2"
# Completed under its original runtime. Its receipts remain readable by status.
LEGACY_PLAN = "e6eb96e11049fbfaa2ee62c15a2b706b8165f1d6b2a300acc71c98af2a412c15"


def prepare(root: Path, parent: Path) -> dict:
    """Authenticate the original plan without opening historical fit endpoints."""
    if root.resolve() == parent.resolve():
        raise ValueError("The refit requires a separate results directory")
    path = root / "plan.json"
    if path.exists() and sealed_read(path)["artifact_sha256"] == LEGACY_PLAN:
        raise ValueError(
            "The completed v1 archive is read-only. Use status to verify its receipts "
            "or prepare a new v2 run in a separate directory."
        )
    previous = sealed_read(parent / "plan.json")
    if previous["artifact_sha256"] != PARENT:
        raise ValueError("Expected the registered 96-formula forecast experiment")
    identity = original.implementation()
    if (
        identity["versions"] != previous["implementation"]["versions"]
        or identity["sources"] != previous["implementation"]["sources"]
    ):
        raise ValueError("Common fitting adapter, graph pipeline or packages differ")
    identity["eligibility"] = file_hash(Path(eligible.__file__))
    contract = public_bounds.matched_bound_contract(previous["cohort"])
    identity["refit_runner"] = file_hash(Path(__file__))
    identity["bound_resolution"] = file_hash(Path(public_bounds.__file__))
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
                **contract,
                "split": "train",
                "locations": "all observation times in all training events",
                "starting_parameters": "frozen constructor and original rank seed",
                "objective": "original training-scaled pooled squared error",
            },
        },
    )
    eligible.register(root)
    return plan


def _fit(row: dict, slot: dict, budget: dict, targets: list[str]) -> dict:
    from autoformalism.research.trajectory_nonnegative_fitting import fit_nonnegative

    train = adapter._training(row["case"], row["dataset"])["evidence"]["trajectories"]
    result = fit_nonnegative(
        row["graph"],
        slot["formula"],
        train,
        seed=slot["seed"],
        targets=targets,
        **budget,
    )
    if result["budget"] != {**budget, "seed": slot["seed"]}:
        raise ValueError("Optimizer used a different registered budget")
    if not 0 <= result["evaluations"] <= budget["max_evaluations"]:
        raise ValueError("Optimizer exceeded its residual evaluation cap")
    if result["status"] == "fitted":
        counts = result["negative_samples"]
        if set(result["constrained_targets"]) != set(targets) or any(
            counts.get(name) != 0 for name in targets
        ):
            raise ValueError("Constrained fit has negative or unchecked outputs")
    return result


def _slot(
    args: tuple, *, root: Path, digest: str, budget: dict, targets: list[str]
) -> str:
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
            result = _fit(row, slot, budget, targets)
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
        _slot,
        root=root,
        digest=plan["artifact_sha256"],
        budget=plan["fit_budget"],
        targets=plan["constraint"]["targets"],
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


def status(root: Path) -> dict:
    """Verify recorded fits against their sealed plan without invoking the fitter."""
    plan = sealed_read(root / "plan.json")
    receipts, counts = {}, Counter()
    for slot in plan["slots"]:
        key = slot["fit_key"]
        if (root / "fits" / f"{key}.json").exists():
            record = original._receipt(root, slot, plan["artifact_sha256"])
            receipts[key] = record["artifact_sha256"]
            counts[record["result"]["status"]] += 1
    scored = (root / "development.json").exists()
    if scored:
        frozen = sealed_read(root / "frozen.json")
        development = sealed_read(root / "development.json")
        if (
            len(receipts) != len(plan["slots"])
            or frozen["plan"] != plan["artifact_sha256"]
            or frozen["fit_receipts"] != receipts
            or development["freeze"] != frozen["artifact_sha256"]
        ):
            raise ValueError("Archived forecast differs from its frozen fit receipts")
    return {
        "plan": plan["artifact_sha256"],
        "verified_fits": len(receipts),
        "expected": len(plan["slots"]),
        "fit_status": counts,
        "scored": scored,
    }


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
        print(json.dumps(status(root)))
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
