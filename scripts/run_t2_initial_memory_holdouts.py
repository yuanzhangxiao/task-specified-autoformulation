"""Compare persistent and fading initial glucose effects on two held events."""

from __future__ import annotations

import argparse
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import scipy

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import positive_insulin_refit_inputs as intake
from scripts import t2_initial_memory_folds as folds
from scripts import t2_initial_memory_forecast as forecast

HERE = Path(__file__).resolve().parent
TRAIN = Path("output/trajectory-functions-20261123/inputs/train.json")
ARCHIVE = Path("output/t2-initial-order-20261123/orion-initial-order-handoff.zip")
OUTPUT = Path("output/t2-initial-memory-holdouts-20261124")
ARMS = ("persistent", "fading")
COLD = ("short", "long")
STARTS = (*COLD, "nested")
BUDGET = 200


def _checkpoint(path: Path, identity: dict, produce):
    """Resume only if the data, seed and frozen protocol agree."""
    if path.exists():
        value = sealed_read(path)
        if value["identity"] != identity:
            raise ValueError(f"Checkpoint identity changed at {path}")
        return value["result"]
    value = sealed_write(path, {"identity": identity, "result": produce()})
    return value["result"]


def _best(attempts: list[dict]) -> dict:
    """Select the feasible fit with the lowest independent training replay error."""
    qualified = [
        a
        for a in attempts
        if a["replay"]["eligible"]
        and a["replay"]["summary"]["pooled_nmse"] is not None
        and math.isfinite(a["replay"]["summary"]["pooled_nmse"])
    ]
    if not qualified:
        raise ValueError(
            "No candidate passed the fifteen-event replay and ordering checks"
        )
    return min(
        qualified,
        key=lambda a: (a["replay"]["summary"]["pooled_nmse"], STARTS.index(a["start"])),
    )


def _fold(job: tuple) -> dict:
    """Freeze both selected models from exactly fifteen events."""
    from scripts import t2_initial_memory_fit as kernel

    root, train, held, graph, plan_id = job
    directory = root / held
    rows = folds.load_fit_rows(train, held)
    identity = {"plan": plan_id, "held": held, "training_rows": content_hash(rows)}
    setup = _checkpoint(
        directory / "fold.json", identity, lambda: folds.fit_inputs(rows, held)
    )
    if sorted(setup["fit_ids"]) != list(folds.fit_ids(held)):
        raise ValueError("Insulin fit and model fitting IDs differ")
    setup = setup | {"graph": graph}
    nested_seed = None
    selected = {}
    for arm in ARMS:
        candidates = []
        for start in STARTS:
            if arm == "persistent" and start == "nested":
                nested_seed = _best(candidates)["model"]
            seed = nested_seed if start == "nested" else None
            key = identity | {
                "arm": arm,
                "start": start,
                "seed": content_hash(seed),
                "setup": content_hash(setup),
            }

            attempt = _checkpoint(
                directory / f"{arm}-{start}-fit.json",
                key,
                partial(
                    kernel.fit,
                    rows,
                    setup,
                    arm,
                    start,
                    budget=BUDGET,
                    persistent_model=seed,
                ),
            )
            model = attempt["model"]
            if model is None:
                print(f"{held} {arm} {start}: {attempt['termination']}", flush=True)
                continue
            replay = _checkpoint(
                directory / f"{arm}-{start}-replay.json",
                key | {"model": content_hash(model)},
                partial(forecast.training_replay, model, rows),
            )
            print(
                f"{held} {arm} {start} NMSE {replay['summary']['pooled_nmse']}"
                f" eligible={replay['eligible']}",
                flush=True,
            )
            candidates.append({"model": model, "start": start, "replay": replay})
        winner = _best(candidates)
        selected[arm] = {k: v for k, v in winner.items() if k != "replay"} | {
            "train_summary": winner["replay"]["summary"],
            "ordering": winner["replay"]["ordering"],
            "domain": winner["replay"]["domain"],
        }
    return sealed_write(
        directory / "selected.json", {"identity": identity, "models": selected}
    )


def _future(root: Path, selections: dict, train: Path) -> dict:
    """Freeze all four predictions before scoring any omitted target."""
    if set(selections) != set(folds.HOLDOUTS) or any(
        set(s["models"]) != set(ARMS) for s in selections.values()
    ):
        raise ValueError(
            "All four training-only selections must be frozen before forecasting"
        )
    immutable = {k: s["artifact_sha256"] for k, s in selections.items()}
    truth = {row["trajectory_id"]: row for row in intake.load_training(train)}

    def predict_all():
        return {
            held: {
                arm: forecast.predict(
                    selections[held]["models"][arm]["model"],
                    forecast.public_event(truth[held]),
                )
                for arm in ARMS
            }
            for held in folds.HOLDOUTS
        }

    predictions = _checkpoint(root / "forecasts.json", immutable, predict_all)
    scores = {}
    for held in folds.HOLDOUTS:
        scores[held] = {
            arm: forecast.score(
                selections[held]["models"][arm]["model"],
                predictions[held][arm],
                truth[held],
            )
            for arm in ARMS
        }
    sealed_write(
        root / "scores.json",
        {
            "frozen_models": immutable,
            "forecasts": content_hash(predictions),
            "scores": scores,
        },
    )
    forecast.plot(scores, root / "curves.png")
    return scores


def run(
    root: Path, train: Path = TRAIN, archive: Path = ARCHIVE, workers: int = 2
) -> dict:
    """Run or deterministically resume the two-event comparison."""
    handoff = folds.authenticate_handoff(archive)
    code = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            HERE / "t2_initial_memory_folds.py",
            HERE / "t2_initial_memory_fit.py",
            HERE / "t2_initial_memory_forecast.py",
        ]
    }
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-initial-memory-holdout-1",
                "code": code,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "training_artifact": intake.original.campaign.TRAIN,
                "structure_source": handoff["model_sha256"],
                "fit_ids": {h: list(folds.fit_ids(h)) for h in folds.HOLDOUTS},
                "arms": list(ARMS),
                "starts_per_arm": list(STARTS),
                "budget_per_start": BUDGET,
                "third_start": "best eligible persistent cold fit",
                "minimum_observed_initial_gap_fraction": 0.5,
                "selection": "lowest pooled 15-event replay NMSE among feasible fits",
                "held_initial_measurements": ["Gp", "I"],
                "official_validation_opened": False,
                "test_opened": False,
            },
        )
        jobs = [
            (root, train, h, handoff["graph"], plan["artifact_sha256"])
            for h in folds.HOLDOUTS
        ]
        if workers == 1:
            results = list(map(_fold, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_fold, jobs))
        return _future(root, dict(zip(folds.HOLDOUTS, results, strict=True)), train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.root, workers=args.workers)
    print(args.root / "scores.json")


if __name__ == "__main__":
    main()
