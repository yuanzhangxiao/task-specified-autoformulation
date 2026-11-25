"""Test a frozen constructor against synthetic ODEs with known forcing laws."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import shape_equation_nomination as constructor
from autoformalism.research import trajectory_function_fitting as fitting
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash

DEFAULT = Path(__file__).resolve().parents[1] / "output/shape-recovery-20261125"
SEEDS = tuple(range(8))
LAWS = ("affine", "hinge a", "tanh a", "square a", "product a b", "sine")
GRAPH = {
    "sources": {"x": ["x", "a", "b"]},
    "variable_types": {"x": "differential"},
    "initializers": {"x": {"mode": "observed_initial", "channel": "x"}},
}
STEP = dict.fromkeys(("a", "b"), "previous_complete_knots")
NTRAIN, NHELD, INTERVALS = 6, 6, 48


def event(law: str, seed: int, number: int, *, held: bool) -> dict:
    """Make one noise-free event; share schedules across laws for a given seed."""
    rng = np.random.default_rng(np.random.SeedSequence([20261125, seed, held, number]))
    dt = rng.uniform(0.08, 0.4, INTERVALS)
    a, b = rng.uniform(-np.sqrt(3), np.sqrt(3), (2, INTERVALS + 1))
    extra = {
        "affine": np.zeros_like(a),
        "hinge a": np.maximum(a, 0),
        "tanh a": np.tanh(a),
        "square a": a**2,
        "product a b": a * b,
        "sine": np.sin(3 * a + 2 * b),
    }[law]
    drive = 0.3 + 0.5 * a - 0.25 * b + extra
    x = [float(rng.uniform(-1, 1))]
    for width, u in zip(dt, drive[:-1], strict=True):
        decay = np.exp(-0.6 * width)
        x.append(float(decay * x[-1] - np.expm1(-0.6 * width) * u / 0.6))
    return {
        "trajectory_id": f"{'held' if held else 'train'}_{number:03}",
        "time": np.r_[0, np.cumsum(dt)].tolist(),
        "targets": {"x": x},
        "inputs": {"a": a.tolist(), "b": b.tolist()},
        "auxiliaries": {},
        "covariates": {},
    }


def forecast(formula: dict, events: list[dict], scale: float) -> dict:
    """Use returned starts and a step-held input on each irregular RK4 interval."""
    params = formula["parameters"]
    curves = []
    for steps in (4, 8):
        model = fitting._Rollout(GRAPH, formula, ["x"], steps)
        theta = np.array([params[k] for k in model.parameters])
        predictions = []
        for event in events:
            dt = np.diff(event["time"])
            forcings = np.array([event["inputs"][k][:-1] for k in model.forcings])
            # Inputs are constant up to each right endpoint, then may jump.
            # Only the observed initial x is passed to the propagator.
            y = model.step.mapaccum(len(dt))(
                event["targets"]["x"][0],
                np.repeat(theta[:, None], len(dt), axis=1),
                forcings,
                forcings,
                dt.reshape(1, -1),
            )
            predictions.extend(np.asarray(y).ravel())
        curves.append(np.asarray(predictions))
    if not all(np.isfinite(v).all() for v in curves):
        return {"status": "nonfinite"}
    delta = float(np.max(np.abs(curves[1] - curves[0])) / scale)
    if delta > 0.01:
        return {"status": "resolution_unverified", "grid_difference": delta}
    target = np.concatenate([ev["targets"]["x"][1:] for ev in events])
    return {
        "status": "scored",
        "nmse": float(np.mean(((curves[1] - target) / scale) ** 2)),
        "grid_difference": delta,
    }


def trial(law: str, seed: int) -> dict:
    training = [event(law, seed, i, held=False) for i in range(NTRAIN)]
    menu = constructor.build_shape_shortlist(GRAPH, training, interpolation=STEP)
    held = [event(law, seed, i, held=True) for i in range(NHELD)]
    scale = float(np.std(np.concatenate([r["targets"]["x"] for r in training])))
    baseline = np.concatenate(
        [r["targets"]["x"][0] - np.array(r["targets"]["x"][1:]) for r in held]
    )
    return {
        "truth": law,
        "seed": seed,
        "train_sha256": content_hash(training),
        "held_sha256": content_hash(held),
        "candidates": menu["candidates"],
        "ordered_families": [f["x"] for f in menu["candidate_families"]],
        "first_forecast": forecast(menu["candidates"][0], held, scale),
        "persistence_nmse": float(np.mean((baseline / scale) ** 2)),
    }


def run(root: Path) -> dict:
    with _lock(root / "write-lock"):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "anonymous-synthetic-equation-recovery-1",
                "laws": list(LAWS),
                "outside_menu": "sine",
                "seeds": list(SEEDS),
                "training_events": NTRAIN,
                "forecast_events": NHELD,
                "intervals_per_event": INTERVALS,
                "graph": GRAPH,
                "interpolation": STEP,
                "sources": {
                    path.name: file_hash(path)
                    for path in (
                        Path(__file__),
                        Path(constructor.__file__),
                        Path(constructor.base.__file__),
                        Path(fitting.__file__),
                    )
                },
            },
        )
        summaries = []
        for index, law in enumerate(LAWS):
            rows = []
            for seed in SEEDS:
                path = root / "trials" / f"law{index}-seed{seed}.json"
                if path.exists():
                    r = sealed_read(path)
                else:
                    r = sealed_write(
                        path, {"plan": plan["artifact_sha256"], **trial(law, seed)}
                    )
                if (r["plan"], r["truth"], r["seed"]) != (
                    plan["artifact_sha256"],
                    law,
                    seed,
                ):
                    raise ValueError("Trial differs from the frozen synthetic plan")
                rows.append(r)
            scores = [
                r["first_forecast"]["nmse"]
                for r in rows
                if r["first_forecast"]["status"] == "scored"
            ]
            over = len(rows)
            summaries.append(
                {
                    "truth": law,
                    "seeds": over,
                    "top1": sum(r["ordered_families"][0] == law for r in rows),
                    "top2": sum(law in r["ordered_families"] for r in rows),
                    "chosen": dict(Counter(r["ordered_families"][0] for r in rows)),
                    "forecasts_scored": len(scores),
                    "median_held_nmse": median(scores) if scores else None,
                    "median_persistence_nmse": median(
                        r["persistence_nmse"] for r in rows
                    ),
                }
            )
        return sealed_write(
            root / "summary.json", {"plan": plan["artifact_sha256"], "rows": summaries}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().root), indent=2))
