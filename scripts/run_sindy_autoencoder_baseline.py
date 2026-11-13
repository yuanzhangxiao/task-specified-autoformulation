"""Train, select and seal the delay SINDy autoencoder baseline on public cells.

    plan    freeze the settings, cells, seeds and public data identities
    train   fit one (cell, seed, latent dimension, window) task; a sealed task is kept
    select  replay every task's checkpoint on validation; keep the best per seed

Development data only: public training trajectories fit the model, public
validation trajectories choose the refinement checkpoint, the latent dimension
and the window length. No test or private file is opened.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from autoformalism.baselines import latent_ode as lo
from autoformalism.baselines import sindy_autoencoder as sa
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.phase_c_baselines import BASELINE_CELLS, load_cell
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PLAN_SCHEMA = "sindy-autoencoder-plan-1"


def plan(root: Path, release: Path, config_path: Path) -> dict:
    """Seal the frozen plan with the identities of the public data it reads."""
    frozen = json.loads(config_path.read_text())
    if frozen.get("status") != "frozen_before_training":
        raise ValueError("only a frozen plan can be run")
    config = sa.SindyAutoencoderConfig.model_validate(frozen["settings"])
    cells = {}
    for name in frozen["cells"]:
        if name not in BASELINE_CELLS:
            raise ValueError(f"{name} is not a Phase C baseline cell")
        cell = load_cell(release, name)
        if cell.identity["release_summary_sha256"] != frozen["release_summary_sha256"]:
            raise ValueError("the release differs from the plan's")
        targets, inputs = lo.channels(cell.context)
        cells[name] = {
            "identity": cell.identity,
            "targets": list(targets),
            "inputs": list(inputs),
            "context": cell.context.model_dump(mode="json"),
        }
    tasks = [
        {"index": i, "cell": cell, "seed": seed, "latent_dimension": d, "delays": n}
        for i, (cell, seed, d, n) in enumerate(
            (c, s, d, n)
            for c in frozen["cells"]
            for s in frozen["seeds"]
            for d in config.latent_dimensions
            for n in config.delays
        )
    ]
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "protocol": sa.PROTOCOL,
            "source_config": {
                "path": str(config_path),
                "sha256": public.content_sha256(frozen),
            },
            "settings": config.model_dump(mode="json"),
            "release": str(release.resolve()),
            "cells": cells,
            "seeds": frozen["seeds"],
            "tasks": tasks,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def _splits(plan_record: dict, name: str):
    cell = load_cell(Path(plan_record["release"]), name)
    if cell.identity != plan_record["cells"][name]["identity"]:
        raise ValueError(f"public data of {name} changed since the plan")
    targets, inputs = lo.channels(cell.context)
    return (
        lo.arrays(cell.dataset.train, targets, inputs),
        lo.arrays(cell.dataset.validation, targets, inputs),
    )


def train(root: Path, index: int) -> dict:
    """Fit one task; a task with a sealed result is never fitted again."""
    record = sealed_read(root / "plan.json")
    task = record["tasks"][index]
    path = root / "tasks" / f"{index:03d}.json"
    if path.exists():
        return sealed_read(path)
    config = sa.SindyAutoencoderConfig.model_validate(record["settings"])
    train_data, validation = _splits(record, task["cell"])
    result = sa.train(
        train_data,
        validation,
        task["latent_dimension"],
        task["delays"],
        task["seed"],
        config,
    )
    return sealed_write(
        path, {"plan_sha256": record["artifact_sha256"], "task": task, **result}
    )


def replay_nmse(model: dict, data: lo.Arrays, settings: dict) -> float:
    """Free-rollout error of a saved model, as it is selected and scored."""
    replay = sa.SindyAutoencoder(model)
    predicted = np.stack(
        [
            replay.simulate(
                data.time,
                data.targets[b, 0],
                data.inputs[b],
                method="Radau",
                rtol=settings["replay_rtol"],
                atol=settings["replay_atol"],
                seconds=settings["replay_seconds"],
            )[0]
            for b in range(len(data.targets))
        ]
    )
    return lo.nmse(predicted, data.targets, model["standardization"]["target_sd"])


def select(root: Path) -> dict:
    """Per cell and seed: the task whose checkpoint has the lowest validation error."""
    record = sealed_read(root / "plan.json")
    settings = record["settings"]
    rows = []
    for name, cell in record["cells"].items():
        _, validation = _splits(record, name)
        for seed in record["seeds"]:
            candidates = []
            for task in record["tasks"]:
                if (task["cell"], task["seed"]) != (name, seed):
                    continue
                path = root / "tasks" / f"{task['index']:03d}.json"
                if not path.exists():
                    raise ValueError(f"task {task['index']} has no result yet")
                result = sealed_read(path)
                best = result["best"]
                entry = {
                    "task": task["index"],
                    "latent_dimension": task["latent_dimension"],
                    "delays": task["delays"],
                    "kept_terms": result["kept_terms"],
                    "epoch": best["epoch"],
                    "validation_nmse_rk4": best["validation_nmse_rk4"],
                    "validation_nmse": math.inf,
                    "replay_error": None,
                }
                if best["snapshot"] is not None:
                    model = {
                        "targets": cell["targets"],
                        "inputs": cell["inputs"],
                        "delays": task["delays"],
                        "sample_interval": result["sample_interval"],
                        "standardization": result["standardization"],
                        **best["snapshot"],
                    }
                    try:
                        entry["validation_nmse"] = replay_nmse(
                            model, validation, settings
                        )
                    except (ValueError, ArithmeticError, TimeoutError) as exc:
                        entry["replay_error"] = str(exc)[:300]
                    entry["model"] = model
                candidates.append(entry)
            finite = [c for c in candidates if math.isfinite(c["validation_nmse"])]
            chosen = min(finite, key=lambda c: c["validation_nmse"]) if finite else None
            rows.append(
                {
                    "benchmark_id": name,
                    "repetition": seed,
                    "status": "complete" if chosen else "rollout_failed",
                    "selected_latent_dimension": chosen and chosen["latent_dimension"],
                    "selected_delays": chosen and chosen["delays"],
                    "kept_terms": chosen and chosen["kept_terms"],
                    "validation_nmse": chosen["validation_nmse"] if chosen else None,
                    "equations": chosen
                    and sa.SindyAutoencoder(chosen["model"]).equations(),
                    "model": chosen
                    and {
                        **chosen["model"],
                        "semantics": "sindy_autoencoder",
                        "context": cell["context"],
                    },
                    "candidates": [
                        {k: v for k, v in c.items() if k != "model"} for c in candidates
                    ],
                }
            )
    return sealed_write(
        root / "selection.json",
        {
            "plan_sha256": record["artifact_sha256"],
            "protocol": sa.PROTOCOL,
            "selection_rule": (
                "per cell and seed, among the tasks of every latent dimension and "
                "window length, the one whose refinement checkpoint has the lowest "
                "validation free-rollout NMSE in the adaptive replay"
            ),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "train", "select"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cstr_hard_sindy_autoencoder_v1.json"),
    )
    parser.add_argument("--index", type=int)
    a = parser.parse_args()
    if a.stage == "plan":
        out = plan(a.root, a.release, a.config)
        print(out["artifact_sha256"], len(out["tasks"]), "tasks")
    elif a.stage == "train":
        out = train(a.root, a.index)
        print(
            json.dumps(
                {k: out[k] for k in ("task", "seconds", "kept_terms", "diverged")}
                | {k: out["best"][k] for k in ("epoch", "validation_nmse_rk4")}
            )
        )
    else:
        out = select(a.root)
        print(
            json.dumps(
                [
                    {
                        k: r[k]
                        for k in (
                            "benchmark_id",
                            "repetition",
                            "status",
                            "selected_latent_dimension",
                            "selected_delays",
                            "kept_terms",
                            "validation_nmse",
                        )
                    }
                    for r in out["rows"]
                ],
                indent=1,
            )
        )
