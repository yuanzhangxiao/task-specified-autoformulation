"""Test a total-glucose balance against archived fading fits on two T2 folds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import run_t2_initial_memory_holdouts as previous
from scripts.construction_functions import file_hash

ROOT = Path("output/t2-total-glucose-holdouts-20261124")
PARENT_PLAN = "dfa24b49570324a431453d1fb80da4c0d390309aae6eb063dbd04350f168c79a"
PARENTS = {
    "train_011": "1bf4052e4772edd3757cbe0af0f448aa64110b518acf4753a3881d69dc2eabc9",
    "train_012": "c371666fc0e177ce621d4b33bf6192ba384ba783f655c85ef108adb7120776bd",
}
FOLD_RECEIPTS = {
    "train_011": "9652e15d0f17fd50b653a1597d38929cb714f02c6faec665009b6fe7e4b5f7db",
    "train_012": "d44b2cbae129eadf67f138292b7bbd2a9060b8e707ee619da413eb39f78e6905",
}
STARTS = ("short", "long", "warm")
ARMS = ("fading", "total")


def fold_inputs(held: str):
    """Authenticate each parent, its shared insulin law and its 15 training rows."""
    rows = previous.folds.load_fit_rows(previous.TRAIN, held)
    folder = previous.OUTPUT / held
    inputs = sealed_read(folder / "fold.json")
    selected = sealed_read(folder / "selected.json")
    ident = {"held": held, "training_rows": content_hash(rows), "plan": PARENT_PLAN}
    setup = inputs["result"]
    ids = list(previous.folds.fit_ids(held))
    if (
        inputs["artifact_sha256"] != FOLD_RECEIPTS[held]
        or selected["artifact_sha256"] != PARENTS[held]
        or selected["identity"] != ident
        or inputs["identity"] != ident
        or setup["fit_ids"] != ids
        or setup["training_rows_sha256"] != content_hash(rows)
    ):
        raise ValueError("Different fold, insulin fit or fitted fading control")
    control = selected["models"]["fading"]
    m = control["model"]
    if (
        m["fit_training_ids"] != ids
        or m["scales"] != setup["scales"]
        or m["scalar_model"] != setup["insulin"]["scalar_model"]
    ):
        raise ValueError("Parent and insulin fitting fold disagree")
    return rows, setup, control


def fit_fold(job: tuple) -> dict:
    """Fit and select a total-balance equation without opening omitted outcomes."""
    from scripts import t2_total_glucose_fit as kernel
    from scripts import t2_total_glucose_replay as replay

    root, held, plan = job
    rows, setup, control = fold_inputs(held)
    folder = root / held
    runs = {}
    for start in STARTS:
        ident = {"plan": plan, "held": held, "start": start, "parent": PARENTS[held]}
        attempt = previous._checkpoint(
            folder / f"{start}-fit.json",
            ident,
            partial(
                kernel.fit,
                rows,
                setup,
                start,
                budget=previous.BUDGET,
                warm_model=control["model"] if start == "warm" else None,
            ),
        )
        model = attempt["model"]
        if model is None:
            print(held, start, attempt["termination"], flush=True)
            continue
        check = previous._checkpoint(
            folder / f"{start}-replay.json",
            ident | {"model": content_hash(model)},
            partial(replay.training_replay, model, rows),
        )
        loss = check["summary"]["pooled_nmse"]
        if loss is None or not check["eligible"]:
            print(held, start, "unqualified", flush=True)
            continue
        if not np.isclose(loss, attempt["training_nmse"], atol=1e-6, rtol=1e-5):
            raise ValueError("Optimizer and independent full-graph rollout disagree")
        runs[start] = {
            "model": model,
            "train_summary": check["summary"],
            "ordering": check["ordering"],
            "domain": check["domain"],
            "start": start,
            "checkpoints": {
                "fit": f"{start}-fit.json",
                "replay": f"{start}-replay.json",
            },
        }
        print(held, start, loss, flush=True)
    if not runs:
        raise ValueError(
            f"No total-balance endpoint qualified on {held}'s fitting fold"
        )
    best = min(runs.values(), key=lambda r: r["train_summary"]["pooled_nmse"])
    return sealed_write(
        folder / "selected.json",
        {"plan": plan, "held": held, "models": {"fading": control, "total": best}},
    )


def _frozen_models(selections):
    if set(selections) != set(PARENTS) or any(
        set(s["models"]) != set(ARMS) or s["held"] != h for h, s in selections.items()
    ):
        raise ValueError("Both folds must have both models frozen before scoring")
    return {
        h: {k: v["model"] for k, v in s["models"].items()}
        for h, s in selections.items()
    }


def held_scores(root: Path, selections: dict) -> dict:
    """Generate and seal all forecasts before computing any held-event errors."""
    from scripts import t2_total_glucose_replay as replay

    models = _frozen_models(selections)
    ident = {h: s["artifact_sha256"] for h, s in selections.items()}
    rows = {
        r["trajectory_id"]: r for r in previous.intake.load_training(previous.TRAIN)
    }

    def predictions():
        result = {}
        for h, fitted in models.items():
            public = previous.forecast.public_event(rows[h])
            result[h] = {
                "fading": previous.forecast.predict(fitted["fading"], public),
                "total": replay.predict(fitted["total"], public),
            }
        return result

    forecast = previous._checkpoint(root / "forecasts.json", ident, predictions)
    original = sealed_read(previous.OUTPUT / "forecasts.json")
    if original["identity"] != PARENTS:
        raise ValueError("Different archived comparison")
    scores = {}
    for h in models:
        if forecast[h]["fading"] != original["result"][h]["fading"]:
            raise ValueError("Control forecast has changed")
        scores[h] = {
            k: previous.forecast.score(models[h][k], forecast[h][k], rows[h])
            for k in ARMS
        }
    return sealed_write(
        root / "scores.json",
        {"selections": ident, "forecasts": content_hash(forecast), "scores": scores},
    )


def _plot_held(scores: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        2, 2, figsize=(9, 5.6), sharex=True, sharey="row", layout="constrained"
    )
    for col, h in enumerate(PARENTS):
        arms = scores[h]
        axes[0, col].set_title(f"Omitted {h}", loc="left", weight="bold")
        for row, (k, unit) in enumerate(
            (("Gp", "Plasma glucose (mg/kg)"), ("U", "Disposal (mg/kg/min)"))
        ):
            ax = axes[row, col]
            first = arms["fading"]
            ax.plot(
                first["time"],
                first["observed"][k],
                color="#242933",
                lw=2.4,
                label="Observed",
            )
            for a, label, color, dash in (
                ("fading", "Fading plasma balance", "#c46a35", "--"),
                ("total", "Fading total balance", "#166e9a", "-."),
            ):
                ax.plot(
                    arms[a]["time"],
                    arms[a]["predicted"][k],
                    color=color,
                    ls=dash,
                    lw=1.7,
                    label=label,
                )
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(axis="y", alpha=0.16)
            if col == 0:
                ax.set_ylabel(unit)
        axes[1, col].set_xlabel("Time (min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(root: Path, workers: int = 2):
    from scripts import t2_total_glucose_fit as kernel
    from scripts import t2_total_glucose_meals as meals
    from scripts import t2_total_glucose_replay as replay

    with _lock(root):
        baseline = sealed_read(previous.OUTPUT / "plan.json")
        if baseline["artifact_sha256"] != PARENT_PLAN:
            raise ValueError("Unexpected fading baseline protocol")
        files = [Path(__file__), Path(kernel.__file__), Path(replay.__file__)]
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "total-glucose-fifteen-event-holdouts-1",
                "code": {p.name: file_hash(p) for p in files},
                "packages": {p: version(p) for p in ("numpy", "scipy", "casadi")},
                "parent_plan": PARENT_PLAN,
                "parent_selections": PARENTS,
                "insulin_training_receipts": FOLD_RECEIPTS,
                "fit_ids": baseline["fit_ids"],
                "arms": list(ARMS),
                "total_balance": "S'=EGP+gain*D_meal-U-E; Gp=S-Gt",
                "initial_total": "Gp(0)+Gt(0)",
                "initial_disposal_gap_fraction": 0.5,
                "starts": list(STARTS),
                "budget_per_start": previous.BUDGET,
                "selection": "lowest feasible 15-event replay NMSE",
                "forecast_initial_readings": ["Gp", "I"],
                "meal_scope": "all fitting events with positive ingestion",
                "official_validation_opened": False,
                "test_opened": False,
            },
        )
        jobs = [(root, h, plan["artifact_sha256"]) for h in PARENTS]
        if workers == 1:
            fits = list(map(fit_fold, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                fits = list(pool.map(fit_fold, jobs))
        selections = dict(zip(PARENTS, fits, strict=True))
        scores = held_scores(root, selections)
        _plot_held(scores["scores"], root / "held-curves.png")
        models = _frozen_models(selections)
        training = {h: previous.folds.load_fit_rows(previous.TRAIN, h) for h in PARENTS}

        def curves():
            return {
                h: {
                    k: replay.training_curves(m, training[h]) for k, m in fitted.items()
                }
                for h, fitted in models.items()
            }

        all_curves = previous._checkpoint(
            root / "training-curves.json", scores["selections"], curves
        )
        meals.review_meals(all_curves, training, root)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.root, args.workers)
    print(args.root / "scores.json")
