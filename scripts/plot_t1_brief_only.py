#!/usr/bin/env python3
"""Plot the brief-only endpoint beside both previously inspected Sol fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.plot_t1_intervention_probe import finish, save
from scripts.probe_t1_full_interventions import copy_sealed
from scripts.probe_t1_interventions import effect_error
from scripts.replay_t1_curves import write_csv

PRIMARY = "cell01_seed1_brief_only"
MODELS = (
    (PRIMARY, "Brief-only (our method), seed 1", "#009E73"),
    ("obfuscated_raw_data_agent_rep0", "Sol: no extra latent states", "#D55E00"),
    ("obfuscated_raw_data_agent_rep1", "Sol: three extra latent states", "#0072B2"),
)


def prepare_comparators(root: Path, source: Path, probe: Path) -> None:
    """Bundle unchanged comparator checkpoints with explicit source identities."""
    plan = sealed_read(root / "plan.json")
    old_plan = sealed_read(probe / "plan.json")
    models = sealed_read(source / "models.json")
    if old_plan["artifact_sha256"] != plan["probe_plan_sha256"]:
        raise ValueError("comparator diagnostic suite differs")
    if old_plan["source_models_sha256"] != models["artifact_sha256"]:
        raise ValueError("comparator source model inventory differs")
    hashes = {}
    ids = {m[0] for m in MODELS[1:]}
    for model in sorted(ids):
        for split, count in (("train", 16), ("validation", 4)):
            for index in range(count):
                relative = f"replays/{model}/{split}/{index:03d}.json"
                hashes[relative] = copy_sealed(
                    source / relative, root / "comparators" / relative
                )
        for case in plan["cases"]:
            source_path = probe / "replays" / model / f"{case['id']}.json"
            relative = f"replays/{model}/diagnostic/{case['id']}.json"
            hashes[relative] = copy_sealed(source_path, root / "comparators" / relative)
    sealed_write(
        root / "comparators/manifest.json",
        {
            "source_models_sha256": models["artifact_sha256"],
            "probe_plan_sha256": old_plan["artifact_sha256"],
            "artifacts": hashes,
            "models": [m for m in models["models"] if m["model_id"] in ids],
            "parameter_fitting_performed": False,
        },
    )


def records(root: Path, split: str, key: str) -> list[dict]:
    """Check physical references and times before overlaying any predictions."""
    rows = [
        sealed_read(
            (root if model == PRIMARY else root / "comparators")
            / "replays"
            / model
            / split
            / f"{key}.json"
        )
        for model, _, _ in MODELS
    ]
    if any(
        r["time"] != rows[0]["time"] or r["observed"] != rows[0]["observed"]
        for r in rows[1:]
    ):
        raise ValueError("comparison trajectories differ")
    return rows


def draw(ax, rows: list[dict], title: str) -> None:
    """Draw absolute values; keep failed rollouts explicit."""
    ax.plot(
        rows[0]["time"], rows[0]["observed"], color="#222222", lw=2.3, label="Reference"
    )
    for (_, label, color), row in zip(MODELS, rows, strict=True):
        if row["success"]:
            ax.plot(row["time"], row["predicted"], label=label, color=color, lw=1.7)
        else:
            ax.text(0.02, 0.8, f"{label}: rollout failed", transform=ax.transAxes)
    finish(ax, title)


def run(root: Path, source: Path, probe: Path) -> None:
    """Show fixed illustrative cases and every training/validation/probe trajectory."""
    prepare_comparators(root, source, probe)
    plan = sealed_read(root / "plan.json")
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    panels = (
        ("train", "002", "Training: 60 g meal at 0 min"),
        ("train", "010", "Training: 30 g at 0 min + 60 g at 120 min"),
        ("diagnostic", "fasting_gt120", "No meal: initial tissue glucose +20%"),
        ("diagnostic", "split_meal_gt100", "Two 30 g meals at 60 and 90 min"),
    )
    for ax, (split, key, title) in zip(axes.flat, panels, strict=True):
        draw(ax, records(root, split, key), title)
    fig.suptitle(
        "Brief-only: strong training fit, intervention-dependent generalization\n"
        "Absolute trajectories; all parameters frozen",
        fontsize=14,
    )
    save(fig, folder, "brief_only_overview")

    exported = []
    for split in ("train", "validation", "diagnostic"):
        keys = (
            [case["id"] for case in plan["cases"]]
            if split == "diagnostic"
            else [f"{i:03d}" for i in range(16 if split == "train" else 4)]
        )
        cols = 4 if split == "train" else 2
        fig, axes = plt.subplots(
            (len(keys) + cols - 1) // cols,
            cols,
            figsize=(16 if cols == 4 else 12, 12 if len(keys) > 4 else 7),
            squeeze=False,
            layout="constrained",
        )
        for ax, key in zip(axes.flat, keys, strict=False):
            rows = records(root, split, key)
            scores = " / ".join(
                f"{r['nmse']:.3g}" if r["success"] else "failed" for r in rows
            )
            title = f"{split}_{key}\nNMSE: {scores}"
            draw(ax, rows, title)
            ax.set_title(title, fontsize=9 if split == "train" else 10)
            for (model, _, _), row in zip(MODELS, rows, strict=True):
                for i, time in enumerate(row["time"]):
                    exported.append(
                        {
                            "model_id": model,
                            "split": split,
                            "trajectory_id": row["trajectory_id"],
                            "time_min": time,
                            "reference_Gp": row["observed"][i],
                            "predicted_Gp": row["predicted"][i]
                            if row["success"]
                            else None,
                        }
                    )
        for ax in list(axes.flat)[len(keys) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"Brief-only and Sol: all {split} trajectories\n"
            "NMSE order: Brief-only / Sol without extra states / Sol with three states",
            fontsize=14,
        )
        save(fig, folder, f"all_{split}")
    write_csv(root / "comparison_curves.csv", exported)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, case in zip(
        axes, ("fasting_gt80", "fasting_gt120", "split_meal_gt100"), strict=True
    ):
        control = next(c["paired_control"] for c in plan["cases"] if c["id"] == case)
        changed = records(root, "diagnostic", case)
        base = records(root, "diagnostic", control)
        for i, ((_, label, color), r, c) in enumerate(
            zip(MODELS, changed, base, strict=True)
        ):
            effect = effect_error(r, c)
            if i == 0:
                ax.plot(
                    r["time"],
                    [a - b for a, b in zip(r["observed"], c["observed"], strict=True)],
                    color="#222222",
                    lw=2.3,
                    label="Reference",
                )
            if effect["status"] == "complete":
                ax.plot(
                    r["time"],
                    effect["predicted_delta"],
                    color=color,
                    label=label,
                    lw=1.7,
                )
        finish(ax, f"{case}\nDifference from its matched control", effect=True)
    save(fig, folder, "paired_effects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.probe)
