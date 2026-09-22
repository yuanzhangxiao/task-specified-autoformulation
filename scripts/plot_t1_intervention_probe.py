#!/usr/bin/env python3
"""Plot all outcomes of the frozen exploratory T1 intervention suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read

MODELS = (
    ("obfuscated_raw_data_agent_rep0", "Sol: no latent states", "#D55E00"),
    ("obfuscated_raw_data_agent_rep1", "Sol: meal-memory states", "#0072B2"),
    ("cell01_seed0_no_spec", "No specification: ignores auxiliaries", "#009E73"),
)


def finish(ax, title, *, effect=False, xmax=300):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(
        (r"Change in $G_p$" if effect else r"Plasma glucose mass $G_p$")
        + r" (mg kg$^{-1}$)"
    )
    ax.set_xlim(0, xmax)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)


def plot_records(ax, records):
    first = records[0]
    ax.plot(
        first["time"], first["observed"], color="#222222", lw=2.3, label="Reference"
    )
    for (_, label, color), record in zip(MODELS, records, strict=True):
        if record["success"]:
            ax.plot(
                record["time"], record["predicted"], color=color, lw=1.7, label=label
            )


def save(fig, folder, name):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(folder / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def run(source: Path, root: Path):
    plan = sealed_read(root / "plan.json")
    report = sealed_read(root / "summary.json")
    if {job["model_id"] for job in plan["models"]} != {row[0] for row in MODELS}:
        raise ValueError("unexpected frozen model inventory")
    folder = root / "figures"
    folder.mkdir(exist_ok=True)

    def training(index):
        return [
            sealed_read(source / "replays" / model / "train" / f"{index:03d}.json")
            for model, _, _ in MODELS
        ]

    def diagnostic(case_id):
        return [
            sealed_read(root / "replays" / model / f"{case_id}.json")
            for model, _, _ in MODELS
        ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, index, title in zip(
        axes[0],
        (2, 10),
        (
            "Training: 60 g meal at 0 min",
            "Training: 30 g at 0 min + 60 g at 120 min",
        ),
        strict=True,
    ):
        plot_records(ax, training(index))
        finish(ax, title)
    for ax, case, label in zip(
        axes[1], ("fasting_gt80", "fasting_gt120"), ("-20%", "+20%"), strict=True
    ):
        rows = [
            next(
                r
                for r in report["rows"]
                if r["model_id"] == model and r["case_id"] == case
            )
            for model, _, _ in MODELS
        ]
        time = diagnostic(case)[0]["time"]
        ax.plot(time, rows[0]["effect"]["reference_delta"], lw=2.3, color="#222222")
        for (_, _, color), row in zip(MODELS, rows, strict=True):
            ax.plot(time, row["effect"]["predicted_delta"], lw=1.7, color=color)
        finish(
            ax,
            f"New initial condition: tissue glucose {label}\n"
            "Response relative to the matched unperturbed control",
            effect=True,
            xmax=120,
        )
    fig.suptitle(
        "Similar training fit does not determine the intervention response\n"
        "Saved models; all parameters frozen; no refitting",
        fontsize=15,
    )
    save(fig, folder, "training_and_intervention_effects")

    fig, axes = plt.subplots(4, 4, figsize=(17, 13), layout="constrained")
    for i, ax in enumerate(axes.flat):
        records = training(i)
        plot_records(ax, records)
        scores = " / ".join(f"{r['nmse']:.3g}" for r in records)
        finish(ax, f"train_{i:03d}\nNMSE: {scores}")
    fig.suptitle(
        "All 16 training trajectories\n"
        "NMSE order: Sol no latent / Sol meal-memory / No specification",
        fontsize=15,
    )
    save(fig, folder, "all_training")

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), layout="constrained")
    for ax, case in zip(axes.flat, plan["cases"], strict=False):
        plot_records(ax, diagnostic(case["id"]))
        meals = "; ".join(
            f"{grams:g} g at {time:g} min" for time, grams in case["meals"]
        )
        finish(ax, f"{case['id']}: {meals or 'no meal'}")
    axes.flat[-1].set_visible(False)
    fig.suptitle(
        "Every declared diagnostic case: absolute glucose trajectories\n"
        "Independent tissue-glucose preparation; coherent regenerated auxiliaries",
        fontsize=14,
    )
    save(fig, folder, "all_diagnostic_trajectories")

    cases = [case for case in plan["cases"] if case["paired_control"]]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for ax, case in zip(axes.flat, cases, strict=False):
        rows = [
            next(
                r
                for r in report["rows"]
                if r["model_id"] == model and r["case_id"] == case["id"]
            )
            for model, _, _ in MODELS
        ]
        time = diagnostic(case["id"])[0]["time"]
        ax.plot(
            time,
            rows[0]["effect"]["reference_delta"],
            color="#222222",
            label="Reference",
            lw=2.3,
        )
        scores = []
        for (_, label, color), row in zip(MODELS, rows, strict=True):
            ax.plot(
                time, row["effect"]["predicted_delta"], color=color, label=label, lw=1.7
            )
            scores.append(f"{row['effect']['relative_squared_error']:.3g}")
        finish(
            ax,
            f"{case['id']} minus {case['paired_control']}\n"
            f"Relative effect error: {' / '.join(scores)}",
            effect=True,
        )
    axes.flat[-1].set_visible(False)
    fig.suptitle(
        "All paired intervention effects, including meal-spacing control\n"
        "Effect error uses reference response energy, not training variance",
        fontsize=14,
    )
    save(fig, folder, "all_paired_effects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.root)
