#!/usr/bin/env python3
"""Show absolute Full trajectories and clarify the paired meal-spacing response."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.plot_t1_intervention_probe import finish, save

COLORS = ("#0072B2", "#D55E00")
TITLES = {
    "fasting_gt100": "No meal: unperturbed control",
    "fasting_gt80": "No meal: initial tissue glucose -20%",
    "fasting_gt120": "No meal: initial tissue glucose +20%",
    "meal_gt100": "60 g meal at 60 min: unperturbed control",
    "meal_gt80": "60 g meal: initial tissue glucose -20%",
    "meal_gt120": "60 g meal: initial tissue glucose +20%",
    "split_meal_gt100": "Two 30 g meals: 60 and 90 min",
}


def plot_absolute(ax, root: Path, models: list[tuple[str, str]], case: str) -> None:
    """Plot the physical reference and each saved-model free rollout."""
    for i, (model, label) in enumerate(models):
        record = sealed_read(root / "replays" / model / f"{case}.json")
        if i == 0:
            ax.plot(
                record["time"],
                record["observed"],
                color="#222222",
                lw=2.2,
                label="Reference",
            )
        if record["success"]:
            ax.plot(
                record["time"],
                record["predicted"],
                color=COLORS[i],
                lw=1.7,
                label=label,
            )
        else:
            ax.text(0.05, 0.9 - 0.1 * i, f"{label}: failed", transform=ax.transAxes)
    finish(ax, TITLES[case])


def run(root: Path) -> None:
    """Include every Full endpoint and case; keep named/obfuscated models separate."""
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    plan = sealed_read(root / "plan.json")
    summary = sealed_read(root / "summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for cell, variant in enumerate(("Named", "Obfuscated")):
        models = [
            (f"cell{cell:02d}_seed{seed}_full", f"Full seed {seed}") for seed in (0, 1)
        ]
        for ax, case in zip(
            axes[cell], ("fasting_gt120", "split_meal_gt100"), strict=True
        ):
            plot_absolute(ax, root, models, case)
            ax.set_title(f"{variant} — {TITLES[case]}")
    fig.suptitle(
        "Full: absolute trajectories under new interventions\n"
        "All four available endpoints; saved parameters; no refitting",
        fontsize=14,
    )
    save(fig, folder, "full_overview")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained"
    )
    for cell, variant in enumerate(("Named", "Obfuscated")):
        models = [
            (f"cell{cell:02d}_seed{seed}_full", f"Full seed {seed}") for seed in (0, 1)
        ]
        for ax, case in zip(
            axes[cell], ("fasting_gt100", "fasting_gt120"), strict=True
        ):
            plot_absolute(ax, root, models, case)
            ax.set_title(f"{variant} — {TITLES[case]}")
    fig.suptitle(
        "Full: the intervention and its matched control\n"
        "Absolute trajectories; same initial plasma glucose; same saved parameters",
        fontsize=14,
    )
    save(fig, folder, "full_matched_control")

    for cell, variant in enumerate(("named", "obfuscated")):
        models = [
            (f"cell{cell:02d}_seed{seed}_full", f"Full seed {seed}") for seed in (0, 1)
        ]
        fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout="constrained")
        for ax, case in zip(axes.flat, plan["cases"], strict=False):
            plot_absolute(ax, root, models, case["id"])
        axes.flat[-1].set_visible(False)
        fig.suptitle(
            f"Full ({variant}): all seven diagnostic cases\n"
            "Absolute trajectories; identical inputs and frozen parameters",
            fontsize=14,
        )
        save(fig, folder, f"full_{variant}_all_cases")

    models = [
        ("obfuscated_raw_data_agent_rep1", "Sol: three extra latent states"),
        ("obfuscated_raw_data_agent_rep0", "Sol: no extra latent states"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, case in zip(axes[:2], ("meal_gt100", "split_meal_gt100"), strict=True):
        plot_absolute(ax, root, models, case)
    for i, (model, _label) in enumerate(models):
        row = next(
            r
            for r in summary["rows"]
            if r["model_id"] == model and r["case_id"] == "split_meal_gt100"
        )
        record = sealed_read(root / "replays" / model / "split_meal_gt100.json")
        if i == 0:
            axes[2].plot(
                record["time"],
                row["effect"]["reference_delta"],
                color="#222222",
                lw=2.2,
            )
        if row["effect"]["status"] == "complete":
            axes[2].plot(
                record["time"],
                row["effect"]["predicted_delta"],
                color=COLORS[i],
                lw=1.7,
            )
    finish(
        axes[2],
        "Effect of delaying half the meal\nSplit meal - single meal",
        effect=True,
    )
    fig.suptitle(
        "Meal-spacing probe: same 60 g total, different timing\n"
        "Supplied physiological auxiliaries are regenerated for each schedule",
        fontsize=14,
    )
    save(fig, folder, "meal_spacing_absolute_and_effect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
