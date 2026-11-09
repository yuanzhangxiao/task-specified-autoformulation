"""Plot every planned endpoint's meal-response errors from the T1 comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from scripts.replay_t1_curves import CELLS

METHODS = (
    ("full", "Full"),
    ("no_latent", "No latent"),
    ("no_spec", "Prediction only"),
    ("sindy", "SINDy"),
    ("pysr", "PySR"),
    ("d3", "D3"),
    ("sol", "GPT-5.6 Sol"),
)
PANELS = (
    ("meal_vs_fasting_gt100", "60 g meal vs fasting"),
    ("split_meal_gt100", "Two 30 g meals vs one 60 g meal"),
)
MARKERS = ("o", "^", "s")


def plot(directory: Path) -> None:
    """Create a two-cell, two-contrast plot including failures and repetitions."""
    with (directory / "paired_responses.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), sharex=True)
    for i, cell in enumerate(CELLS):
        for j, (contrast, _) in enumerate(PANELS):
            ax = axes[i, j]
            selected = [
                r for r in rows if r["cell"] == cell and r["contrast"] == contrast
            ]
            displayed = [
                (m, label)
                for m, label in METHODS
                if any(r["method"] == m for r in selected)
            ]
            labels = []
            for y, (method, label) in enumerate(displayed):
                reps = sorted(
                    (r for r in selected if r["method"] == method),
                    key=lambda r: int(r["seed"]),
                )
                complete = sum(r["status"] == "complete" for r in reps)
                labels.append(f"{label}   {complete}/{len(reps)}")
                for r, offset in zip(
                    reps, np.linspace(-0.22, 0.22, len(reps)), strict=True
                ):
                    if r["status"] != "complete":
                        continue
                    ax.scatter(
                        float(r["effect_rse"]),
                        y + offset,
                        s=41,
                        marker=MARKERS[int(r["seed"])],
                        color="#00786d" if method == "full" else "#34506c",
                        zorder=3,
                    )
            ax.set_xscale("log")
            ax.axvline(1, color="#a34b2f", ls="--", lw=1.3, zorder=1)
            ax.set_yticks(range(len(labels)), labels if j == 0 else [])
            ax.set_ylim(len(labels) - 0.55, -0.55)
            ax.grid(axis="x", which="major", color="#d6dcdf", lw=0.7)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", which="minor", length=0)
            ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_xlim(0.004, 11)
            if i == 0:
                ax.set_title(PANELS[j][1], loc="left", fontsize=12, pad=15)
            if j == 1:
                ax.text(
                    1.025,
                    0.50,
                    ("Named channels", "Obfuscated channels")[i],
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    fig.suptitle(
        "Glucose response to meal interventions",
        x=0.19,
        y=0.967,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.19,
        0.912,
        "Canonical T1-easy   •   Frozen, auxiliary-conditioned rollouts",
        fontsize=10,
        color="#415467",
    )
    fig.text(
        0.55,
        0.092,
        "Paired response relative squared error (log scale)",
        ha="center",
        fontsize=10,
    )
    keys = [
        Line2D(
            [], [], marker=m, ms=6, color="#34506c", ls="none", label=f"Seed/rep {n}"
        )
        for n, m in enumerate(MARKERS)
    ]
    keys.append(
        Line2D([], [], color="#a34b2f", ls="--", label="Zero-response benchmark")
    )
    fig.legend(
        handles=keys,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.55, 0.024),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.19, right=0.94, top=0.84, bottom=0.15, wspace=0.08, hspace=0.21
    )
    fig.savefig(
        directory / "paired_meal_responses.png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)
