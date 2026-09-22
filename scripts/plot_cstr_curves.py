#!/usr/bin/env python3
"""Render complete contact sheets from verified frozen CSTR curve exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {"full": "#1975b8", "no_latent": "#d16b22"}
LABELS = {"full": "Full", "no_latent": "No latent"}


def read_curves(root: Path) -> dict:
    """Read aligned numeric arrays, retaining the exact model/trajectory grouping."""
    groups = {}
    with (root / "trajectories.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), row["split"], row["trajectory_id"], row["arm"])
            groups.setdefault(key, []).append(
                [float(row[k]) for k in ("time", "observed", "predicted", "residual")]
            )
    return {key: np.asarray(values) for key, values in groups.items()}


def legend() -> list:
    return [Line2D([], [], color="#242424", lw=1.8, label="Reference")] + [
        Line2D(
            [],
            [],
            color=COLORS[arm],
            lw=1.8,
            linestyle="-" if arm == "full" else "--",
            label=LABELS[arm],
        )
        for arm in COLORS
    ]


def draw(axis, curves, metrics, seed, split, trajectory, residual=False) -> None:
    """Draw both complete free rollouts on the same reference and axes."""
    full = curves[(seed, split, trajectory, "full")]
    if residual:
        axis.axhline(0, color="#777777", lw=1)
    else:
        axis.plot(full[:, 0], full[:, 1], color="#242424", lw=1.8, zorder=2)
    for arm in COLORS:
        values = curves[(seed, split, trajectory, arm)]
        if not np.array_equal(values[:, :2], full[:, :2]):
            raise ValueError("models have different observation arrays")
        axis.plot(
            values[:, 0],
            values[:, 3 if residual else 2],
            color=COLORS[arm],
            lw=1.5,
            linestyle="-" if arm == "full" else "--",
            zorder=3,
        )
    axis.grid(alpha=0.2)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_xlim(full[0, 0], full[-1, 0])
    axis.set_xlabel("Time (min)")
    axis.set_ylabel("Prediction - reference (K)" if residual else "Temperature (K)")
    values = [metrics[(seed, split, trajectory, arm)] for arm in COLORS]
    axis.set_title(
        f"{trajectory}  |  NMSE: {values[0]:.4g} / {values[1]:.4g}", fontsize=10
    )


def save(figure, root: Path, name: str) -> None:
    figure.savefig(root / f"{name}.png", dpi=170, facecolor="white")
    figure.savefig(root / f"{name}.svg", facecolor="white")
    plt.close(figure)


def plot_reference_forcings(root: Path, figures: Path, curves: dict) -> None:
    """Show original input changes alongside the original reference responses."""
    channels = {}
    with (root / "inputs_auxiliaries.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "validation" and row["role"] == "external_inputs":
                key = (row["trajectory_id"], row["channel"])
                channels.setdefault(key, []).append(
                    [float(row["time"]), float(row["value"])]
                )
    ids = sorted({key[0] for key in channels})
    figure, axes = plt.subplots(2, 4, figsize=(15, 7))
    for column, trajectory in enumerate(ids):
        for name, scale, color in (
            ("Cf", 0.25, "#1975b8"),
            ("Tf", 10, "#d16b22"),
            ("Tjf", 12, "#7a4eab"),
        ):
            values = np.asarray(channels[(trajectory, name)])
            axes[0, column].plot(
                values[:, 0],
                (values[:, 1] - values[0, 1]) / scale,
                label=name,
                color=color,
            )
        reference = curves[(0, "validation", trajectory, "full")]
        axes[1, column].plot(
            reference[:, 0], reference[:, 1] - reference[0, 1], color="#242424"
        )
        axes[0, column].set_title(trajectory)
        axes[0, column].set_ylim(-0.6, 0.6)
        axes[1, column].set_ylim(-8, 8)
        for axis in axes[:, column]:
            axis.set_xlim(0, 30)
            axis.set_xlabel("Time (min)")
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Input change / channel scale")
    axes[1, 0].set_ylabel("Reference T(t) - T(0) (K)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("CSTR · Saved validation inputs and reference responses", y=0.99)
    figure.text(
        0.5,
        0.015,
        "Original stored arrays, no regenerated data. "
        "Input scales: Cf = 0.25; Tf = 10 K; Tjf = 12 K.\n"
        "Step/pulse reference responses are nearly flat despite changing inputs; "
        "the sinusoidal case responds.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.90))
    save(figure, figures, "saved_reference_forcing_audit")


def run(root: Path) -> None:
    """Plot all saved cases, with exploratory status and error definitions visible."""
    report = json.loads((root / "summary.json").read_text())
    if not report["all_scores_match"]:
        raise ValueError("refuse to plot unverified replay as reproduced results")
    curves = read_curves(root)
    with (root / "trajectory_metrics.csv").open() as stream:
        metrics = {
            (int(r["seed"]), r["split"], r["trajectory_id"], r["arm"]): float(r["nmse"])
            for r in csv.DictReader(stream)
        }
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    plot_reference_forcings(root, figures, curves)
    footer = (
        "INSPECTION ONLY: saved step/pulse references need a forcing audit.\n"
        "Frozen round-12 models; no refitting or target resets. "
        "Conditional on supplied C(t), Tj(t).\n"
        "NMSE labels: Full / No latent, using one pooled training scale. "
        "Validation was used for model selection."
    )
    for seed in (0, 1):
        for split, shape, size in (
            ("train", (4, 4), (15, 12)),
            ("validation", (2, 2), (12, 8)),
        ):
            ids = sorted(
                {key[2] for key in curves if key[0] == seed and key[1] == split}
            )
            for residual in (False, True):
                figure, axes = plt.subplots(*shape, figsize=size)
                for axis, trajectory in zip(axes.flat, ids, strict=True):
                    draw(axis, curves, metrics, seed, split, trajectory, residual)
                noun = "Residuals" if residual else "Free-rollout trajectories"
                figure.suptitle(
                    f"CSTR · Seed {seed} · {split.capitalize()} · {noun}",
                    y=0.985,
                    fontsize=17,
                )
                figure.legend(
                    handles=legend()[1:] if residual else legend(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=3,
                    frameon=False,
                )
                figure.text(
                    0.5,
                    0.012,
                    footer,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#444444",
                )
                figure.tight_layout(rect=(0, 0.08, 1, 0.91))
                save(
                    figure,
                    figures,
                    f"seed{seed}_{split}_{'residuals' if residual else 'curves'}",
                )
        # This overview includes every validation case and two clearly labeled
        # training examples; the complete training contact sheet is also saved.
        train_ids = sorted(
            {key[2] for key in curves if key[0] == seed and key[1] == "train"}
        )
        ranges = {
            t: np.ptp(curves[(seed, "train", t, "full")][:, 1]) for t in train_ids
        }
        nonconstant = sorted(
            (t for t in train_ids if ranges[t] > 1e-3), key=lambda t: (ranges[t], t)
        )
        # Median reference excursion, independent of model errors.
        chosen = nonconstant[
            (len(nonconstant) - 1) // 2 : (len(nonconstant) - 1) // 2 + 2
        ]
        val_ids = sorted(
            {key[2] for key in curves if key[0] == seed and key[1] == "validation"}
        )
        figure, axes = plt.subplots(3, 2, figsize=(12, 11))
        for axis, (split, trajectory) in zip(
            axes.flat,
            [("train", t) for t in chosen] + [("validation", t) for t in val_ids],
            strict=True,
        ):
            draw(axis, curves, metrics, seed, split, trajectory)
        figure.suptitle(
            f"CSTR · Seed {seed} · Training examples and all validation cases",
            y=0.985,
            fontsize=16,
        )
        figure.legend(
            handles=legend(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.956),
            ncol=3,
            frameon=False,
        )
        figure.text(
            0.5,
            0.012,
            "Top: two middle-ranked nonconstant training reference excursions; "
            "below: all validation cases.\n" + footer,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#444444",
        )
        figure.tight_layout(rect=(0, 0.085, 1, 0.92))
        save(figure, figures, f"seed{seed}_overview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
