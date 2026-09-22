#!/usr/bin/env python3
"""Plot all saved T1 replay trajectories without selecting models or refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
METHODS = {"sindy": "SINDy", "pysr": "PySR", "raw_data_agent": "GPT-5.6 Sol"}


def records(root: Path, model_id: str, split: str) -> list[dict]:
    """Load checked trajectory records in their original order."""
    return [
        sealed_read(p)
        for p in sorted(
            (root / "replays" / model_id / split).glob("[0-9][0-9][0-9].json")
        )
    ]


def decorate(ax, record: dict, title: str) -> None:
    """Use original physical target units for named and verified aliases alike."""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"Plasma glucose mass (mg kg$^{-1}$)")
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(record["time"][0], record["time"][-1])


def draw(ax, record: dict, label: str, color: str, **kwargs) -> None:
    if record["success"]:
        ax.plot(
            record["time"],
            record["predicted"],
            label=label,
            color=color,
            linewidth=1.6,
            **kwargs,
        )
    else:
        ax.text(
            0.02,
            0.9,
            f"{label}: rollout failed",
            transform=ax.transAxes,
            color=color,
            fontsize=8,
        )


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def comparison(
    root: Path,
    folder: Path,
    ids: list[str],
    labels: list[str],
    name: str,
    split: str,
    title: str,
) -> None:
    groups = [records(root, name, split) for name in ids]
    count = len(groups[0])
    ncols = 4 if count > 4 else 2
    fig, axes = plt.subplots(
        (count + ncols - 1) // ncols,
        ncols,
        figsize=(15 if count > 4 else 11, 11 if count > 4 else 7),
        squeeze=False,
        layout="constrained",
    )
    for i, ax in enumerate(axes.flat):
        if i >= count:
            ax.set_visible(False)
            continue
        ref = groups[0][i]
        ax.plot(
            ref["time"], ref["observed"], color="#222222", label="Reference", lw=2.2
        )
        score_labels = []
        for group, label, color in zip(groups, labels, COLORS, strict=False):
            record = group[i]
            if record["observed"] != ref["observed"] or record["time"] != ref["time"]:
                raise ValueError("comparison trajectories differ")
            draw(ax, record, label, color)
            value = record["nmse"]
            score_labels.append(
                f"{label} {value:.3g}" if value is not None else f"{label} failed"
            )
        decorate(ax, ref, ref["trajectory_id"] + "\nNMSE: " + "; ".join(score_labels))
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"{title} — {split}\nFrozen parameters; free rollout; all trajectories shown",
        fontsize=14,
    )
    save(fig, folder / f"{name}_{split}")


def external_gallery(root: Path, folder: Path, variant: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), layout="constrained")
    for j, (method, label) in enumerate(METHODS.items()):
        groups = [
            records(root, f"{variant}_{method}_rep{s}", "validation") for s in range(3)
        ]
        for i, ax in enumerate(axes[j]):
            ref = groups[0][i]
            ax.plot(
                ref["time"], ref["observed"], color="#222222", label="Reference", lw=2.2
            )
            scores = []
            for seed, group in enumerate(groups):
                draw(
                    ax,
                    group[i],
                    f"Repetition {seed}",
                    COLORS[seed],
                    linestyle=("-", "--", ":")[seed],
                )
                nmse = group[i]["nmse"]
                scores.append(f"{nmse:.3g}" if nmse is not None else "failed")
            decorate(
                ax,
                ref,
                f"{label} · {ref['trajectory_id']}\nNMSE: " + " / ".join(scores),
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"T1 canonical {variant}: external baselines — validation\n"
        "All three repetitions; saved parameters; supplied auxiliaries available",
        fontsize=15,
    )
    save(fig, folder / f"{variant}_external_validation")


def run(root: Path) -> None:
    """Export every ablation pair and all external repetitions, including failures."""
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    for seed in (0, 1):
        if seed == 0:
            for split in ("train", "validation"):
                comparison(
                    root,
                    folder,
                    ["cell00_seed0_full", "cell00_seed0_no_latent"],
                    ["Full", "No latent"],
                    "named_seed0",
                    split,
                    "T1 canonical named · seed 0",
                )
        for split in ("train", "validation"):
            comparison(
                root,
                folder,
                [f"cell01_seed{seed}_full", f"cell01_seed{seed}_no_spec"],
                ["Full", "No specification"],
                f"obfuscated_seed{seed}",
                split,
                f"T1 canonical obfuscated · seed {seed}",
            )
    for variant in ("named", "obfuscated"):
        external_gallery(root, folder, variant)
        for method, label in METHODS.items():
            comparison(
                root,
                folder,
                [f"{variant}_{method}_rep{s}" for s in range(3)],
                [f"Rep {s}" for s in range(3)],
                f"{variant}_{method}",
                "train",
                f"T1 canonical {variant} · {label}",
            )
    comparison(
        root,
        folder,
        ["cell00_seed1_full"],
        ["Full"],
        "named_seed1",
        "validation",
        "T1 canonical named · seed 1; no No-latent endpoint",
    )
    print(
        json.dumps(
            {"figures": str(folder), "png_count": len(list(folder.glob("*.png")))}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
