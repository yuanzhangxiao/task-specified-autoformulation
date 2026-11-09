"""Plot the paired glucose responses for both historical Full-prompt seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.iclr_anchor_fidelity import LEGACY
from scripts.iclr_latent_clamp import CONTRASTS, FULL, PANEL


def plot(root: Path, panel: Path, output: Path) -> None:
    """Plot the frozen numerical results on their physical scale."""
    summary = sealed_read(root / "summary.json")
    plan = sealed_read(root / "plan.json")
    if summary["plan"] != plan["artifact_sha256"] or plan["source"] != PANEL:
        raise ValueError("clamp results come from a different panel")
    jobs = {}
    for path in sorted((root / "jobs").glob("*.json")):
        job = sealed_read(path)
        if job["artifact_sha256"] not in summary["job_seals"]:
            raise ValueError("unbound numerical output")
        if job["method"] == FULL:
            jobs[job["seed"], job["contrast"]] = job
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.55))
    fig.subplots_adjust(
        left=0.091, right=0.98, top=0.79, bottom=0.102, wspace=0.25, hspace=0.38
    )
    fig.suptitle(
        "Glucose responses with hidden states on their fasting paths",
        x=0.09,
        ha="left",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    for col, case in enumerate(CONTRASTS):
        refs = []
        for k in ("fasting", case):
            ref = sealed_read(panel / "references" / LEGACY / (k + ".json"))
            if ref["artifact_sha256"] != plan["references"][f"{LEGACY}/{k}"]:
                raise ValueError("simulator reference changed")
            refs.append(ref["reference"])
        time = np.asarray(refs[0]["time"])
        for seed in (0, 1):
            ax = axes[seed, col]
            job = jobs[seed, case]
            record = next(
                r
                for r in job["rows"]
                if r["condition"] == "all_latents" and r["target"] == "Gp"
            )
            if record["status"] != "complete":
                raise ValueError("required plotted response unavailable")
            tr = job["solver_target_curves"][0]
            for y, name, color, style in (
                (
                    np.subtract(refs[1]["Gp"], refs[0]["Gp"]),
                    "Simulator",
                    "#202f42",
                    "-",
                ),
                (
                    np.subtract(tr["intact"]["Gp"], tr["control"]["Gp"]),
                    "Fitted model",
                    "#197a75",
                    "-",
                ),
                (
                    np.subtract(tr["all_latents"]["Gp"], tr["control"]["Gp"]),
                    "Hidden states held on fasting path",
                    "#b44d26",
                    "--",
                ),
            ):
                ax.plot(time, y, color=color, ls=style, lw=1.9, label=name)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e2e6ea", lw=0.65)
            condition = "48 g meal" if case == "fed" else "17 min insulin pulse"
            ax.set_title(f"Seed {seed} · {condition}", loc="left", fontsize=11)
            ax.set_xlim(time[0], time[-1])
            if not col:
                ax.set_ylabel("Glucose change (mg/kg)")
            if seed:
                ax.set_xlabel("Time (min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.944),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("clamp", "fidelity", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    plot(a.clamp, a.fidelity, a.output)
