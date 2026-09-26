#!/usr/bin/env python3
"""Plot every frozen perturbed-T1 meal diagnostic and export an explicit close-up."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.compare_t1_perturbed_meals import IDS

STYLES = (
    ("Full seed 0 (R12)", "#0072B2", "-"),
    ("Full seed 1 (R12)", "#009E73", "-"),
    ("Sol repetition 0", "#CC79A7", "--"),
    ("Sol repetition 1", "#D55E00", "--"),
    ("Sol repetition 2", "#E69F00", "-"),
)


def save(fig, root: Path, name: str) -> None:
    """Export a readable preview and publication-editable vector copies."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(root / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot(root: Path) -> None:
    """Keep outliers visible in full views and label any reduced-comparator view."""
    plan = sealed_read(root / "plan.json")
    summary = sealed_read(root / "summary.json")
    if summary["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("summary and frozen design differ")
    records = {
        (m, c["id"]): sealed_read(root / "replays" / m / f"{c['id']}.json")
        for m in IDS
        for c in plan["cases"]
    }
    rows = {(r["model_id"], r["case_id"]): r for r in summary["rows"]}
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    for effect in (False, True):
        for focus in (False, True):
            cases = [c for c in plan["cases"] if not focus or not c["is_control"]]
            if effect:
                cases = [c for c in cases if c["control"]]
            columns = 2 if focus else 3
            height = (len(cases) + columns - 1) // columns
            fig, axes = plt.subplots(
                height,
                columns,
                figsize=(13, 3 * height),
                layout="constrained",
                squeeze=False,
            )
            for ax, case in zip(axes.flat, cases, strict=False):
                key = case["id"]
                first = records[IDS[0], key]
                ref = (
                    rows[IDS[0], key]["response"]["reference_delta"]
                    if effect
                    else first["observed"]
                )
                ax.plot(first["time"], ref, color="#222222", lw=2.2, label="Reference")
                for index, (model, (label, color, ls)) in enumerate(
                    zip(IDS, STYLES, strict=True)
                ):
                    if focus and index in (2, 3):
                        continue
                    record = records[model, key]
                    if record["success"]:
                        y = (
                            rows[model, key]["response"]["predicted_delta"]
                            if effect
                            else record["predicted"]
                        )
                        ax.plot(
                            record["time"], y, label=label, color=color, ls=ls, lw=1.7
                        )
                    else:
                        ax.text(
                            0.02,
                            0.95 - 0.05 * index,
                            f"{label}: rollout failed",
                            color=color,
                            transform=ax.transAxes,
                        )
                for time, _grams in case["meals"]:
                    ax.axvline(time, color="#777777", lw=0.7, alpha=0.4)
                if not focus:
                    ax.set_yscale("symlog", linthresh=100)
                meal_label = " + ".join(f"{g} g @ {t} min" for t, g in case["meals"])
                ax.set_title(
                    meal_label or f"Fasting ({case['duration']} min)", fontsize=10
                )
                ax.set_xlabel("Time (min)")
                ax.set_ylabel(
                    ("Change from fasting" if effect else "Glucose mass") + " (mg/kg)"
                )
                ax.grid(alpha=0.17)
                ax.spines[["top", "right"]].set_visible(False)
            for ax in list(axes.flat)[len(cases) :]:
                ax.set_visible(False)
            view = (
                "Full vs Sol repetition 2; repetitions 0/1 shown in all-model views"
                if focus
                else "All five endpoints, including outliers; symmetric-log y axis"
            )
            fig.suptitle(
                "Perturbed-obfuscated T1: frozen parameters\n"
                + view
                + (
                    "\nSol 1 fasting controls are solver-sensitive; primary Radau shown"
                    if not focus
                    else ""
                ),
                fontsize=13,
            )
            name = ("relative" if effect else "absolute") + (
                "-focus" if focus else "-all"
            )
            save(fig, folder, name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for index, (model, (label, color, ls)) in enumerate(zip(IDS, STYLES, strict=True)):
        cr = sealed_read(root / "conditional" / f"{model}.json")
        if cr["delta"] is None:
            continue
        axes[0].plot(cr["time"], cr["delta"], label=label, color=color, ls=ls)
        if index not in (2, 3):
            axes[1].plot(cr["time"], cr["delta"], label=label, color=color, ls=ls)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("All endpoints; symmetric-log axis")
    axes[1].set_title("Full vs Sol repetition 2; linear axis")
    for ax in axes:
        ax.axhline(0, color="#333333", lw=0.7)
        ax.axvline(60, color="#777777", lw=0.7)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Model change from no-meal control (mg/kg)")
        ax.grid(alpha=0.17)
    fig.suptitle(
        "Conditional 60 g pulse: fasting auxiliary histories held fixed\n"
        "Component diagnostic; no physical ground-truth accuracy score",
        fontsize=13,
    )
    save(fig, folder, "conditional-response")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    plot(parser.parse_args().root)
