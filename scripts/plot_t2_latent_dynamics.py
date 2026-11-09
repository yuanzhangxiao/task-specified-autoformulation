"""Plot T2-hard action responses from frozen evaluator and pairing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.evaluate_t2_latent_dynamics import PROTOCOL, decode


def projection(
    root: Path, index: int, case: str, plan_sha: str, result: dict
) -> np.ndarray:
    """Read the exact saved latent trajectory and its calibration."""
    rec = sealed_read(root / "subjects" / f"{index:03d}" / (case + ".json"))
    if rec["plan_sha256"] != plan_sha:
        raise ValueError("curve from another frozen plan")
    return decode(result["probes"]["X_from_latent"], rec["curves"])


def plot(root: Path, pairs: Path) -> None:
    """Show both enabled seeds and every matched comparator for each prompt."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = sealed_read(root / "plan.json")
    summary = sealed_read(root / "summary.json")
    matches = sealed_read(pairs)
    if (
        summary["protocol"] != PROTOCOL
        or summary["plan_sha256"] != plan["artifact_sha256"]
        or matches["subject_bundle_sha256"] != plan["source_bundle_sha256"]
    ):
        raise ValueError("incompatible plot inputs")
    paired_ids = {
        p[side] for p in matches["pairs"] if p["eligible"] for side in ("left", "right")
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), sharex=True, sharey=True)
    titles = {
        "early_short": "Early pulse · 50 to 65 min",
        "late_long": "Late pulse · 120 to 150 min",
    }
    styles = {0: ("#3776ae", "-"), 1: ("#df761f", "--")}
    handles = {}
    for row_index, prompt in enumerate(("full", "brief_only")):
        for col_index, (case, title) in enumerate(titles.items()):
            ax = axes[row_index, col_index]
            ref = sealed_read(root / "reference" / (case + ".json"))["reference"]
            control = sealed_read(root / "reference/fasting_control.json")["reference"]
            time = ref["time"]
            span = next(c["insulin"][0][:2] for c in plan["cases"] if c["id"] == case)
            ax.axvspan(*span, color="#dce5ec", alpha=0.8, linewidth=0)
            ax.axhline(0, color="#92999d", linewidth=0.7)
            truth = np.asarray(ref["X"]) - control["X"]
            handles["Simulator"] = ax.plot(time, truth, color="#222222", linewidth=2.6)[
                0
            ]
            unavailable = []
            for i, (subject, result) in enumerate(
                zip(plan["rows"], summary["models"], strict=True)
            ):
                method = subject["method"]
                if not method.startswith(f"autoformalism:{prompt}:"):
                    continue
                enabled = method.endswith(":c1v1s1")
                if not enabled and subject["source_id"] not in paired_ids:
                    continue
                seed = subject["repetition"]
                arm = method.rsplit(":", 1)[1]
                if enabled:
                    color, linestyle = styles.get(seed, ("#8c564b", "-."))
                    label = f"Critic + verifier · seed {seed}"
                else:
                    color, linestyle = "#219187", ":"
                    label = f"Matched {arm} · seed {seed}"
                    if arm == "c0v0s1":
                        label = f"Neither · seed {seed} (matched)"
                scored = next(
                    r
                    for r in result["scores"]
                    if r["case"] == case and r["endpoint"] == "X_from_latent"
                )
                if scored["status"] != "complete":
                    unavailable.append(label)
                    continue

                pulse = projection(root, i, case, plan["artifact_sha256"], result)
                fasting = projection(
                    root, i, "fasting_control", plan["artifact_sha256"], result
                )
                curve = pulse - fasting
                if not np.isclose(
                    abs(curve).max(), abs(scored["predicted_signed_peak"])
                ):
                    raise ValueError("plotted peak disagrees with frozen effects")
                handles[label] = ax.plot(
                    time, curve, color=color, linestyle=linestyle, linewidth=2.1
                )[0]
            if unavailable:
                ax.text(
                    0.02,
                    0.97,
                    "Unavailable: " + ", ".join(unavailable),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            if row_index == 0:
                ax.set_title(title, fontsize=12)
            if col_index == 0:
                display = "Full" if prompt == "full" else "Brief-only"
                ax.set_ylabel(f"{display} prompt\nRemote action ΔX (pmol/L)")
            if row_index == 1:
                ax.set_xlabel("Time (min)")
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(axis="y", color="#d5dbdd", linewidth=0.6)
            ax.set_xlim(0, 300)
    fig.suptitle("Insulin action after a fasting insulin infusion", fontsize=17, y=0.97)
    order = sorted(handles, key=lambda name: (name != "Simulator", name))
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="lower center",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=9.8,
    )
    fig.subplots_adjust(
        left=0.11, bottom=0.20, right=0.985, top=0.865, hspace=0.16, wspace=0.10
    )
    for ext in ("pdf", "png"):
        fig.savefig(root / ("action-response." + ext), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.pairs)
