"""Export a figure and per-endpoint summaries of the conditional meal-flux assay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from autoformalism.rebuttal.prefit_replay import sealed_read

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LABELS = {
    "full": "Full",
    "no_latent": "No latent",
    "no_spec": "Prediction only",
    "raw_data_agent": "GPT-5.6 Sol",
    "pysr": "PySR",
    "sindy": "SINDy",
}


def report(root: Path) -> None:
    """Summarize all planned meal schedules, withholding incomplete endpoint means."""
    plan = sealed_read(root / "plan.json")
    ref = sealed_read(root / "reference.json")
    if ref["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("unmatched reference")
    scenario = "meal_45"
    records = []
    for identifier in plan["models"]:
        model = sealed_read(root / "models" / (identifier + ".json"))
        if (
            model["model_id"] != identifier
            or model["plan_sha256"] != plan["artifact_sha256"]
        ):
            raise ValueError("unmatched subject record")
        rows = [model["cases"][k] for k in plan["meal_schedules"] if k != "fasting"]
        available = [r for r in rows if r["status"] == "ok"]
        summary = {
            "presentation": "Named"
            if "canonical_named_easy" in model["cell"]
            else "Obfuscated",
            "method": LABELS[model["method"]],
            "repetition": model["repetition"],
            "model_id": identifier,
            "scored": len(available),
            "planned": len(rows),
        }
        for field in (
            "response_rse",
            "fasting_memory_rse",
            "peak_time_error_min",
            "area_ratio",
        ):
            values = [r[field] for r in available if r[field] is not None]
            summary[field] = (
                float(np.median(values)) if len(values) == len(rows) else None
            )
        records.append(summary)
    endpoints = pd.DataFrame.from_records(records)
    endpoints.to_csv(root / "endpoints.csv", index=False)

    # Each trace panel shows every available endpoint for the specified method(s).
    # All method repetitions, including unresolved ones, also appear in endpoints.csv.
    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 5.7), sharex=True, layout="constrained"
    )
    time = np.arange(plan["horizon_min"] + 1)
    truth = np.subtract(ref["cases"][scenario]["ra"], ref["cases"]["fasting"]["ra"])
    colors = {
        "Full": "#126b75",
        "Prediction only": "#a14c19",
        "No latent": "#a14c19",
        "GPT-5.6 Sol": "#614da4",
    }
    for i, presentation in enumerate(("Named", "Obfuscated")):
        for j, methods in enumerate(
            (("Full", "No latent", "Prediction only"), ("GPT-5.6 Sol",))
        ):
            ax = axes[i, j]
            ax.plot(time, truth, color="#171717", lw=2, label="Simulator Ra")
            subset = endpoints[
                (endpoints.presentation == presentation)
                & endpoints.method.isin(methods)
            ]
            for model in subset.itertuples(index=False):
                result = sealed_read(root / "models" / (model.model_id + ".json"))[
                    "cases"
                ][scenario]
                if result["status"] == "ok":
                    ax.plot(
                        time,
                        result["delta_flux"],
                        color=colors[model.method],
                        lw=1.45,
                        linestyle=("-", "--", ":")[model.repetition],
                        label=f"{model.method} {model.repetition}",
                    )
            ax.axhline(0, color="0.75", lw=0.6, zorder=0)
            ax.set_title(
                f"{presentation} · "
                + ("Full and ablation" if j == 0 else "external agent"),
                loc="left",
                fontsize=11,
            )
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, plan["horizon_min"])
    fig.suptitle("Glucose appearance after a 45 g meal at minute 37", fontsize=13)
    fig.supxlabel("Time (minutes)", fontsize=10)
    fig.supylabel("Meal-driven plasma glucose rate (mg/kg/min)", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"meal-appearance.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    report(parser.parse_args().root)
