"""Render absolute and paired responses for the frozen full and fixed-state arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import iclr_anchor_fidelity as assay
from scripts import iclr_reference_fits as fits

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def reference_effects(root: Path, plan: dict) -> list[dict]:
    """Record diagnostic strength even when a candidate is unavailable or unstable."""
    rows = []
    for cell, schedules in plan["schedules"].items():
        c = assay.base_cell(cell)
        for name, item in schedules.items():
            if item["role"] != "contrast":
                continue
            records = []
            for k in (name, item["control"]):
                stored = sealed_read(root / "references" / cell / (k + ".json"))
                if stored["plan"] != plan["artifact_sha256"]:
                    raise ValueError("reference does not match frozen test version")
                records.append(stored["reference"])
            y, control = records
            for quantity in fits.TARGETS[c] + fits.HIDDEN[c]:
                values = np.asarray(y[quantity])
                base = np.asarray(control[quantity])
                delta = values - base
                t = np.asarray(y["time"])
                i = int(np.abs(delta).argmax())
                floor = 1e-8 * max(1.0, max(abs(values)), max(abs(base)))
                rows.append(
                    {
                        "benchmark_id": cell,
                        "case": name,
                        "control": item["control"],
                        "quantity": quantity,
                        "unit": assay.UNITS[quantity],
                        "time_unit": "normalized reactor time"
                        if c == fits.CSTR
                        else "min",
                        "peak_effect": delta[i],
                        "peak_time": t[i] if abs(delta[i]) > floor else None,
                        "effect_auc": float(np.trapezoid(delta, t)),
                        "amplitude_floor": floor,
                        "reference_diagnostic": bool(abs(delta[i]) > floor),
                    }
                )
    return rows


def plot(root: Path) -> Path:
    plan = sealed_read(root / "frozen_selection_manifest.json")
    summary = sealed_read(root / "summary.json")
    if summary["plan"] != plan["artifact_sha256"]:
        raise ValueError("mismatched figure inputs")
    output = root / "absolute_and_difference_trajectories.pdf"
    assay.csv_table(root / "reference_effects.csv", reference_effects(root, plan))
    with PdfPages(output) as pdf:
        for cell, cases in plan["schedules"].items():
            c = assay.base_cell(cell)
            case = (
                "feed_high"
                if c == fits.CSTR
                else ("fed_long" if c == fits.T2 else "split")
            )
            control = cases[case]["control"]
            truth, base = (
                sealed_read(root / "references" / cell / (k + ".json"))["reference"]
                for k in (case, control)
            )
            for quantity in fits.TARGETS[c] + fits.HIDDEN[c]:
                fig, axs = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
                time = truth["time"]
                axs[0].plot(
                    time, truth[quantity], color="black", lw=2, label="Simulator"
                )
                axs[1].plot(
                    time,
                    np.subtract(truth[quantity], base[quantity]),
                    color="black",
                    lw=2,
                )
                for i, row in enumerate(plan["roster"]):
                    if (
                        row["benchmark_id"] != cell
                        or row["status"] != "ready"
                        or row["method"]
                        not in (
                            "full",
                            "fixed_gamma",
                            "fitted_reference_structure",
                            "autoformalism:full:c1v1s1",
                        )
                    ):
                        continue
                    stats = summary["subjects"][i]
                    eligible = [
                        s
                        for s in stats["scores"]
                        if s["case"] == case
                        and s["quantity"] == quantity
                        and s["status"] in ("complete", "negligible_reference_effect")
                    ]
                    if not eligible:
                        continue
                    kind = eligible[0]["readout"]
                    folder = root / "subjects" / f"{i:04d}"
                    pred, base_pred = (
                        sealed_read(folder / (k + ".json"))["curves"]
                        for k in (case, control)
                    )
                    if kind == "oracle_affine":
                        proj = stats["training_probes"][quantity]
                        p, q = (assay.probe.decode(proj, d) for d in (pred, base_pred))
                    else:
                        key = "direct." + quantity if kind == "direct" else quantity
                        p, q = (np.asarray(d[key]) for d in (pred, base_pred))
                    method = row["method"].replace(
                        "autoformalism:full:c1v1s1", "historical Full"
                    )
                    label = f"{method} {row['repetition']}" + (
                        " (affine)" if kind == "oracle_affine" else ""
                    )
                    line = axs[0].plot(time, p, label=label, lw=1.1)[0]
                    axs[1].plot(time, p - q, color=line.get_color(), lw=1.1)
                axs[0].set_title("Absolute trajectory")
                axs[1].set_title("Matched change")
                for ax in axs:
                    ax.set_xlabel(
                        "Normalized reactor time"
                        if c == fits.CSTR
                        else "Time (minutes)"
                    )
                    ax.set_ylabel(f"{quantity} ({assay.UNITS[quantity]})")
                    ax.spines[["top", "right"]].set_visible(False)
                axs[0].legend(fontsize=7, frameon=False)
                fig.suptitle(cell.replace("phase_", "") + "\n" + case.replace("_", " "))
                pdf.savefig(fig)
                plt.close(fig)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    print(plot(p.parse_args().root))
