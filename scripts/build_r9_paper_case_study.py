#!/usr/bin/env python3
"""Build the assisted R9 paper figures from sealed saved rollouts; never refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.plot_t1_four_model_interventions import sealed, write_csv

CELL = "phase_b_anonymous_system_t1_canonical_obfuscated_easy"
MODELS = {
    "r9": ("Assisted R9", "#009E73", "-"),
    "no_spec": ("No specification", "#D55E00", "--"),
    "sol0": ("Sol 0 (validation-selected)", "#7651A8", "-."),
    "sol1": ("Sol 1", "#0072B2", ":"),
    "sol2": ("Sol 2", "#B07A00", (0, (5, 2))),
}
PRIMARY = ("r9", "no_spec", "sol0")
CASES = (
    "fasting_gt100",
    "fasting_gt80",
    "fasting_gt120",
    "meal_gt100",
    "meal_gt80",
    "meal_gt120",
    "split_meal_gt100",
)
CONTROLS = {
    "fasting_gt80": "fasting_gt100",
    "fasting_gt120": "fasting_gt100",
    "meal_gt80": "meal_gt100",
    "meal_gt120": "meal_gt100",
    "split_meal_gt100": "meal_gt100",
}
TITLES = {
    "fasting_gt100": "Fasting control",
    "fasting_gt80": r"Fasting: initial $G_t$ -20%",
    "fasting_gt120": r"Fasting: initial $G_t$ +20%",
    "meal_gt100": "60 g meal: basal control",
    "meal_gt80": r"60 g meal: initial $G_t$ -20%",
    "meal_gt120": r"60 g meal: initial $G_t$ +20%",
    "split_meal_gt100": "30 + 30 g: 30 min spacing",
}


def validate_records(records: dict) -> None:
    """Check full reference/grid identity and finiteness, allowing solver roundoff."""
    first = next(iter(records.values()))
    for name, r in records.items():
        if not r["success"] or r["time"] != first["time"]:
            raise ValueError(f"Failed rollout or time-grid mismatch: {name}")
        for field in ("observed", "predicted"):
            if len(r[field]) != len(r["time"]) or not np.isfinite(r[field]).all():
                raise ValueError(f"Incomplete/nonfinite {field}: {name}")
        if not np.allclose(r["observed"], first["observed"], rtol=0, atol=1e-8):
            raise ValueError(f"Physical reference differs: {name}")


def score(record: dict, scale: float, control: dict | None = None) -> dict:
    """Recompute absolute and own-control response errors from complete curves."""
    y, p = np.asarray(record["observed"]), np.asarray(record["predicted"])
    result = {"nmse": float(np.mean(((p - y) / scale) ** 2)), "response_rse": None}
    if not np.isclose(result["nmse"], record["nmse"], rtol=1e-9, atol=1e-12):
        raise ValueError("Saved absolute NMSE does not match the curve")
    if control is not None:
        if record["time"] != control["time"]:
            raise ValueError("Control grid differs")
        dy = y - control["observed"]
        dp = p - control["predicted"]
        denom = float(dy @ dy)
        if denom <= 1e-8:
            raise ValueError("Negligible reference intervention effect")
        result["response_rse"] = float(np.sum((dp - dy) ** 2) / denom)
    return result


def selected_sol(scores: list[dict]) -> str:
    """Choose a repetition from original validation only, never from probes."""
    sol = [r for r in scores if r["model"].startswith("sol")]
    return min(sol, key=lambda r: (r["validation_nmse"], r["model"]))["model"]


def record_path(artifacts: Path, name: str, split: str, case: str) -> Path:
    """Resolve the known canonical sources without scanning other experiments."""
    if name == "r9":
        root = artifacts / "dalla-canonical-rescue-inspection-2026-09-25"
        return root / "replays/brief_canonical_r9_repaired" / split / f"{case}.json"
    if name == "no_spec":
        if split == "probes":
            root = artifacts / "t1-intervention-probe-2026-09-22"
            return root / "replays/cell01_seed0_no_spec" / f"{case}.json"
        root = artifacts / "t1-curves-round12-2026-09-22"
        return root / "replays/cell01_seed0_no_spec" / split / f"{case}.json"
    root = artifacts / "t1-external-intervention-replay-2026-09-25-v1"
    return root / f"replays/cell1_sol_rep{name[-1]}" / split / f"{case}.json"


def pooled(records: list[dict], scale: float) -> float:
    """Pool all target samples, not an unweighted subset of trajectories."""
    return sum(score(r, scale)["nmse"] * len(r["time"]) for r in records) / sum(
        len(r["time"]) for r in records
    )


def draw(ax, records, names, title, controls=None):
    """Draw native-unit absolute curves or each model's own paired difference."""
    first = records["r9"]
    y = np.asarray(first["observed"])
    if controls:
        y = y - controls["r9"]["observed"]
    ax.plot(first["time"], y, color="#202020", lw=1.8, label="Reference", zorder=7)
    for name in names:
        label, color, style = MODELS[name]
        p = np.asarray(records[name]["predicted"])
        if controls:
            p = p - controls[name]["predicted"]
        ax.plot(first["time"], p, color=color, ls=style, lw=1.35, label=label)
    if controls:
        ax.axhline(0, color="#999999", lw=0.5, zorder=0)
    ax.set(title=title, xlabel="Time (min)", xlim=(0, 300))
    ax.set_ylabel(
        r"$\Delta G_p$ (mg kg$^{-1}$)" if controls else r"$G_p$ (mg kg$^{-1}$)"
    )
    ax.grid(alpha=0.16, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def finish(fig, axes, output, stem, names, footnote):
    """Export vector figures and a directly inspectable PNG."""
    import matplotlib.pyplot as plt

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4 if len(names) == 3 else 3,
        frameon=False,
        fontsize=8,
    )
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=7, color="#555555")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            output / f"{stem}.{ext}",
            dpi=240,
            metadata={"Creator": "Autoformalism frozen-curve case study"}
            if ext == "pdf"
            else None,
        )
    plt.close(fig)


def render(data: dict, schedules: dict, output: Path) -> None:
    """Main six-panel figure and complete supporting trajectory grids."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.8))
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.93, bottom=0.21, wspace=0.39, hspace=0.48
    )
    for row, case in enumerate(("002", "010")):
        title = (
            "a  Training: one 60 g meal" if row == 0 else "d  Training: 30 + 60 g meals"
        )
        draw(axes[row, 0], data["train"][case], PRIMARY, title)
        for time, grams in schedules["train"][case]:
            axes[row, 0].axvline(time, color="#777777", lw=0.7, alpha=0.5)
            axes[row, 0].text(
                time + 5,
                0.93,
                f"{grams:g} g",
                fontsize=7,
                transform=axes[row, 0].get_xaxis_transform(),
            )
    for col, case in enumerate(("fasting_gt80", "fasting_gt120"), start=1):
        direction = "-20%" if col == 1 else "+20%"
        draw(
            axes[0, col],
            data["probes"][case],
            PRIMARY,
            f"{'b' if col == 1 else 'c'}  Held out: initial $G_t$ {direction}",
        )
        draw(
            axes[1, col],
            data["probes"][case],
            PRIMARY,
            f"{'e' if col == 1 else 'f'}  Change from fasting control",
            data["probes"]["fasting_gt100"],
        )
    for row in (0, 1):
        lo = min(axes[row, c].get_ylim()[0] for c in (1, 2))
        hi = max(axes[row, c].get_ylim()[1] for c in (1, 2))
        for col in (1, 2):
            axes[row, col].set_ylim(lo, hi)
    finish(
        fig,
        axes,
        output,
        "r9_case_study",
        PRIMARY,
        "Canonical obfuscated T1-easy | assisted endpoint | frozen parameters",
    )
    for split, shape in (("train", (4, 4)), ("validation", (2, 2))):
        fig, axes = plt.subplots(
            *shape, figsize=(9.2, 9.6 if split == "train" else 6.4)
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            top=0.90 if split == "train" else 0.86,
            bottom=0.18 if split == "train" else 0.24,
            wspace=0.42,
            hspace=0.72,
        )
        for ax, (case, records) in zip(axes.flat, data[split].items(), strict=True):
            meals = schedules[split][case]
            if len(meals) > 2 and len({g for _, g in meals}) == 1:
                times = "/".join(f"{t:g}" for t, _ in meals)
                schedule = f"{meals[0][1]:g} g at {times} min"
            else:
                schedule = ", ".join(f"{g:g}g@{t:g}" for t, g in meals) or "no meal"
            draw(ax, records, tuple(MODELS), f"{split} {case}\n{schedule}")
        fig.suptitle(
            f"All {len(data[split])} original {split} trajectories", fontsize=12
        )
        finish(
            fig,
            axes,
            output,
            f"r9_all_{split}",
            tuple(MODELS),
            "All frozen Sol repetitions retained; no trajectory-specific refitting",
        )
    for relative in (False, True):
        cases = tuple(CONTROLS) if relative else CASES
        fig, axes = plt.subplots(
            2 if relative else 3, 3, figsize=(9.2, 6.4 if relative else 8.1)
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            top=0.86 if relative else 0.90,
            bottom=0.24 if relative else 0.18,
            wspace=0.36,
            hspace=0.63,
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, case in zip(axes.flat, cases, strict=False):
            ax.set_visible(True)
            title = TITLES[case]
            if relative and case == "split_meal_gt100":
                title = "Split minus single 60 g meal"
            draw(
                ax,
                data["probes"][case],
                tuple(MODELS),
                title,
                data["probes"][CONTROLS[case]] if relative else None,
            )
        fig.suptitle(
            "All intervention responses"
            if relative
            else "All absolute intervention and control trajectories",
            fontsize=12,
        )
        finish(
            fig,
            axes,
            output,
            "r9_all_responses" if relative else "r9_all_absolute",
            tuple(MODELS),
            "Exploratory probes; all directions and repetitions shown",
        )


def build(artifacts: Path, output: Path) -> dict:
    """Verify 135 saved rollouts, export their complete values and render figures."""
    sources = {}
    old = artifacts / "t1-curves-round12-2026-09-22"
    rescue = artifacts / "dalla-canonical-rescue-inspection-2026-09-25"
    probe = sealed(artifacts / "t1-intervention-probe-2026-09-22/plan.json", sources)
    original = sealed(old / "inputs/ours/plan.json", sources)
    if any(m["cell"] != CELL for m in probe["models"]):
        raise ValueError("Intervention plan is not the canonical obfuscated cell")
    r9_plan = sealed(rescue / "plan.json", sources)
    r9 = next(
        m for m in r9_plan["models"] if m["model_id"] == "brief_canonical_r9_repaired"
    )
    if r9["cell"] != CELL or r9_plan["cases"] != probe["cases"]:
        raise ValueError("R9 used a different cell or intervention suite")
    scale = probe["training_scale"]
    data, scores, curves, schedules = {}, [], [], {}
    for split, cases in (
        ("train", [f"{i:03d}" for i in range(16)]),
        ("validation", [f"{i:03d}" for i in range(4)]),
        ("probes", CASES),
    ):
        data[split], schedules[split] = {}, {}
        for case in cases:
            records = {
                name: sealed(record_path(artifacts, name, split, case), sources)
                for name in MODELS
            }
            validate_records(records)
            data[split][case] = records
            if split != "probes":
                key = "training" if split == "train" else "validation"
                row = original["cells"][CELL][key]["rows"][int(case)]
                schedules[split][case] = [
                    [t, u]
                    for t, u in zip(
                        row["time"], row["external_inputs"]["u01"], strict=True
                    )
                    if u > 0
                ]
            for name, record in records.items():
                for t, y, p in zip(
                    record["time"], record["observed"], record["predicted"], strict=True
                ):
                    curves.append(
                        {
                            "split": split,
                            "case": case,
                            "model": name,
                            "time_min": t,
                            "reference": y,
                            "prediction": p,
                        }
                    )
                control = (
                    data[split][CONTROLS[case]][name] if case in CONTROLS else None
                )
                scores.append(
                    dict(
                        model=name,
                        split=split,
                        case=case,
                        **score(record, scale, control),
                    )
                )
    if schedules["train"]["002"] != [[0.0, 60.0]] or schedules["train"]["010"] != [
        [0.0, 30.0],
        [120.0, 60.0],
    ]:
        raise ValueError("The two previously displayed training schedules changed")
    summary = []
    for name in MODELS:
        paired = [
            r for r in scores if r["model"] == name and r["response_rse"] is not None
        ]
        summary.append(
            {
                "model": name,
                "train_nmse": pooled([r[name] for r in data["train"].values()], scale),
                "validation_nmse": pooled(
                    [r[name] for r in data["validation"].values()], scale
                ),
                "worst_intervention_nmse": max(r["nmse"] for r in paired),
                "worst_response_rse": max(r["response_rse"] for r in paired),
            }
        )
    if selected_sol(summary) != "sol0":
        raise ValueError("Original-validation selection no longer identifies Sol 0")
    differences = []
    for case, control in CONTROLS.items():
        for name in MODELS:
            a, b = data["probes"][case][name], data["probes"][control][name]
            for i, t in enumerate(a["time"]):
                differences.append(
                    {
                        "case": case,
                        "control": control,
                        "model": name,
                        "time_min": t,
                        "reference_delta": a["observed"][i] - b["observed"][i],
                        "predicted_delta": a["predicted"][i] - b["predicted"][i],
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "curves.csv", curves)
    write_csv(output / "response_curves.csv", differences)
    write_csv(output / "scores.csv", scores)
    write_csv(output / "summary.csv", summary)
    (output / "plot_data.json").write_text(
        json.dumps({"data": data, "schedules": schedules})
    )
    render(data, schedules, output)
    manifest = {
        "protocol": "assisted-r9-paper-case-study-1",
        "cell": CELL,
        "summary": summary,
        "selected_sol": selected_sol(summary),
        "selection_rule": "minimum original validation NMSE",
        "primary_models": list(PRIMARY),
        "all_models": list(MODELS),
        "displayed_training_cases": ["002", "010"],
        "training_case_selection": "Same two meal examples as the prior figure",
        "cases": probe["cases"],
        "training_scale": scale,
        "source_files_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verified_rollouts": len(scores),
        "new_integrations": 0,
        "fitting_calls": 0,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "limitation": (
            "Selected assisted historical case study, "
            "not a matched autonomous ablation."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Build from saved project artifacts, or replot a portable data export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if args.replot:
        saved = json.loads((args.output / "plot_data.json").read_text())
        render(saved["data"], saved["schedules"], args.output)
    else:
        result = build(args.artifacts, args.output)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("verified_rollouts", "selected_sol", "summary")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
