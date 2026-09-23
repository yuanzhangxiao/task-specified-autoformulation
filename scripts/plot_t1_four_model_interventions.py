#!/usr/bin/env python3
"""Compare four frozen T1 models using existing replay checkpoints only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

CELL = "phase_b_anonymous_system_t1_canonical_obfuscated_easy"
PROBE_HASH = "8035569cea1a51f09bf0f895f6e425eafe937a8ac49341c39fe472398ab3b72d"
BRIEF_ID = "brief_canonical_obfuscated_s1_r13_trial"
MODELS = {
    "brief_only_r13": (BRIEF_ID, "Brief-only R13 (unretained)", "#009E73", "-"),
    "sol_no_latent": (
        "obfuscated_raw_data_agent_rep0",
        "Sol rep 0: no latent states",
        "#E69F00",
        "--",
    ),
    "sol_three_latent": (
        "obfuscated_raw_data_agent_rep1",
        "Sol rep 1: three latent states",
        "#CC79A7",
        "-",
    ),
    "no_spec": ("cell01_seed0_no_spec", "No specification", "#D55E00", "-."),
}


def sealed(path: Path, sources: dict) -> dict:
    """Validate original content seals and retain exact file provenance."""
    payload = path.read_bytes()
    value = json.loads(payload)
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in value.items() if k != "artifact_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if digest != value.get("artifact_sha256"):
        raise ValueError(f"Artifact seal differs: {path}")
    sources[str(path)] = hashlib.sha256(payload).hexdigest()
    return value


def aligned(records: dict[str, dict]) -> None:
    """Reject failed, nonfinite, misaligned, or different-reference rollouts."""
    reference = next(iter(records.values()))
    for key, row in records.items():
        if not row["success"] or row["time"] != reference["time"]:
            raise ValueError(f"Failed or mismatched time grid: {key}")
        if row["observed"] != reference["observed"]:
            raise ValueError(f"Different physical reference: {key}")
        if len(row["predicted"]) != len(row["time"]):
            raise ValueError(f"Incomplete prediction: {key}")
        if len(row["observed"]) != len(row["time"]):
            raise ValueError(f"Incomplete reference: {key}")
        if not all(
            math.isfinite(x)
            for field in ("time", "observed", "predicted")
            for x in row[field]
        ):
            raise ValueError(f"Nonfinite trajectory: {key}")


def wide_rows(case: str, records: dict[str, dict]) -> list[dict]:
    """Export native-unit trajectories without resampling or rounding."""
    aligned(records)
    first = next(iter(records.values()))
    return [
        {
            "case_id": case,
            "time_min": t,
            "reference": first["observed"][i],
            **{key: row["predicted"][i] for key, row in records.items()},
        }
        for i, t in enumerate(first["time"])
    ]


def differences(rows: list[dict], controls: list[dict]) -> list[dict]:
    """Subtract each model's own control, never a shared reference control."""
    if [r["time_min"] for r in rows] != [r["time_min"] for r in controls]:
        raise ValueError("Intervention and control time grids differ")
    return [
        {
            "case_id": r["case_id"],
            "control_case_id": c["case_id"],
            "time_min": r["time_min"],
            **{key: r[key] - c[key] for key in ("reference", *MODELS)},
        }
        for r, c in zip(rows, controls, strict=True)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Preserve round-trippable float values in a simple wide CSV."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output: Path, absolute: dict, relative: dict, xmax: int) -> None:
    """Show controls and both directions with consistent colors and axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for name, data, keys, title, ylabel in (
        (
            "absolute",
            absolute,
            ("fasting_gt100", "fasting_gt80", "fasting_gt120"),
            "Absolute plasma-glucose trajectories",
            r"$G_p$ (mg kg$^{-1}$)",
        ),
        (
            "relative_change",
            relative,
            ("fasting_gt80", "fasting_gt120"),
            "Change from each model's own unperturbed control",
            r"$\Delta G_p$ (mg kg$^{-1}$)",
        ),
    ):
        fig, axes = plt.subplots(1, len(keys), figsize=(12, 4.6), sharey=True)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.29, wspace=0.13)
        for ax, key in zip(axes, keys, strict=True):
            rows = [r for r in data[key] if r["time_min"] <= xmax]
            times = [r["time_min"] for r in rows]
            ax.plot(
                times,
                [r["reference"] for r in rows],
                color="#222222",
                lw=2.5,
                label="Ground truth",
                zorder=5,
            )
            for column, (_, label, color, style) in MODELS.items():
                ax.plot(
                    times,
                    [r[column] for r in rows],
                    color=color,
                    ls=style,
                    lw=1.8,
                    label=label,
                )
            subtitle = {
                "fasting_gt100": "Unperturbed control",
                "fasting_gt80": "Initial tissue glucose: -20%",
                "fasting_gt120": "Initial tissue glucose: +20%",
            }[key]
            ax.set(title=subtitle, xlabel="Time (min)", xlim=(0, xmax))
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, y=0.98, fontsize=15)
        fig.text(
            0.5,
            0.875,
            "Canonical obfuscated T1-easy | fasting | same initial plasma glucose",
            ha="center",
            fontsize=10,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            frameon=False,
            fontsize=10,
        )
        fig.text(
            0.5,
            0.015,
            "Exploratory diagnostic; frozen fitted models. "
            "Brief-only R13 is an unretained trial.",
            ha="center",
            fontsize=9,
        )
        suffix = "" if xmax == 300 else "_first120min"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}{suffix}.{extension}", dpi=180)
        plt.close(fig)


def build(artifacts: Path, output: Path) -> dict:
    """Bind the same physical cases, export all data, and render comparisons."""
    sources: dict[str, str] = {}
    probe = artifacts / "t1-intervention-probe-2026-09-22"
    brief = artifacts / "t1-round13-inspection-2026-09-22"
    original = artifacts / "t1-curves-round12-2026-09-22"
    plan = sealed(probe / "plan.json", sources)
    brief_plan = sealed(brief / "plan.json", sources)
    if plan["artifact_sha256"] != PROBE_HASH:
        raise ValueError("Unexpected diagnostic plan")
    if brief_plan["probe_plan_sha256"] != PROBE_HASH:
        raise ValueError("Brief-only used a different intervention plan")
    brief_job = next(m for m in brief_plan["models"] if m["model_id"] == BRIEF_ID)
    if brief_job["cell"] != CELL or any(m["cell"] != CELL for m in plan["models"]):
        raise ValueError("Models belong to different benchmark cells")
    if brief_plan["cases"] != plan["cases"]:
        raise ValueError("Different intervention definitions")
    output.mkdir(parents=True, exist_ok=True)
    absolute, relative, scores, development = {}, {}, [], []
    for split, case_ids in (
        ("diagnostic", [c["id"] for c in plan["cases"]]),
        ("train", [f"{i:03d}" for i in range(16)]),
        ("validation", [f"{i:03d}" for i in range(4)]),
    ):
        for case in case_ids:
            records = {}
            for key, (model_id, *_rest) in MODELS.items():
                if key == "brief_only_r13":
                    path = brief / "replays" / model_id / split / f"{case}.json"
                elif split == "diagnostic":
                    path = probe / "replays" / model_id / f"{case}.json"
                else:
                    path = original / "replays" / model_id / split / f"{case}.json"
                records[key] = sealed(path, sources)
                scores.append(
                    {
                        "model_id": key,
                        "split": split,
                        "case_id": case,
                        "nmse": records[key]["nmse"],
                    }
                )
            rows = wide_rows(case, records)
            if split == "diagnostic":
                absolute[case] = rows
            else:
                development.extend({"split": split, **r} for r in rows)
    for case in plan["cases"]:
        if control := case["paired_control"]:
            relative[case["id"]] = differences(absolute[case["id"]], absolute[control])
    write_csv(output / "absolute.csv", [r for rows in absolute.values() for r in rows])
    write_csv(
        output / "relative_change.csv", [r for rows in relative.values() for r in rows]
    )
    write_csv(output / "training_validation.csv", development)
    write_csv(output / "scores.csv", scores)
    for xmax in (300, 120):
        plot(output, absolute, relative, xmax)
    manifest = {
        "protocol": "t1-four-model-plot-export-1",
        "cell": CELL,
        "probe_plan_sha256": PROBE_HASH,
        "brief_plan_sha256": brief_plan["artifact_sha256"],
        "brief_source_model_id": brief_job["source_model_id"],
        "cases": plan["cases"],
        "models": MODELS,
        "training_scale": plan["training_scale"],
        "source_files_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": {
            "physical_references_equal": True,
            "time_grids_equal": True,
            "all_values_finite": True,
            "rollouts_read": len(scores),
        },
        "new_integrations": 0,
        "refitting_calls": 0,
        "live_llm_calls": 0,
        "test_data_opened": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Create reproducible plots and a self-contained data handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.artifacts_root.resolve(), args.output.resolve())
    shutil.copy2(__file__, args.output / "build_from_checkpoints.py")
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
