"""Pair frozen Full and fixed-state outcomes within each hard anchor and seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.iclr_search_controls import ARMS as TREATMENTS
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_active_history import endpoint_complexity
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.iclr_specification_results import confirmed_scores
from scripts.iclr_submission_evidence import table

PRIMARY = {"acquisition_00": ("short_high", "Gp"), "sensor_00": ("jacket_warm", "T")}
ARMS = ("full", "fixed_gamma")
SCORES = (
    "absolute_nmse",
    "effect_rse",
    "reference_signed_peak",
    "predicted_signed_peak",
    "reference_window_auc",
    "predicted_window_auc",
    "peak_time_error",
)


def pair_outcomes(arms: dict, scores: dict) -> list[dict]:
    """Keep both outcomes, denominators and row-level numerical statuses."""
    indexed = {}
    for arm, model in arms.items():
        entries = {}
        for r in scores[model["source_id"]]:
            key = (r["case"], r["quantity"], r["readout"])
            if key in entries:
                raise ValueError("duplicate subject-condition-readout")
            entries[key] = r
        indexed[arm] = entries
    if not indexed["full"] or set(indexed["full"]) != set(indexed["fixed_gamma"]):
        raise ValueError("incomplete paired independent outcomes")
    results = []
    for key in sorted(indexed["full"]):
        row = dict(zip(("case", "quantity", "readout"), key, strict=True))
        for arm in ARMS:
            val = indexed[arm][key]
            row[arm + "_status"] = val["status"]
            row.update({arm + "_" + k: val.get(k) for k in SCORES})
        for k in ("absolute_nmse", "effect_rse"):
            a, b = (row[arm + "_" + k] for arm in ARMS)
            row["full_minus_fixed_" + k] = (
                a - b if a is not None and b is not None else None
            )
        results.append(row)
    return results


def run(search: Path, fidelity: Path, output: Path):
    """Authenticate each planned lineage, join the scores, and report paired results."""
    source = search / "frozen_models.json"
    bundle = sealed_read(source)
    plan = sealed_read(search / "plan.json")
    if bundle["protocol"] != "iclr-controlled-search-models-1":
        raise ValueError("require a final controlled-search export")
    verify_controlled_bundle(source, bundle)
    selected = {r["source_id"]: r for r in bundle["rows"]}
    all_scores, score_id = confirmed_scores(
        fidelity, [bundle["artifact_sha256"]], selected
    )
    arms = {}
    for block in plan["blocks"]:
        for task in block["tasks"]:
            m = selected[task["task_id"]]
            k = (task["cell"], task["seed"], task["treatment"])
            if k in arms or (m["benchmark_id"], m["repetition"], m["method"]) != k:
                raise ValueError("mismatched assignment")
            arms[k] = m
    if set(arms) != {(c, s, a) for c in PRIMARY for s in (0, 1, 2) for a in TREATMENTS}:
        raise ValueError("require the entire three-seed eight-arm anchor matrix")
    pairs = []
    endpoints = []
    for c in PRIMARY:
        for s in range(3):
            pair = {a: arms[c, s, a] for a in ARMS}
            for a, m in pair.items():
                source_id = m["source_id"]
                if source_id not in all_scores:
                    raise ValueError("missing assessment even for an unavailable model")
                d = {
                    "cell": c,
                    "seed": s,
                    "arm": a,
                    "source_id": source_id,
                    "model": m.get("model_sha256"),
                    "receipt": m["source_sha256"],
                    "status": m["status"],
                    "validation_nmse": m.get("validation_nmse"),
                    "shared_initial_receipt": pair["fixed_gamma"][
                        "shared_initial_result_sha256"
                    ],
                }
                if m["status"] == "ready":
                    if public.content_sha256(m["candidate"]) != d["model"]:
                        raise ValueError("model hash changed")
                    d.update(endpoint_complexity(m))
                endpoints.append(d)
            rows = pair_outcomes(pair, all_scores)
            if (PRIMARY[c][0], PRIMARY[c][1], "target") not in {
                (r["case"], r["quantity"], r["readout"]) for r in rows
            }:
                raise ValueError("the prespecified intervention is missing")
            pairs.extend(
                {
                    "cell": c,
                    "seed": s,
                    "full_source": pair["full"]["source_id"],
                    "fixed_source": pair["fixed_gamma"]["source_id"],
                    **r,
                }
                for r in rows
            )
    summary = sealed_write(
        output / "comparison.json",
        {
            "protocol": "iclr-paired-frozen-anchor-1",
            "plan": plan["artifact_sha256"],
            "bundle": bundle["artifact_sha256"],
            "confirmation": score_id,
            "primary_panel": {cell: list(values) for cell, values in PRIMARY.items()},
            "endpoints": endpoints,
            "outcomes": pairs,
            "parameter_cap": plan["max_parameters"],
        },
    )
    table(output / "paired_independent_outcomes.csv", pairs)
    table(output / "paired_endpoint_complexity.csv", endpoints)
    plot(endpoints, pairs, output)
    return summary


def plot(endpoints: list, pairs: list, output: Path) -> None:
    """Show paired seeds, model availability and primary pulse results."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2), layout="constrained")
    for i, (cell, (case, target)) in enumerate(PRIMARY.items()):
        points = {(e["seed"], e["arm"]): e for e in endpoints if e["cell"] == cell}
        p = {
            r["seed"]: r
            for r in pairs
            if (r["cell"], r["case"], r["quantity"], r["readout"])
            == (cell, case, target, "target")
        }
        for j, field in enumerate(
            ("absolute_nmse", "effect_rse", "target_reachable_state_count", "ready")
        ):
            ax = axes[i, j]
            for s, color in enumerate(("#256e92", "#b85529", "#398165")):
                vals = []
                for a in ARMS:
                    e = points[s, a]
                    if j < 2:
                        val = p[s].get(a + "_" + field)
                    elif j == 2:
                        val = e.get(field)
                    else:
                        val = int(e["status"] == "ready")
                    vals.append(float("nan") if val is None else val)
                ax.plot(
                    (0, 1), vals, "o-", label=f"Seed {s}", lw=1, color=color, alpha=0.8
                )
            ax.set_xticks((0, 1), ("Full", "Fixed"))
            ax.set_xlim(-0.25, 1.25)
            if j < 2:
                ax.set_yscale("symlog", linthresh=0.01)
            if j == 3:
                ax.set_yticks((0, 1), ("Unavailable", "Fitted"))
                ax.set_ylim(-0.25, 1.25)
            if not i:
                ax.set_title(
                    (
                        "Absolute NMSE",
                        "Response RSE",
                        "Reachable states",
                        "Final status",
                    )[j]
                )
            if not j:
                ax.set_ylabel(
                    "T2 · insulin pulse" if not i else "CSTR · jacket warming"
                )
            ax.spines[["top", "right"]].set_visible(False)
        if not i:
            axes[i, 0].legend(fontsize=7)
    fig.savefig(output / "paired-anchor-outcomes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("search", "fidelity", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    run(a.search, a.fidelity, a.output)
