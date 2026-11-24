"""Compare frozen free rollouts of the immediate and delayed-meal T2 balances."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.qualify_t2_disposal_revision import TARGETS, paired

MODELS = {
    "feasible_reference": "Supplied feasible fit",
    "immediate_refit": "Refitted original balance",
    "delayed_refit": "Refitted delayed balance",
}


def _case_ids(split: str) -> tuple[str, ...]:
    if split not in {"train", "val"}:
        raise ValueError("Expected public training or development split")
    return tuple(f"{split}_{i:03}" for i in range(16 if split == "train" else 4))


def _index(result: dict, ids: tuple[str, ...]) -> dict:
    events = {e["event"]: e for e in result["events"]}
    if set(events) != set(ids) or len(events) != len(result["events"]):
        raise ValueError("Missing, duplicate or unexpected event in registered split")
    return events


def _nmse_difference(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in TARGETS}


def summarize(scores: dict[str, dict], split: str) -> dict:
    """Compare each matched event and preserve numerical and domain qualifications."""
    ids = _case_ids(split)
    if not scores.keys() >= MODELS.keys():
        raise ValueError("Need both refits and the supplied feasible comparator")
    by_event = {name: _index(result, ids) for name, result in scores.items()}
    comparisons = {}
    for reference in ("immediate_refit", "feasible_reference"):
        before, after, differences = [], [], {}
        for name in ids:
            a, b = by_event[reference][name], by_event["delayed_refit"][name]
            if a["status"] != "scored" or b["status"] != "scored":
                differences[name] = {"status": "unresolved"}
                continue
            if a["time"] != b["time"] or a["observed"] != b["observed"]:
                raise ValueError(f"Inconsistent observations or time grid in {name}")
            before.append(a)
            after.append(b)
            differences[name] = {
                "status": "scored",
                "pooled_nmse": b["pooled_nmse"] - a["pooled_nmse"],
                "per_target_nmse": _nmse_difference(
                    a["per_target_nmse"], b["per_target_nmse"]
                ),
            }
        comparison = paired(before, after) if before else {"events": 0, "counts": {}}
        whole = None
        if len(before) == len(ids):
            a, b = scores[reference]["summary"], scores["delayed_refit"]["summary"]
            whole = {
                "pooled_nmse": b["pooled_nmse"] - a["pooled_nmse"],
                "per_target_nmse": _nmse_difference(
                    a["available_per_target_nmse"], b["available_per_target_nmse"]
                ),
            }
        comparisons[reference] = comparison | {
            "candidate": "delayed_refit",
            "candidate_minus_reference_by_event": differences,
            "candidate_minus_reference_whole_cohort": whole,
            "reference_domain_eligible": scores[reference]["domain"]["eligible"],
            "candidate_domain_eligible": scores["delayed_refit"]["domain"]["eligible"],
        }
    return {"split": split, "models": scores, "comparisons": comparisons}


def plot(report_path: Path, image: Path) -> None:
    """Draw paired glucose and disposal trajectories from sealed development scores."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    payload = sealed_read(report_path)
    report = payload["report"]
    if not payload["freeze"] or report["split"] != "val":
        raise ValueError("Expected report linked to a frozen training selection")
    ids = _case_ids("val")
    cases = {k: _index(report["models"][k], ids) for k in MODELS}
    styles = {
        "feasible_reference": {"color": "#867447", "ls": ":", "lw": 1.65},
        "immediate_refit": {"color": "#bf542d", "ls": "--", "lw": 1.6},
        "delayed_refit": {"color": "#116ca5", "ls": "-", "lw": 1.6},
    }
    fig = Figure(figsize=(12.6, 5.35))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, len(ids), sharex="col")
    for col, name in enumerate(ids):
        for r, target in enumerate(("Gp", "U")):
            ax = axes[r, col]
            reference = None
            for arm, label in MODELS.items():
                event = cases[arm][name]
                if not report["models"][arm]["domain"]["eligible"]:
                    label += " (domain unresolved)"
                if event["status"] != "scored":
                    raise ValueError(f"Unresolved forecast for {arm}/{name}")
                observations = (event["time"], event["observed"][target])
                if reference is None:
                    reference = observations
                    ax.plot(*observations, label="Observed", color="#202124", lw=2)
                elif observations != reference:
                    raise ValueError(f"Unmatched observations or times for {name}")
                ax.plot(
                    event["time"],
                    event["predicted"][target],
                    label=label,
                    **styles[arm],
                )
            if r == 0:
                ax.set_title(name, fontsize=10)
            ax.grid(axis="y", alpha=0.18)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Plasma glucose\n(mg/kg)", fontsize=10)
    axes[1, 0].set_ylabel("Glucose disposal\n(mg/kg/min)", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.11, top=0.79, wspace=0.27)
    fig.suptitle(
        "Glucose and disposal on the four development trajectories", fontsize=14, y=0.97
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        fontsize=10,
    )
    fig.supxlabel("Time (min)", y=0.02, fontsize=10)
    fig.savefig(image, dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    plot(args.report, args.image)
