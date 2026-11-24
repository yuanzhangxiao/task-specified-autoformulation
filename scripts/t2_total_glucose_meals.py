"""Post-fit comparison of the two balances on registered T2 training meal events."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research.meal_revision_metrics import INFUSION, MEAL, _meal_events
from autoformalism.staged_topology import content_hash
from scripts.t2_initial_memory_folds import HOLDOUTS, TARGETS, fit_ids

ARMS = ("fading", "total")


def meal_schedule(row: dict) -> dict:
    """Describe intake by integrating the supplied, linearly interpolated rate."""
    t = np.asarray(row["time"], dtype=float)
    q = np.asarray(row["inputs"][MEAL], dtype=float)
    insulin = np.asarray(row["inputs"][INFUSION], dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or not np.all(np.diff(t) > 0)
        or t.shape != q.shape
        or t.shape != insulin.shape
        or not np.all(np.isfinite([t, q, insulin]))
    ):
        raise ValueError("Invalid public input schedule")
    pulses = _meal_events(t, q)
    external = bool(np.any(insulin != 0))
    if not pulses:
        group = "no meal"
    elif external:
        group = "meal with external insulin"
    elif len(pulses) > 1:
        group = "multiple meals"
    elif pulses[0]["start_min"] > t[0]:
        group = "delayed single meal"
    else:
        group = "single meal at zero"
    return {
        "pulses": pulses,
        "total_g": sum(p["dose_g"] for p in pulses),
        "spacing_min": np.diff([p["start_min"] for p in pulses]).tolist(),
        "external_insulin": external,
        "group": group,
    }


def _measure(curve: dict, row: dict, schedule: dict) -> dict:
    """Record per-event NMSE and the time and size of the largest native-unit miss."""
    event = row["trajectory_id"]
    if (
        curve["event"] != event
        or curve["time"] != row["time"]
        or not np.isclose(curve["meal_grams"], schedule["total_g"])
        or not curve["qualification"]["numerically_resolved"]
    ):
        raise ValueError(f"Curve and meal schedule do not match for {event}")
    targets = {}
    for target in TARGETS:
        observed = np.asarray(row["targets"][target], dtype=float)
        pred = np.asarray(curve["predicted"][target], dtype=float)
        if (
            not np.array_equal(curve["observed"][target], observed)
            or pred.shape != observed.shape
            or not np.all(np.isfinite(pred))
        ):
            raise ValueError(f"Unmatched observations or predictions for {target}")
        i = int(np.argmax(abs(pred - observed)))
        ymax = int(np.argmax(observed))
        imax = int(np.argmax(pred))
        targets[target] = {
            "nmse": curve["per_target_nmse"][target],
            "worst_absolute_error": float(abs(pred[i] - observed[i])),
            "time_min": curve["time"][i],
            "observed": float(observed[i]),
            "predicted": float(pred[i]),
            "observed_peak": float(observed[ymax]),
            "observed_peak_min": curve["time"][ymax],
            "predicted_peak": float(pred[imax]),
            "predicted_peak_min": curve["time"][imax],
        }
    return {
        "samples": len(row["time"]),
        "targets": targets,
        "domain_eligible": curve["qualification"]["eligible"],
    }


def _summary(events: dict) -> dict:
    """Summarize event-weighted and sample-weighted errors on a matched set of meals."""
    n = sum(e["fading"]["samples"] for e in events.values())
    summary = {"events": list(events), "samples": n}
    for target in TARGETS:
        summary[target] = {}
        for arm in ARMS:
            worst_id = max(
                events,
                key=lambda k: events[k][arm]["targets"][target]["worst_absolute_error"],
            )
            worst = events[worst_id][arm]["targets"][target]
            nmse = [e[arm]["targets"][target]["nmse"] for e in events.values()]
            summary[target][arm] = {
                "pooled_nmse": sum(
                    e[arm]["targets"][target]["nmse"] * e[arm]["samples"]
                    for e in events.values()
                )
                / n,
                "macro_nmse": float(np.mean(nmse)),
                "worst_absolute_error": worst["worst_absolute_error"],
                "worst_event": worst_id,
                "time_min": worst["time_min"],
                "observed": worst["observed"],
                "predicted": worst["predicted"],
            }
        diff = np.array(
            [
                e["total"]["targets"][target]["nmse"]
                - e["fading"]["targets"][target]["nmse"]
                for e in events.values()
            ]
        )
        tol = 1e-10
        summary[target]["total_lower"] = int(sum(diff < -tol))
        summary[target]["total_higher"] = int(sum(diff > tol))
        summary[target]["tied_within_1e-10"] = int(sum(abs(diff) <= tol))
    return summary


def _fold(held: str, curves: dict, rows: list[dict]) -> dict:
    ids = fit_ids(held)
    if sorted(r["trajectory_id"] for r in rows) != list(ids) or set(curves) != set(
        ARMS
    ):
        raise ValueError("Expected registered fitting events and both frozen models")
    if any(set(curves[a]) != set(ids) for a in ARMS):
        raise ValueError("A frozen training curve was missing or from another fold")
    events = {}
    for row in sorted(rows, key=lambda r: r["trajectory_id"]):
        event = row["trajectory_id"]
        schedule = meal_schedule(row)
        if schedule["pulses"]:
            events[event] = {"schedule": schedule} | {
                arm: _measure(curves[arm][event], row, schedule) for arm in ARMS
            }
    if not events:
        raise ValueError("No observed meal trajectories")
    groups = sorted({e["schedule"]["group"] for e in events.values()})
    return {
        "all_meals": _summary(events),
        "by_schedule": {
            group: _summary(
                {k: e for k, e in events.items() if e["schedule"]["group"] == group}
            )
            for group in groups
        },
        "events": events,
    }


def _plot(held: str, curves: dict, events: dict, path: Path) -> None:
    """Plot all shared meals on a common scale per target."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(14.4, 3.2 * int(np.ceil(len(events) / 3))))
    tiles = fig.add_gridspec(int(np.ceil(len(events) / 3)), 3, wspace=0.22, hspace=0.36)
    ref = {}
    for j, (event, detail) in enumerate(events.items()):
        panel = tiles[j // 3, j % 3].subgridspec(2, 1, hspace=0.06)
        schedule = detail["schedule"]
        title = f"{event}    " + ", ".join(
            f"{p['dose_g']:g} g at {p['start_min']:g} min" for p in schedule["pulses"]
        )
        if schedule["external_insulin"]:
            title += " + insulin"
        for i, target, label in ((0, "Gp", "Gp (mg/kg)"), (1, "U", "U (mg/kg/min)")):
            ax = fig.add_subplot(panel[i], sharey=ref.get(target))
            ref.setdefault(target, ax)
            a, b = curves["fading"][event], curves["total"][event]
            for pulse in schedule["pulses"]:
                ax.axvspan(
                    pulse["start_min"], pulse["end_min"], color="0.75", alpha=0.3, lw=0
                )
            ax.plot(
                a["time"],
                a["observed"][target],
                color="#242429",
                lw=1.9,
                label="Observed",
            )
            ax.plot(
                a["time"],
                a["predicted"][target],
                "--",
                color="#be542d",
                lw=1.55,
                label="Fading, plasma balance",
            )
            ax.plot(
                b["time"],
                b["predicted"][target],
                "-.",
                color="#00758b",
                lw=1.65,
                label="Fading, total balance",
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9", lw=0.55)
            if i == 0:
                ax.set_title(title, fontsize=9.5, loc="left")
                ax.tick_params(labelbottom=False)
            if j % 3 == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Training meal trajectories with {held} omitted from fitting",
        y=0.993,
        fontsize=14,
    )
    handles, labels = ref["Gp"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.977),
    )
    fig.subplots_adjust(top=0.922, bottom=0.05)
    fig.supxlabel("Time (min)", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def review_meals(curves_by_fold: dict, rows_by_fold: dict, output_dir: Path) -> dict:
    """Compare fixed predictions on the meals used in both fits."""
    if set(curves_by_fold) != set(HOLDOUTS) or set(rows_by_fold) != set(HOLDOUTS):
        raise ValueError("Both registered folds are required")
    folds = {h: _fold(h, curves_by_fold[h], rows_by_fold[h]) for h in HOLDOUTS}
    a, b = (folds[h]["events"] for h in HOLDOUTS)
    if a.keys() != b.keys() or any(a[k]["schedule"] != b[k]["schedule"] for k in a):
        raise ValueError("The meal-event cohorts differ between folds")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = sealed_write(
        output_dir / "meals.json",
        {
            "protocol": "t2-fixed-balance-meal-comparison-1",
            "scope": "reconstruction of training meal events",
            "training_curves_sha256": content_hash(curves_by_fold),
            "folds": folds,
        },
    )
    for held in HOLDOUTS:
        _plot(
            held,
            curves_by_fold[held],
            folds[held]["events"],
            output_dir / f"meals-{held}.png",
        )
    return result
