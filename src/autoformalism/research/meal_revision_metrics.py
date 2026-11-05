"""Training reconstruction and meal-peak evidence for the candidate-15 comparison."""

from collections.abc import Mapping, Sequence
from html import escape
from math import isclose
from statistics import median

import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import expm

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory

PROTOCOL = "candidate-15-meal-revision-metrics-1"
CHANNELS = ("Gp", "I", "U")
MEAL = "meal_rate_g_per_min"
INFUSION = "insulin_pmol_per_kg_min"
SHIFTED = {"train_003": 0.0, "train_006": 60.0, "train_007": 120.0}
DOSES = (30, 60, 90, 120, 150)


def _robust(values: Sequence[float]) -> dict:
    """Summarize equally weighted trajectories, including an explicit empty set."""
    if not values:
        return {"count": 0, "median": None, "mad": None}
    center = median(values)
    return {
        "count": len(values),
        "median": center,
        "mad": median(abs(x - center) for x in values),
    }


def _meal_events(time: np.ndarray, rates: np.ndarray) -> list[dict]:
    """Identify positive input runs and integrate their supplied linear knots."""
    if np.any(rates < 0):
        raise ValueError("negative ingestion rate")
    positive = rates > 0
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.flatnonzero(positive & ~np.r_[positive[1:], False])
    events = []
    for start, end in zip(starts, ends, strict=True):
        first = max(0, start - 1)
        last = min(len(time) - 1, end + 1)
        interval = slice(first, last + 1)
        events.append(
            {
                "start_min": float(time[first]),
                "end_min": float(time[last]),
                "dose_g": float(trapezoid(rates[interval], time[interval])),
            }
        )
    return events


def _grid_width(time: np.ndarray, index: int) -> float:
    points = time[max(index - 1, 0) : index + 2]
    return float(np.max(np.diff(points)))


def _peak(values: np.ndarray) -> tuple[int, float, bool]:
    """Use the earliest sampled global maximum and flag unresolved near-ties."""
    index = int(np.argmax(values))
    max_value = float(values[index])
    unique = (
        np.count_nonzero(np.isclose(values, max_value, rtol=1e-10, atol=1e-12)) == 1
    )
    return index, max_value, bool(unique)


def _reference_meal(
    tid: str, time: np.ndarray, rates: np.ndarray, events: list
) -> bool:
    """Match each predeclared 90 g triangle by identity, schedule, and amount."""
    if tid not in SHIFTED or len(events) != 1:
        return False
    start = SHIFTED[tid]
    triangle = np.maximum(0.0, 18.0 * (1.0 - np.abs(time - start - 5.0) / 5.0))
    return bool(
        events[0]["start_min"] == start
        and events[0]["end_min"] == start + 10.0
        and isclose(events[0]["dose_g"], 90.0, rel_tol=1e-10)
        and np.allclose(rates, triangle, rtol=1e-10, atol=1e-12)
    )


def _trajectory(
    row: PublicTrajectory,
    block: np.ndarray,
    channels: Sequence[str],
    insulin_scale: float,
) -> dict:
    t = np.asarray(row.time)
    rates = np.asarray(row.external_inputs[MEAL])
    events = _meal_events(t, rates)
    observed = np.asarray(row.targets["I"])
    predicted = observed + insulin_scale * block[:, channels.index("I")]
    if not np.isfinite(predicted).all():
        raise ValueError("nonfinite reconstructed insulin")
    oi, om, ou = _peak(observed)
    pi, pm, pu = _peak(predicted)
    return {
        "trajectory": row.trajectory_id,
        "meal_events": events,
        "external_insulin_zero": all(x == 0 for x in row.external_inputs[INFUSION]),
        "matches_shifted_90g_schedule": _reference_meal(
            row.trajectory_id, t, rates, events
        ),
        "target_training_nmse": dict(
            zip(channels, np.mean(block**2, axis=0).tolist(), strict=True)
        ),
        "observed_insulin_peak_min": float(t[oi]),
        "predicted_insulin_peak_min": float(t[pi]),
        "observed_insulin_max": om,
        "predicted_insulin_max": pm,
        "observed_peak_unique": ou,
        "predicted_peak_unique": pu,
        "grid_tolerance_min": max(_grid_width(t, oi), _grid_width(t, pi)),
        "peak_time_error_min": float(t[pi] - t[oi]),
        "peak_height_error": pm - om,
    }


def _group(rows: list[dict]) -> dict:
    """Keep every trajectory; summarize timing only for uniquely located peaks."""
    return {
        "trajectories": [row["trajectory"] for row in rows],
        "target_nmse": {
            c: _robust([row["target_training_nmse"][c] for row in rows])
            for c in CHANNELS
        },
        "absolute_peak_time_error_min": _robust(
            [
                abs(row["peak_time_error_min"])
                for row in rows
                if row["observed_peak_unique"] and row["predicted_peak_unique"]
            ]
        ),
        "absolute_peak_height_error": _robust(
            [abs(row["peak_height_error"]) for row in rows]
        ),
    }


def _dose_groups(rows: list[dict]) -> list[dict]:
    single = [r for r in rows if len(r["meal_events"]) == 1]
    # Doses within 1e-6 g share a display group; all exact integrals remain in rows.
    doses = sorted({round(r["meal_events"][0]["dose_g"], 6) for r in single})
    out = []
    for dose in doses:
        group = [r for r in single if round(r["meal_events"][0]["dose_g"], 6) == dose]
        out.append(
            {
                "dose_g": dose,
                "all": _group(group),
                "unconfounded_at_zero": _group(
                    [
                        r
                        for r in group
                        if r["external_insulin_zero"]
                        and r["meal_events"][0]["start_min"] == 0
                    ]
                ),
            }
        )
    return out


def summarize(
    training: PublicSplit,
    channels: Sequence[str],
    scales: Mapping[str, float],
    residual: np.ndarray,
    *,
    glucose_gain: float | None = None,
    clearance: float | None = None,
) -> dict:
    """Score the complete original-equation residual in row/time/channel order."""
    if training.name != "train":
        raise ValueError("meal revision metrics require public training data")
    channels = tuple(channels)
    if len(channels) != len(CHANNELS) or set(channels) != set(CHANNELS):
        raise ValueError("expected exactly Gp, I and U")
    if set(scales) != set(CHANNELS) or any(
        not np.isfinite(s) or s <= 0 for s in scales.values()
    ):
        raise ValueError("invalid training scales")
    n = sum(len(row.time) for row in training.rows)
    values = np.asarray(residual, dtype=float)
    if values.shape != (n * len(channels),) or not np.isfinite(values).all():
        raise ValueError("training residual must be complete, flat and finite")
    blocks = values.reshape(n, len(channels))
    nmse = np.mean(blocks**2, axis=0)
    if not np.isfinite(nmse).all():
        raise ValueError("nonfinite squared training residual")
    rows, offset = [], 0
    for row in training.rows:
        if (
            set(row.targets) != set(CHANNELS)
            or {MEAL, INFUSION} - row.external_inputs.keys()
        ):
            raise ValueError("required training channels missing or unexpected target")
        stop = offset + len(row.time)
        rows.append(_trajectory(row, blocks[offset:stop], channels, scales["I"]))
        offset = stop
    no_meal = [r for r in rows if not r["meal_events"]]
    groups = _dose_groups(rows)
    dose_set = {
        g["dose_g"] for g in groups if g["unconfounded_at_zero"]["trajectories"]
    }
    total = float(np.sum(nmse))
    result = {
        "protocol": PROTOCOL,
        "training_sha256": content_sha256(training),
        "scope": "Public training reconstruction with fitted training initial values.",
        "training_scales": {c: float(scales[c]) for c in channels},
        "training_nmse": float(np.mean(nmse)),
        "target_training_nmse": dict(zip(channels, nmse.tolist(), strict=True)),
        "normalized_sse_share": {
            c: (float(loss / total) if total else None)
            for c, loss in zip(channels, nmse, strict=True)
        },
        "rows": rows,
        "across_trajectories": _group(rows),
        "single_meal_by_dose": groups,
        "all_five_unconfounded_doses_at_zero": set(DOSES) <= dose_set,
        "shifted_90g": _group(
            [
                r
                for r in rows
                if r["matches_shifted_90g_schedule"] and r["external_insulin_zero"]
            ]
        ),
        "no_meal_controls": {
            "all": _group(no_meal),
            "zero_external_insulin": _group(
                [r for r in no_meal if r["external_insulin_zero"]]
            ),
            "with_external_insulin": _group(
                [r for r in no_meal if not r["external_insulin_zero"]]
            ),
        },
        "peak_convention": "Earliest whole-trajectory sampled maximum. "
        "Near-ties use rtol=1e-10 and atol=1e-12 in insulin units.",
    }
    if (glucose_gain is None) != (clearance is None):
        raise ValueError("both conditional insulin coefficients are required")
    if glucose_gain is not None:
        result["conditional_postmeal"] = conditional_postmeal(
            training, result, glucose_gain=glucose_gain, clearance=clearance
        )
    return result


def conditional_postmeal(
    training: PublicSplit, summary: dict, *, glucose_gain: float, clearance: float
) -> dict:
    """Replay I'=a*observed_Gp-lambda*I with linear glucose between samples."""
    if training.name != "train" or summary["training_sha256"] != content_sha256(
        training
    ):
        raise ValueError("conditional replay requires the same public training rows")
    if not np.isfinite([glucose_gain, clearance]).all():
        raise ValueError("invalid conditional insulin coefficients")
    operator = np.array([[-clearance, glucose_gain, 0.0], [0, 0, 1], [0, 0, 0]])
    transitions, rows = {}, []
    by_id = {row.trajectory_id: row for row in training.rows}
    for score in summary["rows"]:
        if not (
            score["matches_shifted_90g_schedule"] and score["external_insulin_zero"]
        ):
            continue
        tid = score["trajectory"]
        row = by_id[tid]
        first = row.time.index(score["meal_events"][0]["end_min"])
        last = row.time.index(score["observed_insulin_peak_min"])
        if last <= first or not score["observed_peak_unique"]:
            continue
        insulin = row.targets["I"][first]
        glucose = row.targets["Gp"]
        for i in range(first, last):
            h = row.time[i + 1] - row.time[i]
            if h not in transitions:
                transitions[h] = expm(operator * h)[0]
            insulin = float(
                transitions[h]
                @ (insulin, glucose[i], (glucose[i + 1] - glucose[i]) / h)
            )
        if not np.isfinite(insulin):
            raise ValueError("nonfinite conditional postmeal result")
        rows.append(
            {
                "trajectory": tid,
                "meal_end_min": row.time[first],
                "observed_peak_min": row.time[last],
                "observed_I_at_meal_end": row.targets["I"][first],
                "observed_I_at_peak": row.targets["I"][last],
                "conditional_I_at_peak": insulin,
            }
        )
    return {
        "rows": rows,
        "glucose_gain": float(glucose_gain),
        "clearance": float(clearance),
        "scope": "Conditional training diagnostic with linearly interpolated "
        "observed glucose and observed insulin at meal end.",
    }


def _route(rows: list[dict]) -> dict:
    witnesses = [
        r
        for r in rows
        if r["matches_shifted_90g_schedule"] and r["external_insulin_zero"]
    ]
    supported = {w["trajectory"] for w in witnesses} == set(SHIFTED) and all(
        w["predicted_peak_unique"]
        and w["observed_peak_unique"]
        and w["observed_insulin_peak_min"] > w["meal_events"][0]["end_min"]
        for w in witnesses
    )
    persistent = supported and all(
        w["peak_time_error_min"] < -w["grid_tolerance_min"] for w in witnesses
    )
    status = "persistent_early_peak" if persistent else "early_peak_criterion_cleared"
    return {
        "status": status if supported else "insufficient_evidence",
        "eligible_for_proposer": persistent,
        "witnesses": witnesses,
        "criterion": "All three prespecified 90 g zero-infusion schedules have unique "
        "sampled peaks, an observed peak after ingestion ends, and fitted peaks "
        "earlier than observed by more than one local sampling interval. "
        "Numerical completion and verification are prerequisites.",
    }


def compare(parent: dict, child: dict) -> dict:
    """Compare like-for-like training replays and identify persistent early peaks."""
    if any(r["protocol"] != PROTOCOL for r in (parent, child)):
        raise ValueError("meal revision metrics protocol differs")
    if any(parent[k] != child[k] for k in ("training_sha256", "training_scales")):
        raise ValueError("parent and child training data or scales differ")
    deltas = []
    for p, c in zip(parent["rows"], child["rows"], strict=True):
        if p["trajectory"] != c["trajectory"]:
            raise ValueError("training trajectory pairing differs")
        deltas.append(
            {
                "trajectory": p["trajectory"],
                "target_nmse_delta": {
                    t: c["target_training_nmse"][t] - p["target_training_nmse"][t]
                    for t in CHANNELS
                },
                "absolute_peak_time_error_delta_min": abs(c["peak_time_error_min"])
                - abs(p["peak_time_error_min"])
                if all(
                    r["observed_peak_unique"] and r["predicted_peak_unique"]
                    for r in (p, c)
                )
                else None,
                "absolute_peak_height_error_delta": abs(c["peak_height_error"])
                - abs(p["peak_height_error"]),
            }
        )
    no_meal = set(parent["no_meal_controls"]["all"]["trajectories"])
    return {
        "training_nmse_delta": child["training_nmse"] - parent["training_nmse"],
        "target_training_nmse_delta": {
            t: child["target_training_nmse"][t] - parent["target_training_nmse"][t]
            for t in CHANNELS
        },
        "no_meal_target_nmse_delta": {
            t: _robust(
                [
                    d["target_nmse_delta"][t]
                    for d in deltas
                    if d["trajectory"] in no_meal
                ]
            )
            for t in CHANNELS
        },
        "trajectory_deltas": deltas,
        "routing": _route(child["rows"]),
    }


def _median_cell(summary: dict) -> str:
    if not summary["count"]:
        return "n/a"
    return f"{summary['median']:.6g} [{summary['mad']:.3g}]"


def render_comparison(parent: dict, child: dict) -> str:
    """Render training errors and meal/control witnesses for a matched restart."""
    paired = compare(parent, child)
    lines = [
        "### Training reconstruction",
        "",
        "| Measure | Parent | Child | Child minus parent |",
        "| --- | ---: | ---: | ---: |",
        f"| Overall NMSE | {parent['training_nmse']:.8g} | {child['training_nmse']:.8g}"
        f" | {paired['training_nmse_delta']:+.6g} |",
    ]
    for c in CHANNELS:
        lines.append(
            f"| {c} NMSE | {parent['target_training_nmse'][c]:.6g} | "
            f"{child['target_training_nmse'][c]:.6g} | "
            f"{paired['target_training_nmse_delta'][c]:+.6g} |"
        )
    lines.extend(
        [
            "",
            "Per-channel NMSE across trajectories, median [unscaled MAD].",
            "",
            "| Group | Arm | Count | Gp | I | U |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, arm in (("Parent", parent), ("Child", child)):
        for group, stats in (
            ("All", arm["across_trajectories"]),
            ("No meal", arm["no_meal_controls"]["all"]),
            (
                "No meal, zero infusion",
                arm["no_meal_controls"]["zero_external_insulin"],
            ),
            (
                "No meal, with infusion",
                arm["no_meal_controls"]["with_external_insulin"],
            ),
        ):
            cells = " | ".join(_median_cell(stats["target_nmse"][c]) for c in CHANNELS)
            lines.append(
                f"| {group} | {name} | {len(stats['trajectories'])} | {cells} |"
            )
    lines.extend(
        [
            "",
            "### Sampled insulin peaks on every single-meal schedule",
            "",
            "Times are minutes from trajectory start. Parentheses contain peak insulin "
            "concentration. A dagger marks an ambiguous fitted or observed peak time.",
            "",
            "| Trajectory | Dose (g) | Meal interval (min) | External insulin | "
            "Observed time (height) | Parent time (height) | Child time (height) |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for p, c in zip(parent["rows"], child["rows"], strict=True):
        if len(p["meal_events"]) != 1:
            continue
        meal = p["meal_events"][0]
        readings = []
        for row, kind in ((p, "observed"), (p, "predicted"), (c, "predicted")):
            mark = "" if row[f"{kind}_peak_unique"] else "†"
            readings.append(
                f"{row[f'{kind}_insulin_peak_min']:g}{mark} "
                f"({row[f'{kind}_insulin_max']:.6g})"
            )
        tid = escape(p["trajectory"]).replace("|", "\\|").replace("\n", " ")
        infusion = "zero" if p["external_insulin_zero"] else "present"
        lines.append(
            f"| {tid} | {meal['dose_g']:g} | "
            f"{meal['start_min']:g}-{meal['end_min']:g} | "
            f"{infusion} | {' | '.join(readings)} |"
        )
    if all("conditional_postmeal" in arm for arm in (parent, child)):
        lines.extend(
            [
                "",
                "### Conditional insulin at the observed peak",
                "",
                "Each calculation starts at observed meal-end insulin and uses "
                "linearly interpolated observed glucose with fixed coefficients.",
                "",
                "| Trajectory | End of meal I | Observed peak I | Parent | Child |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        lookup = {r["trajectory"]: r for r in child["conditional_postmeal"]["rows"]}
        for p in parent["conditional_postmeal"]["rows"]:
            if p["trajectory"] not in lookup:
                continue
            c = lookup[p["trajectory"]]
            lines.append(
                f"| {p['trajectory']} | {p['observed_I_at_meal_end']:.6g} | "
                f"{p['observed_I_at_peak']:.6g} | {p['conditional_I_at_peak']:.6g} | "
                f"{c['conditional_I_at_peak']:.6g} |"
            )
    lines.extend(["", f"Peak-timing decision: `{paired['routing']['status']}`."])
    return "\n".join(lines) + "\n"
