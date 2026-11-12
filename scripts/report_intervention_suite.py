"""Per-event metrics and exposure counts for a scored intervention suite.

Reads a root that ``scripts/score_baseline_interventions.py`` froze on a suite
declaring events and a validity rule (``configs/cstr_combined_interventions_v1.json``)
and then scored: its plan, validity record, references, scores and per-subject
checkpoints. Nothing is rolled out again.

For an event window ``a <= t < b`` the event's effect is the response's change
since ``a``, ``e(t) = d(t) - d(a)``, where ``d`` is the temperature minus the
matched control's. Direction: the signs of the true and predicted effects at
their largest magnitudes agree; a predicted effect below the study's floor
counts as wrong. Peak delay, for timing events only: time of the predicted
largest effect minus time of the true one. Event RSE: the squared error of the
effect over the window divided by the true effect's.

A model is exposed by a trajectory when it fails there (any status but
complete, a seed that retained no model included), its response RSE is 1 or
more, or it gets the direction of any event wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

QUANTITY = "T"
#: A response RSE at or above this is no better than predicting no response.
NO_BETTER_THAN_NONE = 1.0
#: The reference's own effect was below the study's floor: nothing to assess.
UNASSESSABLE = frozenset({"negligible_reference_effect"})


def event_metrics(
    time, truth, control_truth, predicted, control_predicted, window, timing
) -> dict:
    """One event's direction, peak delay and RSE of its effect."""
    time, truth, control_truth, predicted, control_predicted = map(
        np.asarray, (time, truth, control_truth, predicted, control_predicted)
    )
    # Half-open, so a window never holds the first sample of the next event.
    inside = (time >= window[0]) & (time < window[1])
    if inside.sum() < 2:
        raise ValueError(f"event window {window} holds fewer than two samples")
    t = time[inside]
    observed = (truth - control_truth)[inside]
    estimate = (predicted - control_predicted)[inside]
    observed, estimate = observed - observed[0], estimate - estimate[0]
    floor = 1e-8 * max(1.0, abs(truth).max(), abs(control_truth).max())
    actual, guess = int(np.argmax(abs(observed))), int(np.argmax(abs(estimate)))
    if abs(observed[actual]) <= floor:
        return {"status": "negligible_reference_effect"}
    resolvable = abs(estimate[guess]) > floor
    return {
        "status": "complete",
        "direction_correct": bool(
            resolvable and np.sign(observed[actual]) == np.sign(estimate[guess])
        ),
        "peak_delay": float(t[guess] - t[actual]) if timing and resolvable else None,
        "true_peak_time": float(t[actual]),
        "true_peak": float(observed[actual]),
        "predicted_peak": float(estimate[guess]),
        "event_rse": float(np.mean((estimate - observed) ** 2) / np.mean(observed**2)),
    }


def _scores(path: Path) -> dict:
    if not path.exists():
        return {}
    return {
        (r["subject"], r["case"]): r
        for r in sealed_read(path)["rows"]
        if r["readout"] == "target" and r["quantity"] == QUANTITY
    }


def evaluate(root: Path) -> dict:
    """Event rows and exposure for every scored subject on every kept trajectory."""
    plan = sealed_read(root / "plan.json")
    if not plan.get("validity_rule"):
        raise ValueError("the plan declares no validity rule")
    verdict = sealed_read(root / "validity.json")
    if verdict["plan"] != plan["artifact_sha256"]:
        raise ValueError("the validity record belongs to another plan")
    (cell,) = plan["schedules"]
    schedule = plan["schedules"][cell]
    contrasts = [
        k
        for k, v in schedule.items()
        if v["role"] == "contrast" and k not in verdict["dropped"]
    ]
    refs = {}
    for case in {*contrasts, *(schedule[k]["control"] for k in contrasts)}:
        record = sealed_read(root / "references" / cell / f"{case}.json")
        if record["plan"] != plan["artifact_sha256"]:
            raise ValueError("stale reference checkpoint")
        refs[case] = record["reference"]
    rows = []
    for folder, subjects in (
        (root, plan["rows"]),
        (root / "study", plan["study_rows"]),
    ):
        scored = _scores(folder / "scores.json")
        if subjects and not scored:
            raise ValueError(f"{folder} has not been scored")
        for i, subject in enumerate(subjects):
            for case in contrasts:
                item, base = schedule[case], schedule[case]["control"]
                found = scored[(i, case)]
                status = found["status"]
                row = {
                    "method": subject["method"],
                    "repetition": subject["repetition"],
                    "source_id": subject["source_id"],
                    "case": case,
                    "group": item["group"],
                    "status": status,
                    "response_rse": found.get("effect_rse"),
                    "trajectory_nmse": found.get("absolute_nmse"),
                    "events": [],
                }
                if status == "complete":
                    checkpoints = folder / "subjects" / f"{i:04d}"
                    curves = {
                        k: sealed_read(checkpoints / f"{k}.json")["curves"][QUANTITY]
                        for k in (case, base)
                    }
                    for event in item.get("events", []):
                        row["events"].append(
                            {
                                "name": event["name"],
                                "timing": event["timing"],
                                **event_metrics(
                                    refs[case]["time"],
                                    refs[case][QUANTITY],
                                    refs[base][QUANTITY],
                                    curves[case],
                                    curves[base],
                                    event["window"],
                                    event["timing"],
                                ),
                            }
                        )
                reasons = []
                if status in UNASSESSABLE:
                    row["exposed"] = None
                else:
                    if status != "complete":
                        reasons.append(f"failed: {status}")
                    elif row["response_rse"] >= NO_BETTER_THAN_NONE:
                        reasons.append("response RSE of 1 or more")
                    wrong = [
                        e["name"]
                        for e in row["events"]
                        if e.get("direction_correct") is False
                    ]
                    if wrong:
                        reasons.append("wrong direction: " + ", ".join(wrong))
                    row["exposed"] = bool(reasons)
                row["reasons"] = reasons
                rows.append(row)
    return {"plan": plan["artifact_sha256"], "kept": contrasts, "rows": rows}


def _median(values) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def summarize(evaluation: dict) -> list[dict]:
    """Per method and set: exposures, median response RSE and NMSE over seeds.

    A failure counts as worse than any value (infinite) in the medians.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in evaluation["rows"]:
        if r["exposed"] is None:
            continue
        for group in (r["group"], "all"):
            groups.setdefault((r["method"], group), []).append(r)
    out = []
    for (method, group), items in sorted(groups.items()):

        def value(r, key):
            return r[key] if r["status"] == "complete" else math.inf

        delays = {}
        for r in items:
            for e in r["events"]:
                if e.get("peak_delay") is not None:
                    delays.setdefault(e["name"], []).append(abs(e["peak_delay"]))
        out.append(
            {
                "method": method,
                "group": group,
                "trajectories": len(items),
                "failed": sum(r["status"] != "complete" for r in items),
                "exposed": sum(r["exposed"] for r in items),
                "median_response_rse": _median(value(r, "response_rse") for r in items),
                "median_trajectory_nmse": _median(
                    value(r, "trajectory_nmse") for r in items
                ),
                "events_wrong_direction": sum(
                    e.get("direction_correct") is False
                    for r in items
                    for e in r["events"]
                ),
                "events_assessed": sum(
                    e.get("direction_correct") is not None
                    for r in items
                    for e in r["events"]
                ),
                "median_abs_peak_delay": {
                    k: _median(v) for k, v in sorted(delays.items())
                },
            }
        )
    return out


def report(root: Path) -> dict:
    """Seal the event rows and the summary; write both as CSV beside them."""
    evaluation = evaluate(root)
    summary = summarize(evaluation)
    with (root / "suite_events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "repetition",
                "case",
                "group",
                "status",
                "response_rse",
                "trajectory_nmse",
                "exposed",
                "event",
                "direction_correct",
                "peak_delay",
                "event_rse",
            ]
        )
        for r in evaluation["rows"]:
            for e in r["events"] or [{}]:
                writer.writerow(
                    [
                        r["method"],
                        r["repetition"],
                        r["case"],
                        r["group"],
                        r["status"],
                        r["response_rse"],
                        r["trajectory_nmse"],
                        r["exposed"],
                        e.get("name"),
                        e.get("direction_correct"),
                        e.get("peak_delay"),
                        e.get("event_rse"),
                    ]
                )
    with (root / "suite_summary.csv").open("w", newline="") as handle:
        keys = [k for k in summary[0] if k != "median_abs_peak_delay"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    return sealed_write(
        root / "suite_report.json",
        {**evaluation, "summary": summary, "test_data_opened": False},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    out = report(parser.parse_args().root)
    print(out["artifact_sha256"], len(out["rows"]), "rows")
    print(json.dumps([s for s in out["summary"] if s["group"] == "all"], indent=1))
