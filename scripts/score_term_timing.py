"""Peak timing of each model's declared mechanisms under the suite's timing events.

The combined suite (``configs/cstr_combined_interventions_v1.json``) times the
temperature's response to its pulses (``scripts/report_intervention_suite.py``).
This times the same events on the quantities the frozen declared-mechanism rule
(``configs/cstr_declared_mechanisms_v1.json``) reads from each model's own
equations: the three terms of its temperature balance and its declared hidden
states, against the simulator's, with the curves ``score_declared_mechanisms.py``
computes. The models, suite, references and rule are the ones already scored.

An event's effect is the quantity's response (minus the matched control's) as a
change since the last sample before the event's window. A term can contain the
pulsed input itself, and an input pulse is already on at the window's first
sample, so the suite's change since that sample would hide the jump. The
temperature is continuous in time, and its peak delays under the suite's own
rule must reproduce the suite report's. Peak error: time of the predicted
largest effect minus time of the true one. Event RSE, the suite's measure of an
event's time course: the squared error of the effect over the window divided by
the true effect's. A hidden state is read through the affine map the mechanism
scores fitted on the training calibration runs; its scale is nonnegative, so the
map moves no peak, and a state whose map could not be fitted is not timed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_anchor_fidelity as anchor
from scripts import report_intervention_suite as suite
from scripts import score_declared_mechanisms as declared

PROTOCOL = "declared-mechanism-timing-1"
QUANTITIES = ("T", *declared.MECHANISMS, *declared.STATES)
FAILURES = (
    ValueError,
    ArithmeticError,
    TimeoutError,
    RuntimeExpressionError,
    ModelValidationError,
)


def event_timing(time, truth, control_truth, predicted, control_predicted, window):
    """One event's direction, peak error and RSE, as a change since the prior sample."""
    time, truth, control_truth, predicted, control_predicted = map(
        np.asarray, (time, truth, control_truth, predicted, control_predicted)
    )
    inside = (time >= window[0]) & (time < window[1])
    first = int(np.argmax(inside))
    if inside.sum() < 2 or first < 1:
        raise ValueError(f"event window {window} needs a sample before it")
    t = time[inside]
    observed = truth - control_truth
    estimate = predicted - control_predicted
    observed = observed[inside] - observed[first - 1]
    estimate = estimate[inside] - estimate[first - 1]
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
        "true_peak_time": float(t[actual]),
        "predicted_peak_time": float(t[guess]) if resolvable else None,
        "peak_error": float(t[guess] - t[actual]) if resolvable else None,
        "true_peak": float(observed[actual]),
        "predicted_peak": float(estimate[guess]),
        "event_rse": float(np.mean((estimate - observed) ** 2) / np.mean(observed**2)),
    }


def state_maps(mechanisms: Path, attribution: dict) -> dict:
    """Each model's fitted readout of its declared C and Tj, from the scored rows."""
    scored = sealed_read(mechanisms / "scores.json")
    if scored["attribution_sha256"] != attribution["artifact_sha256"]:
        raise ValueError("the mechanism scores belong to another assignment")
    maps = {}
    for r in scored["rows"]:
        if r["quantity"] in declared.STATES and r.get("readout") == "affine":
            maps.setdefault((r["source_id"], r["quantity"]), r.get("map"))
    return maps


def suite_delays(root: Path, plan: dict) -> dict:
    """The suite report's temperature peak delay of every timed event."""
    report = sealed_read(root / "suite_report.json")
    if report["plan"] != plan["artifact_sha256"]:
        raise ValueError("the suite report belongs to another plan")
    return {
        (r["source_id"], r["case"], e["name"]): e
        for r in report["rows"]
        for e in r["events"]
        if e.get("timing")
    }


def main(root: Path, mechanisms: Path, out: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    attribution = sealed_read(mechanisms / "attribution.json")
    if attribution["root_plan_sha256"] != plan["artifact_sha256"] or attribution[
        "rule_sha256"
    ] != declared._digest(declared.RULE):
        raise ValueError("the assignment belongs to another root or rule")
    (cell,) = plan["cells"]
    schedule = declared.kept_schedule(root, plan, cell)
    timed = {
        case: item
        for case, item in schedule.items()
        if item["role"] == "contrast" and any(e["timing"] for e in item["events"])
    }
    needed = {*timed, *(item["control"] for item in timed.values())}
    refs = {
        case: sealed_read(root / "references" / cell / f"{case}.json")["reference"]
        for case in needed
    }
    truth = {
        case: {"T": np.asarray(r["T"]), **declared._truth(r)}
        for case, r in refs.items()
    }
    maps = state_maps(mechanisms, attribution)
    delays = suite_delays(root, plan)
    by_id = {r["source_id"]: r for r in declared.subjects(root)}
    rows, parity = [], {"events": 0, "identical": 0, "differ": []}
    for entry in attribution["rows"]:
        base = {k: entry[k] for k in ("source_id", "method", "repetition")}
        if entry["status"] != "declared":
            rows.append(base | {"quantity": "*", "status": entry["status"]})
            continue
        view = declared.view_of(by_id[entry["source_id"]])
        model, failed = {}, {}
        for case in needed:
            try:
                model[case] = declared.curves(
                    view, entry, anchor.trajectory(cell, refs[case])
                )
            except FAILURES as exc:
                failed[case] = str(exc)[:300]
        for quantity in QUANTITIES:
            if quantity in declared.STATES:
                key = "state." + quantity
                present = entry["hidden"][quantity] is not None
                readout = maps.get((entry["source_id"], quantity))
                if present and (readout is None or readout[0] <= 0):
                    rows.append(
                        base | {"quantity": quantity, "status": "readout_unavailable"}
                    )
                    continue
            else:
                key = quantity
                present = quantity == "T" or entry["represented"][quantity]
            if not present:
                rows.append(base | {"quantity": quantity, "status": "not_represented"})
                continue
            for case, item in timed.items():
                control = item["control"]
                for event in item["events"]:
                    if not event["timing"]:
                        continue
                    row = base | {
                        "quantity": quantity,
                        "case": case,
                        "event": event["name"],
                    }
                    if case not in model or control not in model:
                        rows.append(row | {"status": "rollout_unavailable"})
                        continue
                    scale, shift = (
                        readout if quantity in declared.STATES else (1.0, 0.0)
                    )
                    args = (
                        refs[case]["time"],
                        truth[case][quantity],
                        truth[control][quantity],
                        scale * np.asarray(model[case][key]) + shift,
                        scale * np.asarray(model[control][key]) + shift,
                        event["window"],
                    )
                    rows.append(row | event_timing(*args))
                    if quantity == "T":
                        mine = suite.event_metrics(*args, True)
                        theirs = delays[(entry["source_id"], case, event["name"])]
                        parity["events"] += 1
                        if all(
                            mine.get(k) == theirs.get(k)
                            for k in ("peak_delay", "direction_correct")
                        ):
                            parity["identical"] += 1
                        else:
                            parity["differ"].append(
                                [entry["source_id"], case, event["name"]]
                            )
        if failed:
            rows.append(
                base
                | {"quantity": "*", "status": "rollout_failures", "failures": failed}
            )
    out.mkdir(parents=True, exist_ok=True)
    record = sealed_write(
        out / "term_timing.json",
        {
            "protocol": PROTOCOL,
            "root_plan_sha256": plan["artifact_sha256"],
            "attribution_sha256": attribution["artifact_sha256"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scorer_sha256": hashlib.sha256(
                Path(declared.__file__).read_bytes()
            ).hexdigest(),
            "suite_parity": parity,
            "rows": rows,
            "private_reference_opened": True,
            "test_data_opened": False,
        },
    )
    keys = [
        "method",
        "repetition",
        "quantity",
        "case",
        "event",
        "status",
        "direction_correct",
        "true_peak_time",
        "predicted_peak_time",
        "peak_error",
        "true_peak",
        "predicted_peak",
        "event_rse",
    ]
    with (out / "term_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/combined-interventions-cstr-hard-v1")
    )
    parser.add_argument(
        "--mechanisms",
        type=Path,
        default=Path("outputs/declared-mechanisms-combined-cstr-hard-v1"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/term-timing-combined-cstr-hard-v1")
    )
    a = parser.parse_args()
    record = main(a.root, a.mechanisms, a.out)
    print(record["artifact_sha256"], len(record["rows"]), "rows")
    print(json.dumps(record["suite_parity"]))
