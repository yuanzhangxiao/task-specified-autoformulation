"""Direction of each model's declared mechanisms over the suite's trajectories.

The study's direction (``direction_correct``) compares the signs of the
model's and the simulator's responses at each curve's largest magnitude: one
point per trajectory. Many mechanism responses on the combined suite
(``configs/cstr_combined_interventions_v1.json``) have two lobes of comparable
size, so that one point can flip with a small change in either lobe. This
scores the sign of the whole response instead: the share of the simulator's
response, weighted by its size, during which the model's response has the same
sign,

    agreement = sum_t |y(t)| [sign yhat(t) = sign y(t)] / sum_t |y(t)|,

over the whole trajectory, where y and yhat are the simulator's and the model's
quantity in the intervention run minus the matched control, as the response
RSE uses them. A direction counts as correct when the agreement exceeds one
half: the model moves the right way for most of the response. A model that does
not respond agrees nowhere. The quantities are the ones the frozen
declared-mechanism rule (``configs/cstr_declared_mechanisms_v1.json``) reads
from each model's equations, with the curves ``score_declared_mechanisms.py``
computes; the study's peak-sign direction and response RSE are recomputed from
the same curves and must reproduce the mechanism scores'.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_anchor_fidelity as anchor
from scripts import score_declared_mechanisms as declared
from scripts import score_term_timing as timing

PROTOCOL = "declared-mechanism-directions-1"
TOLERANCE = 1e-9


def sign_agreement(truth, control_truth, predicted, control_predicted) -> float:
    """Share of the true response, weighted by its size, with the model's sign."""
    observed = np.asarray(truth) - np.asarray(control_truth)
    estimate = np.asarray(predicted) - np.asarray(control_predicted)
    weight = np.abs(observed)
    total = float(weight.sum())
    if total <= 0:
        raise ValueError("the reference response is zero")
    same = np.sign(estimate) == np.sign(observed)
    return float(weight[same].sum() / total)


def recorded_scores(mechanisms: Path, attribution: dict) -> dict:
    """The mechanism scores' direct-readout rows, by model, quantity and case."""
    scored = sealed_read(mechanisms / "scores.json")
    if scored["attribution_sha256"] != attribution["artifact_sha256"]:
        raise ValueError("the mechanism scores belong to another assignment")
    return {
        (r["source_id"], r["quantity"], r["case"]): r
        for r in scored["rows"]
        if r.get("readout") == "direct" and r["quantity"] in declared.MECHANISMS
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
    contrasts = {k: v for k, v in schedule.items() if v["role"] == "contrast"}
    needed = {*contrasts, *(v["control"] for v in contrasts.values())}
    refs = {
        case: sealed_read(root / "references" / cell / f"{case}.json")["reference"]
        for case in needed
    }
    truth = {case: declared._truth(r) for case, r in refs.items()}
    recorded = recorded_scores(mechanisms, attribution)
    by_id = {r["source_id"]: r for r in declared.subjects(root)}
    rows, parity = [], {"rows": 0, "identical": 0, "differ": []}
    for entry in attribution["rows"]:
        base = {k: entry[k] for k in ("source_id", "method", "repetition")}
        if entry["status"] != "declared":
            rows.append(base | {"quantity": "*", "status": entry["status"]})
            continue
        view = declared.view_of(by_id[entry["source_id"]])
        model = {}
        for case in needed:
            with contextlib.suppress(timing.FAILURES):
                model[case] = declared.curves(
                    view, entry, anchor.trajectory(cell, refs[case])
                )
        for quantity in declared.MECHANISMS:
            if not entry["represented"][quantity]:
                rows.append(base | {"quantity": quantity, "status": "not_represented"})
                continue
            for case, item in contrasts.items():
                control = item["control"]
                row = base | {"quantity": quantity, "case": case}
                if case not in model or control not in model:
                    rows.append(row | {"status": "rollout_unavailable"})
                    continue
                args = (
                    truth[case][quantity],
                    truth[control][quantity],
                    model[case][quantity],
                    model[control][quantity],
                )
                study = anchor.scores(refs[case]["time"], *args, item["window"], 1.0)
                agreement = sign_agreement(*args)
                rows.append(
                    row
                    | {
                        "status": "complete",
                        "peak_direction_correct": study["direction_correct"],
                        "effect_rse": study["effect_rse"],
                        "sign_agreement": agreement,
                        "direction_correct": agreement > 0.5,
                    }
                )
                theirs = recorded[(entry["source_id"], quantity, case)]
                parity["rows"] += 1
                if theirs["status"] == "complete" and (
                    theirs["direction_correct"] == study["direction_correct"]
                    and abs(theirs["effect_rse"] - study["effect_rse"])
                    <= TOLERANCE * max(1.0, abs(theirs["effect_rse"]))
                ):
                    parity["identical"] += 1
                else:
                    parity["differ"].append([entry["source_id"], quantity, case])
    out.mkdir(parents=True, exist_ok=True)
    record = sealed_write(
        out / "term_directions.json",
        {
            "protocol": PROTOCOL,
            "root_plan_sha256": plan["artifact_sha256"],
            "attribution_sha256": attribution["artifact_sha256"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scorer_sha256": hashlib.sha256(
                Path(declared.__file__).read_bytes()
            ).hexdigest(),
            "study_parity": parity,
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
        "status",
        "peak_direction_correct",
        "sign_agreement",
        "direction_correct",
        "effect_rse",
    ]
    with (out / "term_directions.csv").open("w", newline="") as handle:
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
        "--out",
        type=Path,
        default=Path("outputs/term-directions-combined-cstr-hard-v1"),
    )
    a = parser.parse_args()
    record = main(a.root, a.mechanisms, a.out)
    print(record["artifact_sha256"], len(record["rows"]), "rows")
    parity = record["study_parity"]
    print(json.dumps({k: parity[k] for k in ("rows", "identical")}))
    print(json.dumps(parity["differ"][:10]))
