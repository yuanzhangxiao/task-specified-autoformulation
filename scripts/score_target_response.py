"""Direction accuracy and shape error of the temperature response, for every model.

The combined suite's report (``scripts/report_intervention_suite.py``) gives each
model's temperature response RSE on every kept trajectory. This adds the two
bounded measures used for the mechanisms, from the same sealed rollout
checkpoints, so nothing is simulated again: the size-weighted sign agreement of
the temperature response over the whole trajectory
(``score_term_directions.sign_agreement``) and, at each of the suite's timing
events, the shape error of the response measured from the sample before the
event (``score_term_timing.event_timing``). The response RSE is recomputed from
the same curves and must reproduce the report's. Every subject of the first
root, the study's arms included, is read, and so is every subject of each extra
root scored on the same suite with the same references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_anchor_fidelity as anchor
from scripts import score_declared_mechanisms as declared
from scripts import score_term_directions as directions
from scripts import score_term_timing as timing

PROTOCOL = "target-response-measures-1"
QUANTITY = "T"
TOLERANCE = 1e-9


def subject_folders(root: Path, plan: dict) -> dict[str, Path]:
    """Each subject's checkpoint folder by source id, as the suite report reads it."""
    folders = {}
    for folder, subjects in (
        (root, plan["rows"]),
        (root / "study", plan["study_rows"]),
    ):
        for i, row in enumerate(subjects):
            folders[row["source_id"]] = folder / "subjects" / f"{i:04d}"
    return folders


def references(root: Path, plan: dict) -> tuple[dict, dict]:
    """The kept contrasts and the reference record of every case they use."""
    (cell,) = plan["cells"]
    schedule = declared.kept_schedule(root, plan, cell)
    contrasts = {k: v for k, v in schedule.items() if v["role"] == "contrast"}
    needed = {*contrasts, *(v["control"] for v in contrasts.values())}
    refs = {}
    for case in needed:
        record = sealed_read(root / "references" / cell / f"{case}.json")
        if record["plan"] != plan["artifact_sha256"]:
            raise ValueError("stale reference checkpoint")
        refs[case] = record["reference"]
    return contrasts, refs


def main(roots: list[Path], out: Path) -> dict:
    first = sealed_read(roots[0] / "plan.json")
    contrasts, refs = references(roots[0], first)
    rows, parity, sources = [], {"rows": 0, "identical": 0, "differ": []}, []
    for root in roots:
        plan = sealed_read(root / "plan.json")
        if plan["suite"]["sha256"] != first["suite"]["sha256"]:
            raise ValueError(f"{root} was scored on another suite")
        theirs, their_refs = references(root, plan)
        if set(theirs) != set(contrasts) or any(
            their_refs[k]["time"] != refs[k]["time"]
            or their_refs[k][QUANTITY] != refs[k][QUANTITY]
            for k in refs
        ):
            raise ValueError(f"{root} has other kept trajectories or references")
        report = sealed_read(root / "suite_report.json")
        if report["plan"] != plan["artifact_sha256"]:
            raise ValueError(f"{root}: the suite report belongs to another plan")
        sources.append(
            {
                "root": str(root),
                "plan_sha256": plan["artifact_sha256"],
                "suite_report_sha256": report["artifact_sha256"],
            }
        )
        folders = subject_folders(root, plan)
        for r in report["rows"]:
            base = {
                k: r[k] for k in ("method", "repetition", "source_id", "case", "status")
            }
            if r["status"] != "complete":
                rows.append(base)
                continue
            item = contrasts[r["case"]]
            control = item["control"]
            curves = {
                k: sealed_read(folders[r["source_id"]] / f"{k}.json")["curves"][
                    QUANTITY
                ]
                for k in (r["case"], control)
            }
            args = (
                refs[r["case"]][QUANTITY],
                refs[control][QUANTITY],
                curves[r["case"]],
                curves[control],
            )
            study = anchor.scores(refs[r["case"]]["time"], *args, item["window"], 1.0)
            parity["rows"] += 1
            if abs(study["effect_rse"] - r["response_rse"]) <= TOLERANCE * max(
                1.0, abs(r["response_rse"])
            ):
                parity["identical"] += 1
            else:
                parity["differ"].append([r["source_id"], r["case"]])
            events = [
                {"name": e["name"]}
                | timing.event_timing(
                    refs[r["case"]]["time"],
                    refs[r["case"]][QUANTITY],
                    refs[control][QUANTITY],
                    np.asarray(curves[r["case"]]),
                    np.asarray(curves[control]),
                    e["window"],
                )
                for e in item["events"]
                if e["timing"]
            ]
            rows.append(
                base
                | {
                    "response_rse": r["response_rse"],
                    "sign_agreement": directions.sign_agreement(*args),
                    "events": events,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    record = sealed_write(
        out / "target_response.json",
        {
            "protocol": PROTOCOL,
            "sources": sources,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rse_parity": parity,
            "rows": rows,
            "private_reference_opened": True,
            "test_data_opened": False,
        },
    )
    with (out / "target_response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "repetition",
                "case",
                "status",
                "response_rse",
                "sign_agreement",
                "event",
                "shape_rse",
                "peak_error",
            ]
        )
        for r in rows:
            for e in r.get("events") or [{}]:
                writer.writerow(
                    [
                        r["method"],
                        r["repetition"],
                        r["case"],
                        r["status"],
                        r.get("response_rse"),
                        r.get("sign_agreement"),
                        e.get("name"),
                        e.get("shape_rse"),
                        e.get("peak_error"),
                    ]
                )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        required=True,
        help="a scored suite root; repeat for extra roots on the same suite",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/target-response-combined-cstr-hard-v1"),
    )
    a = parser.parse_args()
    record = main(a.root, a.out)
    print(record["artifact_sha256"], len(record["rows"]), "rows")
    parity = record["rse_parity"]
    print(json.dumps({k: parity[k] for k in ("rows", "identical")}))
    print(json.dumps(parity["differ"][:10]))
