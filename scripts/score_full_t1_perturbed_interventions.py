"""Score the perturbed T1-easy Full models on the anchor panel.

The planning agent ran Full on perturbed T1-easy with the anchor study's
settings (search array 2353678) and exported the three frozen seeds. This
script checks that export against its plan and freeze receipts and against
the public data the panel checks, and recomputes each seed's recorded
training and validation NMSE with the reference fits' definition (free-rollout
squared error over the training variance, averaged over trajectories), so
that both can share one column. It then replays the panel's two T1 reference
fits and every Full subject the panel scored, requiring identical records and
metric rows, and scores the three seeds with the evaluator, schedules and
saved references that scored the textbook fits in
``score_expanded_bound_interventions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as panel
from scripts import score_expanded_bound_interventions as textbook

PROTOCOL = "t1-perturbed-full-interventions-1"
CONTROLLED = "iclr-controlled-search-models-1"
CELL = textbook.fits.T1P
SEEDS = (0, 1, 2)
FIRST_SUBJECT = 200
TOLERANCE = 1e-6


def references(root: Path, plan: dict, cell: str) -> dict:
    """The panel's saved reference response of every case of one cell."""
    refs = {}
    for case in plan["schedules"][cell]:
        record = sealed_read(root / "references" / cell / f"{case}.json")
        if record["plan"] != plan["artifact_sha256"]:
            raise ValueError("reference response from another panel")
        refs[case] = record["reference"]
    return refs


def load_full(source: Path, public_root: Path) -> tuple[dict, list[dict]]:
    """The frozen Full seeds, checked against their receipts and public data."""
    bundle = sealed_read(source)
    if bundle["protocol"] != CONTROLLED:
        raise ValueError("not a controlled search export")
    panel.verify_controlled_bundle(source, bundle)
    rows = sorted(bundle["rows"], key=lambda r: r["repetition"])
    if [
        (r["benchmark_id"], r["method"], r["repetition"], r["status"]) for r in rows
    ] != [(CELL, "full", s, "ready") for s in SEEDS]:
        raise ValueError("expected three ready Full seeds on perturbed T1-easy")
    subjects = []
    for r in rows:
        if r.get("semantics", "continuous_time") != "continuous_time":
            raise ValueError("unsupported search semantics")
        checked = panel.canonicalize(r, public_root, None)
        if checked.get("source_benchmark_id") != CELL:
            raise ValueError("public data identity was not checked")
        subjects.append(
            {
                **checked,
                "source_protocol": CONTROLLED,
                "source_bundle_sha256": bundle["artifact_sha256"],
                "direct_readouts": {},
            }
        )
    return bundle, subjects


def rollout_nmse(row: dict, split, scales: dict) -> float:
    """Mean over trajectories of each target's rollout MSE over training variance."""
    model = compile_candidate(
        CandidateModel.model_validate(row["candidate"]),
        ValidationContext.model_validate(row["context"]),
    )
    config = FitConfig(
        integration_method="Radau",
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
        allow_derivative_regression=False,
    )
    errors = []
    for tr in split.rows:
        sim = simulate_trajectory(
            model,
            tr,
            row["parameters"],
            row["initials"],
            config,
            reset_observed_states=False,
        )
        if not sim.success:
            return float("inf")
        errors.append(
            [
                np.mean(((np.asarray(sim.predictions[n]) - tr.targets[n]) / s) ** 2)
                for n, s in scales.items()
            ]
        )
    return float(np.mean(np.mean(errors, axis=0)))


def check_nmse(subjects: list[dict], public_root: Path, scales: dict) -> dict:
    """Require each recorded NMSE to match the reference fits' definition."""
    _, training, validation = load_public_cell(public_root / CELL)
    report = {}
    for row in subjects:
        entry = {}
        for split, data, recorded in (
            ("training", training, row["training_nmse"]),
            ("validation", validation, row["validation_nmse"]),
        ):
            value = rollout_nmse(row, data, scales)
            if not abs(value - recorded) <= TOLERANCE * abs(recorded):
                raise ValueError(f"seed {row['repetition']} {split} NMSE differs")
            entry[split] = {"recorded": recorded, "recomputed": value}
        report[str(row["repetition"])] = entry
    return report


def replay_full_subjects(root: Path, out: Path, plan: dict, scored: dict) -> dict:
    """Replay every Full subject the panel scored; require identical results."""
    report = {}
    for i, row in enumerate(plan["roster"]):
        if row["source_protocol"] != CONTROLLED or row["method"] != "full":
            continue
        cell = row["benchmark_id"]
        cases = plan["schedules"][cell]
        result = panel.evaluate_one(
            (
                out,
                i,
                row,
                cases,
                references(root, plan, cell),
                plan["artifact_sha256"],
                plan["training_sd"][cell],
            )
        )
        recorded = [r for r in scored["rows"] if r["subject"] == i]
        if json.loads(json.dumps(result["scores"])) != recorded:
            raise ValueError(f"replayed metrics differ for subject {i}")
        for case in cases:
            name = Path("subjects") / f"{i:04d}" / f"{case}.json"
            if sealed_read(out / name) != sealed_read(root / name):
                raise ValueError(f"replayed record differs for subject {i} {case}")
        report[str(i)] = {
            "benchmark_id": cell,
            "repetition": row["repetition"],
            "metric_rows": len(recorded),
            "cases": len(cases),
            "failed_cases": sorted(result["failures"]),
            "replay": "identical records and metrics",
        }
    if len(report) != 6:
        raise ValueError("expected six Full subjects on the panel")
    return report


def load_textbook(root: Path, plan: dict, scored: dict) -> dict:
    """The textbook fits' sealed scores, from this panel and this scorer."""
    identity = sealed_read(root / "plan.json")
    record = sealed_read(root / "scores.json")
    if (
        record["plan"] != identity["artifact_sha256"]
        or identity["panel_manifest_sha256"] != plan["artifact_sha256"]
        or identity["panel_scored_rows_sha256"] != scored["artifact_sha256"]
        or identity["script_sha256"]
        != hashlib.sha256(Path(textbook.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("textbook scores come from another panel or scorer")
    return record


def gp_contrasts(rows: list[dict]) -> list[dict]:
    """The perturbed cell's Gp target rows: fed vs fasting, split vs fed."""
    chosen = [
        r
        for r in rows
        if r["benchmark_id"] == CELL
        and r["quantity"] == "Gp"
        and r["readout"] == "target"
    ]
    if sorted((r["case"], r["control"]) for r in chosen) != [
        ("fed", "fasting"),
        ("split", "fed"),
    ]:
        raise ValueError("expected one Gp row per meal contrast")
    return chosen


def main(panel_root: Path, bundle: Path, public_root: Path, fits: Path, out: Path):
    plan, refs, scored = textbook.load_panel(panel_root)
    if plan["mode"] != "confirmation":
        raise ValueError("expected the confirmation panel")
    export, subjects = load_full(bundle, public_root)
    tb = load_textbook(fits, plan, scored)
    nmse = check_nmse(subjects, public_root, plan["training_sd"][CELL])
    with public._lock(out):
        identity = sealed_write(
            out / "plan.json",
            {
                "protocol": PROTOCOL,
                "panel_manifest_sha256": plan["artifact_sha256"],
                "panel_scored_rows_sha256": scored["artifact_sha256"],
                "textbook_scores_sha256": tb["artifact_sha256"],
                "full_export_sha256": export["artifact_sha256"],
                "full_plan_sha256": export["plan_sha256"],
                "full_freeze_sha256": export["selection_freeze_sha256"],
                "evaluator": panel.source_hash(),
                "textbook_script_sha256": hashlib.sha256(
                    Path(textbook.__file__).read_bytes()
                ).hexdigest(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "runtime": public._runtime(),
                "public_files": subjects[0]["public_file_sha256"],
                "nmse_check": nmse,
                "subjects": {
                    str(FIRST_SUBJECT + r["repetition"]): {
                        "seed": r["repetition"],
                        "source_id": r["source_id"],
                        "model_sha256": r["model_sha256"],
                    }
                    for r in subjects
                },
            },
        )
        rehearsal = {
            "reference_fits": textbook.rehearse(
                panel_root, out / "rehearsal", plan, refs, scored
            )[0],
            "full_subjects": replay_full_subjects(
                panel_root, out / "rehearsal", plan, scored
            ),
        }
        rows, failures = [], {}
        for r in subjects:
            seed = r["repetition"]
            result = textbook.score(out / "full", FIRST_SUBJECT + seed, r, plan, refs)
            failures[str(seed)] = result["failures"]
            rows += [{"model": "full", "seed": seed, **s} for s in result["scores"]]
        table = [
            {"model": p, "seed": None, "validation_nmse": v, **s}
            for p, v in tb["validation_nmse"].items()
            for s in gp_contrasts([r for r in tb["rows"] if r["profile"] == p])
        ] + [
            {
                "model": "full",
                "seed": r["repetition"],
                "validation_nmse": r["validation_nmse"],
                **s,
            }
            for r in subjects
            for s in gp_contrasts([x for x in rows if x["seed"] == r["repetition"]])
        ]
        record = sealed_write(
            out / "scores.json",
            {
                "protocol": PROTOCOL,
                "plan": identity["artifact_sha256"],
                "rehearsal": rehearsal,
                "failures": failures,
                "rows": rows,
                "gp_contrasts": table,
            },
        )
        panel.csv_table(out / "intervention_effects.csv", rows)
        panel.csv_table(out / "gp_contrasts.csv", table)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("output/iclr-anchor-results-2337694/fidelity"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("output/iclr-t1-perturbed-full-2353678/frozen/frozen_models.json"),
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=Path("output/iclr-nonbaseline-reader/phase-c-public"),
    )
    parser.add_argument(
        "--textbook",
        type=Path,
        default=Path("outputs/t1-perturbed-expanded-bound-interventions"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/t1-perturbed-full-interventions")
    )
    a = parser.parse_args()
    record = main(a.panel, a.bundle, a.public, a.textbook, a.out)
    for r in record["gp_contrasts"]:
        print(
            r["model"],
            r["seed"],
            f"{r['case']} vs {r['control']}",
            r["status"],
            r["validation_nmse"],
            r.get("effect_rse"),
            r.get("reference_signed_peak"),
            r.get("predicted_signed_peak"),
            r.get("peak_time_error"),
            r.get("direction_correct"),
        )
    print(json.dumps(record["failures"]))
