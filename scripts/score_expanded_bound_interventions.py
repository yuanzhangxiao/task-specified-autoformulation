"""Score the expanded-bound perturbed T1 reference fits on the anchor panel.

The anchor panel (``scripts/iclr_anchor_fidelity.py``, confirmation schedules)
scored the privileged reference fits, whose coefficients were bounded to half
a log unit around their scales. The bound-sensitivity study
(``scripts/iclr_transfer_sensitivity.py``) refitted the same canonical
structure to perturbed T1-easy within wider bounds. This script replays the
panel's two T1 reference-fit subjects with the panel's own evaluator, frozen
schedules and saved reference responses, requires the replay to reproduce
every recorded curve and metric exactly, and then scores the
validation-selected fit of each expanded-bound profile the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_anchor_fidelity as panel
from scripts import iclr_reference_fits as fits

PROTOCOL = "expanded-bound-interventions-1"
CELLS = (fits.T1, fits.T1P)
FIRST_NEW_SUBJECT = 100
ORIGINAL = "original_bounds"


def load_panel(root: Path) -> tuple[dict, dict, dict]:
    """Read the frozen panel, its reference responses and its scored rows."""
    plan = sealed_read(root / "frozen_selection_manifest.json")
    if (
        plan["protocol"] != panel.VERSION
        or plan["source"] != panel.source_hash()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("evaluator or runtime differs from the panel's")
    refs = {}
    for cell in CELLS:
        refs[cell] = {}
        for case in plan["schedules"][cell]:
            record = sealed_read(root / "references" / cell / f"{case}.json")
            if record["plan"] != plan["artifact_sha256"]:
                raise ValueError("reference response from another panel")
            refs[cell][case] = record["reference"]
    scored = sealed_read(root / "scored_rows.json")
    if scored["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("scored rows from another panel")
    return plan, refs, scored


def reference_subjects(plan: dict) -> dict[int, dict]:
    """The panel's original-bound reference fit of each T1 cell, by subject."""
    found = {
        i: r
        for i, r in enumerate(plan["roster"])
        if r["source_protocol"] == fits.VERSION and r["benchmark_id"] in CELLS
    }
    if sorted(r["benchmark_id"] for r in found.values()) != sorted(CELLS):
        raise ValueError("expected one reference fit per T1 cell")
    return found


def expanded_subjects(transfer: Path, original: dict) -> list[dict]:
    """The validation-selected fit of each expanded-bound profile, as a subject."""
    plan = sealed_read(transfer / "transfer_plan.json")
    summary = sealed_read(transfer / "transfer_summary.json")
    if (
        summary["plan"] != plan["artifact_sha256"]
        or plan["cell"] != original["benchmark_id"]
        or plan["original_plan"] != original["fit"]["plan_sha256"]
        or plan["scripts"][1] != panel.source_hash()["iclr_reference_fits.py"]
    ):
        raise ValueError("bound study does not refit the panel's reference structure")
    rows = []
    for profile in plan["profiles"]:
        chosen = [
            r for r in summary["rows"] if r["profile"] == profile and r["selected"]
        ]
        if len(chosen) != 1:
            raise ValueError(f"no single selected fit for {profile}")
        start = chosen[0]["optimizer_start"]
        fit = sealed_read(
            transfer / "transfer-fits" / profile / str(start) / "fit.json"
        )
        if (
            fit["plan"] != plan["artifact_sha256"]
            or fit["artifact_sha256"] != chosen[0]["fit_seal"]
            or fit["status"] != "ready"
        ):
            raise ValueError("selected fit differs from the bound study's summary")
        rows.append(
            {
                **original,
                "method": "fitted_reference_structure_expanded_bounds",
                "profile": profile,
                "source_id": f"{original['benchmark_id']}:{profile}",
                "source_bundle_sha256": summary["artifact_sha256"],
                "fit": {
                    "artifact_sha256": fit["artifact_sha256"],
                    "plan_sha256": plan["artifact_sha256"],
                    "start": start,
                    "status": fit["status"],
                    "scores": fit["scores"],
                    "theta": fit["theta"],
                },
            }
        )
    return rows


def score(out: Path, subject: int, row: dict, plan: dict, refs: dict) -> dict:
    """Score one subject with the panel's evaluator, schedules and references."""
    cell = row["benchmark_id"]
    result = panel.evaluate_one(
        (
            out,
            subject,
            row,
            plan["schedules"][cell],
            refs[cell],
            plan["artifact_sha256"],
            plan["training_sd"][cell],
        )
    )
    return json.loads(json.dumps(result))


def curves(root: Path, subject: int, case: str) -> dict:
    record = sealed_read(root / "subjects" / f"{subject:04d}" / f"{case}.json")
    return {k: record.get(k) for k in ("curves", "verification_curves", "solver_gap")}


def rehearse(panel_root: Path, out: Path, plan: dict, refs: dict, scored: dict):
    """Require the replay to reproduce the panel's curves and metrics exactly."""
    report, rows = {}, []
    for subject, row in reference_subjects(plan).items():
        result = score(out, subject, row, plan, refs)
        recorded = [r for r in scored["rows"] if r["subject"] == subject]
        cases = plan["schedules"][row["benchmark_id"]]
        if result["failures"] or result["scores"] != recorded:
            raise ValueError(f"replayed metrics differ for subject {subject}")
        if any(
            curves(out, subject, k) != curves(panel_root, subject, k) for k in cases
        ):
            raise ValueError(f"replayed curves differ for subject {subject}")
        report[row["benchmark_id"]] = {
            "subject": subject,
            "fit_seal": row["fit"]["artifact_sha256"],
            "metric_rows": len(recorded),
            "cases": len(cases),
            "replay": "identical curves and metrics",
        }
        rows += [{"profile": ORIGINAL, **r} for r in result["scores"]]
    return report, rows


def validation_nmse(row: dict) -> float:
    return float(np.mean(row["fit"]["scores"]["validation"]))


def main(panel_root: Path, transfer: Path, out: Path) -> dict:
    plan, refs, scored = load_panel(panel_root)
    original = reference_subjects(plan)
    perturbed = next(r for r in original.values() if r["benchmark_id"] == fits.T1P)
    new = expanded_subjects(transfer, perturbed)
    with public._lock(out):
        identity = sealed_write(
            out / "plan.json",
            {
                "protocol": PROTOCOL,
                "panel_manifest_sha256": plan["artifact_sha256"],
                "panel_scored_rows_sha256": scored["artifact_sha256"],
                "evaluator": panel.source_hash(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "runtime": public._runtime(),
                "rehearsal_subjects": {
                    str(i): r["fit"]["artifact_sha256"] for i, r in original.items()
                },
                "new_subjects": {
                    str(FIRST_NEW_SUBJECT + k): {
                        "profile": r["profile"],
                        "start": r["fit"]["start"],
                        "fit_seal": r["fit"]["artifact_sha256"],
                        "validation_nmse": validation_nmse(r),
                    }
                    for k, r in enumerate(new)
                },
            },
        )
        rehearsal, rows = rehearse(panel_root, out / "rehearsal", plan, refs, scored)
        failures = {}
        for k, row in enumerate(new):
            result = score(out / "expanded", FIRST_NEW_SUBJECT + k, row, plan, refs)
            failures[row["profile"]] = result["failures"]
            rows += [{"profile": row["profile"], **r} for r in result["scores"]]
        record = sealed_write(
            out / "scores.json",
            {
                "protocol": PROTOCOL,
                "plan": identity["artifact_sha256"],
                "rehearsal": rehearsal,
                "failures": failures,
                "validation_nmse": {
                    ORIGINAL: validation_nmse(perturbed),
                    **{r["profile"]: validation_nmse(r) for r in new},
                },
                "rows": rows,
            },
        )
        panel.csv_table(out / "intervention_effects.csv", rows)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("output/iclr-anchor-results-2337694/fidelity"),
    )
    parser.add_argument(
        "--transfer", type=Path, default=Path("output/iclr-local-completion/historical")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/t1-perturbed-expanded-bound-interventions"),
    )
    a = parser.parse_args()
    record = main(a.panel, a.transfer, a.out)
    for r in record["rows"]:
        if r["readout"] == "target":
            print(
                r["benchmark_id"].removeprefix("phase_c_dalla_man_"),
                r["profile"],
                f"{r['case']} vs {r['control']}",
                r["quantity"],
                r["readout"],
                r["status"],
                r.get("effect_rse"),
                r.get("direction_correct"),
                r.get("peak_time_error"),
            )
    print(json.dumps(record["validation_nmse"]), json.dumps(record["failures"]))
