"""Score frozen external-baseline picks on the study's held-out test panel.

The study scored its frozen reactor models on six held-out test protocols of the
corrected CSTR-hard cell (``scripts/iclr_stress_evaluation.py``): each test
schedule and a quiet control were simulated once by the private reference and
sealed, and every model is rolled out from the initial reading alone. Test NMSE
is the mean over the six schedules of the squared error of the reactor
temperature divided by its training variance; the paired response RSE and its
direction come from the same rollouts.

This script puts the external baselines' picks, as frozen for the intervention
scoring (``scripts/score_baseline_interventions.py`` roots), through that same
``score`` with the study's sealed references, cell and training scales. Only the
prediction differs: each baseline is rolled out by the intervention scoring's
predictor under the study's forecast budget and two-solver agreement check (a
native one-step map has one replay). Every model of the study's frozen bundle on
the cell is scored through the unchanged evaluator as well, so the sealed
results of Full and the fixed-state control can be reproduced beside the rest.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.rebuttal.phase_c_baselines import load_cell
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_stress_data as data
from scripts import iclr_stress_evaluation as stress
from scripts import score_baseline_interventions as scoring

PROTOCOL = "baseline-test-panel-scores-1"
CELL = "sensor_00"
#: The study's forecast budget for one panel rollout (``stress.predict``).
SECONDS = 120.0


def baseline_predictor(row: dict):
    """A ``stress.score`` predictor that rolls out one frozen baseline pick."""
    native = row["model"]["semantics"] == "native_increment"
    solvers = ("native",) if native else ("Radau", "BDF")
    targets = row["model"]["context"]["targets"]

    def run(endpoint: dict, cell: dict, pair: dict) -> tuple[dict, dict]:
        runs = []
        for key in ("perturbed", "control"):
            trajectory = stress.public_input(pair[key], cell)
            curves = []
            for solver in solvers:
                value = scoring.predict(row, trajectory, solver, seconds=SECONDS)
                curves.append({c: np.asarray(value[c], dtype=float) for c in targets})
            runs.append(curves)
        primary, control = runs
        # The study's agreement rule for two independent solvers.
        for c in targets if not native else ():
            obs = np.asarray(pair["perturbed"]["targets"][c])
            abs_gap = np.max(np.abs(primary[0][c] - primary[1][c]))
            effect = primary[0][c] - control[0][c]
            effect_gap = np.max(np.abs(effect - (primary[1][c] - control[1][c])))
            truth_effect = obs - pair["control"]["targets"][c]
            if abs_gap > 1e-4 * max(
                1.0, max(abs(obs)), max(abs(primary[0][c]))
            ) or effect_gap > 5e-3 * max(
                1e-8, max(abs(effect)), max(abs(truth_effect))
            ):
                raise stress.SolverDisagreement(f"paired solver disagreement on {c}")
        return primary[0], control[0]

    return run


def same_training(release: Path, cell_root: Path, source: str) -> None:
    """The panel cell's training records are the ones the baselines fitted."""
    rows = read_seal(cell_root / "train.json")["rows"]
    fitted = load_cell(release, source).dataset.train.trajectories
    if len(rows) != len(fitted):
        raise ValueError("the panel cell has another number of training records")
    for row, tr in zip(rows, fitted, strict=True):
        same = (
            np.allclose(row["time"], tr.time, rtol=0, atol=1e-12)
            and all(
                np.allclose(row["targets"][c], tr.targets[c], rtol=0, atol=1e-9)
                for c in row["targets"]
            )
            and all(
                np.allclose(row["external_inputs"][c], tr.external_inputs[c], atol=1e-9)
                for c in row["external_inputs"]
            )
        )
        if not same:
            raise ValueError(f"training record {row['trajectory_id']} differs")


def score(
    out: Path,
    roots: list[Path],
    references: Path,
    cell_root: Path,
    manifest_path: Path,
    study_models: Path,
    release: Path,
) -> dict:
    """Score every study model and every frozen baseline pick on the panel."""
    manifest = sealed_read(manifest_path)
    cell = next(c for c in manifest["cells"] if c["cell"] == CELL)
    for name, digest in cell["public_files"].items():
        if data.digest(cell_root / name) != digest:
            raise ValueError(f"{name} is not the panel cell's public file")
    same_training(release, cell_root, cell["source"])
    sealed_refs = read_seal(references)
    refs = sealed_refs["records"]
    scales = data.train_scales(read_seal(cell_root / "train.json")["rows"])
    bundle = sealed_read(study_models)
    subjects = [("study", r) for r in bundle["rows"] if r["benchmark_id"] == CELL]
    for root in roots:
        plan = sealed_read(root / "plan.json")
        if plan["protocol"] != scoring.PROTOCOL:
            raise ValueError(f"{root} is not a frozen baseline intervention root")
        subjects += [
            ("baseline", r) for r in plan["rows"] if r["benchmark_id"] == cell["source"]
        ]
    rows = []
    for kind, r in subjects:
        key = f"{r['method']}-{r['repetition']}"
        path = out / "results" / f"{key}.json"
        if not path.exists():
            record = {
                "kind": kind,
                "method": r["method"],
                "repetition": r["repetition"],
                "source_id": r.get("source_id"),
                "evaluation_status": "model_unavailable",
            }
            if r["status"] == "ready":
                predictor = stress.predict if kind == "study" else baseline_predictor(r)
                record.update(stress.score(r, cell, refs, scales, predictor))
            sealed_write(path, record)
        rows.append(sealed_read(path))
    return sealed_write(
        out / "scores.json",
        {
            "protocol": PROTOCOL,
            "cell": CELL,
            "references_sha256": sealed_refs.get("sha256") or data.digest(references),
            "evaluator_sha256": data.digest(Path(stress.__file__)),
            "design_sha256": manifest["artifact_sha256"],
            "rows": rows,
            "test_data_opened": True,
        },
    )


def report(out: Path) -> list[dict]:
    """Per method: test NMSE by seed, its median, and the paired responses."""
    rows = sealed_read(out / "scores.json")["rows"]
    summary = []
    for method in dict.fromkeys(r["method"] for r in rows):
        items = sorted(
            (r for r in rows if r["method"] == method), key=lambda r: r["repetition"]
        )
        nmse = [
            r.get("test_nmse") if r.get("test_nmse") is not None else float("inf")
            for r in items
        ]
        rse = [
            c["response_rse"]
            for r in items
            for t in r.get("trajectories", [])
            for c in t.get("channels", {}).values()
            if c.get("response_rse") is not None
        ]
        directions = [
            c["signed_peak_direction_correct"]
            for r in items
            for t in r.get("trajectories", [])
            for c in t.get("channels", {}).values()
            if c.get("signed_peak_direction_correct") is not None
        ]
        summary.append(
            {
                "method": method,
                "kind": items[0]["kind"],
                "test_nmse_by_seed": nmse,
                "median_test_nmse": statistics.median(nmse),
                "complete": sum(
                    r.get("evaluation_status") == "complete" for r in items
                ),
                "seeds": len(items),
                "median_response_rse": statistics.median(rse) if rse else None,
                "correct_directions": f"{sum(directions)}/{len(directions)}",
            }
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--references", type=Path)
    parser.add_argument("--cell-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--study-models", type=Path)
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/phase-c-development-final-v2")
    )
    a = parser.parse_args()
    if a.stage == "score":
        result = score(
            a.out,
            a.root,
            a.references,
            a.cell_root,
            a.manifest,
            a.study_models,
            a.release,
        )
        print(result["artifact_sha256"], len(result["rows"]), "models")
    else:
        print(json.dumps(report(a.out), indent=1))
