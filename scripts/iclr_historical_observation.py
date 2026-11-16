"""Evaluate archived Full T1/T2 easy/hard models on matched held-out schedules.

This is a retrospective assay of the submitted six-visit incumbents. It keeps
their legacy meal-event contract, equations, coefficients and shared initializers.
It does not turn historical fits into prospective Phase C confirmation runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash
from scripts import iclr_anchor_fidelity as anchor
from scripts import iclr_scaling_response as metrics
from scripts.iclr_retained_mechanisms import read_training
from scripts.iclr_search_history import INVENTORY
from scripts.iclr_submission_evidence import ARCHIVE, CUTOFF, PLAN, REPO, load_archive

PROTOCOL = "iclr-historical-observation-1"
CELLS = tuple(
    f"phase_b_dalla_man_{task}_canonical_named_{tier}"
    for task in ("t1", "t2")
    for tier in ("easy", "hard")
)
SCORES = ("test_nmse", "response_rse", "directional_accuracy", "shape_error")


def task_of(cell: str) -> str:
    """Keep only the four canonical named observation regimes."""
    if cell not in CELLS:
        raise ValueError("unsupported historical observation cell")
    return "T1" if "_t1_" in cell else "T2"


def schedules(task: str, panel: str) -> dict:
    """Reuse existing test/confirmation schedules with the historical contract."""
    if task not in ("T1", "T2"):
        raise ValueError("unknown task")
    if panel == "confirmation_schedules":
        cell = anchor.fits.T1 if task == "T1" else anchor.LEGACY
        result = {
            k: v
            for k, v in anchor.cases(cell, "confirmation").items()
            if v["role"] != "train_calibration"
        }
        for item in result.values():
            p = item["protocol"]
            p["input_contract"] = "legacy-events-1"
            p["input_names"] = [
                "meal_event_g" if n == "meal_rate_g_per_min" else n
                for n in p["input_names"]
            ]
        return result
    if panel != "original_heldout":
        raise ValueError("unknown panel")
    result = {}
    for p in phase_b_protocols("dalla_man", task=task):
        if p.split != "test":
            continue
        control = p.protocol_id + "_control"
        quiet = {"meals": []}
        if "initial_shift" in p.specification:
            quiet["initial_shift"] = p.specification["initial_shift"]
        result[control] = {
            "role": "control",
            "protocol": p.model_copy(
                update={
                    "protocol_id": control,
                    "specification": quiet,
                }
            ).model_dump(mode="json"),
        }
        result[p.protocol_id] = {
            "role": "contrast",
            "control": control,
            "protocol": p.model_dump(mode="json"),
        }
    return result


def retained_rows(search: dict, inventory: dict) -> list[dict]:
    """Bind every planned incumbent to the original selected-fit receipt."""
    if inventory["plan_sha256"] != PLAN or inventory["test_data_opened"] is not False:
        raise ValueError("wrong historical inventory")
    snapshots = {
        r["task_id"]: r
        for r in search["rows"]
        if r["round"] == CUTOFF
        and r["cell"] in CELLS
        and r["arm"] == "full"
        and r["task_id"].endswith("_c1v1s1")
    }
    if {(r["cell"], r["seed"]) for r in snapshots.values()} != {
        (c, s) for c in CELLS for s in (0, 1)
    } or len(snapshots) != 8:
        raise ValueError("incomplete planned historical denominator")
    selected = {}
    for entry in inventory["occurrences"]:
        task = entry["task"]["task_id"]
        if task not in snapshots or entry["round"] != CUTOFF:
            continue
        if entry["role"] != "selected":
            continue
        if (
            task in selected
            or entry["source_sha256"] != snapshots[task]["result_sha256"]
        ):
            raise ValueError("duplicate or mismatched selected receipt")
        selected[task] = entry
    rows = []
    for task, old in sorted(snapshots.items()):
        row = {
            "source_id": task,
            "cell": old["cell"],
            "seed": old["seed"],
            "round": CUTOFF,
            "source_result_sha256": old["result_sha256"],
            "recorded_validation_nmse": old["retained_validation_nmse"],
            "status": "no_retained_model"
            if old["retained_validation_nmse"] is None
            else "model_export_missing",
        }
        if task in selected:
            entry = selected[task]
            model, fit = inventory["models"][entry["model_id"]], entry["fit"]
            if (
                content_hash(model) != entry["model_id"]
                or model["cell"] != old["cell"]
                or model["parameters"] != fit["parameters"]
                or model["lowered_candidate_sha256"] != fit["lowered_candidate_sha256"]
                or model["parameter_status"] != "saved_fit_vector"
                or fit["status"] != "complete"
                or fit["validation_initials_fitted"] is not False
                or fit["training_only_parameter_estimation"] is not True
                or fit["validation"]["normalized_mse"]
                != old["retained_validation_nmse"]
            ):
                raise ValueError("model and archived fitting receipt disagree")
            row.update(
                status="ready",
                model_id=entry["model_id"],
                candidate=model["candidate"],
                context=model["lowered_context"],
                parameters=model["parameters"],
                initials={},
                fit_identity=fit["identity"],
                initialization_plan=model["initialization_plan"],
            )
        rows.append(row)
    return rows


def freeze(root: Path, archive: Path, inventory: Path, public_root: Path) -> dict:
    """Freeze selection, schedules, training normalization and numerical sources."""
    search, campaign = load_archive(archive)
    inv = sealed_read(inventory)
    rows = retained_rows(search, inv)
    scales, hashes, training = {}, {}, {}
    for cell in CELLS:
        row = next(r for r in rows if r["cell"] == cell and r["status"] == "ready")
        path = public_root / cell / "train.csv"
        expected = campaign["public_assets"][cell]["train.csv"]
        data = read_training(path, expected, json.dumps(row["context"], sort_keys=True))
        training[cell] = data.trajectories
        scales[cell] = {
            c: float(np.std(np.concatenate([r.targets[c] for r in data.trajectories])))
            for c in row["context"]["targets"]
        }
        hashes[cell] = expected
    for task in ("t1", "t2"):
        easy, hard = [
            f"phase_b_dalla_man_{task}_canonical_named_{tier}"
            for tier in ("easy", "hard")
        ]
        for a, b in zip(training[easy], training[hard], strict=True):
            if a.trajectory_id != b.trajectory_id or not np.array_equal(a.time, b.time):
                raise ValueError("easy/hard training schedules differ")
            for c in b.targets:
                if not np.array_equal(a.targets[c], b.targets[c]):
                    raise ValueError("easy/hard common targets differ")
    return metrics.write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "cohort": "six visits, Full prompt, c1v1s1, pre-pruning",
            "source_archive": metrics.digest(archive),
            "inventory": inv["artifact_sha256"],
            "rows": rows,
            "training_sd": scales,
            "training_files": hashes,
            "schedules": {
                panel: {task: schedules(task, panel) for task in ("T1", "T2")}
                for panel in ("original_heldout", "confirmation_schedules")
            },
            "implementation": metrics.digest(Path(__file__)),
            "metric_source": metrics.digest(Path(metrics.__file__)),
            "source": anchor.source_hash(),
            "runtime": public._runtime(),
            "retraining": False,
            "historical_event_contract_preserved": True,
            "interpretation": (
                "Retrospective evaluation, not prospective Phase C confirmation"
            ),
        },
    )


def public_trajectory(row: dict, record: dict) -> Trajectory:
    """Expose only declared forcing, auxiliaries and initial target readings."""
    ctx = ValidationContext.model_validate(row["context"])
    if ctx.fixed_covariates or ctx.lagged_targets:
        raise ValueError("unsupported historical channel contract")
    t = np.asarray(record["time"])
    fields = record["fields"]
    return Trajectory(
        trajectory_id="heldout",
        time=t,
        targets={c: np.full_like(t, fields[c][0], dtype=float) for c in ctx.targets},
        auxiliaries={c: np.asarray(fields[c]) for c in ctx.auxiliaries},
        external_inputs={c: np.asarray(fields[c]) for c in ctx.external_inputs},
        fixed_covariates={},
        derivatives={},
    )


def reference(protocol: dict) -> dict:
    """Generate and independently verify the reference under the legacy contract."""
    p = PhaseBProtocol.model_validate(protocol)
    if p.input_contract != "legacy-events-1":
        raise ValueError("historical meals must retain their event encoding")
    results = [
        simulate_phase_b(p, solver=ReferenceSolver(method=m))
        for m in ("DOP853", "Radau")
    ]
    first, second = results
    if not np.allclose(first.states, second.states, rtol=2e-6, atol=2e-6):
        raise ValueError("reference state solvers disagree")
    fields = {n: first.states[:, i] for i, n in enumerate(first.state_names)}
    fields.update(first.derived)
    fields.update({n: first.inputs[:, i] for i, n in enumerate(first.input_names)})
    names = ("Gp", "I", "U", "EGP", "Uii", "E", "Gt", *first.input_names)
    for n in ("I", "U", "EGP", "Uii", "E"):
        if not np.allclose(first.derived[n], second.derived[n], rtol=2e-6, atol=2e-6):
            raise ValueError("reference readout solvers disagree")
    return {
        "time": first.time.tolist(),
        "fields": {n: fields[n].tolist() for n in names},
    }


def cached_reference(root: Path, plan: dict, protocol: dict) -> dict:
    """Deduplicate reference schedules and reject changed checkpoint bindings."""
    key = content_hash(protocol)
    path = root / "references" / f"{key}.json"
    binding = {"plan": plan["artifact_sha256"], "protocol": protocol}
    if not path.exists():
        metrics.write(path, binding | {"reference": reference(protocol)})
    value = sealed_read(path)
    if any(value[k] != v for k, v in binding.items()):
        raise ValueError("stale reference checkpoint")
    return value


def predict(row: dict, record: dict) -> dict:
    """Integrate the saved equations and initializers with two independent solvers."""
    tr = public_trajectory(row, record)
    model = compile_candidate(
        CandidateModel.model_validate(row["candidate"]),
        ValidationContext.model_validate(row["context"]),
    )
    curves = []
    for method in ("Radau", "BDF"):
        result = simulate_trajectory(
            model,
            tr,
            row["parameters"],
            row["initials"],
            FitConfig(
                integration_method=method,
                relative_tolerance=1e-8,
                absolute_tolerance=1e-10,
                allow_derivative_regression=False,
            ),
            deadline=monotonic() + 120,
            reset_observed_states=False,
        )
        if not result.success:
            raise ValueError(f"{method}: {result.message}")
        curves.append({k: v.tolist() for k, v in result.predictions.items()})
    gap = anchor.probe.solver_agreement(curves[0], curves[1], 1e-4)
    return {"predicted": curves[0], "alternate": curves[1], "solver_gap": gap}


def cached_prediction(root: Path, plan: dict, row: dict, ref: dict) -> dict:
    """Save successful and failed rollouts with model/reference/plan identities."""
    binding = {
        "plan": plan["artifact_sha256"],
        "model": row["model_id"],
        "reference": ref["artifact_sha256"],
    }
    path = root / "predictions" / row["source_id"] / f"{ref['artifact_sha256']}.json"
    if not path.exists():
        try:
            result = {"status": "complete", **predict(row, ref["reference"])}
        except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as exc:
            result = {"status": "rollout_unavailable", "reason": str(exc)}
        metrics.write(path, binding | result)
    value = sealed_read(path)
    if any(value[k] != v for k, v in binding.items()):
        raise ValueError("stale model checkpoint")
    return value


def score_pair(
    a: dict, b: dict, ref: dict, control: dict, target: str, scale: float, panel: str
) -> dict:
    """Score one target/contrast after the original paired numerical agreement gate."""
    if a["status"] != "complete" or b["status"] != "complete":
        return {"status": "rollout_unavailable"}
    if not np.array_equal(ref["time"], control["time"]):
        raise ValueError("paired reference time grids differ")
    y, c = [np.asarray(r["fields"][target]) for r in (ref, control)]
    p, q = [np.asarray(r["predicted"][target]) for r in (a, b)]
    alt = np.subtract(a["alternate"][target], b["alternate"][target])
    effect, predicted = y - c, p - q
    gap = float(
        np.max(abs(alt - predicted))
        / max(1e-8, np.max(abs(effect)), np.max(abs(predicted)))
    )
    if gap > 5e-3:
        return {"status": "effect_solver_disagreement", "solver_gap": gap}
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training normalization")
    diagnostic = metrics.response_metrics(y, c, p, q)
    active = diagnostic["status"] == "complete"
    time = np.asarray(ref["time"])
    rse = None
    if active:
        numerator, denominator = (predicted - effect) ** 2, effect**2
        rse = float(
            np.trapezoid(numerator, time) / np.trapezoid(denominator, time)
            if panel == "original_heldout"
            else numerator.sum() / denominator.sum()
        )
    return {
        "status": "complete",
        "effect_status": diagnostic["status"],
        "test_nmse": float(np.mean((p - y) ** 2) / scale**2),
        "response_rse": rse,
        "directional_accuracy": diagnostic.get("sign_agreement"),
        "shape_error": diagnostic.get("shape_rse"),
        "shape_gain": diagnostic.get("best_scale"),
        "effect_solver_gap": gap,
    }


def pool(rows: list[dict]) -> dict:
    """Keep all-case numerical failures; pool only defined response metrics."""
    if not rows or any(r["status"] != "complete" for r in rows):
        return {"status": "evaluation_incomplete"}
    result = {"status": "complete"}
    for key in SCORES:
        values = [r[key] for r in rows if r[key] is not None]
        if not values:
            raise ValueError("panel has no defined metric")
        result[key] = (statistics.median if key in SCORES[:2] else statistics.mean)(
            values
        )
    return result


def run(root: Path) -> dict:
    """Score every planned historical seed, retaining missing and failed outcomes."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["implementation"] != metrics.digest(Path(__file__))
        or plan["source"] != anchor.source_hash()
        or plan["metric_source"] != metrics.digest(Path(metrics.__file__))
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("frozen evaluator source or runtime changed")
    records, rows = [], []
    for panel, task_schedules in plan["schedules"].items():
        for endpoint in plan["rows"]:
            task = task_of(endpoint["cell"])
            schedule = task_schedules[task]
            row = {k: endpoint[k] for k in ("source_id", "cell", "seed", "status")}
            row.update(panel=panel, task=task, model_id=endpoint.get("model_id"))
            if endpoint["status"] != "ready":
                rows.append(row)
                continue
            measured = []
            for case, item in schedule.items():
                if item["role"] != "contrast":
                    continue
                refs = [
                    cached_reference(root, plan, schedule[k]["protocol"])
                    for k in (case, item["control"])
                ]
                predictions = [cached_prediction(root, plan, endpoint, r) for r in refs]
                for target in ("Gp",) if task == "T1" else ("Gp", "I"):
                    score = score_pair(
                        *predictions,
                        *(r["reference"] for r in refs),
                        target,
                        plan["training_sd"][endpoint["cell"]][target],
                        panel,
                    )
                    measured.append(score)
                    records.append(
                        row
                        | {
                            "case": case,
                            "control": item["control"],
                            "target": target,
                            "references": [r["artifact_sha256"] for r in refs],
                            **score,
                        }
                    )
            row.update(pool(measured))
            row["case_target_records"] = len(measured)
            row["active_responses"] = sum(
                r.get("effect_status") == "complete" for r in measured
            )
            rows.append(row)
            print(f"{panel}: {endpoint['source_id']}: {row['status']}", flush=True)
    groups = []
    for panel in plan["schedules"]:
        for cell in CELLS:
            cohort = [r for r in rows if r["panel"] == panel and r["cell"] == cell]
            complete = [r for r in cohort if r["status"] == "complete"]
            groups.append(
                {
                    "panel": panel,
                    "cell": cell,
                    "planned": len(cohort),
                    "retained": sum(r.get("model_id") is not None for r in cohort),
                    "scored": len(complete),
                    "statistics": {
                        k: metrics.interval(
                            [r[k] for r in complete],
                            higher_is_better=k == "directional_accuracy",
                        )
                        if complete
                        else None
                        for k in SCORES
                    },
                }
            )
    result = metrics.write(
        root / "summary.json",
        {
            "protocol": PROTOCOL,
            "plan": plan["artifact_sha256"],
            "rows": rows,
            "case_rows": records,
            "groups": groups,
            "definitions": {
                "target": "T1: Gp; T2: common Gp,I (U is not pooled into the easy row)",
                "aggregation": (
                    "Within seed: median NMSE/RSE, mean accuracy/shape over "
                    "intervention-target pairs; across seeds: median [worst,best]"
                ),
                "nmse": (
                    "Mean squared prediction error / original training-channel "
                    "variance; identical scales across tiers within task"
                ),
                "response_rse": (
                    "Paired response squared-error / reference effect energy; "
                    "trapezoid integral for original tests, sample sums for "
                    "later confirmation schedules"
                ),
                "direction": (
                    "Continuous magnitude-weighted sign agreement; no thresholding"
                ),
                "shape": "Whole-response nonnegative gain-adjusted RSE, no time shift",
                "zero_effect": (
                    "Initial-condition-only original test is included in NMSE; "
                    "undefined zero-reference response metrics are excluded"
                ),
                "scope": (
                    "Retrospective six-visit pre-pruning Full models with legacy "
                    "event inputs and references; not the new continuous-rate "
                    "Full cohort"
                ),
            },
        },
    )
    lines = ["Historical Full: observed-variable sensitivity", ""]
    for panel in plan["schedules"]:
        lines += [
            panel,
            "",
            "| Model | Scored/planned | NMSE | RSE | Directional accuracy "
            "| Shape error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for g in (g for g in groups if g["panel"] == panel):
            label = (
                g["cell"]
                .replace("phase_b_dalla_man_", "")
                .replace("_canonical_named_", " ")
                .upper()
            )
            cells = [label, f"{g['scored']}/{g['planned']}"]
            for k in SCORES:
                z = g["statistics"][k]
                cells.append(
                    "unavailable"
                    if z is None
                    else f"{z['median']:.2f} [{z['worst']:.2f}, {z['best']:.2f}]"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"{k}: {v}" for k, v in result["definitions"].items()]
    (root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Run a reproducible local replay without cluster or provider access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--public-root", type=Path, default=REPO)
    args = parser.parse_args()
    freeze(args.root, args.archive, args.inventory, args.public_root)
    run(args.root)


if __name__ == "__main__":
    main()
