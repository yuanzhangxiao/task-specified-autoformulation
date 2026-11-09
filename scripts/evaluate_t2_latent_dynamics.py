"""Measure frozen T2-hard models against physical insulin-action trajectories."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.baselines.d3_rollout import predict
from autoformalism.benchmarks.phase_b_generation import phase_b_protocols
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_man as dm
from autoformalism.rebuttal.mechanism_probes import _native
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel

CELL = "phase_b_dalla_man_t2_canonical_named_hard"
PROTOCOL = "t2-hard-physical-latent-probe-1"
CALIBRATION = ("train_basal", "train_meal_90g_t0", "train_insulin_pulse")
TARGETS = ("Gp", "I")
INPUTS = ("meal_event_g", "insulin_pmol_per_kg_min")


def schedules() -> list[dict]:
    """Use three training designs and two independently specified pulse contrasts."""
    registered = {p.protocol_id: p for p in phase_b_protocols("dalla_man", task="T2")}
    rows = []
    for name in CALIBRATION:
        spec = registered[name].specification
        rows.append(
            {
                "id": name,
                "meals": spec["meals"],
                "insulin": spec.get("insulin", []),
                "calibration": True,
            }
        )
    # Fasting keeps the meal-event representation identical (zero) under both
    # legacy and continuous forcing. For fresh interventions the simulator and
    # learned ODEs see the same linearly interpolated one-minute insulin input.
    for name, insulin in (
        ("fasting_control", []),
        ("early_short", [[50, 65, 0.5]]),
        ("late_long", [[120, 150, 0.35]]),
    ):
        rows.append({"id": name, "meals": [], "insulin": insulin, "calibration": False})
    return rows


def reference(case: dict, solver: str) -> dict:
    """Generate targets, private X and permitted channels from one reference."""
    result = dm.simulate_dalla_man(
        meals=tuple(map(tuple, case["meals"])),
        duration=300.0,
        dt=1.0,
        variant="original",
        external_forcing=dm.DallaManExternalForcing(
            insulin_pmol_per_kg_min=tuple(map(tuple, case["insulin"]))
        ),
        continuous_inputs=not case["calibration"],
        solver=ReferenceSolver(method=solver, rtol=1e-10, atol=1e-12, max_step=0.5),
    )
    return {
        "id": case["id"],
        "time": result.time.tolist(),
        "Gp": result.states[:, dm.STATE_INDEX["Gp"]].tolist(),
        "I": result.derived["I"].tolist(),
        "Uii": result.derived["Uii"].tolist(),
        "X": result.states[:, dm.STATE_INDEX["X"]].tolist(),
        "meal_event_g": result.meal_event_g.tolist(),
        "insulin_pmol_per_kg_min": result.derived["insulin_forcing"].tolist(),
    }


def solver_agreement(first: dict, second: dict, tolerance: float) -> float:
    """Require repeatable states and readouts, with one scale per series."""
    worst = 0.0
    for name in first:
        if name == "id":
            continue
        a, b = np.asarray(first[name]), np.asarray(second[name])
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"invalid {name} series in numerical replay")
        discrepancy = float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(a))))
        worst = max(worst, discrepancy)
    if worst > tolerance:
        raise ValueError(f"solver disagreement {worst:.5g} exceeds {tolerance:g}")
    return worst


def public_trajectory(record: dict) -> Trajectory:
    """Pass only allowed inputs, auxiliaries and initial target measurements."""
    time = np.array(record["time"])
    return Trajectory(
        trajectory_id=record["id"],
        time=time,
        targets={k: np.full_like(time, record[k][0], dtype=float) for k in TARGETS},
        auxiliaries={"Uii": np.array(record["Uii"])},
        external_inputs={k: np.array(record[k]) for k in INPUTS},
        fixed_covariates={},
        derivatives={},
    )


def rollout(row: dict, tr: Trajectory, solver: str, seconds: float) -> dict:
    """Return generated observations and latent trajectories of the frozen model."""
    context = ValidationContext.model_validate(row["context"])
    if (
        set(context.targets) != set(TARGETS)
        or set(context.auxiliaries) != {"Uii"}
        or set(context.external_inputs) != set(INPUTS)
        or context.lagged_targets
        or context.fixed_covariates
    ):
        raise ValueError("source runtime channel contract differs from named T2-hard")
    if row["semantics"] == "native_increment":
        model = _native(row)
        values = predict(model, tr, teacher_forced=False, seconds=seconds)
        return {k: values[:, model.states.index(k)].tolist() for k in TARGETS}
    if row["semantics"] != "continuous_time":
        raise ValueError("unknown execution semantics")
    candidate = CandidateModel.model_validate(row["candidate"])
    compiled = compile_candidate(candidate, context)
    sim = simulate_trajectory(
        compiled,
        tr,
        row["parameters"],
        row["initials"],
        FitConfig(
            integration_backend="solve_ivp",
            integration_method=solver,
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            allow_derivative_regression=False,
        ),
        deadline=monotonic() + seconds,
        reset_observed_states=False,
    )
    if not sim.success or sim.states is None:
        raise ValueError(sim.message or "incomplete rollout")
    values = {k: sim.predictions[k].tolist() for k in TARGETS}
    measured = set(context.targets) | set(context.auxiliaries)
    for mapping in candidate.observation_mappings:
        readout = ast.parse(mapping.expression, mode="eval").body
        if isinstance(readout, ast.Name):
            measured.add(readout.id)
    latent = {s.name for s in candidate.states if s.kind.value == "latent"} - measured
    for i, name in enumerate(compiled.state_names):
        if name in latent:
            values["latent." + name] = sim.states[i].tolist()
    return values


def calibrate(xs: dict[str, np.ndarray], truth: np.ndarray) -> dict:
    """Fit one scalar affine decoder using calibration data only (ties by name)."""
    best = None
    for name, series in sorted(xs.items()):
        center = float(np.mean(series))
        scale = float(np.max(np.abs(series - center)))
        if scale <= 1e-12 * max(1.0, abs(center)) or not np.isfinite(scale):
            continue
        z = (series - center) / scale
        y_mean = float(np.mean(truth))
        gain = float(z @ (truth - y_mean) / (z @ z))
        err = float(np.mean((y_mean + z * gain - truth) ** 2))
        if np.isfinite(err) and (best is None or err < best["calibration_mse"]):
            best = {
                "name": name,
                "center": center,
                "scale": scale,
                "gain": gain,
                "offset": y_mean,
                "calibration_mse": err,
            }
    return best or {"status": "no_varying_coordinate" if xs else "no_latent_coordinate"}


def decode(probe: dict, curves: dict) -> np.ndarray:
    """Use the calibration constants unchanged on an independent trajectory."""
    return probe["offset"] + probe["gain"] * (
        (np.asarray(curves[probe["name"]]) - probe["center"]) / probe["scale"]
    )


def metrics(time, truth, predicted, control_truth, control_predicted) -> dict:
    """Score the absolute curve and the same-model, same-preparation paired effect."""
    time, truth, predicted, control_truth, control_predicted = map(
        np.asarray, (time, truth, predicted, control_truth, control_predicted)
    )
    if (
        time.ndim != 1
        or len(time) < 2
        or np.any(np.diff(time) <= 0)
        or any(
            a.shape != time.shape or not np.isfinite(a).all()
            for a in (time, truth, predicted, control_truth, control_predicted)
        )
    ):
        raise ValueError(
            "paired trajectories must be finite and share an increasing grid"
        )
    delta, observed = predicted - control_predicted, truth - control_truth
    predicted_index, actual_index = (
        int(np.argmax(abs(delta))),
        int(np.argmax(abs(observed))),
    )
    expected, estimate = float(observed[actual_index]), float(delta[predicted_index])
    floor = 1e-8 * max(1.0, abs(truth).max(), abs(control_truth).max())
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        score = {
            "rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
            "reference_signed_peak": expected,
            "predicted_signed_peak": estimate,
            "effect_rse": None,
            "direction_correct": None,
            "peak_time_error_min": None,
        }
        if abs(expected) <= floor:
            return {**score, "status": "negligible_reference_effect"}
        score["effect_rse"] = float(
            np.mean(((delta - observed) / expected) ** 2)
            / np.mean((observed / expected) ** 2)
        )
        resolvable = abs(estimate) > floor
        score["direction_correct"] = bool(
            resolvable and (np.sign(expected) == np.sign(estimate))
        )
        if resolvable:
            score["peak_time_error_min"] = float(
                time[predicted_index] - time[actual_index]
            )
        return {**score, "status": "complete"}


def identity() -> dict:
    """Bind immutable work records to the analysis source and numerical runtime."""
    return {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def prepare(models: Path, root: Path) -> dict:
    """Freeze every supplied T2-hard subject before any reference access."""
    bundle = sealed_read(models)
    if bundle["protocol"] != "public-mechanism-model-bundle-1":
        raise ValueError("require a frozen public-mechanism-model-bundle-1")
    rows = sorted(
        (r for r in bundle["rows"] if r["benchmark_id"] == CELL),
        key=lambda r: (r["method"], r["repetition"]),
    )
    if not rows or len({(r["method"], r["repetition"]) for r in rows}) != len(rows):
        raise ValueError("require distinct planned T2-hard identities")
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "identity": identity(),
                "cases": schedules(),
                "rows": rows,
                "source_bundle_sha256": bundle["artifact_sha256"],
                "trajectory_seconds_per_solver": 20.0,
                "model_solvers": ["Radau", "BDF"],
                "reference_solvers": ["DOP853", "Radau"],
                "scope": "Frozen historical endpoints, evaluated on fresh synthetic "
                "fasting insulin infusions. X is the simulator's remote insulin "
                "action. An affine readout measures accessible trajectory information.",
                "decoder": "One latent coordinate selected by pooled MSE on three "
                "regenerated training schedules. The generated-I control uses the "
                "same affine calibration; model parameters are frozen.",
            },
        )


def references(root: Path, plan: dict) -> dict:
    """Generate and checkpoint references checked against a second solver."""
    records = {}
    for case in plan["cases"]:
        path = root / "reference" / (case["id"] + ".json")
        if not path.exists():
            a, b = (reference(case, solver) for solver in plan["reference_solvers"])
            gap = solver_agreement(a, b, 2e-6)
            sealed_write(
                path,
                {
                    "plan_sha256": plan["artifact_sha256"],
                    "reference": a,
                    "max_solver_discrepancy": gap,
                },
            )
        record = sealed_read(path)
        if record["plan_sha256"] != plan["artifact_sha256"]:
            raise ValueError("reference checkpoint belongs to another plan")
        records[case["id"]] = record["reference"]
    return records


def score_curves(refs, curves, alternates, case, name, probes) -> dict:
    """Check numerical agreement on the pulse difference before scoring it."""
    control = "fasting_control"
    actual = name if name in TARGETS else "X"
    r, r0 = (np.asarray(refs[k][actual]) for k in (case, control))

    def read(records, key):
        if name in TARGETS:
            return np.asarray(records[key][name])
        return decode(probes[name], records[key])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p, c = (read(curves, k) for k in (case, control))
        delta = p - c
        second = read(alternates, case) - read(alternates, control)
        amplitude = max(1e-8, float(abs(r - r0).max()), float(abs(delta).max()))
        gap = float(abs(second - delta).max()) / amplitude
        if gap > 5e-3:
            return {"status": "paired_solver_disagreement", "effect_solver_gap": gap}
        values = metrics(refs[case]["time"], r, p, r0, c)
    return {**values, "effect_solver_gap": gap}


def score_subject(args: tuple) -> dict:
    """Evaluate one fixed endpoint, retaining unavailable and failed cases."""
    root, plan_hash, index, row, refs, seconds, solvers = args
    folder = root / "subjects" / f"{index:03d}"
    curves = {}
    alternates = {}
    errors = {}
    for key, ref in refs.items():
        if row["status"] != "ready":
            errors[key] = row["status"]
            continue
        path = folder / (key + ".json")
        if not path.exists():
            marker = folder / (key + ".started.json")
            if marker.exists():
                if sealed_read(marker)["plan_sha256"] != plan_hash:
                    raise ValueError("rollout reservation belongs to another plan")
                rec = {"error": "interrupted_rollout"}
            else:
                sealed_write(marker, {"plan_sha256": plan_hash, "case": key})
                try:
                    tr = public_trajectory(ref)
                    first = rollout(row, tr, solvers[0], seconds)
                    gap = 0.0
                    if row["semantics"] == "continuous_time":
                        second = rollout(row, tr, solvers[1], seconds)
                        gap = solver_agreement(first, second, 1e-4)
                    rec = {"curves": first, "scaled_solver_discrepancy": gap}
                    if row["semantics"] == "continuous_time" and key not in CALIBRATION:
                        rec["crosscheck"] = second
                except (
                    ArithmeticError,
                    ValueError,
                    RuntimeError,
                    TimeoutError,
                    ModelValidationError,
                    RuntimeExpressionError,
                ) as exc:
                    rec = {"error": f"{type(exc).__name__}: {exc}"}
            sealed_write(path, {"plan_sha256": plan_hash, **rec})
        rec = sealed_read(path)
        if rec["plan_sha256"] != plan_hash:
            raise ValueError("candidate checkpoint belongs to another plan")
        if "curves" in rec:
            curves[key] = rec["curves"]
            alternates[key] = rec.get("crosscheck", rec["curves"])
        else:
            errors[key] = rec["error"]

    probes = {}
    calibration_ready = all(k in curves for k in CALIBRATION)
    if calibration_ready:
        truth = np.concatenate([refs[k]["X"] for k in CALIBRATION])
        names = list(curves[CALIBRATION[0]])
        for label, features in (
            ("X_from_latent", [n for n in names if n.startswith("latent.")]),
            ("X_from_generated_I", ["I"]),
        ):
            data = {
                n: np.concatenate([curves[k][n] for k in CALIBRATION]) for n in features
            }
            probes[label] = calibrate(data, truth)
    rows = []
    control = "fasting_control"
    for case in ("early_short", "late_long"):
        for name in (*TARGETS, "X_from_latent", "X_from_generated_I"):
            if row["status"] != "ready":
                value = {"status": row["status"]}
            elif case in errors or control in errors:
                value = {"status": "rollout_unavailable"}
            elif name not in TARGETS and not calibration_ready:
                value = {"status": "calibration_unavailable"}
            elif name not in TARGETS and "status" in probes[name]:
                value = probes[name]
            else:
                try:
                    value = score_curves(refs, curves, alternates, case, name, probes)
                except (ArithmeticError, ValueError) as exc:
                    value = {"status": "nonfinite_metric", "error": str(exc)}
            rows.append(
                {
                    "method": row["method"],
                    "repetition": row["repetition"],
                    "case": case,
                    "endpoint": name,
                    **value,
                }
            )
    return {
        "method": row["method"],
        "repetition": row["repetition"],
        "source_status": row["status"],
        "source_error": row.get("error"),
        "rollout_errors": errors,
        "probes": probes,
        "scores": rows,
    }


def run(root: Path, workers: int) -> dict:
    """Resume frozen references and per-model rollouts, then export all denominators."""
    if workers < 1:
        raise ValueError("workers must be positive")
    with public._lock(root):
        plan = sealed_read(root / "plan.json")
        if plan["protocol"] != PROTOCOL or plan["identity"] != identity():
            raise ValueError("protocol, evaluator or runtime changed since prepare")
        refs = references(root, plan)
        tasks = [
            (
                root,
                plan["artifact_sha256"],
                i,
                row,
                refs,
                plan["trajectory_seconds_per_solver"],
                plan["model_solvers"],
            )
            for i, row in enumerate(plan["rows"])
        ]
        if workers == 1:
            results = [score_subject(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(score_subject, tasks))
        report = sealed_write(
            root / "summary.json",
            {
                "protocol": PROTOCOL,
                "plan_sha256": plan["artifact_sha256"],
                "cell": CELL,
                "scope": plan["scope"],
                "decoder": plan["decoder"],
                "models": results,
                "test_split_opened": False,
                "model_refits": 0,
                "llm_calls": 0,
            },
        )
        rows = [item for m in results for item in m["scores"]]
        keys = list(dict.fromkeys(k for item in rows for k in item))
        with (root / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("--models", type=Path, required=True)
    prep.add_argument("--root", type=Path, required=True)
    execute = stages.add_parser("run")
    execute.add_argument("--root", type=Path, required=True)
    execute.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        plan = prepare(args.models, args.root)
        print(f"Frozen {len(plan['rows'])} subjects; plan {plan['artifact_sha256']}")
    else:
        report = run(args.root, args.workers)
        print(f"Scored {len(report['models'])} subjects. Results in {args.root}")


if __name__ == "__main__":
    main()
