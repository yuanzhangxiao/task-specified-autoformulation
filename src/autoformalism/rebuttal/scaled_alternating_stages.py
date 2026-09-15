"""Killable stages with immutable results and preserved physical incumbents."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting.conditional_collocation import ConditionalCollocation
from autoformalism.fitting.conditional_optimizer import load_point, optimize
from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
from autoformalism.fitting.sensitivity_probe import symbolic_rollout
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    write_json,
)
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.scaled_alternating import (
    ScaledAlternatingPlan,
    checked,
    inputs,
    verify,
)
from autoformalism.staged_topology import content_hash


def task_identity(frozen, index):
    if index not in (0, 1):
        raise ValueError("index outside paired matrix")
    return content_hash({"freeze": frozen["identity"], "arm": frozen["tasks"][index]})


def best_of(*stages):
    points = [s["best"] for s in stages if s.get("best")]
    return min(points, key=lambda p: p["cost"]) if points else None


def oracle_for(system, training, common, plan, root, seconds, sensitivities):
    budget = EvaluationBudget(
        monotonic() + seconds, plan.refinement_calls if sensitivities else 2
    )
    return GuardedOracle(
        system,
        training,
        common["output_scale"],
        plan.settings(),
        root / "calls",
        budget.deadline,
        sensitivities=sensitivities,
        budget=budget,
        point_seconds=plan.sensitivity_point_seconds
        if sensitivities
        else plan.screen_point_seconds,
        reject_invalid_trials=True,
    )


def stage(output: Path, index: int, name: str):
    """A stage gets its budget once, including graph construction/startup."""
    started = monotonic()
    frozen = verify(output)
    identity = task_identity(frozen, index)
    root = output / f"results/task_{index:03d}"
    destination = root / name
    checked(destination / "started.json", identity)
    problem, system, training, common, nodes = inputs(output, frozen)
    plan = ScaledAlternatingPlan.model_validate(frozen["plan"])
    ordinary = dict(zip(system.names, common["ordinary_parameters"], strict=True))
    if name == "initializer":
        graph = ConditionalCollocation(system, training, common, nodes, plan.penalty)
        oracle = oracle_for(system, training, common, plan, destination, 1, False)
        r = optimize(
            graph,
            oracle.lower,
            oracle.upper,
            arm=frozen["tasks"][index],
            seconds=max(0.001, plan.initializer_seconds - (monotonic() - started)),
            maximum_iterations=plan.initializer_iterations,
            cycles=plan.alternating_cycles,
            block_iterations=plan.block_iterations,
            root=destination / "native",
            identity=identity,
        )
        result = {"status": "finished", "initializer": r}
    elif name == "screen":
        saved = load_point(root / "initializer/native", identity)
        points = [ordinary]
        if saved:
            record, _, q = saved
            if record["common_identity"] != common["identity"]:
                raise ValueError("saved initializer common units differ")
            proposed = dict(
                zip(
                    system.names,
                    (q * np.asarray(common["parameter_units"])).tolist(),
                    strict=True,
                )
            )
            if proposed != record["parameters"]:
                raise ValueError("initializer parameters/array differ")
            if proposed != ordinary:
                points.append(proposed)
        oracle = oracle_for(
            system,
            training,
            common,
            plan,
            destination,
            max(0.001, plan.screen_seconds - (monotonic() - started)),
            False,
        )
        records = []
        for point in points:
            try:
                oracle(oracle.vector(point))
            except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
                records.append({"error": str(error)})
            else:
                records.append(oracle.last_evaluation)
        result = {
            "status": "finished",
            "best": oracle.best,
            "evaluations": records,
            "calls": oracle.budget.calls,
        }
    elif name == "refinement":
        prior = checked(root / "screen/result.json", identity)
        if not prior.get("best"):
            result = {"status": "no_feasible_screen", "best": None, "calls": 0}
        else:
            oracle = oracle_for(
                system,
                training,
                common,
                plan,
                destination,
                max(0.001, plan.refinement_seconds - (monotonic() - started)),
                True,
            )
            try:
                fit = least_squares(
                    oracle,
                    oracle.vector(prior["best"]["parameters"]),
                    jac=oracle.jacobian,
                    bounds=(oracle.lower, oracle.upper),
                    x_scale=np.asarray(common["parameter_units"]),
                    ftol=None,
                    xtol=1e-8,
                    gtol=1e-8,
                    max_nfev=plan.refinement_calls,
                )
                native = {
                    "success": bool(fit.success),
                    "message": str(fit.message),
                    "nfev": fit.nfev,
                    "optimality": float(fit.optimality),
                    "status": int(fit.status),
                }
            except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
                native = {"success": False, "message": str(error)}
            result = {
                "status": "finished",
                "native": native,
                "best": oracle.best,
                "calls": oracle.budget.calls,
                "valid_calls": oracle.valid_calls,
                "rejected_trials": oracle.rejected_trials,
                "failure_evidence": oracle.failure_evidence,
            }
    elif name.startswith("replay-"):
        _, split, method = name.split("-")
        if split not in ("train", "val") or method not in ("Radau", "BDF"):
            raise ValueError("unsupported replay")
        selected = checked(root / "selected.json", identity)
        if selected["best"] is None:
            result = {"status": "no_parameters"}
        else:
            data = unpack_split(problem["splits"][split])
            theta = np.array([selected["best"]["parameters"][n] for n in system.names])
            predictions = []
            for row in data.trajectories:
                predicted, _, _, _ = symbolic_rollout(
                    system,
                    row,
                    theta,
                    plan.settings(tight=True, method=method),
                    started + plan.replay_seconds,
                    sensitivities=False,
                )
                predictions.extend(predicted[:, 0])
            y = np.concatenate([r.targets["v01"] for r in data.trajectories])
            predicted = np.asarray(predictions)
            if not np.isfinite(predicted).all():
                raise ValueError("nonfinite replay predictions")
            # Predictions only: never serialize or score latent trajectory labels.
            result = {
                "status": "finished",
                "nmse": float(np.mean(((predicted - y) / common["output_scale"]) ** 2)),
                "predictions": predicted.tolist(),
                "split": split,
                "solver": method,
            }
    else:
        raise ValueError("unknown stage")
    result = _finite_payload(
        {
            **result,
            "identity": identity,
            "seconds": monotonic() - started,
            "common_identity": common["identity"],
        }
    )
    write_json(destination / "result.json", result, immutable=True)
    return result


def failed_result(
    root: Path,
    name: str,
    identity: str,
    exit_code: int,
    *,
    status="interrupted",
    message="Stage budget consumed; no fresh attempt on resume",
):
    """Retain incumbents while distinguishing numerical failure from interruption."""
    result = {
        "identity": identity,
        "status": status,
        "exit_code": exit_code,
        "message": message,
    }
    if name in ("screen", "refinement"):
        best = root / name / "calls/best_evaluated.json"
        result["best"] = read_json(best) if best.exists() else None
    if name == "initializer":
        saved = load_point(root / name / "native", identity)
        result["initializer"] = saved[0] if saved else None
    write_json(root / name / "result.json", result, immutable=True)
    return result


def interrupted_result(root: Path, name: str, identity: str, exit_code: int):
    """An interrupted stage is consumed in full, with no automatic rerun."""
    return failed_result(
        root,
        name,
        identity,
        exit_code,
        status="interrupted" if exit_code in (124, 125, -9, -15) else "worker_failed",
    )


def finish(output: Path, index: int):
    frozen = verify(output)
    identity = task_identity(frozen, index)
    root = output / f"results/task_{index:03d}"
    common = read_json(output / "common.json")
    selected = checked(root / "selected.json", identity)
    replay = {}
    passed = True
    for split in ("train", "val"):
        r = [
            checked(root / f"replay-{split}-{method}/result.json", identity)
            for method in ("Radau", "BDF")
        ]
        if any(v["status"] != "finished" for v in r):
            replay[split] = {
                "pass": False,
                "errors": [v.get("message", v["status"]) for v in r],
            }
            passed = False
            continue
        delta = float(
            np.max(abs(np.array(r[0]["predictions"]) - r[1]["predictions"]))
            / common["output_scale"]
        )
        replay[split] = {
            "pass": delta <= 1e-4,
            "maximum_scaled_prediction_difference": delta,
            "nmse": r[0]["nmse"],
            "BDF_nmse": r[1]["nmse"],
        }
        passed &= replay[split]["pass"]
    result = {
        "identity": identity,
        "arm": frozen["tasks"][index],
        "status": ("complete" if passed else "replay_unverified")
        if selected["best"]
        else "fit_failed",
        "common_identity": common["identity"],
        "selected": selected,
        "replay": replay,
        "initializer": checked(root / "initializer/result.json", identity),
        "screen": checked(root / "screen/result.json", identity),
        "refinement": checked(root / "refinement/result.json", identity),
        "test_data_opened": False,
        "llm_calls": 0,
        "hidden_trajectory_labels_used": False,
        "known_hidden_initials_fixed": True,
    }
    for label, threshold in (("strict", 1e-4), ("good", 0.01), ("practical", 0.1)):
        result[label] = bool(
            passed and all(replay[s]["nmse"] <= threshold for s in ("train", "val"))
        )
    write_json(root / "result.json", result, immutable=True)
    return result
