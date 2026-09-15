"""Numerical worker for existing scaled-collocation checkpoints, never new nodes."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from time import monotonic, process_time, time


def event(root: Path, name: str, **details) -> None:
    """Flush progress before imports, compilation, and every physical rollout."""
    value = {
        "event": name,
        "utc_seconds": time(),
        "monotonic": monotonic(),
        "cpu_seconds": process_time(),
        **details,
    }
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps(value), flush=True)


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "pydantic", "scipy")
        },
        "casadi": importlib.metadata.version("casadi"),
    }


def checkpoint_parameters(output: Path, frozen: dict, index: int, common: dict) -> dict:
    """Verify only this checkpoint and read q, not the unused latent node array."""
    import numpy as np
    from scaled_recovery_io import child_path, digest, read, sha

    task = frozen["tasks"][index]
    if task["array_error"]:
        raise ValueError(task["array_error"])
    root = output / "points" / task["arm"]
    record = read(root / "checkpoint.json")
    if (
        record.get("metadata_sha256")
        != digest({k: v for k, v in record.items() if k != "metadata_sha256"})
        or record["identity"] != task["source_task_identity"]
        or record["common_identity"] != common["identity"]
    ):
        raise ValueError("checkpoint metadata/common identity differs")
    path = child_path(root, record["array"])
    if sha(path) != record["array_sha256"]:
        raise ValueError("checkpoint array digest differs")
    with np.load(path, allow_pickle=False) as arrays:
        q = np.asarray(arrays["q"]).ravel()
    if len(q) != len(common["parameter_names"]) or not np.isfinite(q).all():
        raise ValueError("invalid checkpoint parameter vector")
    values = dict(
        zip(
            common["parameter_names"],
            (q * common["parameter_units"]).tolist(),
            strict=True,
        )
    )
    if values != record["parameters"]:
        raise ValueError("checkpoint parameter/array mismatch")
    return values


def execute(output: Path, root: Path, index: int, name: str, seconds: float) -> dict:
    """Startup is explicit; numerical budgets begin after the ready handshake."""
    from scaled_recovery_io import checked, read, write

    started = monotonic()
    event(root, "numerical_imports_begin")
    import casadi
    import numpy as np
    import scipy
    from scipy.optimize import least_squares

    from autoformalism.fitting.conditional_scaling import system_for
    from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
    from autoformalism.fitting.sensitivity_probe import symbolic_rollout
    from autoformalism.rebuttal.piecewise_campaign import unpack_split
    from autoformalism.rebuttal.scaled_alternating import ScaledAlternatingPlan

    event(root, "numerical_imports_end")
    frozen = read(output / "freeze.json")
    identity = frozen["identity"]
    versions = runtime_versions()
    if any(versions[k] != frozen["runtime"][k] for k in versions):
        raise ValueError("source numerical runtime versions differ")
    event(
        root,
        "runtime_checked",
        versions=versions,
        origins={
            "numpy": np.__file__,
            "scipy": scipy.__file__,
            "casadi": casadi.__file__,
        },
    )
    problem, common = read(output / "problem.json"), read(output / "common.json")
    event(root, "model_construction_begin")
    system = system_for(problem)
    training = unpack_split(problem["splits"]["train"])
    plan = ScaledAlternatingPlan.model_validate(frozen["plan"])
    if list(system.names) != common["parameter_names"]:
        raise ValueError("parameter order differs")
    event(root, "model_construction_end")
    if name == "screen-ordinary":
        parameters = dict(zip(system.names, common["ordinary_parameters"], strict=True))
    elif name == "screen-checkpoint":
        event(root, "checkpoint_parameters_begin")
        parameters = checkpoint_parameters(output, frozen, index, common)
        event(root, "checkpoint_parameters_end")
        if parameters == dict(
            zip(system.names, common["ordinary_parameters"], strict=True)
        ):
            return {
                "identity": identity,
                "status": "duplicate_ordinary",
                "calls": 0,
                "best": None,
                "numerical_seconds": 0.0,
                "setup_seconds": monotonic() - started,
            }
    else:
        parameters = checked(root.parent / "selected.json", identity)["best"][
            "parameters"
        ]

    class LoggedOracle(GuardedOracle):
        def __call__(self, values):
            event(
                root,
                "rollout_begin",
                next_call=self.calls + 1,
                sensitivities=self.with_sensitivities,
            )
            try:
                return super().__call__(values)
            finally:
                event(
                    root, "rollout_end", calls=self.calls, valid_calls=self.valid_calls
                )

    numerical_started = monotonic()
    deadline = numerical_started + seconds
    write(
        root / "ready.json",
        {
            "identity": identity,
            "monotonic": numerical_started,
            "budget_seconds": seconds,
            "setup_seconds": numerical_started - started,
        },
    )
    event(root, "numerical_begin", budget_seconds=seconds)
    if name.startswith("screen-") or name == "refinement":
        sensitivity = name == "refinement"
        budget = EvaluationBudget(deadline, plan.refinement_calls if sensitivity else 1)
        oracle = LoggedOracle(
            system,
            training,
            common["output_scale"],
            plan.settings(),
            root / "calls",
            deadline,
            sensitivities=sensitivity,
            budget=budget,
            point_seconds=plan.sensitivity_point_seconds
            if sensitivity
            else plan.screen_point_seconds,
            reject_invalid_trials=True,
        )
        native = None
        error = None
        try:
            values = oracle.vector(parameters)
            if sensitivity:
                fit = least_squares(
                    oracle,
                    values,
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
            else:
                oracle(values)
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "status": "finished" if oracle.best else "no_feasible_evaluation",
            "best": oracle.best,
            "calls": oracle.budget.calls,
            "valid_calls": oracle.valid_calls,
            "native": native,
            "error": error,
            "failure_evidence": oracle.failure_evidence,
        }
    else:
        _, split, method = name.split("-")
        if split not in ("train", "val") or method not in ("Radau", "BDF"):
            raise ValueError("unsupported replay")
        data = unpack_split(problem["splits"][split])
        theta = np.array([parameters[n] for n in system.names])
        predictions = []
        for trajectory in data.trajectories:
            event(
                root, "replay_trajectory_begin", trajectory_id=trajectory.trajectory_id
            )
            predicted, _, _, _ = symbolic_rollout(
                system,
                trajectory,
                theta,
                plan.settings(tight=True, method=method),
                deadline,
                sensitivities=False,
            )
            predictions.extend(predicted[:, 0])
        y = np.concatenate([row.targets["v01"] for row in data.trajectories])
        predicted = np.asarray(predictions)
        if not np.isfinite(predicted).all():
            raise ValueError("nonfinite replay predictions")
        result = {
            "status": "finished",
            "split": split,
            "solver": method,
            "nmse": float(np.mean(((predicted - y) / common["output_scale"]) ** 2)),
            "predictions": predicted.tolist(),
        }
    result.update(
        identity=identity,
        numerical_seconds=monotonic() - numerical_started,
        setup_seconds=numerical_started - started,
    )
    event(root, "numerical_end", status=result["status"])
    return result
