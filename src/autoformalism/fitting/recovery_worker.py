"""M12 training-only rollout optimization and independent prediction checks."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.identifiable_refinement import RefinementPolicy, refine
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.public_fitting import PublicFitRequest


def certify(payload: dict, folder: Path) -> dict:
    """Check every training trajectory with two solvers; no parameter certificate."""
    launched = public._read(folder / "launch.json")["monotonic"]
    deadline = launched + payload["seconds"]
    train = public.unpack_split(TrainingOnlySplit.model_validate(payload["training"]))
    request = PublicFitRequest.model_validate(payload["request"])
    model = public._lower(request)[0]
    parameters = payload["parameters"]
    if set(parameters) != set(model.parameter_names):
        raise ValueError("incomplete certificate vector")
    scales = scale_for(train)
    rows = []
    for trajectory in train.trajectories:
        row = {"trajectory": trajectory.trajectory_id, "complete": False}
        rows.append(row)
        predictions = []
        try:
            for method in ("Radau", "DOP853"):
                if monotonic() >= deadline:
                    raise TimeoutError("training certificate deadline")
                result = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    FitConfig(
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    reset_observed_states=False,
                    deadline=deadline,
                )
                if not result.success:
                    raise ValueError(result.message)
                predictions.append(result.predictions)
            scores = [
                float(np.mean(((p[c] - trajectory.targets[c]) / scales[c]) ** 2))
                for p in predictions
                for c in request.context.targets
            ]
            difference = max(
                float(np.max(abs(predictions[0][c] - predictions[1][c]))) / scales[c]
                for c in request.context.targets
            )
            if not np.isfinite(scores).all() or not np.isfinite(difference):
                raise ValueError("nonfinite training certificate")
            row.update(
                complete=True, worst_nmse=max(scores), solver_difference=difference
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            row["error"] = str(error)[-500:]
        public._write(folder / "progress.json", {"rows": rows})
    complete = all(r["complete"] for r in rows) and bool(rows)
    worst = max((r["worst_nmse"] for r in rows if r["complete"]), default=None)
    difference = max(
        (r["solver_difference"] for r in rows if r["complete"]), default=None
    )
    # Mean of per-trajectory worst-method/channel scores: all must be good too.
    mean = float(np.mean([r["worst_nmse"] for r in rows])) if complete else None
    result = {
        "payload_sha256": public.content_sha256(payload),
        "parameters": parameters,
        "complete": complete,
        "training_nmse": mean,
        "maximum_trajectory_nmse": worst,
        "maximum_solver_difference": difference,
        "passed": bool(
            complete
            and mean <= payload["training_nmse"]
            and worst <= payload["trajectory_nmse"]
            and difference <= payload["solver_agreement"]
        ),
        "scope": (
            "empirical training prediction; not coefficient recovery "
            "or a global error bound"
        ),
        "rows": rows,
    }
    public._write(folder / "result.json", result)
    return result


def rollout(payload: dict, folder: Path) -> dict:
    """Reuse joint forward-sensitivity TRF with the same physical generic start."""
    train = public.unpack_split(TrainingOnlySplit.model_validate(payload["training"]))
    request = PublicFitRequest.model_validate(payload["request"])
    system = SymbolicODE(public._lower(request)[0], allow_piecewise=True)
    launched = public._read(folder / "launch.json")["monotonic"]
    seconds = max(0.001, payload["seconds"] - (monotonic() - launched) - 1)
    result = refine(
        system,
        train,
        scale_for(train),
        SETTINGS,
        NumericalCoordinates.model_validate(payload["coordinates"]),
        payload["points"],
        RefinementPolicy(
            seconds=min(1200, seconds),
            maximum_calls=payload["maximum_calls"],
            training_nmse=payload["training_nmse"],
            trajectory_nmse=payload["trajectory_nmse"],
        ),
        "joint_stopping",
        folder / "refinement",
        record_history=True,
    )
    result["payload_sha256"] = public.content_sha256(payload)
    public._write(folder / "result.json", result)
    return result
