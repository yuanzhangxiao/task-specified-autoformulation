"""Score a fully fixed public model without an optimizer or sensitivity system."""

from time import monotonic

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory

POLICY = "parameter-free-rollout-1"


def evaluate(model, training, validation, settings: FitConfig) -> dict:
    """Use training scales and causal boundaries; never optimize latent initials."""
    if model.parameter_names:
        raise ValueError("fixed rollout requires no parameters after lowering initials")
    if training.name.value != "train" or validation.name.value != "val":
        raise ValueError("fixed rollout requires public train/validation splits")
    scaler = TrainingScaler().fit(training)
    targets = model.validated.context.targets
    scales = {t: scaler.scales[f"target:{t}"].standard_deviation for t in targets}
    if any(not np.isfinite(s) or s <= 0 for s in scales.values()):
        raise ValueError("training scales must be positive and finite")
    started = monotonic()
    deadline = started + settings.maximum_wall_time_seconds
    result = {
        "execution_mode": POLICY,
        "parameters": {},
        "optimizer_calls": 0,
        "actual_residual_calls": 0,
        "native_optimizer_converged": None,
        "validation_initials_fitted": False,
        "training_only_parameter_estimation": True,
        "independent_replay": "not_performed",
        "scales": scales,
    }
    for name, split in (("training", training), ("validation", validation)):
        squared = {t: [] for t in targets}
        errors = {}
        for row in split.trajectories:
            try:
                sim = simulate_trajectory(
                    model,
                    row,
                    {},
                    {},
                    settings,
                    deadline=deadline,
                    reset_observed_states=False,
                )
            except TimeoutError as error:
                errors[row.trajectory_id] = str(error)
                continue
            if not sim.success:
                errors[row.trajectory_id] = sim.message
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pieces = {
                    t: ((sim.predictions[t] - row.targets[t]) / scales[t]) ** 2
                    for t in targets
                }
            if any(not np.isfinite(v).all() for v in pieces.values()):
                errors[row.trajectory_id] = "nonfinite prediction error"
                continue
            for t, values in pieces.items():
                squared[t].append(values)
        scores = (
            {t: float(np.mean(np.concatenate(v))) for t, v in squared.items()}
            if not errors
            else {}
        )
        result[name] = {
            "normalized_mse": float(np.mean(list(scores.values()))) if scores else None,
            "per_target_normalized_mse": scores,
            "failed_trajectories": list(errors),
            "errors": errors,
        }
    result["seconds"] = monotonic() - started
    result["budget_exhausted"] = monotonic() >= deadline
    return result
