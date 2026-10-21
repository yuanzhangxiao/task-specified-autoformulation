"""Training-only state/output checks of one frozen joint parameter vector.

This opt-in verifier performs no search, sensitivity rollout or rank computation.
The caller binds the supplied parameters to the saved endpoint and owns durable
reservation/resume policy. Independent solver passes share one finite allowance;
only measured synchronous checkpoints are excluded from numerical-active time.
"""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from time import monotonic, process_time

import numpy as np

from autoformalism.data import Trajectory
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


def _unavailable(result: dict) -> None:
    """Keep public availability coherent after late deadline/checkpoint failures."""
    result["success"] = result["fit_success"]
    if not result["success"]:
        result.update(
            normalized_mse=None,
            per_target_normalized_mse={},
            per_row_scores={},
            predictions={},
            parameters=None,
        )


def _validate(
    problem: PrefixFitProblem,
    training: PublicSplit,
    parameters: dict,
    numerical: kernel.NumericalPolicy,
) -> tuple[kernel.TrainingOracle, np.ndarray]:
    """Refuse changed coordinates/domains or tolerances before any integration."""
    if not isinstance(problem, PrefixFitProblem):
        raise TypeError("typed prefix fit problem required")
    if not isinstance(training, PublicSplit) or training.name != "train":
        raise ValueError("saved-parameter verification accepts training only")
    if not isinstance(numerical, kernel.NumericalPolicy):
        raise TypeError("typed numerical policy required")
    if (numerical.rtol, numerical.atol, numerical.solver_tolerance) != (
        1e-9,
        1e-11,
        1e-6,
    ):
        raise ValueError("saved-parameter verification requires frozen tolerances")
    if not isinstance(parameters, dict) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in parameters.values()
    ):
        raise ValueError("saved parameters must be a dictionary of finite numbers")
    oracle = kernel.TrainingOracle(problem, training, numerical)
    return oracle, oracle.vector(parameters)


def _rollout(
    oracle: kernel.TrainingOracle,
    row: Trajectory,
    vector: np.ndarray,
    method: str,
    deadline: float,
    result: dict,
    budget: verification._Budget,
) -> np.ndarray:
    """Retain row-level costs and original solver failure evidence exactly once."""
    result["rollout_calls_attempted"] += 1
    receipt = {
        "started": False,
        "completed": False,
        "elapsed_wall_seconds": 0.0,
        "elapsed_cpu_seconds": 0.0,
        "solver_counters": None,
        "error": None,
        "diagnostic": None,
    }
    result["trajectory_diagnostics"][row.trajectory_id][method] = receipt
    budget.check()
    local_vector = oracle.local_vector(vector, row)
    settings = oracle.settings.model_copy(update={"integration_method": method})
    wall, cpu = monotonic(), process_time()
    receipt["started"] = True
    result["rollout_calls_started"] += 1
    try:
        values, _, states, counters = kernel.symbolic_rollout(
            oracle.system,
            row,
            local_vector,
            settings,
            deadline,
            sensitivities=False,
        )
        receipt["solver_counters"] = deepcopy(counters)
        if (
            values.shape != (len(row.time), len(oracle.system.channels))
            or states.shape != (len(row.time), oracle.system.state_count)
            or not np.isfinite(values).all()
            or not np.isfinite(states).all()
        ):
            raise ValueError("invalid state/output rollout shape or nonfinite values")
        receipt["completed"] = True
        result["rollout_calls_completed"] += 1
        return values
    except BaseException as error:
        result["rollout_calls_failed"] += 1
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["diagnostic"] = deepcopy(getattr(error, "diagnostic", None))
        # A failed integrator may not expose its partial solver counters. Preserve
        # unavailable counters as null rather than inventing zero work.
        if isinstance(receipt["diagnostic"], dict):
            receipt["solver_counters"] = deepcopy(
                receipt["diagnostic"].get("solver_counters")
            )
        raise
    finally:
        receipt["elapsed_wall_seconds"] = monotonic() - wall
        receipt["elapsed_cpu_seconds"] = process_time() - cpu


def _scores(
    oracle: kernel.TrainingOracle, predictions: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict]:
    """Match the joint training objective, weighting samples rather than rows."""
    scales = np.asarray([oracle.scales[name] for name in oracle.system.channels])
    residuals, rows = [], {}
    for row in oracle.rows:
        values = predictions[row.trajectory_id]
        observed = np.column_stack([row.targets[n] for n in oracle.system.channels])
        residual = (values - observed) / scales
        squared = residual**2
        if not np.isfinite(squared).all():
            raise ValueError("nonfinite normalized training residual")
        residuals.append(residual)
        rows[row.trajectory_id] = {
            "normalized_mse": float(np.mean(squared)),
            "per_target_normalized_mse": dict(
                zip(
                    oracle.system.channels,
                    np.mean(squared, axis=0).tolist(),
                    strict=True,
                )
            ),
            "sample_count": len(row.time),
        }
    return np.concatenate(residuals, axis=0), rows


def verify_saved_parameters(
    problem: PrefixFitProblem,
    training: PublicSplit,
    parameters: dict,
    numerical: kernel.NumericalPolicy,
    execution: verification.ExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
) -> dict:
    """Check the exact saved joint endpoint with two state/output-only passes.

    Invalid inputs raise before integration and carry a measured
    ``verification_receipt``. Runtime failures return an unavailable result;
    interruptions propagate with that same terminal receipt. No endpoint is
    adjusted, even if its reconstruction error is large. Parameter provenance
    beyond coordinate/domain validation must be checked by the owning campaign.
    """
    budget = verification._Budget(execution)
    result = verification._base_result(execution) | {
        "protocol": "construction-saved-parameter-verification-1",
        "success": False,
        "fit_success": False,
        "parameters": None,
        "parameters_sha256": None,
        "normalized_mse": None,
        "per_target_normalized_mse": {},
        "per_row_scores": {},
        "predictions": {},
        "solver_difference": None,
        "selection_data": "train_only",
        "sensitivities": False,
        "rank_diagnostic": {
            "assessed": False,
            "reason": "state_output_only_verification",
        },
        "optimizer_calls": 0,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "private_data_opened": False,
        "rollout_calls_attempted": 0,
        "rollout_calls_started": 0,
        "rollout_calls_completed": 0,
        "rollout_calls_failed": 0,
        "trajectory_diagnostics": {},
        "count_scope": (
            "verification_calls_* count whole-training-dataset solver passes; "
            "rollout_calls_* count individual trajectory/method integrations. "
            "Completed rollouts returned finite states/outputs, including returns "
            "later rejected for deadline expiry. calls/completed_calls count search "
            "and remain zero. Missing failure counters mean unavailable, not zero."
        ),
    }
    try:
        frozen = deepcopy(parameters)
        oracle, vector = _validate(problem, training, frozen, numerical)
        result.update(
            parameters_sha256=content_sha256(frozen),
            problem_sha256=content_sha256(problem),
            training_sha256=content_sha256(training),
            policy=numerical.model_dump(mode="json"),
            training_scales=dict(oracle.scales),
            trajectory_diagnostics={row.trajectory_id: {} for row in oracle.rows},
        )
    except BaseException as error:
        verification._invalidate(result, error)
        with suppress(TimeoutError):
            budget.check()
        result["timing"] = budget.snapshot()
        error.verification_receipt = deepcopy(result)
        raise

    def evaluate(
        method: str, deadline: float
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
        values = {}
        for row in oracle.rows:
            values[row.trajectory_id] = _rollout(
                oracle, row, vector, method, deadline, result, budget
            )
            # A returned overrun cannot start another trajectory.
            budget.check()
        residual, row_scores = _scores(oracle, values)
        return values, residual, row_scores

    def qualify(
        solutions: list[tuple[dict[str, np.ndarray], np.ndarray, dict]],
    ) -> None:
        predictions, residual, row_scores = solutions[0]
        other = solutions[1][1]
        difference = float(np.max(np.abs(residual - other), initial=0.0))
        result["solver_difference"] = difference
        if not np.isfinite(difference) or difference > numerical.solver_tolerance:
            result["verification_error"] = "independent training integrators disagree"
            return
        result.update(
            fit_success=True,
            parameters=deepcopy(frozen),
            normalized_mse=float(np.mean(residual**2)),
            per_target_normalized_mse=dict(
                zip(
                    oracle.system.channels,
                    np.mean(residual**2, axis=0).tolist(),
                    strict=True,
                )
            ),
            per_row_scores=row_scores,
            predictions={
                row_id: {
                    name: values[:, index].tolist()
                    for index, name in enumerate(oracle.system.channels)
                }
                for row_id, values in predictions.items()
            },
        )

    def save(snapshot: dict) -> None:
        _unavailable(snapshot)
        if checkpoint is not None:
            checkpoint(snapshot)

    try:
        verification._run(
            result, budget, save if checkpoint is not None else None, evaluate, qualify
        )
    except BaseException as error:
        _unavailable(result)
        error.verification_receipt = deepcopy(result)
        raise
    _unavailable(result)
    return result
