"""Training-only comparison of joint TRF and bounded conditional checkpoint rescue.

No reference values, latent labels or validation arrays enter this module.
All screens and optimization blocks share one evaluation/time budget.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from autoformalism.fitting.feasibility import (
    EvaluationBudget,
    GuardedOracle,
    SensitivityUnavailable,
)
from autoformalism.fitting.public_fitting import _write
from autoformalism.schemas.base import StrictSchema

ARMS = ("joint", "joint_stopping", "conditional_stopping")


class RefinementPolicy(StrictSchema):
    """Frozen limits; accuracy is a training stop, never a recovery certificate."""

    seconds: float = Field(default=120, gt=0, le=1200)
    maximum_calls: int = Field(default=180, ge=5, le=2000)
    training_nmse: float = Field(default=1e-10, gt=0)
    trajectory_nmse: float = Field(default=1e-9, gt=0)
    stall_steps: int = Field(default=4, ge=2, le=20)
    relative_progress: float = Field(default=1e-7, gt=0)
    scaled_step: float = Field(default=1e-5, gt=0)
    conditional_candidates: int = Field(default=3, ge=1, le=5)
    block_calls: int = Field(default=12, ge=2, le=100)


class _Accuracy(Exception):
    pass


class _Stalled(Exception):
    pass


class _BlockLimit(Exception):
    pass


def diverse_points(
    points: list[dict], names: tuple, scales: np.ndarray, count: int
) -> list[dict]:
    """Keep a good rollout and geometrically distinct pairs, not just top NMSEs."""
    ordered = sorted(points, key=lambda p: p.get("cost", float("inf")))
    chosen = ordered[:1]
    remaining = ordered[1:]
    while remaining and len(chosen) < count:

        def distance(point):
            x = np.array([point["parameters"][n] for n in names]) / scales
            return min(
                float(
                    np.linalg.norm(
                        x - np.array([v["parameters"][n] for n in names]) / scales
                    )
                )
                for v in chosen
            )

        index = max(range(len(remaining)), key=lambda i: distance(remaining[i]))
        chosen.append(remaining.pop(index))
    return chosen


def refine(
    system,
    training,
    scales: dict,
    settings,
    coordinates,
    points: list[dict],
    policy: RefinementPolicy,
    arm: Literal["joint", "joint_stopping", "conditional_stopping"],
    directory: Path,
) -> dict:
    """Fit a bounded portfolio and retain the best complete training evaluation."""
    if training.name.value != "train" or arm not in ARMS:
        raise ValueError("expected training-only refinement and a known arm")
    started = monotonic()
    budget = EvaluationBudget(started + policy.seconds, policy.maximum_calls)
    _, units = coordinates.arrays("parameters", system.names)
    stages, screened = [], []
    sizes = [len(row.time) * len(system.channels) for row in training.trajectories]
    stop_enabled = arm != "joint"
    best = None

    def make_oracle(name, sensitivities):
        result = GuardedOracle(
            system,
            training,
            scales,
            settings,
            directory / name,
            budget.deadline,
            sensitivities=sensitivities,
            budget=budget,
            point_seconds=30,
        )
        return result

    def evaluate(oracle, vector):
        nonlocal best
        previous = oracle.valid_calls
        residual = oracle(vector)
        if oracle.valid_calls == previous:
            raise ValueError("unavailable rollout cannot enter optimization")
        cost = float(0.5 * residual @ residual)
        if best is None or cost < best["cost"]:
            best = {
                "parameters": dict(zip(system.names, vector.tolist(), strict=True)),
                "cost": cost,
                "training_nmse": float(np.mean(residual**2)),
                "call": budget.calls,
            }
            _write(directory / "best.json", best)
        per_trajectory = [
            float(np.mean(v**2)) for v in np.split(residual, np.cumsum(sizes)[:-1])
        ]
        if (
            stop_enabled
            and np.mean(residual**2) <= policy.training_nmse
            and max(per_trajectory) <= policy.trajectory_nmse
        ):
            raise _Accuracy
        return residual

    def optimize(point, indices, label, calls, seconds):
        anchor = np.array([point["parameters"][n] for n in system.names])
        oracle = make_oracle(f"stage-{len(stages):02d}", True)
        call_start, stage_start = budget.calls, monotonic()
        stage_deadline = min(budget.deadline, stage_start + seconds)
        record = {
            "stage": label,
            "source": point["source"],
            "initial_parameters": point["parameters"],
            "free_parameters": [system.names[i] for i in indices],
            "maximum_residual_calls": calls,
            "allowance_seconds": seconds,
        }
        last = None
        stalled = 0

        def physical(q):
            vector = anchor.copy()
            vector[indices] = anchor[indices] + units[indices] * q
            return vector

        def fun(q):
            if budget.calls >= call_start + calls or monotonic() >= stage_deadline:
                raise _BlockLimit
            oracle.point_seconds = min(30, max(0.001, stage_deadline - monotonic()))
            return evaluate(oracle, physical(q))

        def jac(q):
            return oracle.jacobian(physical(q))[:, indices] * units[indices]

        def callback(intermediate_result):
            nonlocal last, stalled
            current = (float(intermediate_result.cost), intermediate_result.x.copy())
            if last is not None:
                improvement = (last[0] - current[0]) / max(abs(last[0]), 1e-30)
                step = np.linalg.norm(current[1] - last[1])
                stalled = (
                    stalled + 1
                    if improvement <= policy.relative_progress
                    and step <= policy.scaled_step
                    else 0
                )
                if stop_enabled and stalled >= policy.stall_steps:
                    raise _Stalled
            last = current

        try:
            result = least_squares(
                fun,
                np.zeros(len(indices)),
                jac=jac,
                bounds=(
                    (oracle.lower[indices] - anchor[indices]) / units[indices],
                    (oracle.upper[indices] - anchor[indices]) / units[indices],
                ),
                method="trf",
                ftol=None,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=calls,
                callback=callback,
            )
            record.update(
                stop_reason="optimizer_terminated",
                native_success=bool(result.success),
                native_status=int(result.status),
                message=result.message,
            )
        except _Accuracy:
            record["stop_reason"] = "training_accuracy_reached_pending_replay"
            raise
        except _Stalled:
            record["stop_reason"] = "stalled"
        except _BlockLimit:
            record["stop_reason"] = "block_allowance_exhausted"
        except TimeoutError:
            record["stop_reason"] = "budget_exhausted"
        except SensitivityUnavailable as error:
            record.update(
                stop_reason="point_allowance_exhausted"
                if isinstance(error.__cause__, TimeoutError)
                else "numerical_failure",
                message=str(error)[-1000:],
            )
        except (ValueError, RuntimeError, ArithmeticError) as error:
            record.update(stop_reason="numerical_failure", message=str(error)[-1000:])
        finally:
            record.update(
                calls=budget.calls - call_start, seconds=monotonic() - stage_start
            )
            stages.append(record)
            _write(directory / "stages.json", stages)
        return oracle.best

    reason = "optimizer_terminated"
    try:
        screen = make_oracle("screens", False)
        screen_limit = started + 0.25 * policy.seconds
        for point in points:
            if budget.calls >= policy.maximum_calls // 3 or monotonic() >= screen_limit:
                break
            try:
                screen.point_seconds = max(0.001, screen_limit - monotonic())
                residual = evaluate(screen, screen.vector(point["parameters"]))
                screened.append(
                    {**point, "cost": float(0.5 * residual @ residual), "valid": True}
                )
            except (ValueError, TimeoutError) as error:
                screened.append({**point, "valid": False, "error": str(error)[-500:]})
            _write(directory / "screened.json", screened)
        if arm == "conditional_stopping":
            # Include unscreened candidates too; poor paired NMSE must not
            # automatically eliminate a potentially useful parameter/initial block.
            pool = {p["source"]: p for p in points}
            pool.update({p["source"]: p for p in screened})
            selected = diverse_points(
                list(pool.values()), system.names, units, policy.conditional_candidates
            )
            _write(directory / "conditional_selection.json", selected)
            initial = [
                i
                for i, n in enumerate(system.names)
                if n in system.initial_parameter_names
            ]
            dynamic = [i for i in range(len(system.names)) if i not in initial]
            conditional_deadline = started + 0.65 * policy.seconds
            for point in selected:
                for label, indices in (
                    ("initials_only", initial),
                    ("coefficients_only", dynamic),
                ):
                    if (
                        not indices
                        or monotonic() >= conditional_deadline
                        or budget.calls >= int(0.65 * policy.maximum_calls)
                    ):
                        break
                    found = optimize(
                        point,
                        indices,
                        label,
                        min(
                            policy.block_calls,
                            int(0.65 * policy.maximum_calls) - budget.calls,
                        ),
                        max(0.001, (conditional_deadline - monotonic()) / 2),
                    )
                    if found:
                        point = {**point, **found}
        if best and budget.calls < budget.maximum and monotonic() < budget.deadline:
            optimize(
                {**best, "source": "best_training_pair"},
                list(range(len(system.names))),
                "joint",
                budget.maximum - budget.calls,
                budget.deadline - monotonic(),
            )
            reason = stages[-1]["stop_reason"]
        elif best is None:
            reason = "no_feasible_training_point"
    except _Accuracy:
        reason = "training_accuracy_reached_pending_replay"
    finally:
        _write(directory / "stages.json", stages)
    if reason != "training_accuracy_reached_pending_replay" and (
        budget.calls >= budget.maximum or monotonic() >= budget.deadline
    ):
        reason = "budget_exhausted"
    result = {
        "arm": arm,
        "parameters": best["parameters"] if best else None,
        "training_nmse": best["training_nmse"] if best else None,
        "stop_reason": reason,
        "actual_residual_calls": budget.calls,
        "seconds": monotonic() - started,
        "stages": stages,
        "budget_exhausted": budget.calls >= budget.maximum
        or monotonic() >= budget.deadline,
        "selection": "best_complete_training_rollout_across_all_stages",
        "validation_used_for_fitting": False,
        "reference_values_used": False,
        "initial_values_fitted": list(system.initial_parameter_names),
    }
    _write(directory / "result.json", result)
    return result
