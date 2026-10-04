"""M7 fixed-mesh and restart policies; fitting receives training evidence only."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.identifiable_refinement import RefinementPolicy, refine
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

ARMS = (
    "fixed_dense_collocation",
    "fixed_reduced_collocation",
    "fixed_shooting_exact",
    "fixed_shooting_lbfgs",
    "rollout_continue",
    "rollout_restart",
)
PROTOCOL = "phase-c-fitting-numerical-diagnostic-1"


class NumericalDiagnosticPolicy(StrictSchema):
    """Predeclared limits, mesh resolution and training-only restart rule."""

    target_variables: int = Field(default=6000, ge=20, le=200000)
    minimum_intervals: int = Field(default=24, ge=1, le=1000)
    first_rollout_calls: int = Field(default=41, ge=5, le=500)
    first_rollout_fraction: float = Field(default=2 / 3, gt=0, lt=1)
    stagnation_window: int = Field(default=10, ge=3, le=100)
    stagnation_relative: float = Field(default=0.005, gt=0, lt=1)
    restart_seed: int = Field(default=20261003, ge=0)


def stagnation(history: list[dict], policy: NumericalDiagnosticPolicy) -> dict:
    """Compare incumbent training losses across completed evaluations, not truth."""
    window = policy.stagnation_window
    if len(history) < 2 * window:
        return {"triggered": False, "reason": "insufficient_completed_evaluations"}
    before, after = history[-window - 1]["best_nmse"], history[-1]["best_nmse"]
    if not np.isfinite([before, after]).all() or before <= 0 or after < 0:
        raise ValueError("invalid training history")
    improvement = (before - after) / before
    return {
        "triggered": bool(0 <= improvement <= policy.stagnation_relative),
        "reason": "training_incumbent_progress",
        "relative_improvement": improvement,
        "window_completed_evaluations": window,
        "threshold": policy.stagnation_relative,
        "first_call": history[-window - 1]["call"],
        "last_call": history[-1]["call"],
    }


def generic_restart(layout, start: dict, units: np.ndarray, seed: int) -> dict:
    """Draw once around the original generic values, respecting declared bounds."""
    rng = np.random.default_rng(seed)
    vector = layout.vector(start) * np.exp(rng.uniform(-1.5, 1.5, len(start)))
    for i, name in enumerate(layout.system.names):
        if name in layout.system.initial_parameter_names:
            vector[i] = rng.uniform(-1, 1) * units[i]
    vector = np.clip(vector, layout.lower, layout.upper)
    return dict(zip(layout.system.names, vector.tolist(), strict=True))


def rollout(payload: dict, directory: Path) -> dict:
    """Two bounded TRF phases sharing one total budget and retained incumbent.

    Continuation transfers physical values, not SciPy's trust-region state. Both
    policies share the first phase; only the second start can differ. A restart
    never discards the first phase's best complete training rollout.
    """
    from autoformalism.fitting.transcription_fit import StrategyPolicy

    begun = monotonic()
    policy = StrategyPolicy.model_validate(payload["policy"])
    diagnostic = NumericalDiagnosticPolicy.model_validate(
        payload["numerical_diagnostic"]
    )
    deadline = begun + max(0.01, policy.seconds - 1)
    arm = payload["arm"]
    if arm not in {"rollout_continue", "rollout_restart"}:
        raise ValueError("unknown restart diagnostic arm")
    data = PublicSplit.model_validate(payload["training"])
    if data.name != "train":
        raise ValueError("restart diagnostic requires training data")
    model, start, _ = public._lower(PublicFitRequest.model_validate(payload["request"]))
    system = SymbolicODE(model, allow_piecewise=True)
    training = public.unpack_split(data)
    coordinates = NumericalCoordinates.model_validate(payload["coordinates"])
    scales = scale_for(training)
    layout = SymbolicOracle(
        system,
        training,
        scales,
        SETTINGS,
        directory / "layout",
        None,
        sensitivities=False,
    )
    _, units = coordinates.arrays("parameters", system.names)
    restart = generic_restart(
        layout, start, units, diagnostic.restart_seed + payload["seed"]
    )
    best, calls, stages = None, 0, []
    next_start = start
    decision = {"triggered": False, "reason": "first_phase_unfinished"}
    stop = "no_feasible_training_point"
    for phase in range(2):
        remaining = deadline - monotonic()
        maximum = policy.maximum_rollout_calls - calls
        if remaining <= 1 or maximum < 5:
            break
        allowance = (
            min(remaining, policy.seconds * diagnostic.first_rollout_fraction)
            if phase == 0
            else remaining
        )
        maximum = (
            min(maximum, diagnostic.first_rollout_calls) if phase == 0 else maximum
        )
        folder = directory / f"refinement-{phase}"
        folder.mkdir(parents=True, exist_ok=True)
        public._write(
            directory / "diagnostic.json",
            {
                "stages": stages,
                "decision": decision,
                "active_phase": phase,
                "next_parameters": next_start,
                "calls_completed_before_phase": calls,
                "restart_draw": restart,
                "total_call_limit": policy.maximum_rollout_calls,
                "total_wall_seconds": policy.seconds,
            },
        )
        result = refine(
            system,
            training,
            scales,
            SETTINGS,
            coordinates,
            [
                {
                    "source": "generic"
                    if phase == 0
                    else "restart"
                    if decision.get("used")
                    else "continuation",
                    "parameters": next_start,
                }
            ],
            RefinementPolicy(
                seconds=allowance,
                maximum_calls=maximum,
                training_nmse=policy.training_nmse,
                trajectory_nmse=policy.trajectory_nmse,
            ),
            "joint_stopping",
            folder,
            record_history=True,
        )
        calls += result["actual_residual_calls"]
        stages.append({"phase": phase, "method": "rollout", **result})
        if result["parameters"] is not None and (
            best is None or result["training_nmse"] < best["training_nmse"]
        ):
            best = {
                "parameters": result["parameters"],
                "training_nmse": result["training_nmse"],
                "source": f"rollout_phase_{phase}",
            }
            public._write(directory / "best.json", best)
        stop = result["stop_reason"]
        if stop == "training_accuracy_reached_pending_replay":
            decision = {**decision, "reason": "training_accuracy_reached"}
            break
        if phase == 0:
            history = (
                public._read(folder / "training_history.json")
                if (folder / "training_history.json").exists()
                else []
            )
            decision = stagnation(history, diagnostic)
            decision["used"] = arm == "rollout_restart" and decision["triggered"]
            next_start = (
                restart if decision["used"] else best["parameters"] if best else start
            )
    result = {
        "arm": arm,
        "parameters": best["parameters"] if best else None,
        "training_nmse": best["training_nmse"] if best else None,
        "actual_residual_calls": calls,
        "stages": stages,
        "stop_reason": stop,
        "budget_exhausted": monotonic() >= deadline
        or calls >= policy.maximum_rollout_calls,
        "seconds": monotonic() - begun,
        "decision": decision,
        "selection": "best_complete_training_rollout_across_both_phases",
        "validation_used_for_fitting": False,
        "reference_values_used": False,
    }
    public._write(directory / "diagnostic.json", result)
    public._write(directory / "result.json", result)
    return result
