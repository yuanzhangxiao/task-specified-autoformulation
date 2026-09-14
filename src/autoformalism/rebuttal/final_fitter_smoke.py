"""Numerical gate on a separate small control; difficult data are not fitted."""

from __future__ import annotations

from pathlib import Path

from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.final_fitter import (
    FinalFitterPlan,
    checkpoint_stage,
    fit_task,
    verify,
)
from autoformalism.rebuttal.final_fitter_models import alternative_starts
from autoformalism.rebuttal.fitter_diagnostic import read_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.staged_topology import content_hash


def smoke(directory: Path, plan: FinalFitterPlan) -> dict:
    p, _ = synthetic_problem("shared", 0.0, 0)
    _, guesses = system_for(p)
    p["start"] = {"rate": 0.3, "gain": 0.8, **dict.fromkeys(guesses, 2.0)}
    config = plan.fit.model_copy(
        update={
            "initializer_seconds": 12,
            "refinement_seconds": 30,
            "collocation_maximum_iterations": 100,
            "collocation_minimum_intervals": 1,
            "collocation_diagnostics": False,
            "recovery_screen_seconds": 3,
            "recovery_probe_seconds": 3,
            "recovery_sensitivity_seconds": 5,
        }
    )
    small_plan = FinalFitterPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "total_fit_seconds": 45,
            "selection_seconds": 5,
            "fit": config.model_dump(mode="json"),
        }
    )
    checks = []
    for strategy in (
        "collocation_exact",
        "collocation_limited",
        "direct_multistart",
        "horizon_continuation",
    ):
        task = {
            "strategy": strategy,
            "family": "fixed_shapes",
            "starts": alternative_starts(p, plan.seed),
        }
        result = fit_task(
            p,
            task,
            small_plan,
            directory / strategy,
            content_hash([strategy, small_plan.model_dump(mode="json")]),
        )
        value = result.get("training_nmse")
        checks.append(
            {
                "strategy": strategy,
                "train_nmse": value,
                "pass": value is not None and value < 1e-6,
            }
        )
    return {"checks": checks, "pass": all(c["pass"] for c in checks)}


def gate(output: Path) -> dict:
    frozen = verify(output)
    plan = FinalFitterPlan.model_validate(frozen["plan"])
    result = checkpoint_stage(
        output / "gate",
        frozen["identity"],
        lambda: smoke(output / "gate/control", plan),
    )
    # Gate checks existing task assets without examining their fit outcomes.
    for task in frozen["tasks"]:
        p = read_json(output / f"problems/{task['index']:03d}.json")
        system_for(p)
    return result
