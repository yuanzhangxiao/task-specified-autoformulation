#!/usr/bin/env python3
"""Run a portable synthetic smoke for the optional CasADi initializer."""

from __future__ import annotations

import json

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.schemas import CandidateModel


def _split(name: SplitName, identifier: str, rate: float) -> DatasetSplit:
    time = np.linspace(0.0, 4.0, 41)
    trajectory = Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"target": 2.0 * np.exp(-rate * time)},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    return DatasetSplit(name, (trajectory,), f"{name.value}-fingerprint")


def main() -> None:
    """Recover one positive rate and emit an auditable JSON result."""
    true_rate = 0.55
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "casadi_decay_smoke",
            "parent_candidate_id": None,
            "change_summary": "Portable CasADi initializer smoke.",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed decaying state.",
                }
            ],
            "state_equations": [{"state": "x", "rhs": "-rate * x"}],
            "observation_mappings": [
                {"channel": "target", "expression": "x", "unit": "unit"}
            ],
            "parameters": [
                {
                    "name": "rate",
                    "scope": "global",
                    "role": "rate",
                    "unit": "1/time",
                    "description": "Positive decay rate.",
                }
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "target"}
            ],
        }
    )
    model = compile_candidate(candidate, ValidationContext(targets=("target",)))
    result = fit_candidate(
        model,
        _split(SplitName.TRAIN, "train", true_rate),
        _split(SplitName.VALIDATION, "validation", true_rate),
        FitConfig(
            nonlinear_initializer="casadi_multiple_shooting",
            nonlinear_initializer_failure_policy="raise",
            integration_backend="fixed_rk4",
            maximum_function_evaluations=20,
            casadi_maximum_iterations=100,
            casadi_maximum_wall_time_seconds=30.0,
        ),
    )
    diagnostic = result.initialization_diagnostics[0]
    fitted_rate = result.global_parameters["rate"]
    passed = bool(
        result.success
        and diagnostic.success
        and abs(fitted_rate - true_rate) < 1e-5
        and result.validation_metrics.normalized_mse < 1e-10
    )
    payload = {
        "schema_version": "casadi-multiple-shooting-smoke-1",
        "status": "pass" if passed else "fail",
        "initializer_backend": diagnostic.backend,
        "initializer_iterations": diagnostic.iterations,
        "initializer_objective": diagnostic.objective,
        "initializer_wall_seconds": diagnostic.wall_seconds,
        "true_rate": true_rate,
        "fitted_rate": fitted_rate,
        "absolute_parameter_error": abs(fitted_rate - true_rate),
        "validation_normalized_mse": result.validation_metrics.normalized_mse,
        "final_causal_evaluator": "fixed_rk4",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("CasADi multiple-shooting smoke failed")


if __name__ == "__main__":
    main()
