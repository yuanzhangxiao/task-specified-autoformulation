#!/usr/bin/env python3
"""Exercise affine fitting with generated, never revealed, latent dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import (
    FitConfig,
    evaluate_fitted_candidate,
    fit_candidate,
)
from autoformalism.schemas import CandidateModel


def run_smoke(output: Path) -> dict[str, object]:
    """Fit one hidden fixed basis and verify a frozen held-out rollout."""
    weight = 2.5
    model = compile_candidate(
        _candidate(), ValidationContext(targets=("target",))
    )
    training = _split(SplitName.TRAIN, "train", 0.0, weight)
    validation = _split(SplitName.VALIDATION, "validation", 0.3, weight)
    heldout = _split(SplitName.TEST, "test", -0.2, weight)
    config = FitConfig(
        parameter_fit_strategy="fixed_latent_basis_linear_ridge",
        derivative_ridge_regularization=0.0,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
    )
    fitted = fit_candidate(model, training, validation, config)
    if not fitted.success:
        raise RuntimeError(f"fixed latent-basis fit failed: {fitted.message}")
    _, test_metrics = evaluate_fitted_candidate(
        model,
        heldout,
        global_parameters=fitted.global_parameters,
        global_initial_conditions=fitted.global_initial_conditions,
        target_scales=fitted.target_scales,
        config=config,
    )
    parameter_error = abs(float(fitted.global_parameters["weight"]) - weight)
    if parameter_error > 2e-6 or test_metrics.normalized_mse > 1e-10:
        raise RuntimeError(
            "fixed latent-basis recovery missed its synthetic tolerance: "
            f"parameter_error={parameter_error}, "
            f"test_nmse={test_metrics.normalized_mse}"
        )
    result = {
        "schema_version": "fixed-latent-basis-fitting-smoke-1",
        "status": "pass",
        "backend": fitted.diagnostics[0].backend,
        "exact_observed_derivatives_supplied": True,
        "latent_values_supplied_to_fitter": False,
        "latent_derivatives_supplied_to_fitter": False,
        "true_parameter": weight,
        "fitted_parameter": float(fitted.global_parameters["weight"]),
        "absolute_parameter_error": parameter_error,
        "function_evaluations": fitted.diagnostics[0].function_evaluations,
        "training_normalized_mse": fitted.training_metrics.normalized_mse,
        "validation_normalized_mse": fitted.validation_metrics.normalized_mse,
        "test_normalized_mse": test_metrics.normalized_mse,
        "test_failed_trajectories": list(test_metrics.failed_trajectories),
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "fixed_latent_basis_smoke",
            "parent_candidate_id": None,
            "change_summary": "A fixed hidden relaxation basis drives y.",
            "states": [
                {
                    "name": "y",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed response.",
                },
                {
                    "name": "z",
                    "kind": "latent",
                    "unit": "unit/time",
                    "description": "Unobserved fixed-shape memory.",
                },
            ],
            "state_equations": [
                {"state": "y", "rhs": "weight * z"},
                {"state": "z", "rhs": "-z"},
            ],
            "observation_mappings": [
                {"channel": "target", "expression": "y", "unit": "unit"}
            ],
            "parameters": [
                {
                    "name": "weight",
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 4.0},
                    "initialization_range": {"lower": 0.0, "upper": 4.0},
                    "unit": "1/time",
                    "description": "Affine latent-to-observed weight.",
                }
            ],
            "initial_conditions": [
                {
                    "state": "y",
                    "scope": "global",
                    "initialization_range": {"lower": -2.0, "upper": 2.0},
                },
                {"state": "z", "scope": "global", "fixed_value": 1.0},
            ],
        }
    )


def _split(
    split_name: SplitName,
    identifier: str,
    observed_initial: float,
    weight: float,
) -> DatasetSplit:
    time = np.linspace(0.0, 3.0, 61)
    hidden_truth = np.exp(-time)
    target = observed_initial + weight * (1.0 - hidden_truth)
    trajectory = Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"target": target},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={"target": weight * hidden_truth},
        derivative_provenance=DerivativeProvenance.EXACT,
    )
    return DatasetSplit(split_name, (trajectory,), f"{identifier}-fingerprint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run_smoke(parser.parse_args().output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
