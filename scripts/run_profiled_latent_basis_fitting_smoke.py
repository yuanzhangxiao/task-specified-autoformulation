#!/usr/bin/env python3
"""Exercise profiled fitting of a nonlinear latent shape and affine weight."""

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
from autoformalism.fitting import FitConfig, evaluate_fitted_candidate, fit_candidate
from autoformalism.schemas import CandidateModel


def run_smoke(output: Path) -> dict[str, object]:
    """Recover one latent relaxation rate and one affine response weight."""
    truth = {"weight": 2.5, "tau": 1.4}
    model = compile_candidate(
        _candidate(), ValidationContext(targets=("target",))
    )
    training = _split(SplitName.TRAIN, "train", 0.2, truth)
    validation = _split(SplitName.VALIDATION, "validation", -0.3, truth)
    heldout = _split(SplitName.TEST, "test", 0.6, truth)
    config = FitConfig(
        parameter_fit_strategy="profiled_latent_basis_linear_ridge",
        number_of_starts=2,
        maximum_function_evaluations=100,
        derivative_ridge_regularization=0.0,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
    )
    fitted = fit_candidate(model, training, validation, config)
    if not fitted.success:
        raise RuntimeError(f"profiled latent-basis fit failed: {fitted.message}")
    _, test_metrics = evaluate_fitted_candidate(
        model,
        heldout,
        global_parameters=fitted.global_parameters,
        global_initial_conditions=fitted.global_initial_conditions,
        target_scales=fitted.target_scales,
        config=config,
    )
    errors = {
        name: abs(float(fitted.global_parameters[name]) - value)
        for name, value in truth.items()
    }
    if max(errors.values()) > 2e-5 or test_metrics.normalized_mse > 1e-9:
        raise RuntimeError(
            "profiled latent-basis recovery missed its tolerance: "
            f"parameter_errors={errors}, test_nmse={test_metrics.normalized_mse}"
        )
    result = {
        "schema_version": "profiled-latent-basis-fitting-smoke-1",
        "status": "pass",
        "backend": fitted.diagnostics[fitted.best_start_index].backend,
        "exact_observed_derivatives_supplied": True,
        "latent_values_supplied_to_fitter": False,
        "latent_derivatives_supplied_to_fitter": False,
        "true_parameters": truth,
        "fitted_parameters": dict(fitted.global_parameters),
        "absolute_parameter_errors": errors,
        "outer_function_evaluations": fitted.diagnostics[
            fitted.best_start_index
        ].function_evaluations,
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
            "candidate_id": "profiled_latent_basis_smoke",
            "parent_candidate_id": None,
            "change_summary": "A fitted hidden relaxation basis drives y.",
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
                    "unit": "unit",
                    "description": "Unobserved relaxation memory.",
                },
            ],
            "state_equations": [
                {"state": "y", "rhs": "weight * z"},
                {"state": "z", "rhs": "-z / tau"},
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
                },
                {
                    "name": "tau",
                    "scope": "global",
                    "bounds": {"lower": 0.1, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 2.0},
                    "unit": "1/time",
                    "description": "Latent relaxation time.",
                },
            ],
            "initial_conditions": [
                {
                    "state": "y",
                    "scope": "global",
                    "initialization_range": {"lower": -1.0, "upper": 1.0},
                },
                {"state": "z", "scope": "global", "fixed_value": 1.0},
            ],
        }
    )


def _split(
    split_name: SplitName,
    identifier: str,
    observed_initial: float,
    truth: dict[str, float],
) -> DatasetSplit:
    time = np.linspace(0.0, 3.0, 61)
    hidden_truth = np.exp(-time / truth["tau"])
    target = observed_initial + (
        truth["weight"] * truth["tau"] * (1.0 - hidden_truth)
    )
    trajectory = Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"target": target},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={"target": truth["weight"] * hidden_truth},
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
