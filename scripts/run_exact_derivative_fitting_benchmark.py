#!/usr/bin/env python3
"""Benchmark Eq. (11) exact-derivative ridge against iterative regression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import (
    PiecewiseLinearForcing,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, evaluate_fitted_candidate, fit_candidate
from autoformalism.schemas import CandidateModel


def run_benchmark(config_path: Path, output_root: Path) -> dict[str, object]:
    """Run the frozen synthetic benchmark and preserve the pre-test fit freeze."""
    config_path = config_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    config = _read_object(config_path)
    _validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_sha256 = _sha256(config_path)
    _write_or_validate(
        output_root / "benchmark_freeze.json",
        json.dumps(
            {
                "schema_version": "exact-derivative-fitting-benchmark-freeze-1",
                "status": "frozen_before_fit_or_test_evaluation",
                "plan_path": str(config_path),
                "plan_sha256": plan_sha256,
                "derivative_provenance": "exact",
                "test_data_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    fitted_rows: list[dict[str, object]] = []
    pending: list[tuple[dict[str, Any], object, object, FitConfig, object]] = []
    for case in config["cases"]:
        candidate = _candidate(case)
        context = ValidationContext(
            targets=tuple(case["states"]),
            lagged_targets=tuple(case["states"]),
            external_inputs=tuple(case["external_inputs"]),
        )
        model = compile_candidate(candidate, context)
        truth = {
            item["name"]: float(item["true_value"])
            for item in case["parameters"]
        }
        training = _split(case, "train", SplitName.TRAIN, model, truth)
        validation = _split(
            case, "validation", SplitName.VALIDATION, model, truth
        )
        for repetition in config["repetitions"]:
            for backend in ("exact_derivative_linear_ridge", "bounded_nonlinear"):
                fit_config = _fit_config(config, backend, int(repetition))
                started = time.perf_counter()
                fitted = fit_candidate(model, training, validation, fit_config)
                elapsed = time.perf_counter() - started
                row = {
                    "case_id": case["case_id"],
                    "repetition": int(repetition),
                    "backend": backend,
                    "fit_success": fitted.success,
                    "fit_wall_seconds": elapsed,
                    "training_nmse": fitted.training_metrics.normalized_mse,
                    "validation_nmse": fitted.validation_metrics.normalized_mse,
                    "function_evaluations": sum(
                        item.function_evaluations for item in fitted.diagnostics
                    ),
                    "diagnostic_backends": sorted(
                        {item.backend for item in fitted.diagnostics}
                    ),
                    "true_parameters": truth,
                    "fitted_parameters": dict(fitted.global_parameters),
                    "maximum_absolute_parameter_error": max(
                        abs(float(fitted.global_parameters[name]) - value)
                        for name, value in truth.items()
                    ),
                    "test_nmse": None,
                    "test_failed_trajectories": None,
                }
                fitted_rows.append(row)
                pending.append((case, model, fitted, fit_config, row))

    fit_freeze_path = output_root / "fitted_parameter_freeze.jsonl"
    _write_or_validate(
        fit_freeze_path,
        "".join(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("test_") and key != "fit_wall_seconds"
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in fitted_rows
        ),
    )
    fit_freeze_sha256 = _sha256(fit_freeze_path)

    for case, model, fitted, fit_config, row in pending:
        truth = row["true_parameters"]
        assert isinstance(truth, dict)
        test = _split(case, "test", SplitName.TEST, model, truth)
        _, metrics = evaluate_fitted_candidate(
            model,
            test,
            global_parameters=fitted.global_parameters,
            global_initial_conditions=fitted.global_initial_conditions,
            target_scales=fitted.target_scales,
            config=fit_config,
        )
        row["test_nmse"] = metrics.normalized_mse
        row["test_failed_trajectories"] = list(metrics.failed_trajectories)

    rows_path = output_root / "benchmark_rows.jsonl"
    _write_atomic(
        rows_path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in fitted_rows
        ),
    )
    _write_csv(output_root / "benchmark_rows.csv", fitted_rows)
    summary = _summary(config, fitted_rows, plan_sha256, fit_freeze_sha256)
    _write_atomic(
        output_root / "benchmark_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(output_root / "benchmark_summary.md", _markdown(summary))
    return summary


def _candidate(case: dict[str, Any]) -> CandidateModel:
    states = tuple(str(item) for item in case["states"])
    return CandidateModel.model_validate(
        {
            "candidate_id": f"oracle_{case['case_id']}",
            "parent_candidate_id": None,
            "change_summary": "Frozen synthetic affine-parameter graph.",
            "states": [
                {"name": state, "kind": "observed", "unit": "synthetic_unit"}
                for state in states
            ],
            "state_equations": [
                {"state": state, "rhs": case["rhs"][state]} for state in states
            ],
            "observation_mappings": [
                {"channel": state, "expression": state, "unit": "synthetic_unit"}
                for state in states
            ],
            "parameters": [
                {
                    "name": item["name"],
                    "scope": "global",
                    "bounds": {"lower": item["lower"], "upper": item["upper"]},
                    "initialization_range": {
                        "lower": item["lower"],
                        "upper": item["upper"],
                    },
                    "unit": "synthetic_unit",
                }
                for item in case["parameters"]
            ],
            "initial_conditions": [
                {"state": state, "scope": "global", "expression": state}
                for state in states
            ],
        }
    )


def _split(
    case: dict[str, Any],
    split_key: str,
    split_name: SplitName,
    model: object,
    parameters: dict[str, float],
) -> DatasetSplit:
    trajectories = tuple(
        _trajectory(
            case,
            split_key,
            index,
            item,
            model,
            parameters,
        )
        for index, item in enumerate(case["splits"][split_key])
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "case_id": case["case_id"],
                "split": split_key,
                "trajectories": case["splits"][split_key],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return DatasetSplit(split_name, trajectories, fingerprint)


def _trajectory(
    case: dict[str, Any],
    split_key: str,
    index: int,
    specification: dict[str, Any],
    model: Any,
    parameters: dict[str, float],
) -> Trajectory:
    time_values = np.linspace(0.0, float(case["time_end"]), int(case["time_points"]))
    state_names = tuple(case["states"])
    initial = np.asarray(
        [float(specification["initial"][state]) for state in state_names]
    )
    input_values = {
        name: np.full(len(time_values), float(specification["inputs"][name]))
        for name in case["external_inputs"]
    }
    forcing = PiecewiseLinearForcing(
        time_values,
        input_values,
        allowed_channels=frozenset(case["external_inputs"]),
    )
    solution = solve_ivp(
        lambda current, state: model.rhs(current, state, parameters, forcing),
        (float(time_values[0]), float(time_values[-1])),
        initial,
        t_eval=time_values,
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(f"synthetic truth integration failed: {solution.message}")
    targets = {
        state: np.asarray(solution.y[state_index], dtype=float)
        for state_index, state in enumerate(state_names)
    }
    derivative_matrix = np.column_stack(
        [
            model.rhs(float(current), solution.y[:, sample], parameters, forcing)
            for sample, current in enumerate(time_values)
        ]
    )
    derivatives = {
        state: np.asarray(derivative_matrix[state_index], dtype=float)
        for state_index, state in enumerate(state_names)
    }
    return Trajectory(
        trajectory_id=f"{case['case_id']}__{split_key}_{index:02d}",
        time=time_values,
        targets=targets,
        auxiliaries={},
        external_inputs=input_values,
        fixed_covariates={},
        derivatives=derivatives,
        derivative_provenance=DerivativeProvenance.EXACT,
    )


def _fit_config(
    config: dict[str, Any], backend: str, repetition: int
) -> FitConfig:
    if backend == "exact_derivative_linear_ridge":
        return FitConfig(
            parameter_fit_strategy="exact_derivative_linear_ridge",
            derivative_ridge_regularization=float(config["ridge_regularization"]),
            random_seed=repetition,
            relative_tolerance=1e-9,
            absolute_tolerance=1e-11,
        )
    return FitConfig(
        parameter_fit_strategy="bounded_nonlinear",
        allow_derivative_regression=True,
        number_of_starts=int(config["bounded_nonlinear_starts"]),
        maximum_function_evaluations=int(config["bounded_nonlinear_max_nfev"]),
        random_seed=repetition,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
    )


def _summary(
    config: dict[str, Any],
    rows: list[dict[str, object]],
    plan_sha256: str,
    fit_freeze_sha256: str,
) -> dict[str, object]:
    groups = []
    for backend in ("exact_derivative_linear_ridge", "bounded_nonlinear"):
        selected = [item for item in rows if item["backend"] == backend]
        groups.append(
            {
                "backend": backend,
                "trial_count": len(selected),
                "success_rate": sum(bool(item["fit_success"]) for item in selected)
                / len(selected),
                "fit_wall_seconds_mean": statistics.fmean(
                    float(item["fit_wall_seconds"]) for item in selected
                ),
                "function_evaluations_mean": statistics.fmean(
                    float(item["function_evaluations"]) for item in selected
                ),
                "maximum_absolute_parameter_error_mean": statistics.fmean(
                    float(item["maximum_absolute_parameter_error"])
                    for item in selected
                ),
                "validation_nmse_mean": statistics.fmean(
                    float(item["validation_nmse"]) for item in selected
                ),
                "test_nmse_mean": statistics.fmean(
                    float(item["test_nmse"]) for item in selected
                ),
            }
        )
    return {
        "schema_version": "exact-derivative-fitting-benchmark-summary-1",
        "status": "complete",
        "plan_sha256": plan_sha256,
        "fitted_parameter_freeze_sha256": fit_freeze_sha256,
        "case_count": len(config["cases"]),
        "repetition_count": len(config["repetitions"]),
        "derivative_provenance": "exact",
        "test_evaluated_only_after_parameter_freeze": True,
        "weighted_overall_score_defined": False,
        "groups": groups,
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Exact-derivative affine-parameter fitting benchmark",
        "",
        "Synthetic states and exact derivatives are fully observed. Test rollouts "
        "are evaluated only after fitted parameters are frozen.",
        "",
        "| Backend | Trials | Success | Mean fit seconds | Mean evaluations | "
        "Mean max parameter error | Mean validation NMSE | Mean test NMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["groups"]:  # type: ignore[index]
        lines.append(
            "| {backend} | {trial_count} | {success_rate:.4g} | "
            "{fit_wall_seconds_mean:.4g} | {function_evaluations_mean:.4g} | "
            "{maximum_absolute_parameter_error_mean:.4g} | "
            "{validation_nmse_mean:.4g} | {test_nmse_mean:.4g} |".format(**item)
        )
    return "\n".join(lines) + "\n"


def _validate_config(config: dict[str, object]) -> None:
    if config.get("schema_version") != "exact-derivative-fitting-benchmark-plan-1":
        raise ValueError("unsupported fitting benchmark plan")
    if config.get("status") != "frozen_before_benchmark_run":
        raise ValueError("fitting benchmark plan is not frozen")
    if config.get("derivative_provenance") != "exact":
        raise ValueError("benchmark requires exact derivative provenance")
    if config.get("test_evaluated_only_after_parameter_freeze") is not True:
        raise ValueError("benchmark must freeze fitted parameters before test")
    if config.get("weighted_overall_score_defined") is not False:
        raise ValueError("benchmark must not define a weighted overall score")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fitting benchmark has no cases")
    identifiers = [str(item["case_id"]) for item in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fitting benchmark cases are duplicated")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened = [
        {
            **row,
            "diagnostic_backends": json.dumps(row["diagnostic_backends"]),
            "true_parameters": json.dumps(row["true_parameters"], sort_keys=True),
            "fitted_parameters": json.dumps(
                row["fitted_parameters"], sort_keys=True
            ),
            "test_failed_trajectories": json.dumps(
                row["test_failed_trajectories"]
            ),
        }
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_or_validate(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    _write_atomic(path, text)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run_benchmark(args.config, args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
