#!/usr/bin/env python3
"""Merge separate sealed predictive endpoints for frozen classical baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

from autoformalism.rebuttal.baseline_postfreeze import (
    BaselinePredictiveTestResult,
    FrozenBaselineModel,
)


def main() -> None:
    """Validate task coverage and write method-level endpoint summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-model-root", type=Path, required=True)
    parser.add_argument("--predictive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.final_model_root,
        args.predictive_root,
        args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def summarize(
    final_model_root: Path,
    predictive_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Join every predictive result to exactly one frozen final model."""
    frozen = final_model_root.expanduser().resolve()
    predictive = predictive_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    manifest_path = frozen / "final_model_freeze.json"
    manifest = _read_object(manifest_path)
    models = tuple(
        FrozenBaselineModel.model_validate_json(line)
        for line in (frozen / "frozen_baseline_models.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if (
        manifest.get("status") != "frozen_before_test_or_private_evaluation"
        or manifest.get("frozen_models_sha256")
        != _sha256(frozen / "frozen_baseline_models.jsonl")
        or len(models) != int(manifest.get("model_count", -1))
    ):
        raise ValueError("final-model freeze differs from its manifest")
    results: list[BaselinePredictiveTestResult] = []
    for model in models:
        model_path = frozen / "models" / f"task_{model.task_index:03d}.json"
        result_path = predictive / "tasks" / f"task_{model.task_index:03d}.json"
        result = BaselinePredictiveTestResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if (
            result.task_index,
            result.method,
            result.benchmark_id,
            result.tier,
            result.seed,
            result.frozen_model_sha256,
        ) != (
            model.task_index,
            model.method,
            model.benchmark_id,
            model.tier,
            model.seed,
            _sha256(model_path),
        ):
            raise ValueError(f"predictive result differs: {model.task_index}")
        results.append(result)
    if {item.task_index for item in results} != set(range(len(models))):
        raise ValueError("predictive results do not cover the frozen task matrix")

    groups = []
    for method in ("persistence", "sindy", "pysr"):
        selected = [item for item in results if item.method == method]
        available = [item for item in selected if item.status == "available"]
        scores = [float(item.normalized_mse) for item in available]
        trajectory_total = sum(item.trajectory_count for item in selected)
        groups.append(
            {
                "method": method,
                "evaluation_protocol": (
                    "causal_previous_observation"
                    if method == "persistence"
                    else "causal_one_step_observed_state_reset"
                ),
                "trial_count": len(selected),
                "available_count": len(available),
                "endpoint_coverage": len(available) / len(selected),
                "test_trajectory_success_rate": (
                    sum(item.successful_trajectory_count for item in selected)
                    / trajectory_total
                    if trajectory_total
                    else None
                ),
                "mean_test_normalized_mse": (
                    statistics.fmean(scores) if scores else None
                ),
                "median_test_normalized_mse": (
                    statistics.median(scores) if scores else None
                ),
            }
        )
    summary = {
        "schema_version": "phase-b-public-baseline-predictive-summary-1",
        "status": "complete",
        "task_count": len(results),
        "groups": groups,
        "final_model_freeze_sha256": _sha256(manifest_path),
        "test_data_opened_after_freeze": True,
        "private_reference_opened": False,
        "oracle_derivatives_used": False,
        "oracle_latent_states_used": False,
        "weighted_overall_score_defined": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "predictive_test_results.jsonl"
    _write_atomic(
        results_path,
        "".join(item.model_dump_json() + "\n" for item in results),
    )
    rows = [item.model_dump(mode="json") for item in results]
    csv_path = output / "predictive_test_results.csv"
    _write_csv(csv_path, rows)
    summary_path = output / "predictive_test_summary.json"
    _write_atomic(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_atomic(
        output / "predictive_test_summary.md",
        _markdown(groups),
    )
    run_manifest = {
        "schema_version": "phase-b-public-baseline-predictive-run-1",
        "status": "complete",
        "predictive_results_sha256": _sha256(results_path),
        "predictive_results_csv_sha256": _sha256(csv_path),
        "predictive_summary_sha256": _sha256(summary_path),
        "freeze_receipt_sha256": _sha256(
            predictive / "predictive_test_freeze_receipt.json"
        ),
        "test_data_opened_after_freeze": True,
        "private_reference_opened": False,
    }
    _write_atomic(
        output / "predictive_test_manifest.json",
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    )
    return summary


def _markdown(groups: list[dict[str, object]]) -> str:
    lines = [
        "# Frozen classical baseline predictive evaluation",
        "",
        "Test was opened only after the public final-model freeze. Endpoints are "
        "reported separately; no weighted overall score is defined.",
        "",
        "| Method | Protocol | Coverage | Trajectory success | Mean test NMSE | "
        "Median test NMSE |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for item in groups:
        lines.append(
            "| {method} | {evaluation_protocol} | {endpoint_coverage:.3f} | "
            "{test_trajectory_success_rate:.3f} | {mean} | {median} |".format(
                **item,
                mean=_number(item["mean_test_normalized_mse"]),
                median=_number(item["median_test_normalized_mse"]),
            )
        )
    return "\n".join(lines) + "\n"


def _number(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = tuple(rows[0]) if rows else ("task_index",)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required evaluation artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
