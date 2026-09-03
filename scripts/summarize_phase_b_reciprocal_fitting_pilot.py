#!/usr/bin/env python3
"""Summarize the frozen reciprocal-coordinate fitting pilot by endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.reciprocal_fitting_pilot import (
    canonical_reciprocal_fitting_plan_sha256,
    load_reciprocal_fitting_pilot_plan,
    reciprocal_fitting_task_count,
)


def summarize(experiment_root: Path) -> dict[str, object]:
    """Require the complete matrix and report non-aggregated fit endpoints."""
    experiment_root = experiment_root.expanduser().resolve()
    plan = load_reciprocal_fitting_pilot_plan(
        experiment_root / "frozen" / "plan.json"
    )
    plan_sha256 = canonical_reciprocal_fitting_plan_sha256(plan)
    rows: list[dict[str, Any]] = []
    task_artifacts: list[dict[str, object]] = []
    for index in range(reciprocal_fitting_task_count(plan)):
        path = experiment_root / "tasks" / f"task_{index:03d}.json"
        row = _read_object(path)
        if (
            row.get("schema_version")
            != "phase-b-reciprocal-fitting-pilot-task-1"
            or row.get("status") != "complete"
            or row.get("task_index") != index
            or row.get("plan_sha256") != plan_sha256
        ):
            raise ValueError(f"reciprocal fitting checkpoint differs: {path}")
        rows.append(row)
        task_artifacts.append(
            {
                "path": str(path.relative_to(experiment_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"]["condition_id"])].append(row)
    groups = [_group(name, items) for name, items in sorted(grouped.items())]
    matched = _matched(rows)
    output = experiment_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    task_ledger_path = output / "task_artifact_ledger.jsonl"
    _write_once(
        task_ledger_path,
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in task_artifacts
        ).encode(),
    )
    summary = {
        "schema_version": "phase-b-reciprocal-fitting-pilot-summary-1",
        "status": "complete",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_available_to_fitter": False,
        "exact_training_observed_derivatives_supplied": True,
        "validation_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "weighted_overall_score_defined": False,
        "automatic_winner_selected": False,
        "plan_sha256": plan_sha256,
        "task_artifact_ledger_sha256": _sha256(task_ledger_path),
        "task_count": len(rows),
        "groups": groups,
        "matched_profiled_coordinate_trials": matched,
    }
    json_path = output / "reciprocal_fitting_pilot_summary.json"
    markdown_path = output / "reciprocal_fitting_pilot_summary.md"
    _write_once(
        json_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(markdown_path, _markdown(groups, matched).encode())
    manifest = {
        "schema_version": "phase-b-reciprocal-fitting-pilot-run-1",
        "status": "complete",
        "plan_sha256": plan_sha256,
        "task_artifact_ledger_sha256": _sha256(task_ledger_path),
        "summary_sha256": _sha256(json_path),
        "report_sha256": _sha256(markdown_path),
        "test_data_opened": False,
        "private_reference_available_to_fitter": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = output / "manifest.json"
    _write_once(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(
        output / "manifest.json.sha256",
        f"{_sha256(manifest_path)}  manifest.json\n".encode(),
    )
    return summary


def _group(condition_id: str, rows: list[dict[str, Any]]) -> dict[str, object]:
    compatible = [row for row in rows if row["fit_contract_compatible"]]
    successful = [row for row in compatible if row["fit_success"]]
    return {
        "condition_id": condition_id,
        "trial_count": len(rows),
        "fit_contract_compatibility_rate": len(compatible) / len(rows),
        "fit_success_rate": len(successful) / len(rows),
        "median_training_normalized_mse_conditional_on_success": _median(
            [float(row["training_normalized_mse"]) for row in successful]
        ),
        "median_validation_normalized_mse_conditional_on_success": _median(
            [float(row["validation_normalized_mse"]) for row in successful]
        ),
        "total_function_evaluations": sum(
            int(row.get("function_evaluations", 0)) for row in rows
        ),
        "total_integration_failures": sum(
            int(row.get("integration_failures", 0)) for row in rows
        ),
        "total_fit_wall_seconds": sum(float(row["fit_wall_seconds"]) for row in rows),
        "total_fit_process_cpu_seconds": sum(
            float(row["fit_process_cpu_seconds"]) for row in rows
        ),
        "reciprocal_certificate_rate": sum(
            bool(row.get("certified_reciprocal_transformations")) for row in rows
        )
        / len(rows),
        "failure_types": _counts(
            str(row["error_type"])
            for row in rows
            if row.get("error_type") is not None
        ),
    }


def _matched(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    indexed = {
        (
            str(row["condition"]["condition_id"]),
            str(row["benchmark_id"]),
            int(row["repetition"]),
        ): row
        for row in rows
    }
    result: list[dict[str, object]] = []
    identities = sorted(
        {(str(row["benchmark_id"]), int(row["repetition"])) for row in rows}
    )
    for benchmark_id, repetition in identities:
        original = indexed[
            ("profiled_original_coordinate", benchmark_id, repetition)
        ]
        reciprocal = indexed[
            ("profiled_certified_reciprocal", benchmark_id, repetition)
        ]
        if original["candidate_sha256"] != reciprocal["candidate_sha256"]:
            raise ValueError("profiled coordinate arms used different candidates")
        both_successful = bool(original["fit_success"] and reciprocal["fit_success"])
        result.append(
            {
                "benchmark_id": benchmark_id,
                "repetition": repetition,
                "candidate_sha256": original["candidate_sha256"],
                "both_successful": both_successful,
                "reciprocal_certificate_available": bool(
                    reciprocal.get("certified_reciprocal_transformations")
                ),
                "validation_nmse_reciprocal_minus_original": (
                    float(reciprocal["validation_normalized_mse"])
                    - float(original["validation_normalized_mse"])
                    if both_successful
                    else None
                ),
                "function_evaluations_reciprocal_minus_original": (
                    int(reciprocal["function_evaluations"])
                    - int(original["function_evaluations"])
                    if both_successful
                    else None
                ),
                "wall_seconds_reciprocal_minus_original": (
                    float(reciprocal["fit_wall_seconds"])
                    - float(original["fit_wall_seconds"])
                    if both_successful
                    else None
                ),
            }
        )
    return result


def _markdown(
    groups: list[dict[str, object]], matched: list[dict[str, object]]
) -> str:
    lines = [
        "# Certified reciprocal-coordinate fitting pilot",
        "",
        "This development-only comparison uses the same frozen candidate in each "
        "fitting arm. Exact derivatives are supplied only for observed public "
        "channels; latent values and latent derivatives are never supplied. Test "
        "trajectories remain closed.",
        "",
        "| Condition | Contract coverage | Fit success | Median validation NMSE "
        "| Function evaluations | Wall seconds | Reciprocal certificate |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        validation = row["median_validation_normalized_mse_conditional_on_success"]
        lines.append(
            f"| {row['condition_id']} | "
            f"{float(row['fit_contract_compatibility_rate']):.3f} | "
            f"{float(row['fit_success_rate']):.3f} | "
            f"{_number(validation)} | "
            f"{int(row['total_function_evaluations'])} | "
            f"{float(row['total_fit_wall_seconds']):.3f} | "
            f"{float(row['reciprocal_certificate_rate']):.3f} |"
        )
    covered = sum(bool(row["both_successful"]) for row in matched)
    certified = sum(bool(row["reciprocal_certificate_available"]) for row in matched)
    lines.extend(
        [
            "",
            "Matched profiled comparisons with two successful fits: "
            f"{covered}/{len(matched)}.",
            "Matched candidates with a certified reciprocal coordinate: "
            f"{certified}/{len(matched)}.",
            "",
            "No weighted score or automatic winner is defined.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[str(value)] += 1
    return dict(sorted(result.items()))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"reciprocal pilot summary differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    result = summarize(parser.parse_args().experiment_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
