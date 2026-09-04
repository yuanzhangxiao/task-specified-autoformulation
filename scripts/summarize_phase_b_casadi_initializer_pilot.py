#!/usr/bin/env python3
"""Summarize the complete matched CasADi initializer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.casadi_initializer_pilot import (
    canonical_casadi_initializer_plan_sha256,
    casadi_initializer_task_count,
    load_casadi_initializer_pilot_plan,
)


def summarize(experiment_root: Path) -> dict[str, object]:
    """Require all tasks, verify pairing, and report separate endpoints."""
    experiment_root = experiment_root.expanduser().resolve()
    plan = load_casadi_initializer_pilot_plan(
        experiment_root / "frozen" / "plan.json"
    )
    plan_sha256 = canonical_casadi_initializer_plan_sha256(plan)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, object]] = []
    for task_index in range(casadi_initializer_task_count(plan)):
        path = experiment_root / "tasks" / f"task_{task_index:03d}.json"
        row = _read_object(path)
        if (
            row.get("schema_version")
            != "phase-b-casadi-initializer-pilot-task-1"
            or row.get("status") != "complete"
            or row.get("task_index") != task_index
            or row.get("plan_sha256") != plan_sha256
        ):
            raise ValueError(f"CasADi initializer checkpoint differs: {path}")
        rows.append(row)
        artifacts.append(
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
    task_ledger = output / "task_artifact_ledger.jsonl"
    _write_once(
        task_ledger,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifacts).encode(),
    )
    summary = {
        "schema_version": "phase-b-casadi-initializer-pilot-summary-1",
        "status": "complete",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_available_to_fitter": False,
        "observed_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "weighted_overall_score_defined": False,
        "automatic_winner_selected": False,
        "same_candidate_across_conditions": True,
        "equal_total_wall_time_budget": True,
        "plan_sha256": plan_sha256,
        "task_artifact_ledger_sha256": _sha256(task_ledger),
        "task_count": len(rows),
        "groups": groups,
        "matched_trials": matched,
    }
    json_path = output / "casadi_initializer_pilot_summary.json"
    markdown_path = output / "casadi_initializer_pilot_summary.md"
    _write_once(
        json_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(markdown_path, _markdown(groups, matched).encode())
    manifest = {
        "schema_version": "phase-b-casadi-initializer-pilot-run-1",
        "status": "complete",
        "plan_sha256": plan_sha256,
        "task_artifact_ledger_sha256": _sha256(task_ledger),
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
    initialization = [
        item
        for row in rows
        for item in row.get("initialization_diagnostics", [])
    ]
    return {
        "condition_id": condition_id,
        "trial_count": len(rows),
        "fit_contract_compatibility_rate": len(compatible) / len(rows),
        "initializer_activation_rate": len(initialization) / len(rows),
        "initializer_success_rate": (
            sum(bool(item["success"]) for item in initialization) / len(rows)
        ),
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
        "total_initializer_wall_seconds": sum(
            float(item["wall_seconds"]) for item in initialization
        ),
        "total_fit_wall_seconds": sum(float(row["fit_wall_seconds"]) for row in rows),
        "total_fit_process_cpu_seconds": sum(
            float(row["fit_process_cpu_seconds"]) for row in rows
        ),
        "failure_types": dict(
            Counter(
                str(row["error_type"])
                for row in rows
                if row.get("error_type") is not None
            )
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
    identities = sorted(
        {(str(row["benchmark_id"]), int(row["repetition"])) for row in rows}
    )
    result: list[dict[str, object]] = []
    for benchmark_id, repetition in identities:
        baseline = indexed[("runtime_owned_start", benchmark_id, repetition)]
        casadi = indexed[
            ("casadi_multiple_shooting_start", benchmark_id, repetition)
        ]
        if baseline["candidate_sha256"] != casadi["candidate_sha256"]:
            raise ValueError("initializer arms used different source candidates")
        if baseline["fit_candidate_identity"] != casadi["fit_candidate_identity"]:
            raise ValueError("initializer arms used different executable candidates")
        baseline_budget = _total_budget(baseline)
        casadi_budget = _total_budget(casadi)
        if baseline_budget != casadi_budget:
            raise ValueError("initializer arms used different total wall budgets")
        both_successful = bool(baseline["fit_success"] and casadi["fit_success"])
        init_diagnostics = casadi.get("initialization_diagnostics", [])
        result.append(
            {
                "benchmark_id": benchmark_id,
                "repetition": repetition,
                "candidate_sha256": baseline["candidate_sha256"],
                "equal_total_wall_time_budget_seconds": baseline_budget,
                "casadi_initializer_activated": bool(init_diagnostics),
                "casadi_initializer_succeeded": bool(
                    init_diagnostics and init_diagnostics[0]["success"]
                ),
                "both_fits_successful": both_successful,
                "validation_nmse_casadi_minus_runtime_start": (
                    float(casadi["validation_normalized_mse"])
                    - float(baseline["validation_normalized_mse"])
                    if both_successful
                    else None
                ),
                "function_evaluations_casadi_minus_runtime_start": (
                    int(casadi["function_evaluations"])
                    - int(baseline["function_evaluations"])
                    if both_successful
                    else None
                ),
                "wall_seconds_casadi_minus_runtime_start": (
                    float(casadi["fit_wall_seconds"])
                    - float(baseline["fit_wall_seconds"])
                ),
            }
        )
    return result


def _total_budget(row: dict[str, Any]) -> float:
    condition = row["condition"]
    return float(condition["core_fit_wall_time_seconds"]) + float(
        condition["initializer_wall_time_seconds"]
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _markdown(
    groups: list[dict[str, object]], matched: list[dict[str, object]]
) -> str:
    lines = [
        "# CasADi multiple-shooting initializer pilot",
        "",
        (
            "The same frozen candidate and equal total wall-time budget are used "
            "in both arms. Public train/validation trajectories only; no exact "
            "derivatives, latent values, test data, or private reference are supplied."
        ),
        "",
        (
            "| Condition | Contract coverage | Initializer success | Fit success | "
            "Median validation NMSE | Function evaluations | Total wall seconds |"
        ),
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            "| {condition_id} | {coverage:.3f} | {initializer:.3f} | "
            "{success:.3f} | {nmse} | {evaluations} | {wall:.3f} |".format(
                condition_id=row["condition_id"],
                coverage=row["fit_contract_compatibility_rate"],
                initializer=row["initializer_success_rate"],
                success=row["fit_success_rate"],
                nmse=_display(row["median_validation_normalized_mse_conditional_on_success"]),
                evaluations=row["total_function_evaluations"],
                wall=row["total_fit_wall_seconds"],
            )
        )
    both = sum(bool(item["both_fits_successful"]) for item in matched)
    init_success = sum(
        bool(item["casadi_initializer_succeeded"]) for item in matched
    )
    lines.extend(
        [
            "",
            f"Matched trials with two successful final fits: {both}/{len(matched)}.",
            (
                "Matched trials with successful CasADi initialization: "
                f"{init_success}/{len(matched)}."
            ),
            "",
            "No weighted score or automatic winner is defined.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"CasADi pilot artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
