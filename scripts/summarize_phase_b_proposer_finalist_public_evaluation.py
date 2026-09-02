#!/usr/bin/env python3
"""Summarize the paired public finalist evaluation without a total score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.proposer_finalist_evaluation import (
    canonical_plan_sha256,
    finalist_task_count,
    load_proposer_finalist_evaluation_plan,
)


def main() -> None:
    """Require all task checkpoints and emit endpoint-wise paired summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    plan_path = root / "frozen" / "plan.json"
    plan = load_proposer_finalist_evaluation_plan(plan_path)
    plan_sha = canonical_plan_sha256(plan)
    freeze = _read_object(root / "frozen" / "freeze_manifest.json")
    if freeze.get("plan_sha256") != plan_sha:
        raise ValueError("frozen finalist plan SHA-256 differs")

    missing: list[str] = []
    tasks: list[dict[str, Any]] = []
    for index in range(finalist_task_count(plan)):
        path = root / "tasks" / f"task_{index:03d}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = _read_object(path)
        if (
            payload.get("schema_version")
            != "phase-b-proposer-finalist-public-evaluation-task-1"
            or payload.get("status") != "complete"
            or payload.get("task_index") != index
            or payload.get("plan_sha256") != plan_sha
        ):
            raise ValueError(f"finalist task checkpoint differs: {path}")
        tasks.append(payload)
    if missing:
        raise ValueError(
            "finalist evaluation is incomplete; missing=" + json.dumps(missing)
        )

    condition_rows = _condition_summaries(tasks)
    paired_rows = _paired_rows(tasks, plan)
    flat_rows = [_flat_row(item) for item in tasks]
    summary = {
        "schema_version": "phase-b-proposer-finalist-public-evaluation-summary-1",
        "status": "complete",
        "development_only": True,
        "new_llm_calls_made": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
        "automatic_operating_point_selected": False,
        "plan_sha256": plan_sha,
        "task_count": len(tasks),
        "matched_pair_count": len(paired_rows),
        "conditions": condition_rows,
        "paired_trials": paired_rows,
    }
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, object]] = []
    for path, data, role in (
        (
            output / "finalist_public_evaluation_summary.json",
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
            "endpoint_summary",
        ),
        (
            output / "finalist_public_evaluation_summary.md",
            _markdown(condition_rows, paired_rows).encode(),
            "human_report",
        ),
        (
            output / "finalist_public_evaluation_rows.jsonl",
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in tasks).encode(),
            "task_rows",
        ),
    ):
        _write_once(path, data)
        artifacts.append(_artifact(path, root, role))
    csv_path = output / "finalist_public_evaluation_rows.csv"
    _write_csv_once(csv_path, flat_rows)
    artifacts.append(_artifact(csv_path, root, "flat_task_metrics"))
    paired_path = output / "finalist_public_evaluation_pairs.csv"
    _write_csv_once(paired_path, paired_rows)
    artifacts.append(_artifact(paired_path, root, "paired_metrics"))
    for index in range(finalist_task_count(plan)):
        path = root / "tasks" / f"task_{index:03d}.json"
        artifacts.append(_artifact(path, root, "task_checkpoint"))
    for path, role in (
        (plan_path, "frozen_plan"),
        (root / "frozen" / "freeze_manifest.json", "freeze_manifest"),
    ):
        artifacts.append(_artifact(path, root, role))
    ledger_path = output / "artifact_ledger.jsonl"
    _write_once(
        ledger_path,
        "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in sorted(artifacts, key=lambda item: str(item["path"]))
        ).encode(),
    )
    run_manifest = {
        "schema_version": "phase-b-proposer-finalist-public-evaluation-run-1",
        "status": "complete",
        "plan_sha256": plan_sha,
        "source_replay_artifact_ledger_sha256": (
            plan.source_replay.artifact_ledger_sha256
        ),
        "result_artifact_ledger_sha256": _sha256(ledger_path),
        "result_artifact_count": len(artifacts),
        "new_llm_calls_made": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = output / "finalist_public_evaluation_manifest.json"
    _write_once(
        manifest_path,
        (json.dumps(run_manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(
        output / "finalist_public_evaluation_manifest.json.sha256",
        f"{_sha256(manifest_path)}  {manifest_path.name}\n".encode(),
    )
    print((output / "finalist_public_evaluation_summary.md").read_text())


def _condition_summaries(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in tasks:
        condition = item["condition"]
        grouped[(condition["reasoning_effort"], condition["max_output_tokens"])].append(
            item
        )
    rows = []
    for (effort, budget), items in grouped.items():
        successful = [item for item in items if item["fit_success"]]
        validation = [
            float(item["selected_fit"]["validation_normalized_mse"])
            for item in successful
        ]
        training = [
            float(item["selected_fit"]["training_normalized_mse"])
            for item in successful
        ]
        attempts = [attempt for item in items for attempt in item["fit_attempts"]]
        complexities = [item["complexity"] for item in items]
        mechanism = [item["public_mechanism"] for item in items]
        profile_counts = Counter(
            str(item["selected_fit_profile"])
            for item in successful
        )
        rows.append(
            {
                "reasoning_effort": effort,
                "max_output_tokens": budget,
                "trial_count": len(items),
                "runtime_validity_rate": _rate(
                    sum(item["runtime"]["valid"] for item in items), items
                ),
                "public_target_pass_rate": _rate(
                    sum(item["public_target"]["passed"] for item in items), items
                ),
                "mean_public_mechanism_compliance": _mean(
                    [float(item["mechanism_compliance"]) for item in mechanism]
                ),
                "complete_public_mechanism_assessment_rate": _rate(
                    sum(
                        item["mechanism_compliance_complete"]
                        for item in mechanism
                    ),
                    mechanism,
                ),
                "fit_success_rate": _rate(len(successful), items),
                "primary_fit_success_rate": _rate(
                    sum(
                        bool(item["fit_attempts"])
                        and item["fit_attempts"][0].get("success") is True
                        for item in items
                    ),
                    items,
                ),
                "mean_training_normalized_mse_conditional_on_fit": _mean(training),
                "median_training_normalized_mse_conditional_on_fit": _median(training),
                "mean_validation_normalized_mse_conditional_on_fit": _mean(validation),
                "median_validation_normalized_mse_conditional_on_fit": _median(
                    validation
                ),
                "selected_fit_profile_counts": dict(sorted(profile_counts.items())),
                "total_function_evaluations": sum(
                    int(item.get("function_evaluations", 0)) for item in attempts
                ),
                "total_integration_failures": sum(
                    int(item.get("integration_failures", 0)) for item in attempts
                ),
                "total_fit_wall_seconds": sum(
                    float(item["wall_seconds"]) for item in attempts
                ),
                "total_process_cpu_seconds": sum(
                    float(item["process_cpu_seconds"]) for item in attempts
                ),
                "allocated_cpu_core_hours": sum(
                    float(attempt["wall_seconds"])
                    * int(item.get("allocated_cpus") or 1)
                    / 3600.0
                    for item in items
                    for attempt in item["fit_attempts"]
                ),
                "mean_state_count": _mean(
                    [float(item["state_count"]) for item in complexities]
                ),
                "mean_latent_state_count": _mean(
                    [float(item["latent_state_count"]) for item in complexities]
                ),
                "mean_parameter_count": _mean(
                    [float(item["parameter_count"]) for item in complexities]
                ),
                "mean_additive_term_count": _mean(
                    [
                        float(item["state_equation_additive_term_count"])
                        for item in complexities
                    ]
                ),
                "mean_expression_ast_node_count": _mean(
                    [
                        float(item["total_expression_ast_node_count"])
                        for item in complexities
                    ]
                ),
            }
        )
    return rows


def _paired_rows(tasks: list[dict[str, Any]], plan: object) -> list[dict[str, Any]]:
    if len(plan.conditions) != 2:
        raise ValueError("paired summary requires exactly two conditions")
    labels = [item.directory_name for item in plan.conditions]
    indexed = {
        (
            f"{item['condition']['reasoning_effort']}_"
            f"{item['condition']['max_output_tokens']:06d}",
            item["benchmark_id"],
            item["repetition"],
        ): item
        for item in tasks
    }
    rows = []
    for cell in plan.cells:
        for repetition in plan.repetitions:
            first = indexed[(labels[0], cell.benchmark_id, repetition)]
            second = indexed[(labels[1], cell.benchmark_id, repetition)]
            first_nmse = _selected_nmse(first)
            second_nmse = _selected_nmse(second)
            if first_nmse is not None and second_nmse is not None:
                if first_nmse < second_nmse:
                    fit_outcome = labels[0]
                elif second_nmse < first_nmse:
                    fit_outcome = labels[1]
                else:
                    fit_outcome = "tie"
            elif first_nmse is not None:
                fit_outcome = f"{labels[0]}_only_fit"
            elif second_nmse is not None:
                fit_outcome = f"{labels[1]}_only_fit"
            else:
                fit_outcome = "neither_fit"
            rows.append(
                {
                    "benchmark_id": cell.benchmark_id,
                    "tier": cell.tier,
                    "repetition": repetition,
                    "first_condition": labels[0],
                    "second_condition": labels[1],
                    "first_fit_success": first["fit_success"],
                    "second_fit_success": second["fit_success"],
                    "first_validation_normalized_mse": first_nmse,
                    "second_validation_normalized_mse": second_nmse,
                    "validation_nmse_ratio_first_over_second": (
                        first_nmse / second_nmse
                        if first_nmse is not None
                        and second_nmse is not None
                        and second_nmse != 0.0
                        else None
                    ),
                    "log10_validation_nmse_ratio_first_over_second": (
                        math.log10(first_nmse / second_nmse)
                        if first_nmse is not None
                        and second_nmse is not None
                        and first_nmse > 0.0
                        and second_nmse > 0.0
                        else None
                    ),
                    "fit_outcome": fit_outcome,
                    "first_mechanism_compliance": first["public_mechanism"][
                        "mechanism_compliance"
                    ],
                    "second_mechanism_compliance": second["public_mechanism"][
                        "mechanism_compliance"
                    ],
                    "first_ast_node_count": first["complexity"][
                        "total_expression_ast_node_count"
                    ],
                    "second_ast_node_count": second["complexity"][
                        "total_expression_ast_node_count"
                    ],
                }
            )
    return rows


def _flat_row(item: dict[str, Any]) -> dict[str, Any]:
    selected = item["selected_fit"]
    condition = item["condition"]
    mechanism = item["public_mechanism"]
    complexity = item["complexity"]
    attempts = item["fit_attempts"]
    return {
        "task_index": item["task_index"],
        "source_task_index": item["source_task_index"],
        "reasoning_effort": condition["reasoning_effort"],
        "max_output_tokens": condition["max_output_tokens"],
        "benchmark_id": item["benchmark_id"],
        "tier": item["tier"],
        "repetition": item["repetition"],
        "candidate_sha256": item["candidate_sha256"],
        "runtime_valid": item["runtime"]["valid"],
        "public_target_passed": item["public_target"]["passed"],
        "mechanism_compliance": mechanism["mechanism_compliance"],
        "mechanism_compliance_complete": mechanism[
            "mechanism_compliance_complete"
        ],
        "fit_success": item["fit_success"],
        "selected_fit_profile": item["selected_fit_profile"],
        "training_normalized_mse": (
            None if selected is None else selected["training_normalized_mse"]
        ),
        "validation_normalized_mse": (
            None if selected is None else selected["validation_normalized_mse"]
        ),
        "fit_attempt_count": len(attempts),
        "fit_wall_seconds": sum(float(row["wall_seconds"]) for row in attempts),
        "process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in attempts
        ),
        "function_evaluations": sum(
            int(row.get("function_evaluations", 0)) for row in attempts
        ),
        "integration_failures": sum(
            int(row.get("integration_failures", 0)) for row in attempts
        ),
        **complexity,
    }


def _markdown(
    conditions: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> str:
    lines = [
        "# Repaired proposer finalist public evaluation",
        "",
        "This development comparison reused frozen candidates. It made no new "
        "LLM or scientific-judge calls and opened no test or private data. Metrics "
        "are reported separately; no weighted overall score or automatic operating "
        "point is defined.",
        "",
        "| Reasoning | Tokens | Fit success | Primary fit | Target pass | "
        "Mechanism compliance | Median validation NMSE* | Fit wall (s) | "
        "Process CPU (s) |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions:
        validation = item[
            "median_validation_normalized_mse_conditional_on_fit"
        ]
        lines.append(
            f"| {item['reasoning_effort']} | {item['max_output_tokens']} | "
            f"{item['fit_success_rate']:.3f} | "
            f"{item['primary_fit_success_rate']:.3f} | "
            f"{item['public_target_pass_rate']:.3f} | "
            f"{item['mean_public_mechanism_compliance']:.3f} | "
            f"{_format_optional(validation)} | "
            f"{item['total_fit_wall_seconds']:.1f} | "
            f"{item['total_process_cpu_seconds']:.1f} |"
        )
    outcomes = Counter(str(item["fit_outcome"]) for item in paired)
    lines.extend(
        [
            "",
            "\\* Conditional on a successful common-budget fit; fit coverage is "
            "reported separately.",
            "",
            "## Paired public-validation outcomes",
            "",
            f"- Matched pairs: {len(paired)}",
            *(
                f"- `{key}`: {value}"
                for key, value in sorted(outcomes.items())
            ),
            "",
            "This report is descriptive. Selection of the production proposer "
            "condition requires reviewing fit availability, validation NMSE, "
            "mechanism compliance, complexity, numerical stability, and cost as "
            "separate endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def _selected_nmse(item: dict[str, Any]) -> float | None:
    selected = item["selected_fit"]
    return None if selected is None else float(selected["validation_normalized_mse"])


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rate(numerator: int, denominator: list[object]) -> float:
    return numerator / len(denominator) if denominator else 0.0


def _format_optional(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = temporary.read_bytes()
    temporary.unlink()
    _write_once(path, data)


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"evaluation artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def _artifact(path: Path, root: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
