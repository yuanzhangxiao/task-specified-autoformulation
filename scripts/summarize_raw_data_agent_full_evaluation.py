#!/usr/bin/env python3
"""Combine numerical and fit-free scientific evaluation for raw-agent runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _expected_keys(config: dict[str, Any]) -> tuple[tuple[str, str, int], ...]:
    repetitions = config["repetitions"]
    return tuple(
        (item["benchmark_id"], item["tier"], repetition)
        for item in config["benchmarks"]
        for repetition in range(repetitions)
    )


def _run_index(root: Path) -> dict[tuple[str, str, int], Path]:
    result: dict[tuple[str, str, int], Path] = {}
    for config_path in sorted(root.glob("*/run_config.json")):
        config = _read_object(config_path)
        key = (
            str(config["benchmark_id"]),
            str(config["tier"]),
            int(config["repetition"]),
        )
        if key in result:
            raise ValueError(f"duplicate raw-agent run key: {key}")
        result[key] = config_path.parent
    return result


def _audit_index(
    manifest_path: Path | None, summary_path: Path | None
) -> dict[str, dict[str, Any]]:
    if manifest_path is None or summary_path is None:
        return {}
    manifest = _read_object(manifest_path)
    summary = _read_object(summary_path)
    outcome_index = {
        item["pair_id"]: item for item in summary.get("outcomes", [])
    }
    result = {}
    for source in manifest.get("source_runs", []):
        pair_id = source.get("pair_id")
        run = source.get("run")
        if isinstance(pair_id, str) and isinstance(run, str):
            result[Path(run).name] = outcome_index.get(
                pair_id,
                {"pair_id": pair_id, "status": "missing_audit_outcome"},
            )
    return result


def _tool_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["run"]: row for row in csv.DictReader(handle)}


def _metric_payload(
    evaluation: dict[str, Any], split: str
) -> dict[str, Any]:
    value = evaluation.get(f"{split}_metrics")
    return value if isinstance(value, dict) else {}


def _constraint_max(metrics: dict[str, Any], field: str) -> float | None:
    constraints = metrics.get("soft_constraint_violations")
    if not isinstance(constraints, dict):
        return None
    values = [
        float(item[field])
        for item in constraints.values()
        if isinstance(item, dict) and isinstance(item.get(field), (int, float))
    ]
    return max(values) if values else 0.0


def _scientific_counts(outcome: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(item["verdict"])
        for item in outcome.get("scientific_absolute_assessments", [])
        if isinstance(item, dict) and "verdict" in item
    )


def _verdict_units(
    outcome: dict[str, Any], field: str, verdict: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{item.get('criterion', 'unknown')}:{item.get('subject_id', 'unknown')}"
            for item in outcome.get(field, [])
            if isinstance(item, dict) and item.get("verdict") == verdict
        )
    )


def build_rows(
    *,
    expected_keys: tuple[tuple[str, str, int], ...],
    run_index: dict[tuple[str, str, int], Path],
    audit_index: dict[str, dict[str, Any]],
    tool_index: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Return one combined evaluation record per expected agent run."""
    rows = []
    for benchmark_id, tier, repetition in expected_keys:
        run = run_index.get((benchmark_id, tier, repetition))
        if run is None:
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "tier": tier,
                    "repetition": repetition,
                    "run": "",
                    "agent_status": "missing",
                }
            )
            continue
        status = _read_object(run / "status.json")
        evaluation = (
            _read_object(run / "evaluation.json")
            if (run / "evaluation.json").is_file()
            else {}
        )
        candidate = (
            _read_object(run / "candidate.json")
            if (run / "candidate.json").is_file()
            else {}
        )
        train = _metric_payload(evaluation, "training")
        validation = _metric_payload(evaluation, "validation")
        audit = audit_index.get(
            run.name, {"status": "not_requested", "task_compliance": ""}
        )
        verdicts = _scientific_counts(audit)
        runtime_failures = _verdict_units(
            audit, "deterministic_assessments", "fail"
        )
        runtime_indeterminate = _verdict_units(
            audit, "deterministic_assessments", "indeterminate"
        )
        scientific_failures = _verdict_units(
            audit, "scientific_absolute_assessments", "fail"
        )
        scientific_indeterminate = _verdict_units(
            audit, "scientific_absolute_assessments", "indeterminate"
        )
        strict_scientific_pass = bool(verdicts) and not (
            verdicts["fail"] or verdicts["indeterminate"]
        )
        tool = tool_index.get(run.name, {})
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "tier": tier,
                "repetition": repetition,
                "run": run.name,
                "agent_status": status.get("status", "missing"),
                "candidate_id": status.get("candidate_id", ""),
                "state_count": len(candidate.get("state_equations", [])),
                "process_count": len(candidate.get("processes", [])),
                "parameter_count": len(candidate.get("parameters", [])),
                "parameter_refit_applied": status.get(
                    "parameter_refit_applied", ""
                ),
                "training_normalized_mse": train.get("normalized_mse", ""),
                "training_per_target_normalized_mse": json.dumps(
                    train.get("per_target_normalized_mse", {}), sort_keys=True
                ),
                "validation_normalized_mse": validation.get(
                    "normalized_mse", ""
                ),
                "validation_per_target_normalized_mse": json.dumps(
                    validation.get("per_target_normalized_mse", {}),
                    sort_keys=True,
                ),
                "training_failed_trajectories": len(
                    train.get("failed_trajectories", [])
                ),
                "validation_failed_trajectories": len(
                    validation.get("failed_trajectories", [])
                ),
                "validation_maximum_normalized_constraint_violation": (
                    _constraint_max(
                        validation, "maximum_normalized_violation"
                    )
                ),
                "validation_maximum_violating_fraction": _constraint_max(
                    validation, "violating_fraction"
                ),
                "agent_latency_seconds": status.get(
                    "agent_latency_seconds", ""
                ),
                "input_tokens": (status.get("usage") or {}).get(
                    "input_tokens", ""
                ),
                "output_tokens": (status.get("usage") or {}).get(
                    "output_tokens", ""
                ),
                "total_tokens": (status.get("usage") or {}).get(
                    "total_tokens", ""
                ),
                "contract_repair_count": len(evaluation.get("repairs", [])),
                "validation_warning_count": len(
                    evaluation.get("validation_warnings", [])
                ),
                "processed_tool_calls": tool.get(
                    "raw_processed_code_interpreter_calls", ""
                ),
                "tool_limit_exceeded": tool.get("limit_exceeded", ""),
                "scientific_audit_status": audit.get("status", "not_requested"),
                "task_compliance": audit.get("task_compliance", ""),
                "deterministic_failures": json.dumps(runtime_failures),
                "deterministic_indeterminate": json.dumps(
                    runtime_indeterminate
                ),
                "scientific_pass": verdicts["pass"],
                "scientific_fail": verdicts["fail"],
                "scientific_indeterminate": verdicts["indeterminate"],
                "scientific_not_applicable": verdicts["not_applicable"],
                "scientific_failure_units": json.dumps(scientific_failures),
                "scientific_indeterminate_units": json.dumps(
                    scientific_indeterminate
                ),
                "scientific_all_applicable_pass": strict_scientific_pass,
            }
        )
    return rows


def _finite(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate coverage and quality without treating judge output as gold."""
    numerical = [row for row in rows if row.get("agent_status") == "complete"]
    audited = [
        row for row in rows if row.get("scientific_audit_status") == "complete"
    ]
    nmse = _finite([row.get("validation_normalized_mse") for row in numerical])
    verdicts: Counter[str] = Counter()
    for row in audited:
        for verdict in ("pass", "fail", "indeterminate", "not_applicable"):
            verdicts[verdict] += int(row.get(f"scientific_{verdict}", 0))
    return {
        "schema_version": "raw-data-agent-full-evaluation-summary-1",
        "expected_run_count": len(rows),
        "agent_status_counts": dict(
            sorted(Counter(str(row["agent_status"]) for row in rows).items())
        ),
        "numerical_completion_rate": len(numerical) / len(rows) if rows else None,
        "validation_normalized_mse": {
            "count": len(nmse),
            "mean": statistics.fmean(nmse) if nmse else None,
            "median": statistics.median(nmse) if nmse else None,
            "minimum": min(nmse) if nmse else None,
            "maximum": max(nmse) if nmse else None,
        },
        "scientific_audit_coverage": len(audited) / len(rows) if rows else None,
        "task_compliance_counts": dict(
            sorted(
                Counter(str(row["task_compliance"]) for row in audited).items()
            )
        ),
        "scientific_absolute_verdict_counts": dict(sorted(verdicts.items())),
        "scientific_all_applicable_pass_count": sum(
            bool(row.get("scientific_all_applicable_pass")) for row in audited
        ),
        "complete_numerical_and_scientific_count": sum(
            row.get("agent_status") == "complete"
            and row.get("scientific_audit_status") == "complete"
            for row in rows
        ),
        "test_data_opened": False,
        "scientific_accuracy_claimed": False,
    }


def _benchmark_markdown(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark_id"]), str(row["tier"]))].append(row)
    lines = [
        "# GPT-5.6 fitted raw-data agent evaluation",
        "",
        (
            "Parameter values are supplied by the agent and are never refit "
            "by the evaluator. Scientific verdicts are descriptive judge "
            "outputs, not gold-label accuracy. Test data remain unopened."
        ),
        "",
        (
            "| Benchmark | Tier | Numerical complete | Validation NMSE median "
            "| Range | Scientific coverage | Hard task pass | Any scientific "
            "fail |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (benchmark_id, tier), items in sorted(grouped.items()):
        complete = [item for item in items if item["agent_status"] == "complete"]
        nmse = _finite(
            [item.get("validation_normalized_mse") for item in complete]
        )
        audited = [
            item for item in items if item["scientific_audit_status"] == "complete"
        ]
        median = f"{statistics.median(nmse):.4g}" if nmse else "N/A"
        span = f"{min(nmse):.4g}-{max(nmse):.4g}" if nmse else "N/A"
        lines.append(
            "| "
            + " | ".join(
                (
                    benchmark_id,
                    tier,
                    f"{len(complete)}/{len(items)}",
                    median,
                    span,
                    f"{len(audited)}/{len(items)}",
                    str(sum(item.get("task_compliance") == "pass" for item in audited)),
                    str(
                        sum(
                            int(item.get("scientific_fail", 0)) > 0
                            for item in audited
                        )
                    ),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument("--tool-budget-csv", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = _read_object(args.protocol_config)
    expected = _expected_keys(config)
    rows = build_rows(
        expected_keys=expected,
        run_index=_run_index(args.runs_root),
        audit_index=_audit_index(args.audit_manifest, args.audit_summary),
        tool_index=_tool_index(args.tool_budget_csv),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "full_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    (args.output_root / "full_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "full_evaluation_summary.md").write_text(
        _benchmark_markdown(rows), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
