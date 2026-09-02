#!/usr/bin/env python3
"""Summarize public-only baseline outcomes and realized resource usage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask


def main() -> None:
    """Collect every planned task without opening test or private artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    tasks = tuple(
        BaselinePilotTask.model_validate_json(line)
        for line in args.task_plan.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    rows = [_task_row(task, args.runs_root) for task in tasks]
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "realized_resource_ledger.jsonl"
    csv_path = output / "baseline_development_results.csv"
    summary_path = output / "baseline_development_summary.json"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _task_row(task: BaselinePilotTask, runs_root: Path) -> dict[str, object]:
    run = (
        runs_root.expanduser().resolve()
        / task.method
        / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
    )
    result = _read_object(run / "result.json")
    status = _read_object(run / "run_status.json")
    elapsed = _number(
        result.get("elapsed_wall_seconds")
        if result
        else status.get("elapsed_wall_seconds")
    )
    llm = _llm_accounting(run / "llm_events.jsonl")
    return {
        "schema_version": "phase-b-baseline-resource-ledger-1",
        "task_index": task.task_index,
        "method": task.method,
        "benchmark_id": task.benchmark_id,
        "tier": task.tier,
        "repetition": task.repetition,
        "platform": task.platform,
        "status": result.get("status") or status.get("status") or "missing",
        "training_normalized_mse": result.get("training_normalized_mse"),
        "validation_normalized_mse": result.get("validation_normalized_mse"),
        "elapsed_seconds": elapsed,
        "cpu_core_hours": (
            None if elapsed is None else elapsed * task.cpus_per_task / 3600.0
        ),
        "gpu_hours": (
            None if elapsed is None else elapsed * task.gpu_count / 3600.0
        ),
        "logical_llm_calls": llm["logical_calls"],
        "provider_attempts": llm["provider_attempts"],
        "logical_input_tokens": llm["input_tokens"],
        "logical_output_tokens": llm["output_tokens"],
        "logical_total_tokens": llm["total_tokens"],
        "queue_seconds": None,
        "monetary_cost": None,
        "monetary_cost_policy": "not_priced_report_hardware_time",
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def _llm_accounting(path: Path) -> dict[str, int]:
    events = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
    responses = [event for event in events if event.get("event") == "llm_response"]
    failures = [event for event in events if event.get("event") == "llm_failure"]
    provider_attempts = sum(
        int(event.get("attempts", 0))
        for event in responses
        if not event.get("cache_hit")
    )
    successful_hashes = {
        event.get("request_hash")
        for event in responses
        if isinstance(event.get("request_hash"), str)
    }
    provider_attempts += sum(
        1
        for event in failures
        if event.get("request_hash") not in successful_hashes
    )
    input_tokens = output_tokens = total_tokens = 0
    for event in responses:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            raw = event.get("raw_response")
            usage = raw.get("usage") if isinstance(raw, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens += _integer(usage.get("input_tokens", usage.get("prompt_tokens")))
        output_tokens += _integer(
            usage.get("output_tokens", usage.get("completion_tokens"))
        )
        total_tokens += _integer(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "logical_calls": len(responses),
        "provider_attempts": provider_attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    groups = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        completed = [row for row in items if row["status"] == "development_complete"]
        validation = [
            float(row["validation_normalized_mse"])
            for row in completed
            if row["validation_normalized_mse"] is not None
        ]
        groups.append(
            {
                "method": method,
                "planned_trials": len(items),
                "completed_trials": len(completed),
                "completion_rate": len(completed) / len(items),
                "mean_validation_normalized_mse": (
                    None if not validation else mean(validation)
                ),
                "logical_llm_calls": sum(
                    int(row["logical_llm_calls"]) for row in items
                ),
                "provider_attempts": sum(
                    int(row["provider_attempts"]) for row in items
                ),
                "logical_total_tokens": sum(
                    int(row["logical_total_tokens"]) for row in items
                ),
                "cpu_core_hours": sum(
                    float(row["cpu_core_hours"] or 0.0) for row in items
                ),
                "gpu_hours": sum(
                    float(row["gpu_hours"] or 0.0) for row in items
                ),
            }
        )
    return {
        "schema_version": "phase-b-public-baseline-development-summary-1",
        "status": (
            "complete"
            if all(row["status"] == "development_complete" for row in rows)
            else "partial"
        ),
        "planned_task_count": len(rows),
        "groups": groups,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
    }


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


if __name__ == "__main__":
    main()
