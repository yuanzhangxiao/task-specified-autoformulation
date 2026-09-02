#!/usr/bin/env python3
"""Freeze completed public-only baseline results before oracle comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from autoformalism.baselines.models import (
    BaselineDevelopmentResult,
    BaselineRunStatus,
)
from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask

_SUMMARY_FILES = (
    "baseline_development_results.csv",
    "baseline_development_summary.json",
    "realized_resource_ledger.jsonl",
)
_FROZEN_INPUT_FILES = (
    "freeze_manifest.json",
    "plan.json",
    "planned_resource_ledger.jsonl",
    "task_plan.jsonl",
)


def main() -> None:
    """Validate, copy, and hash-bind one completed development experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-code-commit", required=True)
    args = parser.parse_args()
    manifest = freeze_development_results(
        args.experiment_root,
        args.output_root,
        source_code_commit=args.source_code_commit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def freeze_development_results(
    experiment_root: Path,
    output_root: Path,
    *,
    source_code_commit: str,
) -> dict[str, object]:
    """Freeze public train/validation selections and resource accounting."""
    if re.fullmatch(r"[0-9a-f]{7,40}", source_code_commit) is None:
        raise ValueError("source code commit must be a 7-40 character hex id")
    experiment = experiment_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    task_plan = experiment / "frozen" / "task_plan.jsonl"
    tasks = _load_tasks(task_plan)
    if not tasks:
        raise ValueError("frozen task plan is empty")
    if {task.task_index for task in tasks} != set(range(len(tasks))):
        raise ValueError("frozen task indices must be contiguous from zero")

    source_freeze = _read_object(experiment / "frozen" / "freeze_manifest.json")
    if (
        source_freeze.get("test_data_opened") is not False
        or source_freeze.get("private_reference_opened") is not False
        or source_freeze.get("task_count") != len(tasks)
        or source_freeze.get("task_plan_sha256") != _sha256(task_plan)
        or source_freeze.get("plan_sha256")
        != _sha256(experiment / "frozen" / "plan.json")
        or source_freeze.get("planned_resource_ledger_sha256")
        != _sha256(experiment / "frozen" / "planned_resource_ledger.jsonl")
    ):
        raise ValueError("source freeze is not the expected public-only task set")

    submission = _read_object(experiment / "submission_manifest.json")
    if (
        submission.get("test_data_opened") is not False
        or submission.get("private_reference_opened") is not False
    ):
        raise ValueError("submission manifest is not public-only")

    summary = _read_object(
        experiment / "summary" / "baseline_development_summary.json"
    )
    groups = summary.get("groups")
    if (
        summary.get("status") != "complete"
        or summary.get("planned_task_count") != len(tasks)
        or summary.get("test_data_opened") is not False
        or summary.get("private_reference_opened") is not False
        or not isinstance(groups, list)
        or not all(isinstance(item, dict) for item in groups)
        or sum(_integer(item.get("completed_trials")) for item in groups) != len(tasks)
    ):
        raise ValueError("development summary is not complete and public-only")

    copied: list[dict[str, object]] = []
    for name in _FROZEN_INPUT_FILES:
        copied.append(
            _copy_artifact(
                experiment / "frozen" / name,
                output / "inputs" / name,
                output,
                role="frozen_input",
            )
        )
    copied.append(
        _copy_artifact(
            experiment / "submission_manifest.json",
            output / "inputs" / "submission_manifest.json",
            output,
            role="submission_manifest",
        )
    )
    for name in _SUMMARY_FILES:
        copied.append(
            _copy_artifact(
                experiment / "summary" / name,
                output / "summary" / name,
                output,
                role="development_summary",
            )
        )

    methods: set[str] = set()
    benchmarks: set[str] = set()
    for task in tasks:
        run = (
            experiment
            / "runs"
            / task.method
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        result_path = run / "result.json"
        status_path = run / "run_status.json"
        result = BaselineDevelopmentResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        status = BaselineRunStatus.model_validate_json(
            status_path.read_text(encoding="utf-8")
        )
        if (
            (
                result.method,
                result.benchmark_id,
                result.tier,
                result.seed,
            )
            != (
                task.method,
                task.benchmark_id,
                task.tier,
                task.repetition,
            )
            or result.status != "development_complete"
            or result.test_data_opened is not False
            or status.status != "complete"
            or not result.equations
        ):
            raise ValueError(f"baseline task is not complete: {task.task_index}")
        methods.add(result.method)
        benchmarks.add(result.benchmark_id)
        copied.append(
            _copy_artifact(
                result_path,
                output / "tasks" / f"task_{task.task_index:03d}.json",
                output,
                role="selected_development_result",
            )
        )
        copied.append(
            _copy_artifact(
                status_path,
                output / "statuses" / f"task_{task.task_index:03d}.json",
                output,
                role="terminal_status",
            )
        )

    artifact_ledger = output / "artifact_ledger.jsonl"
    ledger_text = "".join(
        json.dumps(item, sort_keys=True) + "\n"
        for item in sorted(copied, key=lambda item: str(item["path"]))
    )
    _write_once(artifact_ledger, ledger_text.encode("utf-8"))
    manifest = {
        "schema_version": "phase-b-public-baseline-development-result-freeze-1",
        "status": "frozen_before_test_or_oracle_evaluation",
        "source_code_commit": source_code_commit,
        "task_count": len(tasks),
        "selected_result_count": len(tasks),
        "methods": sorted(methods),
        "benchmark_ids": sorted(benchmarks),
        "derivative_provenance": "estimated",
        "derivative_estimator": "numpy.gradient",
        "derivative_edge_order": "2_if_at_least_3_samples_else_1",
        "target_representation": "each_target_as_observed_dynamic_state",
        "validation_protocol": "causal_one_step_observed_state_reset",
        "artifact_count": len(copied),
        "artifact_ledger_sha256": _sha256(artifact_ledger),
        "source_freeze_manifest_sha256": _sha256(
            experiment / "frozen" / "freeze_manifest.json"
        ),
        "source_task_plan_sha256": _sha256(task_plan),
        "development_summary_sha256": _sha256(
            experiment / "summary" / "baseline_development_summary.json"
        ),
        "test_data_opened": False,
        "private_reference_opened": False,
        "oracle_derivatives_used": False,
        "oracle_latent_states_used": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = output / "development_result_freeze.json"
    _write_once(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_once(
        output / "development_result_freeze.json.sha256",
        f"{_sha256(manifest_path)}  development_result_freeze.json\n".encode(),
    )
    return manifest


def _load_tasks(path: Path) -> tuple[BaselinePilotTask, ...]:
    return tuple(
        BaselinePilotTask.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _copy_artifact(
    source: Path,
    destination: Path,
    output_root: Path,
    *,
    role: str,
) -> dict[str, object]:
    data = source.read_bytes()
    _write_once(destination, data)
    return {
        "role": role,
        "path": str(destination.relative_to(output_root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


if __name__ == "__main__":
    main()
