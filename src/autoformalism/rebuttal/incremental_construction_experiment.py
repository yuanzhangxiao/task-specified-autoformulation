"""Frozen public-only experiment contract for incremental construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import NonEmptyText, StrictSchema


class IncrementalConstructionCell(StrictSchema):
    """One public benchmark and its prompt-derived contracts."""

    benchmark_id: NonEmptyText
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mechanism_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IncrementalConstructionModelContract(StrictSchema):
    """Pinned local proposer transport settings."""

    model: NonEmptyText
    reasoning_effort: Literal["low", "medium"]
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=128, le=32768)
    served_context_tokens: int = Field(ge=32768)
    request_timeout_seconds: float = Field(gt=0.0)
    maximum_provider_attempts: int = Field(default=3, ge=1, le=11)


class IncrementalConstructionBudget(StrictSchema):
    """Bounded provider-action budget for each task."""

    topology_branch_count: int = Field(ge=1, le=8)
    function_children_per_topology: int = Field(ge=1, le=8)
    maximum_topology_action_steps: int = Field(ge=1, le=16)
    maximum_functional_action_steps: int = Field(ge=1, le=32)


class IncrementalConstructionExperimentPlan(StrictSchema):
    """Immutable public-only incremental-construction plan."""

    schema_version: Literal["phase-b-incremental-construction-plan-1"]
    status: Literal["frozen_before_proposer_calls"]
    purpose: NonEmptyText
    development_only: Literal[True]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    parameter_fitting_performed: Literal[False]
    scientific_judge_called: Literal[False]
    cells: tuple[IncrementalConstructionCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    model_contract: IncrementalConstructionModelContract
    construction_budget: IncrementalConstructionBudget

    @model_validator(mode="after")
    def matrix_is_unique(self) -> IncrementalConstructionExperimentPlan:
        """Reject duplicate cells or repetitions before provider calls."""
        cell_ids = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("incremental construction cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        return self


class IncrementalConstructionTask(StrictSchema):
    """One independently executable public construction task."""

    task_index: int = Field(ge=0)
    benchmark_id: NonEmptyText
    tier: Literal["easy", "medium", "hard"]
    repetition: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mechanism_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_incremental_construction_plan(
    path: Path,
) -> IncrementalConstructionExperimentPlan:
    """Load and validate an incremental-construction plan."""
    return IncrementalConstructionExperimentPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def build_incremental_construction_tasks(
    plan: IncrementalConstructionExperimentPlan,
) -> tuple[IncrementalConstructionTask, ...]:
    """Build a stable cell-major task ledger."""
    return tuple(
        IncrementalConstructionTask(
            task_index=index,
            benchmark_id=cell.benchmark_id,
            tier=cell.tier,
            repetition=repetition,
            public_prompt_sha256=cell.public_prompt_sha256,
            public_target_contract_sha256=(
                cell.public_target_contract_sha256
            ),
            public_mechanism_spec_sha256=cell.public_mechanism_spec_sha256,
        )
        for index, (cell, repetition) in enumerate(
            (cell, repetition)
            for cell in plan.cells
            for repetition in plan.repetitions
        )
    )


def freeze_incremental_construction_experiment(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path,
    target_contract_root: Path,
    mechanism_spec_root: Path,
) -> dict[str, object]:
    """Verify public inputs and write an immutable task ledger."""
    source = config_path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    plan = load_incremental_construction_plan(source)
    for cell in plan.cells:
        _require_sha(
            public_data_root
            / "phase_b_v1"
            / cell.benchmark_id
            / "proposer_prompt.txt",
            cell.public_prompt_sha256,
            "public prompt",
        )
        _require_sha(
            target_contract_root / "specs" / f"{cell.benchmark_id}.json",
            cell.public_target_contract_sha256,
            "public target contract",
        )
        _require_sha(
            mechanism_spec_root / "specs" / f"{cell.benchmark_id}.json",
            cell.public_mechanism_spec_sha256,
            "public mechanism specification",
        )
    tasks = build_incremental_construction_tasks(plan)
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    task_path = output / "task_plan.jsonl"
    plan_path.write_bytes(source.read_bytes())
    task_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in tasks
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "phase-b-incremental-construction-freeze-1",
        "status": "frozen_before_proposer_calls",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "task_count": len(tasks),
        "cell_count": len(plan.cells),
        "repetition_count": len(plan.repetitions),
        "plan_sha256": _sha(plan_path),
        "task_plan_sha256": _sha(task_path),
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (plan_path, task_path, manifest_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{_sha(path)}  {path.name}\n", encoding="utf-8"
        )
    return manifest


def summarize_incremental_construction_experiment(
    plan_path: Path,
    task_plan_path: Path,
    result_root: Path,
) -> dict[str, object]:
    """Summarize construction endpoints without fitting or private data."""
    plan = load_incremental_construction_plan(plan_path)
    tasks = tuple(
        IncrementalConstructionTask.model_validate_json(line)
        for line in task_plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if tasks != build_incremental_construction_tasks(plan):
        raise ValueError("task ledger differs from the frozen plan")
    rows: list[dict[str, object]] = []
    for task in tasks:
        run_name = _task_run_name(task)
        result_path = result_root / run_name / "construction_result.json"
        payload = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else None
        )
        attempts = payload.get("attempts", []) if payload else []
        candidates = payload.get("candidates", []) if payload else []
        failures = Counter(
            str(item["failure_class"])
            for item in attempts
            if item.get("failure_class")
        )
        rows.append(
            {
                "task_index": task.task_index,
                "benchmark_id": task.benchmark_id,
                "tier": task.tier,
                "repetition": task.repetition,
                "result_present": payload is not None,
                "construction_status": (
                    payload.get("status") if payload else None
                ),
                "complete_topology_count": (
                    payload.get("complete_topology_count", 0)
                    if payload
                    else 0
                ),
                "complete_candidate_count": len(candidates),
                "applied_incomplete_action_count": sum(
                    item.get("status") == "applied_incomplete"
                    for item in attempts
                ),
                "rejected_action_count": sum(
                    item.get("status") == "rejected" for item in attempts
                ),
                "failure_class_counts": dict(sorted(failures.items())),
                "public_target_pass_count": sum(
                    item.get("public_target_evaluation", {}).get("passed")
                    is True
                    for item in candidates
                ),
                "graph_mechanism_pass_count": sum(
                    item.get("public_mechanism_evaluation", {}).get(
                        "graph_mechanism_compliance"
                    )
                    == 1.0
                    for item in candidates
                ),
            }
        )
    candidate_count = sum(int(item["complete_candidate_count"]) for item in rows)
    return {
        "schema_version": "phase-b-incremental-construction-summary-1",
        "status": (
            "complete"
            if all(item["result_present"] for item in rows)
            else "incomplete"
        ),
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "planned_task_count": len(tasks),
        "result_task_count": sum(bool(item["result_present"]) for item in rows),
        "complete_construction_task_count": sum(
            item["construction_status"] == "complete" for item in rows
        ),
        "complete_candidate_count": candidate_count,
        "public_target_pass_rate": (
            sum(int(item["public_target_pass_count"]) for item in rows)
            / candidate_count
            if candidate_count
            else None
        ),
        "graph_mechanism_pass_rate": (
            sum(int(item["graph_mechanism_pass_count"]) for item in rows)
            / candidate_count
            if candidate_count
            else None
        ),
        "rows": rows,
    }


def _task_run_name(task: IncrementalConstructionTask) -> str:
    return f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = _sha(path)
    if observed != expected:
        raise ValueError(
            f"{label} does not match plan: path={path}, "
            f"observed={observed}, expected={expected}"
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "IncrementalConstructionExperimentPlan",
    "IncrementalConstructionTask",
    "build_incremental_construction_tasks",
    "freeze_incremental_construction_experiment",
    "load_incremental_construction_plan",
    "summarize_incremental_construction_experiment",
]
