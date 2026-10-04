"""Freeze the Phase C classical baseline matrix before any baseline runs.

The Phase B freezer identifies its inputs by Phase B prompts, target contracts
and the proposer transport plan, none of which exist for Phase C. Here the
release receipt and each cell's published prompt identify the inputs, the cells
are the Phase C roster, and each method keeps its Phase B settings unchanged.

Only the classical methods are planned here. The LLM baselines have their own
launchers and endpoints and are frozen separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.baseline_pilot import (
    BaselinePilotMethod,
    BaselinePilotResourceAccounting,
    BaselinePilotTask,
    _sha256,
    _write_once,
)
from autoformalism.rebuttal.phase_c_baselines import PROTOCOL, tier_of, verify_release
from autoformalism.research.phase_c_inputs import ROSTER

CLASSICAL_METHODS = frozenset({"persistence", "sindy", "pysr"})


class PhaseCBaselineCell(BaseModel):
    """One released public cell, identified by its published prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "hard", "fixed"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PhaseCBaselinePlan(BaseModel):
    """Immutable development-only matrix over one Phase C release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-c-public-baseline-plan-1"]
    status: Literal["frozen_before_baseline_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    release_protocol: Literal["phase-c-development-2"]
    release_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[PhaseCBaselineCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    methods: tuple[BaselinePilotMethod, ...] = Field(min_length=1)
    resource_accounting: BaselinePilotResourceAccounting
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    weighted_overall_score_defined: Literal[False]

    @model_validator(mode="after")
    def matrix_is_the_roster(self) -> PhaseCBaselinePlan:
        """Require roster cells with their own tiers, and classical CPU methods."""
        names = [cell.benchmark_id for cell in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("Phase C baseline cells must be unique")
        for cell in self.cells:
            if cell.benchmark_id not in ROSTER:
                raise ValueError(f"{cell.benchmark_id} is outside the Phase C roster")
            if cell.tier != tier_of(cell.benchmark_id):
                raise ValueError(f"{cell.benchmark_id} declares the wrong tier")
        methods = [method.method for method in self.methods]
        if len(methods) != len(set(methods)):
            raise ValueError("Phase C baseline methods must be unique")
        if not set(methods) <= CLASSICAL_METHODS or any(
            method.platform != "delta_cpu" for method in self.methods
        ):
            raise ValueError("this plan runs classical methods on Delta CPUs only")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            seed < 0 for seed in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        return self


def load_phase_c_baseline_plan(path: Path) -> PhaseCBaselinePlan:
    """Load one strict Phase C baseline plan."""
    return PhaseCBaselinePlan.model_validate_json(path.read_text(encoding="utf-8"))


def build_phase_c_baseline_tasks(
    plan: PhaseCBaselinePlan,
) -> tuple[BaselinePilotTask, ...]:
    """Expand methods, cells and seeds in deterministic method-major order."""
    tasks: list[BaselinePilotTask] = []
    for method in plan.methods:
        for cell in plan.cells:
            for repetition in plan.repetitions:
                tasks.append(
                    BaselinePilotTask(
                        task_index=len(tasks),
                        method=method.method,
                        comparison_role=method.comparison_role,
                        platform=method.platform,
                        benchmark_id=cell.benchmark_id,
                        tier=cell.tier,
                        repetition=repetition,
                        cpus_per_task=method.cpus_per_task,
                        gpu_type=method.gpu_type,
                        gpu_count=method.gpu_count,
                        wall_timeout_seconds=method.wall_timeout_seconds,
                        maximum_llm_calls=method.maximum_llm_calls,
                        sindy_thresholds=method.sindy_thresholds,
                        pysr_iterations=method.pysr_iterations,
                        maximum_expression_size=method.maximum_expression_size,
                    )
                )
    return tuple(tasks)


def freeze_phase_c_baseline_plan(
    config_path: Path, output_root: Path, release: Path
) -> dict[str, object]:
    """Verify the release against the plan, then write the immutable task ledger.

    The receipt digest and every cell's published prompt must match the plan,
    so a job can only ever run on the release the plan was written for.
    """
    plan = load_phase_c_baseline_plan(config_path)
    release = release.expanduser().resolve()
    names = tuple(cell.benchmark_id for cell in plan.cells)
    receipt = verify_release(release, names)
    if receipt != plan.release_summary_sha256:
        raise ValueError("release receipt differs from the baseline plan")
    for cell in plan.cells:
        prompt = release / "public" / cell.benchmark_id / "proposer_prompt.txt"
        if _sha256(prompt) != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")

    tasks = build_phase_c_baseline_tasks(plan)
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    task_path = root / "task_plan.jsonl"
    _write_once(plan_path, config_path.read_text(encoding="utf-8"))
    _write_once(task_path, "".join(task.model_dump_json() + "\n" for task in tasks))
    manifest = {
        "schema_version": "phase-c-public-baseline-freeze-1",
        "status": "frozen_before_baseline_calls",
        "release_protocol": PROTOCOL,
        "release_summary_sha256": receipt,
        "plan_sha256": _sha256(plan_path),
        "task_plan_sha256": _sha256(task_path),
        "task_count": len(tasks),
        "tasks_by_method": {
            method.method: sum(task.method == method.method for task in tasks)
            for method in plan.methods
        },
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = root / "freeze_manifest.json"
    _write_once(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (plan_path, task_path, manifest_path):
        _write_once(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest
