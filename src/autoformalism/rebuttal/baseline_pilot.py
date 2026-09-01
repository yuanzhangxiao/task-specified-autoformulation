"""Frozen public-only baseline pilot distributed across ACES resources."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.targets import PublicTargetContract


class BaselinePilotCell(BaseModel):
    """One public train/validation cell in the matched pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselinePilotMethod(BaseModel):
    """One baseline adapter and its predeclared compute boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["sindy", "pysr", "d3_native_no_tools"]
    comparison_role: Literal[
        "classical_partial_observability_control",
        "matched_llm_discovery_agent",
    ]
    platform: Literal["aces_cpu", "aces_h100x2"]
    cpus_per_task: int = Field(ge=1)
    gpu_type: Literal["none", "h100"]
    gpu_count: int = Field(ge=0, le=2)
    wall_timeout_seconds: float = Field(gt=0.0)
    maximum_llm_calls: int = Field(ge=0)
    model: str | None = None
    reasoning_effort: Literal["high"] | None = None
    requires_selected_proposer_operating_point: bool
    pysr_iterations: int | None = Field(default=None, ge=1)
    maximum_expression_size: int | None = Field(default=None, ge=3)
    d3_generations: int | None = Field(default=None, ge=1)
    d3_patience: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def method_matches_resources(self) -> BaselinePilotMethod:
        """Prevent method labels from silently changing compute or LLM access."""
        if self.method == "d3_native_no_tools":
            if (
                self.platform != "aces_h100x2"
                or self.gpu_type != "h100"
                or self.gpu_count != 2
                or not self.model
                or self.reasoning_effort != "high"
                or not self.requires_selected_proposer_operating_point
                or self.maximum_llm_calls < 1
                or self.d3_generations != self.maximum_llm_calls
                or self.d3_patience is None
            ):
                raise ValueError("D3 method has an inconsistent LLM/compute contract")
            if self.pysr_iterations is not None:
                raise ValueError("D3 method must not define PySR settings")
            return self
        if (
            self.platform != "aces_cpu"
            or self.gpu_type != "none"
            or self.gpu_count != 0
            or self.model is not None
            or self.reasoning_effort is not None
            or self.requires_selected_proposer_operating_point
            or self.maximum_llm_calls != 0
            or self.d3_generations is not None
            or self.d3_patience is not None
        ):
            raise ValueError("classical method has an inconsistent compute contract")
        if self.method == "pysr" and (
            self.pysr_iterations is None or self.maximum_expression_size is None
        ):
            raise ValueError("PySR method requires iteration and expression budgets")
        if self.method == "sindy" and (
            self.pysr_iterations is not None
            or self.maximum_expression_size is not None
        ):
            raise ValueError("SINDy method must not define PySR settings")
        return self


class BaselinePilotResourceAccounting(BaseModel):
    """Resource fields required for later cross-method reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-baseline-resource-ledger-1"]
    process_wall_time_required: Literal[True]
    slurm_elapsed_and_queue_time_required: Literal[True]
    cpu_core_hours_required: Literal[True]
    gpu_hours_required: Literal[True]
    logical_llm_tokens_required_when_applicable: Literal[True]
    provider_attempts_required_when_applicable: Literal[True]
    local_model_monetary_cost_policy: Literal[
        "not_priced_report_hardware_time"
    ]


class BaselinePilotPlan(BaseModel):
    """Immutable two-cell baseline development matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-public-baseline-pilot-plan-1"]
    status: Literal["frozen_before_baseline_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    cells: tuple[BaselinePilotCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    methods: tuple[BaselinePilotMethod, ...] = Field(min_length=1)
    proposer_transport_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_overlay_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_contract_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_accounting: BaselinePilotResourceAccounting
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    weighted_overall_score_defined: Literal[False]

    @model_validator(mode="after")
    def matrix_is_unique(self) -> BaselinePilotPlan:
        """Require unique cells, seeds, and methods before execution."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        methods = [item.method for item in self.methods]
        if len(cells) != len(set(cells)):
            raise ValueError("baseline pilot cells must be unique")
        if len(methods) != len(set(methods)):
            raise ValueError("baseline pilot methods must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            seed < 0 for seed in self.repetitions
        ):
            raise ValueError(
                "baseline pilot repetitions must be unique and nonnegative"
            )
        return self


class BaselinePilotTask(BaseModel):
    """One public-only baseline array task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0)
    method: Literal["sindy", "pysr", "d3_native_no_tools"]
    comparison_role: str
    platform: Literal["aces_cpu", "aces_h100x2"]
    benchmark_id: str
    tier: Literal["easy", "medium", "hard"]
    repetition: int = Field(ge=0)
    cpus_per_task: int = Field(ge=1)
    gpu_type: Literal["none", "h100"]
    gpu_count: int = Field(ge=0, le=2)
    wall_timeout_seconds: float = Field(gt=0.0)
    maximum_llm_calls: int = Field(ge=0)
    model: str | None = None
    reasoning_effort: str | None = None
    pysr_iterations: int | None = None
    maximum_expression_size: int | None = None
    d3_generations: int | None = None
    d3_patience: int | None = None


def load_baseline_pilot_plan(path: Path) -> BaselinePilotPlan:
    """Load one strict baseline pilot plan."""
    return BaselinePilotPlan.model_validate_json(path.read_text(encoding="utf-8"))


def build_baseline_pilot_tasks(
    plan: BaselinePilotPlan,
) -> tuple[BaselinePilotTask, ...]:
    """Expand methods, cells, and seeds in deterministic method-major order."""
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
                        model=method.model,
                        reasoning_effort=method.reasoning_effort,
                        pysr_iterations=method.pysr_iterations,
                        maximum_expression_size=method.maximum_expression_size,
                        d3_generations=method.d3_generations,
                        d3_patience=method.d3_patience,
                    )
                )
    return tuple(tasks)


def freeze_baseline_pilot(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path,
    target_contract_root: Path,
    prompt_overlay_config_path: Path,
    proposer_transport_plan_path: Path,
) -> dict[str, object]:
    """Validate public inputs and freeze tasks plus planned resource fields."""
    plan = load_baseline_pilot_plan(config_path)
    public_root = public_data_root.expanduser().resolve()
    contract_root = target_contract_root.expanduser().resolve()
    if _sha256(prompt_overlay_config_path.expanduser().resolve()) != (
        plan.prompt_overlay_config_sha256
    ):
        raise ValueError("prompt-overlay config differs from the baseline plan")
    if _sha256(contract_root / "manifest.json") != (
        plan.target_contract_manifest_sha256
    ):
        raise ValueError("target-contract manifest differs from the baseline plan")
    if _sha256(proposer_transport_plan_path.expanduser().resolve()) != (
        plan.proposer_transport_plan_sha256
    ):
        raise ValueError("proposer transport plan differs from the baseline plan")
    for cell in plan.cells:
        prompt = (
            public_root
            / "phase_b_v1"
            / cell.benchmark_id
            / "proposer_prompt.txt"
        )
        contract_path = contract_root / "specs" / f"{cell.benchmark_id}.json"
        if _sha256(prompt) != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")
        if _sha256(contract_path) != cell.public_target_contract_sha256:
            raise ValueError(f"target contract differs: {cell.benchmark_id}")
        contract = PublicTargetContract.model_validate_json(
            contract_path.read_text(encoding="utf-8")
        )
        if (contract.benchmark_id, contract.tier) != (
            cell.benchmark_id,
            cell.tier,
        ) or contract.public_prompt_sha256 != cell.public_prompt_sha256:
            raise ValueError(f"target contract identity differs: {cell.benchmark_id}")

    tasks = build_baseline_pilot_tasks(plan)
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    task_path = root / "task_plan.jsonl"
    resource_path = root / "planned_resource_ledger.jsonl"
    _write_once(plan_path, config_path.read_text(encoding="utf-8"))
    _write_once(
        task_path,
        "".join(task.model_dump_json() + "\n" for task in tasks),
    )
    _write_once(
        resource_path,
        "".join(
            json.dumps(
                {
                    "schema_version": plan.resource_accounting.schema_version,
                    "task_index": task.task_index,
                    "method": task.method,
                    "benchmark_id": task.benchmark_id,
                    "tier": task.tier,
                    "repetition": task.repetition,
                    "platform": task.platform,
                    "cpus_per_task": task.cpus_per_task,
                    "gpu_type": task.gpu_type,
                    "gpu_count": task.gpu_count,
                    "wall_timeout_seconds": task.wall_timeout_seconds,
                    "maximum_llm_calls": task.maximum_llm_calls,
                    "logical_llm_tokens": None,
                    "provider_attempts": None,
                    "queue_seconds": None,
                    "elapsed_seconds": None,
                    "cpu_core_hours": None,
                    "gpu_hours": None,
                    "monetary_cost": None,
                    "monetary_cost_policy": (
                        plan.resource_accounting.local_model_monetary_cost_policy
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for task in tasks
        ),
    )
    manifest = {
        "schema_version": "phase-b-public-baseline-pilot-freeze-1",
        "status": "frozen_before_baseline_calls",
        "plan_sha256": _sha256(plan_path),
        "task_plan_sha256": _sha256(task_path),
        "planned_resource_ledger_sha256": _sha256(resource_path),
        "task_count": len(tasks),
        "cpu_task_count": sum(task.platform == "aces_cpu" for task in tasks),
        "d3_task_count": sum(task.method == "d3_native_no_tools" for task in tasks),
        "matched_trial_count": len(plan.cells) * len(plan.repetitions),
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = root / "freeze_manifest.json"
    _write_once(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    for path in (plan_path, task_path, resource_path, manifest_path):
        _write_once(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest


def freeze_baseline_llm_operating_point(
    baseline_plan_path: Path,
    proposer_plan_path: Path,
    proposer_analysis_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Bind the D3 pilot to the passing proposer model operating point."""
    plan = load_baseline_pilot_plan(baseline_plan_path)
    if _sha256(proposer_plan_path.expanduser().resolve()) != (
        plan.proposer_transport_plan_sha256
    ):
        raise ValueError("proposer plan differs from the baseline pilot")
    proposer_plan = json.loads(proposer_plan_path.read_text(encoding="utf-8"))
    analysis = json.loads(proposer_analysis_path.read_text(encoding="utf-8"))
    selected = analysis.get("selected_max_output_tokens")
    if (
        analysis.get("schema_version")
        != "phase-b-proposer-transport-calibration-analysis-1"
        or analysis.get("status") != "pass"
        or not isinstance(selected, int)
    ):
        raise ValueError("proposer operating point has not passed calibration")
    d3 = next(item for item in plan.methods if item.method == "d3_native_no_tools")
    model_contract = proposer_plan.get("model_contract", {})
    if (
        model_contract.get("model") != d3.model
        or model_contract.get("reasoning_effort") != d3.reasoning_effort
        or analysis.get("selected_reasoning_effort") != d3.reasoning_effort
    ):
        raise ValueError("D3 model settings differ from proposer calibration")
    payload = {
        "schema_version": "phase-b-baseline-llm-operating-point-1",
        "status": "frozen_before_d3_calls",
        "baseline_plan_sha256": _sha256(baseline_plan_path.expanduser().resolve()),
        "proposer_plan_sha256": _sha256(proposer_plan_path.expanduser().resolve()),
        "proposer_analysis_sha256": _sha256(
            proposer_analysis_path.expanduser().resolve()
        ),
        "model": d3.model,
        "reasoning_effort": d3.reasoning_effort,
        "max_output_tokens": selected,
        "maximum_llm_calls_per_trial": d3.maximum_llm_calls,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    _write_once(
        output_path.expanduser().resolve(),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing frozen input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)
