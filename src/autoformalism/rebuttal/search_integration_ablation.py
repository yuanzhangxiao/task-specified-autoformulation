"""Frozen planning for the paired-judge search integration ablation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterRequest
from autoformalism.rebuttal.final_evaluation_pilot import (
    HiddenAuditRequirement,
    validate_hidden_audit,
)
from autoformalism.targets import PublicTargetContract


class SearchAblationCell(BaseModel):
    """One public benchmark/tier cell selected before search calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str | None = None
    public_target_contract_sha256: str | None = None


class SearchAblationArm(BaseModel):
    """One matched selection arm in the judge integration experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: Literal["paired_question_consensus", "no_judge"]
    selection_policy: Literal["incumbent_relative_hybrid", "validation_only"]
    use_judge: bool
    hybrid_science_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def policy_matches_arm(self) -> SearchAblationArm:
        """Prevent an arm label from silently changing its search policy."""
        expected = {
            "paired_question_consensus": ("incumbent_relative_hybrid", True),
            "no_judge": ("validation_only", False),
        }[self.arm_id]
        if (self.selection_policy, self.use_judge) != expected:
            raise ValueError(f"{self.arm_id} arm has a mismatched selection policy")
        if self.use_judge and self.hybrid_science_weight is None:
            raise ValueError("paired judge arm requires a science weight")
        if not self.use_judge and self.hybrid_science_weight is not None:
            raise ValueError("no-judge arm must not define a science weight")
        return self


class SearchBudget(BaseModel):
    """Budgets held identical across both ablation arms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration_budget: int = Field(ge=1)
    beam_size: Literal[1]
    fit_starts: int = Field(ge=1)
    fit_max_nfev: int = Field(ge=1)
    fit_timeout_seconds: float = Field(gt=0.0)
    fit_retry_starts: int | None = Field(default=None, ge=1)
    fit_retry_max_nfev: int | None = Field(default=None, ge=1)
    fit_retry_timeout_seconds: float | None = Field(default=None, gt=0.0)
    final_fit_max_nfev: int = Field(ge=1)
    final_fit_timeout_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def retry_budget_is_complete_and_non_decreasing(self) -> SearchBudget:
        """Require a complete, genuinely expanded retry budget when present."""
        retry = (
            self.fit_retry_starts,
            self.fit_retry_max_nfev,
            self.fit_retry_timeout_seconds,
        )
        if all(value is None for value in retry):
            return self
        if any(value is None for value in retry):
            raise ValueError("fit retry budget fields must be supplied together")
        assert self.fit_retry_starts is not None
        assert self.fit_retry_max_nfev is not None
        assert self.fit_retry_timeout_seconds is not None
        if self.fit_retry_starts < self.fit_starts:
            raise ValueError("fit retry starts must not be lower than the primary")
        if self.fit_retry_max_nfev < self.fit_max_nfev:
            raise ValueError("fit retry nfev must not be lower than the primary")
        if self.fit_retry_timeout_seconds < self.fit_timeout_seconds:
            raise ValueError("fit retry timeout must not be lower than the primary")
        return self


class SearchModelContract(BaseModel):
    """Frozen local proposer and judge serving contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposer_model: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    reasoning_effort: Literal["low"] | None = None
    proposer_reasoning_effort: Literal["low", "medium", "high"] | None = None
    judge_reasoning_effort: Literal["low", "medium", "high"] | None = None
    temperature: float = Field(ge=0.0, le=2.0)
    proposer_max_output_tokens: int = Field(ge=128)
    request_timeout_seconds: float = Field(gt=0.0)
    judge_protocol_config_sha256: str | None = None

    @model_validator(mode="after")
    def reasoning_contract_is_unambiguous(self) -> SearchModelContract:
        """Require either the legacy shared effort or both role-specific efforts."""
        split = (
            self.proposer_reasoning_effort,
            self.judge_reasoning_effort,
        )
        if self.reasoning_effort is not None:
            if any(value is not None for value in split):
                raise ValueError(
                    "shared and role-specific reasoning efforts cannot be mixed"
                )
            return self
        if any(value is None for value in split):
            raise ValueError(
                "both proposer and judge reasoning efforts are required"
            )
        return self

    @property
    def effective_proposer_reasoning_effort(self) -> str:
        """Return the proposer effort for legacy and split configurations."""
        return self.proposer_reasoning_effort or self.reasoning_effort or "low"

    @property
    def effective_judge_reasoning_effort(self) -> str:
        """Return the judge effort for legacy and split configurations."""
        return self.judge_reasoning_effort or self.reasoning_effort or "low"


class SearchPublicInputContract(BaseModel):
    """Immutable public prompt-overlay and target-contract boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_version: Literal["phase_b_v1"]
    prompt_overlay_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_contract_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_cell_count: Literal[40]
    require_non_proposer_files_byte_identical: Literal[True]
    require_production_registered: Literal[True]
    require_sealed_test: Literal[True]


class SearchResourceAccountingContract(BaseModel):
    """Predeclared resource ledger semantics shared across search arms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-search-resource-ledger-1"]
    logical_cached_usage_counted: Literal[True]
    provider_attempts_reported_separately: Literal[True]
    queue_time_separate_from_execution_time: Literal[True]
    development_cost_separate_from_marginal_deployment_cost: Literal[True]
    local_model_monetary_cost_policy: Literal[
        "not_priced_report_hardware_time"
    ]


class SearchIntegrationAblationPlan(BaseModel):
    """Predeclared matched integration matrix for judge causal attribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "phase-b-search-integration-ablation-plan-1",
        "phase-b-search-integration-ablation-plan-2",
        "phase-b-search-integration-ablation-plan-3",
    ]
    status: Literal["frozen_before_search_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    cells: tuple[SearchAblationCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    arms: tuple[SearchAblationArm, ...] = Field(min_length=2)
    model_contract: SearchModelContract
    search_budget: SearchBudget
    hidden_contract_audit: HiddenAuditRequirement
    shared_initial_request_cache: Literal[True]
    test_data_opened_during_search: Literal[False]
    evaluation_endpoints: tuple[str, ...] = Field(min_length=1)
    weighted_overall_score_defined: Literal[False]
    public_input_contract: SearchPublicInputContract | None = None
    deterministic_runtime_checks: tuple[str, ...] = ()
    resource_accounting: SearchResourceAccountingContract | None = None

    @model_validator(mode="after")
    def matrix_is_complete(self) -> SearchIntegrationAblationPlan:
        """Require a unique matrix and exactly the two prespecified arms."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("search-ablation cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            value < 0 for value in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        arm_ids = [item.arm_id for item in self.arms]
        expected = ["paired_question_consensus", "no_judge"]
        if arm_ids != expected:
            raise ValueError(
                "arms must be ordered as paired_question_consensus then no_judge"
            )
        if len(self.evaluation_endpoints) != len(set(self.evaluation_endpoints)):
            raise ValueError("evaluation endpoints must be unique")
        if self.schema_version.endswith(("-2", "-3")):
            if self.public_input_contract is None:
                raise ValueError("version 2+ requires a public-input contract")
            if self.resource_accounting is None:
                raise ValueError("version 2+ requires resource accounting")
            if not self.deterministic_runtime_checks:
                raise ValueError(
                    "version 2+ requires deterministic runtime checks"
                )
            for cell in self.cells:
                if (
                    cell.public_prompt_sha256 is None
                    or cell.public_target_contract_sha256 is None
                ):
                    raise ValueError(
                        "version 2+ cells require prompt and target-contract hashes"
                    )
            if self.model_contract.judge_protocol_config_sha256 is None:
                raise ValueError("version 2+ requires a judge-protocol hash")
        if (
            self.schema_version.endswith("-3")
            and self.model_contract.reasoning_effort is not None
        ):
            raise ValueError("version 3 requires role-specific reasoning")
        if (
            self.schema_version.endswith("-3")
            and self.search_budget.fit_retry_starts is None
        ):
            raise ValueError("version 3 requires a deterministic screening fit retry")
        return self


class SearchIntegrationTask(BaseModel):
    """One array task derived deterministically from the frozen plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0)
    arm_id: Literal["paired_question_consensus", "no_judge"]
    benchmark_id: str
    tier: Literal["easy", "medium", "hard"]
    repetition: int = Field(ge=0)
    selection_policy: Literal["incumbent_relative_hybrid", "validation_only"]
    use_judge: bool
    hybrid_science_weight: float | None = None
    public_prompt_sha256: str | None = None
    public_target_contract_sha256: str | None = None


class FrozenSearchAblationSource(BaseModel):
    """One planned search outcome frozen before sealed evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    method_label: str
    arm_id: Literal["paired_question_consensus", "no_judge"]
    benchmark_id: str
    tier: str
    repetition: int
    source_path: str
    artifact_status: Literal["available", "missing"]
    source_sha256: str | None = None


def load_search_integration_plan(path: Path) -> SearchIntegrationAblationPlan:
    """Load and validate a frozen judge-integration plan."""
    return SearchIntegrationAblationPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def build_search_integration_tasks(
    plan: SearchIntegrationAblationPlan,
) -> tuple[SearchIntegrationTask, ...]:
    """Expand the exact arm/cell/repetition cross-product."""
    tasks: list[SearchIntegrationTask] = []
    for arm in plan.arms:
        for cell in plan.cells:
            for repetition in plan.repetitions:
                tasks.append(
                    SearchIntegrationTask(
                        task_index=len(tasks),
                        arm_id=arm.arm_id,
                        benchmark_id=cell.benchmark_id,
                        tier=cell.tier,
                        repetition=repetition,
                        selection_policy=arm.selection_policy,
                        use_judge=arm.use_judge,
                        hybrid_science_weight=arm.hybrid_science_weight,
                        public_prompt_sha256=cell.public_prompt_sha256,
                        public_target_contract_sha256=(
                            cell.public_target_contract_sha256
                        ),
                    )
                )
    expected = len(plan.arms) * len(plan.cells) * len(plan.repetitions)
    if len(tasks) != expected:
        raise AssertionError("internal search-ablation task count differs")
    return tuple(tasks)


def freeze_search_integration_plan(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path | None = None,
    target_contract_root: Path | None = None,
    prompt_overlay_config_path: Path | None = None,
) -> dict[str, object]:
    """Write immutable plan, task ledger, hashes, and a no-test manifest."""
    resolved_config = config_path.expanduser().resolve()
    plan = load_search_integration_plan(resolved_config)
    public_input_validation = _validate_search_public_inputs(
        plan,
        public_data_root=public_data_root,
        target_contract_root=target_contract_root,
        prompt_overlay_config_path=prompt_overlay_config_path,
    )
    tasks = build_search_integration_tasks(plan)
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    tasks_path = root / "task_plan.jsonl"
    _write_once(plan_path, resolved_config.read_text(encoding="utf-8"))
    _write_once(
        tasks_path,
        "".join(task.model_dump_json() + "\n" for task in tasks),
    )
    manifest = {
        "schema_version": "phase-b-search-integration-ablation-freeze-1",
        "status": "frozen_before_search_calls",
        "plan_sha256": _sha256(plan_path),
        "task_plan_sha256": _sha256(tasks_path),
        "task_count": len(tasks),
        "matched_trial_count": len(plan.cells) * len(plan.repetitions),
        "arm_ids": [item.arm_id for item in plan.arms],
        "arm_launch_order": "judge_arm_then_no_judge_arm",
        "shared_initial_request_cache": plan.shared_initial_request_cache,
        "development_only": plan.development_only,
        "test_data_opened": False,
        "weighted_overall_score_defined": plan.weighted_overall_score_defined,
        "public_input_validation": public_input_validation,
        "resource_accounting_schema_version": (
            None
            if plan.resource_accounting is None
            else plan.resource_accounting.schema_version
        ),
    }
    manifest_path = root / "freeze_manifest.json"
    _write_once(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    for path in (plan_path, tasks_path, manifest_path):
        _write_once(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest


def _validate_search_public_inputs(
    plan: SearchIntegrationAblationPlan,
    *,
    public_data_root: Path | None,
    target_contract_root: Path | None,
    prompt_overlay_config_path: Path | None,
) -> dict[str, object] | None:
    """Bind v2 search cells to the exact prompt overlay and target contracts."""
    boundary = plan.public_input_contract
    if boundary is None:
        return None
    if (
        public_data_root is None
        or target_contract_root is None
        or prompt_overlay_config_path is None
    ):
        raise ValueError(
            "version 2 freeze requires public data, target contracts, and "
            "prompt-overlay config"
        )
    public_root = public_data_root.expanduser().resolve()
    contract_root = target_contract_root.expanduser().resolve()
    overlay_config = prompt_overlay_config_path.expanduser().resolve()
    if _sha256(overlay_config) != boundary.prompt_overlay_config_sha256:
        raise ValueError("prompt-overlay config differs from the frozen plan")
    contract_manifest = contract_root / "manifest.json"
    if _sha256(contract_manifest) != boundary.target_contract_manifest_sha256:
        raise ValueError("target-contract manifest differs from the frozen plan")

    overlay_manifest_path = public_root / "prompt_overlay_manifest.json"
    overlay_manifest = _read_object(overlay_manifest_path)
    expected_overlay = {
        "status": "ready",
        "suite_version": boundary.suite_version,
        "cell_count": boundary.expected_cell_count,
        "non_proposer_files_byte_identical": (
            boundary.require_non_proposer_files_byte_identical
        ),
        "target_contract_manifest_sha256": (
            boundary.target_contract_manifest_sha256
        ),
    }
    for key, expected in expected_overlay.items():
        if overlay_manifest.get(key) != expected:
            raise ValueError(
                f"prompt-overlay manifest field differs: {key}="
                f"{overlay_manifest.get(key)!r}, expected={expected!r}"
            )

    suite_root = public_root / boundary.suite_version
    validated_cells: list[dict[str, str]] = []
    for cell in plan.cells:
        prompt_path = suite_root / cell.benchmark_id / "proposer_prompt.txt"
        cell_manifest_path = suite_root / cell.benchmark_id / "manifest.json"
        contract_path = contract_root / "specs" / f"{cell.benchmark_id}.json"
        prompt_sha256 = _sha256(prompt_path)
        contract_sha256 = _sha256(contract_path)
        if prompt_sha256 != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs for {cell.benchmark_id}")
        if contract_sha256 != cell.public_target_contract_sha256:
            raise ValueError(f"target contract differs for {cell.benchmark_id}")
        contract = PublicTargetContract.model_validate_json(
            contract_path.read_text(encoding="utf-8")
        )
        if (
            contract.benchmark_id,
            contract.tier,
            contract.public_prompt_sha256,
        ) != (cell.benchmark_id, cell.tier, prompt_sha256):
            raise ValueError(
                f"target contract identity differs for {cell.benchmark_id}"
            )
        cell_manifest = _read_object(cell_manifest_path)
        if boundary.require_production_registered and (
            cell_manifest.get("status") != "production_registered"
        ):
            raise ValueError(f"public cell is not registered: {cell.benchmark_id}")
        if boundary.require_sealed_test and not cell_manifest.get("test_sealed"):
            raise ValueError(f"public cell has no sealed test: {cell.benchmark_id}")
        validated_cells.append(
            {
                "benchmark_id": cell.benchmark_id,
                "public_prompt_sha256": prompt_sha256,
                "public_target_contract_sha256": contract_sha256,
            }
        )
    return {
        "status": "validated_before_search_calls",
        "prompt_overlay_manifest_sha256": _sha256(overlay_manifest_path),
        "prompt_overlay_config_sha256": _sha256(overlay_config),
        "target_contract_manifest_sha256": _sha256(contract_manifest),
        "cell_count": len(validated_cells),
        "cells": validated_cells,
        "test_data_opened": False,
    }


def freeze_search_ablation_sources(
    plan_path: Path,
    search_root: Path,
    hidden_audit_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Freeze all planned arm outcomes before test or private evaluation."""
    resolved_plan = plan_path.expanduser().resolve()
    plan = load_search_integration_plan(resolved_plan)
    audit_sha256 = validate_hidden_audit(
        hidden_audit_path,
        plan.hidden_contract_audit,
    )
    tasks = build_search_integration_tasks(plan)
    root = search_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests: list[SourceAdapterRequest] = []
    sources: list[FrozenSearchAblationSource] = []
    for task in tasks:
        method_label = f"autoformalism:{task.arm_id}"
        source = (
            root
            / "searches"
            / task.arm_id
            / "runs"
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
            / "summary.json"
        )
        request_id = (
            f"{task.arm_id}__{task.benchmark_id}__{task.tier}__"
            f"rep{task.repetition}"
        )
        if source.is_file():
            payload = _read_object(source)
            actual_identity = (
                payload.get("benchmark_id"),
                payload.get("tier"),
                payload.get("seed"),
            )
            expected_identity = (
                task.benchmark_id,
                task.tier,
                task.repetition,
            )
            if actual_identity != expected_identity:
                raise ValueError(
                    f"search source identity differs for {request_id}: "
                    f"expected={expected_identity}, actual={actual_identity}"
                )
            if payload.get("evaluation_stage") != "development_selection_frozen":
                raise ValueError(f"search source is not development-frozen: {source}")
            if payload.get("selection_policy") != task.selection_policy:
                raise ValueError(
                    f"search source policy differs for {request_id}: "
                    f"expected={task.selection_policy}, "
                    f"actual={payload.get('selection_policy')}"
                )
            digest = _sha256(source)
            status: Literal["available", "missing"] = "available"
        else:
            digest = None
            status = "missing"
        requests.append(
            SourceAdapterRequest(
                request_id=request_id,
                source_kind="autoformalism",
                source_path=source,
                method_label=method_label,
                expected_benchmark_id=task.benchmark_id,
                expected_tier=task.tier,
                expected_repetition=task.repetition,
            )
        )
        sources.append(
            FrozenSearchAblationSource(
                request_id=request_id,
                method_label=method_label,
                arm_id=task.arm_id,
                benchmark_id=task.benchmark_id,
                tier=task.tier,
                repetition=task.repetition,
                source_path=str(source),
                artifact_status=status,
                source_sha256=digest,
            )
        )
    requests_path = output / "source_adapter_requests.jsonl"
    sources_path = output / "frozen_search_sources.jsonl"
    _write_once(
        requests_path,
        "".join(item.model_dump_json() + "\n" for item in requests),
    )
    _write_once(
        sources_path,
        "".join(item.model_dump_json() + "\n" for item in sources),
    )
    manifest = {
        "schema_version": "phase-b-search-ablation-source-freeze-1",
        "status": "frozen_before_test_or_private_evaluation",
        "plan_sha256": _sha256(resolved_plan),
        "hidden_contract_audit_sha256": audit_sha256,
        "planned_source_count": len(sources),
        "available_source_count": sum(
            item.artifact_status == "available" for item in sources
        ),
        "missing_source_count": sum(
            item.artifact_status == "missing" for item in sources
        ),
        "source_adapter_requests_sha256": _sha256(requests_path),
        "frozen_search_sources_sha256": _sha256(sources_path),
        "selection_frozen": True,
        "test_data_opened": False,
        "private_reference_opened_for_candidate_selection": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = output / "source_freeze_manifest.json"
    _write_once(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    for path in (requests_path, sources_path, manifest_path):
        _write_once(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest


def _write_once(path: Path, content: str) -> None:
    """Write atomically, accepting an existing byte-identical frozen file."""
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
