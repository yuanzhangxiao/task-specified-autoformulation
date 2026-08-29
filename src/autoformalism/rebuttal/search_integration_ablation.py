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


class SearchAblationCell(BaseModel):
    """One public benchmark/tier cell selected before search calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]


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
    final_fit_max_nfev: int = Field(ge=1)
    final_fit_timeout_seconds: float = Field(gt=0.0)


class SearchModelContract(BaseModel):
    """Frozen local proposer and judge serving contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposer_model: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    reasoning_effort: Literal["low"]
    temperature: float = Field(ge=0.0, le=2.0)
    proposer_max_output_tokens: int = Field(ge=128)
    request_timeout_seconds: float = Field(gt=0.0)


class SearchIntegrationAblationPlan(BaseModel):
    """Predeclared matched integration matrix for judge causal attribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-search-integration-ablation-plan-1"]
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
                    )
                )
    expected = len(plan.arms) * len(plan.cells) * len(plan.repetitions)
    if len(tasks) != expected:
        raise AssertionError("internal search-ablation task count differs")
    return tuple(tasks)


def freeze_search_integration_plan(
    config_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Write immutable plan, task ledger, hashes, and a no-test manifest."""
    resolved_config = config_path.expanduser().resolve()
    plan = load_search_integration_plan(resolved_config)
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
