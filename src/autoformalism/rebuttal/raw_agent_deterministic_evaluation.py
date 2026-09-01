"""Frozen source composition for the full GPT-5.6 raw-agent evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    source_identity,
)
from autoformalism.rebuttal.final_evaluation_pilot import (
    FinalEvaluationPilotCell,
    FrozenPilotSource,
    HiddenAuditRequirement,
)


class RawAgentDeterministicEvaluationPlan(BaseModel):
    """Predeclared full-suite evaluation of provider-fitted raw-agent models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-raw-agent-deterministic-evaluation-plan-1"]
    status: Literal["frozen_before_test_or_private_evaluation"]
    method_id: Literal["raw_data_agent:gpt-5.6-sol"]
    provider: Literal["openai"]
    model: Literal["gpt-5.6-sol"]
    full_protocol_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_refresh_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hidden_contract_audit: HiddenAuditRequirement
    repetitions: tuple[int, ...] = Field(min_length=1)
    postfreeze_shard_count: int = Field(ge=1)
    hidden_shard_count: int = Field(ge=1)
    endpoints: tuple[
        Literal[
            "source_completion",
            "runtime_validity",
            "public_mechanism_compliance",
            "sealed_target_nmse",
            "hidden_response_subspace_nmse",
            "intervention_behavior",
            "model_complexity",
            "resource_usage",
        ],
        ...,
    ]
    weighted_overall_score_defined: Literal[False]
    qualitative_llm_requested: Literal[False]
    parameter_refit_applied: Literal[False]
    test_data_opened: Literal[False]

    @model_validator(mode="after")
    def unique_fields(self) -> RawAgentDeterministicEvaluationPlan:
        """Reject duplicate repetitions or endpoints."""
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        if len(self.endpoints) != len(set(self.endpoints)):
            raise ValueError("evaluation endpoints must be unique")
        return self


def load_raw_agent_evaluation_plan(
    path: Path,
) -> RawAgentDeterministicEvaluationPlan:
    """Load the frozen raw-agent evaluation plan."""
    return RawAgentDeterministicEvaluationPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def freeze_raw_agent_sources(
    plan: RawAgentDeterministicEvaluationPlan,
    *,
    full_protocol_config: Path,
    prompt_refresh_config: Path,
    historical_root: Path,
    refresh_root: Path,
    public_data_root: Path,
) -> tuple[
    tuple[SourceAdapterRequest, ...],
    tuple[FrozenPilotSource, ...],
    tuple[FinalEvaluationPilotCell, ...],
]:
    """Compose 90 historical and 30 refreshed runs before sealed evaluation."""
    full_path = full_protocol_config.expanduser().resolve()
    refresh_path = prompt_refresh_config.expanduser().resolve()
    if _sha256(full_path) != plan.full_protocol_config_sha256:
        raise ValueError("full raw-agent protocol differs from the frozen plan")
    if _sha256(refresh_path) != plan.prompt_refresh_config_sha256:
        raise ValueError("prompt-refresh protocol differs from the frozen plan")
    full = _read_object(full_path)
    refresh = _read_object(refresh_path)
    if (full.get("provider"), full.get("model"), full.get("output_contract")) != (
        plan.provider,
        plan.model,
        "fitted_model",
    ):
        raise ValueError("full protocol is not the frozen provider-fitted baseline")
    if (
        refresh.get("provider"),
        refresh.get("model"),
        refresh.get("output_contract"),
    ) != (plan.provider, plan.model, "fitted_model"):
        raise ValueError("refresh protocol is not the frozen provider-fitted baseline")
    if int(full.get("repetitions", -1)) != len(plan.repetitions):
        raise ValueError("full protocol repetition count differs from the plan")
    if int(refresh.get("repetitions", -1)) != len(plan.repetitions):
        raise ValueError("refresh repetition count differs from the plan")

    cells = _cells(full)
    refresh_cells = _cells(refresh, allow_empty=True)
    refresh_by_id = {item.benchmark_id: item for item in refresh_cells}
    full_by_id = {item.benchmark_id: item for item in cells}
    if not set(refresh_by_id) <= set(full_by_id):
        raise ValueError("refresh protocol contains a cell outside the full matrix")
    for identifier, refreshed in refresh_by_id.items():
        if full_by_id[identifier].tier != refreshed.tier:
            raise ValueError(f"refresh tier differs for {identifier}")

    public_root = public_data_root.expanduser().resolve()
    old_root = historical_root.expanduser().resolve()
    new_root = refresh_root.expanduser().resolve() / "runs"
    requests: list[SourceAdapterRequest] = []
    sources: list[FrozenPilotSource] = []
    for cell in cells:
        prompt_path = (
            public_root / "phase_b_v1" / cell.benchmark_id / "proposer_prompt.txt"
        )
        if not prompt_path.is_file():
            raise ValueError(f"public proposer prompt is missing: {prompt_path}")
        expected_prompt_sha256 = _sha256(prompt_path)
        if cell.benchmark_id in refresh_by_id:
            declared = _refresh_prompt_sha256(refresh, cell.benchmark_id)
            if declared != expected_prompt_sha256:
                raise ValueError(
                    f"refresh/public prompt differs for {cell.benchmark_id}"
                )
            selected_root = new_root
            source_generation = "prompt_v3_refresh"
        else:
            selected_root = old_root
            source_generation = "historical_unchanged_prompt"
        for repetition in plan.repetitions:
            run_name = (
                f"openai_gpt-5-6-sol_{cell.benchmark_id}_"
                f"{cell.tier}_rep{repetition}"
            )
            source_path = selected_root / run_name
            relevant = tuple(
                source_path / name
                for name in ("run_config.json", "candidate.json", "evaluation.json")
            )
            absent = tuple(sorted(str(path) for path in relevant if not path.is_file()))
            auxiliary = tuple(
                path
                for path in (
                    source_path / "status.json",
                    source_path / "agent_result.json",
                )
                if path.is_file()
            )
            if not absent:
                _validate_run_config(
                    source_path / "run_config.json",
                    cell=cell,
                    repetition=repetition,
                    prompt_sha256=expected_prompt_sha256,
                    plan=plan,
                )
            request_id = (
                f"raw_data_agent_gpt-5.6-sol__{cell.benchmark_id}__"
                f"{cell.tier}__rep{repetition}"
            )
            request = SourceAdapterRequest(
                request_id=request_id,
                source_kind="raw_data_agent",
                source_path=source_path,
                expected_benchmark_id=cell.benchmark_id,
                expected_tier=cell.tier,
                expected_repetition=repetition,
            )
            expected_identity = (cell.benchmark_id, cell.tier, repetition)
            if source_identity(request) != expected_identity:
                raise ValueError(f"source identity differs for {request_id}")
            requests.append(request)
            sources.append(
                FrozenPilotSource(
                    request_id=request_id,
                    method_id=plan.method_id,
                    benchmark_id=cell.benchmark_id,
                    tier=cell.tier,
                    repetition=repetition,
                    source_kind=f"raw_data_agent:{source_generation}",
                    source_path=str(source_path),
                    artifact_status="missing" if absent else "available",
                    missing_artifacts=absent,
                    artifact_sha256={
                        path.name: _sha256(path)
                        for path in (*relevant, *auxiliary)
                        if path.is_file()
                    },
                )
            )
    expected_count = len(cells) * len(plan.repetitions)
    if len(requests) != expected_count:
        raise AssertionError("raw-agent source composition differs from the plan")
    return tuple(requests), tuple(sources), cells


def _cells(
    payload: dict[str, object], *, allow_empty: bool = False
) -> tuple[FinalEvaluationPilotCell, ...]:
    raw = payload.get("benchmarks")
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise ValueError("raw-agent protocol has no benchmark cells")
    cells = tuple(
        FinalEvaluationPilotCell.model_validate(
            {
                "benchmark_id": item["benchmark_id"],
                "tier": item["tier"],
            }
        )
        for item in raw
    )
    identifiers = [item.benchmark_id for item in cells]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("raw-agent benchmark cells are duplicated")
    return cells


def _refresh_prompt_sha256(payload: dict[str, object], benchmark_id: str) -> str:
    raw = payload.get("benchmarks")
    assert isinstance(raw, list)
    for item in raw:
        if isinstance(item, dict) and item.get("benchmark_id") == benchmark_id:
            return str(item["public_prompt_sha256"])
    raise AssertionError("refresh cell lookup failed")


def _validate_run_config(
    path: Path,
    *,
    cell: FinalEvaluationPilotCell,
    repetition: int,
    prompt_sha256: str,
    plan: RawAgentDeterministicEvaluationPlan,
) -> None:
    config = _read_object(path)
    if (
        config.get("provider"),
        config.get("model"),
        config.get("benchmark_id"),
        config.get("tier"),
        config.get("repetition"),
    ) != (
        plan.provider,
        plan.model,
        cell.benchmark_id,
        cell.tier,
        repetition,
    ):
        raise ValueError(f"raw-agent run identity differs: {path}")
    hashes = config.get("public_input_hashes")
    if (
        not isinstance(hashes, dict)
        or hashes.get("proposer_prompt.txt") != prompt_sha256
    ):
        raise ValueError(f"raw-agent run used a different public prompt: {path}")
    agent = config.get("agent_config")
    if not isinstance(agent, dict) or agent.get("output_contract") != "fitted_model":
        raise ValueError(
            f"raw-agent run does not use the fitted-model contract: {path}"
        )
    if config.get("parameter_refit_applied") is not False:
        raise ValueError(f"raw-agent run applied a parameter refit: {path}")
    if config.get("test_data_opened") is not False:
        raise ValueError(f"raw-agent run opened test data: {path}")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
