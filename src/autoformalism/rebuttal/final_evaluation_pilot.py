"""Frozen source planning for the Phase-B two-cell final-evaluation pilot."""

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


class FinalEvaluationPilotCell(BaseModel):
    """One public benchmark/tier cell selected before pilot method runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]


class HiddenAuditRequirement(BaseModel):
    """Exact pre-method private-contract audit required by the pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-hidden-subspace-contract-audit-2"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_status: Literal["pass"] = "pass"


class FinalEvaluationPilotPlan(BaseModel):
    """Predeclared development pilot matrix and endpoint contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-final-evaluation-pilot-plan-1"]
    status: Literal["frozen_before_autoformalism_pilot_runs"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    cells: tuple[FinalEvaluationPilotCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    methods: tuple[Literal["autoformalism", "raw_data_agent:gpt-5.6-sol"], ...]
    hidden_contract_audit: HiddenAuditRequirement
    endpoints: tuple[
        Literal[
            "source_completion",
            "runtime_validity",
            "public_mechanism_compliance",
            "sealed_target_nmse",
            "hidden_response_subspace_nmse",
            "intervention_behavior",
            "model_complexity",
        ],
        ...,
    ]
    weighted_overall_score_defined: Literal[False]
    qualitative_llm_requested: Literal[False]

    @model_validator(mode="after")
    def matrix_is_unique_and_complete(self) -> FinalEvaluationPilotPlan:
        """Reject duplicated cells, repetitions, methods, and incomplete methods."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("pilot cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            value < 0 for value in self.repetitions
        ):
            raise ValueError("pilot repetitions must be unique and nonnegative")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("pilot methods must be unique")
        expected = {"autoformalism", "raw_data_agent:gpt-5.6-sol"}
        if set(self.methods) != expected:
            raise ValueError(f"pilot methods must be exactly {sorted(expected)}")
        if len(self.endpoints) != len(set(self.endpoints)):
            raise ValueError("pilot endpoints must be unique")
        return self


class FrozenPilotSource(BaseModel):
    """One source selected by the plan and frozen by content hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    method_id: str
    benchmark_id: str
    tier: str
    repetition: int
    source_kind: str
    source_path: str
    artifact_sha256: dict[str, str]


def load_pilot_plan(path: Path) -> FinalEvaluationPilotPlan:
    """Load and validate a frozen pilot plan."""
    return FinalEvaluationPilotPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def validate_hidden_audit(path: Path, requirement: HiddenAuditRequirement) -> str:
    """Verify the exact successful audit artifact and its companion digest."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"hidden contract audit is missing: {resolved}")
    digest = _sha256(resolved)
    if digest != requirement.sha256:
        raise ValueError(
            "hidden contract audit SHA-256 differs; "
            f"expected={requirement.sha256}, actual={digest}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != requirement.schema_version:
        raise ValueError("hidden contract audit schema differs from the pilot plan")
    if payload.get("status") != requirement.required_status:
        raise ValueError("hidden contract audit did not pass")
    digest_path = resolved.with_name(f"{resolved.name}.sha256")
    if not digest_path.is_file():
        raise ValueError(f"hidden contract audit digest is missing: {digest_path}")
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != resolved.name:
        raise ValueError("hidden contract audit companion digest is invalid")
    return digest


def freeze_pilot_sources(
    plan: FinalEvaluationPilotPlan,
    *,
    autoformalism_root: Path,
    raw_agent_root: Path,
) -> tuple[tuple[SourceAdapterRequest, ...], tuple[FrozenPilotSource, ...]]:
    """Resolve the exact planned artifacts and freeze their relevant files."""
    auto_root = autoformalism_root.expanduser().resolve()
    raw_root = raw_agent_root.expanduser().resolve()
    requests: list[SourceAdapterRequest] = []
    sources: list[FrozenPilotSource] = []
    missing: list[str] = []
    for cell in plan.cells:
        for repetition in plan.repetitions:
            for method_id in plan.methods:
                source_kind: Literal["autoformalism", "raw_data_agent"]
                if method_id == "autoformalism":
                    source_kind = "autoformalism"
                    source_path = (
                        auto_root
                        / "runs"
                        / f"{cell.benchmark_id}_{cell.tier}_seed{repetition}"
                        / "summary.json"
                    )
                    relevant = (source_path,)
                else:
                    source_kind = "raw_data_agent"
                    source_path = raw_root / (
                        f"openai_gpt-5-6-sol_{cell.benchmark_id}_"
                        f"{cell.tier}_rep{repetition}"
                    )
                    relevant = tuple(
                        source_path / name
                        for name in (
                            "run_config.json",
                            "candidate.json",
                            "evaluation.json",
                        )
                    )
                absent = [str(path) for path in relevant if not path.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                if method_id == "autoformalism":
                    payload = _read_object(source_path)
                    if payload.get("selection_policy") != "incumbent_relative_hybrid":
                        raise ValueError(
                            "Autoformalism pilot source does not use the frozen "
                            f"selection policy: {source_path}"
                        )
                    if (
                        payload.get("evaluation_stage")
                        != "development_selection_frozen"
                    ):
                        raise ValueError(
                            "Autoformalism pilot source is not a development-frozen "
                            f"selection: {source_path}"
                        )
                else:
                    payload = _read_object(source_path / "run_config.json")
                    if (payload.get("provider"), payload.get("model")) != (
                        "openai",
                        "gpt-5.6-sol",
                    ):
                        raise ValueError(
                            "raw-agent pilot source is not openai:gpt-5.6-sol: "
                            f"{source_path}"
                        )
                request_id = (
                    f"{method_id.replace(':', '_')}__{cell.benchmark_id}__"
                    f"{cell.tier}__rep{repetition}"
                )
                request = SourceAdapterRequest(
                    request_id=request_id,
                    source_kind=source_kind,
                    source_path=source_path,
                )
                identity = source_identity(request)
                expected = (cell.benchmark_id, cell.tier, repetition)
                if identity != expected:
                    raise ValueError(
                        f"source identity differs for {request_id}: "
                        f"expected={expected}, actual={identity}"
                    )
                requests.append(request)
                sources.append(
                    FrozenPilotSource(
                        request_id=request_id,
                        method_id=method_id,
                        benchmark_id=cell.benchmark_id,
                        tier=cell.tier,
                        repetition=repetition,
                        source_kind=source_kind,
                        source_path=str(source_path),
                        artifact_sha256={
                            path.name: _sha256(path) for path in relevant
                        },
                    )
                )
    if missing:
        rendered = "\n".join(f"- {item}" for item in sorted(missing))
        raise ValueError(f"planned source artifacts are missing:\n{rendered}")
    expected_count = len(plan.cells) * len(plan.repetitions) * len(plan.methods)
    if len(requests) != expected_count:
        raise AssertionError("internal pilot source count differs from plan")
    return tuple(requests), tuple(sources)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
