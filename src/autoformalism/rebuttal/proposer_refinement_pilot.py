"""Frozen matched pilot for feedback-rich incumbent refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RefinementPilotCell(BaseModel):
    """One public benchmark cell and its prompt-committed contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mechanism_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RefinementPilotArm(BaseModel):
    """One feedback-matched iterative proposal policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: Literal["rich_exploratory", "rich_incumbent_refinement"]
    proposer_feedback_mode: Literal["rich_v1"]
    proposal_policy: Literal["exploratory", "incumbent_refinement_v1"]
    require_shared_round_zero_cache_hit: bool


class RefinementModelContract(BaseModel):
    """Pinned proposer and judge transport settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    proposer_reasoning_effort: Literal["medium"]
    judge_reasoning_effort: Literal["low"]
    temperature: float = Field(ge=0.0, le=2.0)
    proposer_max_output_tokens: int = Field(ge=128)
    served_context_tokens: int = Field(ge=32768)
    request_timeout_seconds: float = Field(gt=0.0)
    judge_protocol_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RefinementSearchBudget(BaseModel):
    """Common search and numerical budget for both arms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration_budget: int = Field(ge=2)
    beam_size: Literal[1]
    fit_starts: int = Field(ge=1)
    fit_max_nfev: int = Field(ge=1)
    fit_timeout_seconds: float = Field(gt=0.0)
    fit_retry_starts: int = Field(ge=1)
    fit_retry_max_nfev: int = Field(ge=1)
    fit_retry_timeout_seconds: float = Field(gt=0.0)
    final_fit_max_nfev: int = Field(ge=1)
    final_fit_timeout_seconds: float = Field(gt=0.0)
    parameter_fit_strategy: Literal["bounded_nonlinear"]


class ProposerRefinementPilotPlan(BaseModel):
    """Development-only two-arm refinement experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-proposer-refinement-pilot-plan-1"]
    status: Literal["frozen_before_search_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    weighted_overall_score_defined: Literal[False]
    cells: tuple[RefinementPilotCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    arms: tuple[RefinementPilotArm, ...] = Field(min_length=2)
    model_contract: RefinementModelContract
    search_budget: RefinementSearchBudget
    reported_endpoints: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matrix_is_matched_and_unique(self) -> ProposerRefinementPilotPlan:
        """Require one exploratory control followed by one cached refinement arm."""
        cell_keys = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cell_keys) != len(set(cell_keys)):
            raise ValueError("refinement cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        arm_ids = [item.arm_id for item in self.arms]
        if arm_ids != ["rich_exploratory", "rich_incumbent_refinement"]:
            raise ValueError("refinement arms must retain their causal launch order")
        if self.arms[0].require_shared_round_zero_cache_hit:
            raise ValueError("the exploratory arm must populate the shared cache")
        if not self.arms[1].require_shared_round_zero_cache_hit:
            raise ValueError("the refinement arm must reuse the shared round zero")
        if len(self.reported_endpoints) != len(set(self.reported_endpoints)):
            raise ValueError("reported endpoints must be unique")
        return self


class RefinementPilotTask(BaseModel):
    """One independently runnable matched search task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0)
    arm_id: str
    proposer_feedback_mode: str
    proposal_policy: str
    require_shared_round_zero_cache_hit: bool
    benchmark_id: str
    tier: str
    repetition: int = Field(ge=0)
    public_prompt_sha256: str
    public_target_contract_sha256: str
    public_mechanism_spec_sha256: str


def load_refinement_pilot_plan(path: Path) -> ProposerRefinementPilotPlan:
    """Load and validate a frozen refinement plan."""
    return ProposerRefinementPilotPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def build_refinement_pilot_tasks(
    plan: ProposerRefinementPilotPlan,
) -> tuple[RefinementPilotTask, ...]:
    """Build arm-major tasks so the control can populate round-zero caches."""
    tasks: list[RefinementPilotTask] = []
    for arm in plan.arms:
        for cell in plan.cells:
            for repetition in plan.repetitions:
                tasks.append(
                    RefinementPilotTask(
                        task_index=len(tasks),
                        arm_id=arm.arm_id,
                        proposer_feedback_mode=arm.proposer_feedback_mode,
                        proposal_policy=arm.proposal_policy,
                        require_shared_round_zero_cache_hit=(
                            arm.require_shared_round_zero_cache_hit
                        ),
                        benchmark_id=cell.benchmark_id,
                        tier=cell.tier,
                        repetition=repetition,
                        public_prompt_sha256=cell.public_prompt_sha256,
                        public_target_contract_sha256=(
                            cell.public_target_contract_sha256
                        ),
                        public_mechanism_spec_sha256=(
                            cell.public_mechanism_spec_sha256
                        ),
                    )
                )
    return tuple(tasks)


def freeze_refinement_pilot(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path,
    target_contract_root: Path,
    mechanism_spec_root: Path,
    judge_protocol_path: Path,
) -> dict[str, object]:
    """Validate every public input and freeze the task ledger before calls."""
    source = config_path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    plan = load_refinement_pilot_plan(source)
    _validate_judge_protocol(plan, judge_protocol_path)
    cells = []
    for cell in plan.cells:
        prompt = (
            public_data_root
            / "phase_b_v1"
            / cell.benchmark_id
            / "proposer_prompt.txt"
        )
        target = target_contract_root / "specs" / f"{cell.benchmark_id}.json"
        mechanism = mechanism_spec_root / "specs" / f"{cell.benchmark_id}.json"
        _require_sha(prompt, cell.public_prompt_sha256, "public prompt")
        _require_sha(
            target,
            cell.public_target_contract_sha256,
            "public target contract",
        )
        _require_sha(
            mechanism,
            cell.public_mechanism_spec_sha256,
            "public mechanism specification",
        )
        cells.append(
            {
                "benchmark_id": cell.benchmark_id,
                "tier": cell.tier,
                "public_prompt_sha256": cell.public_prompt_sha256,
                "public_target_contract_sha256": (
                    cell.public_target_contract_sha256
                ),
                "public_mechanism_spec_sha256": (
                    cell.public_mechanism_spec_sha256
                ),
            }
        )
    tasks = build_refinement_pilot_tasks(plan)
    task_text = "".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
        for item in tasks
    )
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    task_path = output / "task_plan.jsonl"
    plan_path.write_bytes(source.read_bytes())
    task_path.write_text(task_text, encoding="utf-8")
    manifest = {
        "schema_version": "phase-b-proposer-refinement-pilot-freeze-1",
        "status": "frozen_before_search_calls",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "arm_launch_order": [item.arm_id for item in plan.arms],
        "matched_trial_count": len(plan.cells) * len(plan.repetitions),
        "task_count": len(tasks),
        "plan_sha256": _sha(plan_path),
        "task_plan_sha256": _sha(task_path),
        "judge_protocol_sha256": _sha(judge_protocol_path),
        "public_input_validation": cells,
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (plan_path, task_path, manifest_path):
        (path.with_name(path.name + ".sha256")).write_text(
            f"{_sha(path)}  {path.name}\n",
            encoding="utf-8",
        )
    return manifest


def _validate_judge_protocol(
    plan: ProposerRefinementPilotPlan,
    judge_protocol_path: Path,
) -> None:
    _require_sha(
        judge_protocol_path,
        plan.model_contract.judge_protocol_config_sha256,
        "judge protocol",
    )


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    observed = _sha(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 differs: path={path}, "
            f"expected={expected}, observed={observed}"
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
