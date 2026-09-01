"""Frozen design and analysis for GPT-OSS proposer transport calibration."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.targets import PublicTargetContract


class ProposerCalibrationCell(BaseModel):
    """One public benchmark cell sampled before proposer calibration calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProposerCalibrationModelContract(BaseModel):
    """Model settings held fixed except for the calibrated output budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"]
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_token_budgets: tuple[int, ...] = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0.0)
    maximum_provider_attempts: int = Field(ge=1, le=10)
    served_context_tokens: int = Field(ge=8192)

    @model_validator(mode="after")
    def budgets_are_strictly_increasing_and_fit_context(
        self,
    ) -> ProposerCalibrationModelContract:
        """Prevent duplicated conditions and impossible output-only budgets."""
        budgets = self.max_output_token_budgets
        if tuple(sorted(set(budgets))) != budgets:
            raise ValueError("output-token budgets must be unique and increasing")
        if any(value < 128 for value in budgets):
            raise ValueError("every output-token budget must be at least 128")
        if budgets[-1] >= self.served_context_tokens:
            raise ValueError("output-token budget must be smaller than context")
        return self


class ProposerCalibrationGates(BaseModel):
    """Predeclared transport and deterministic-validity operating-point gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_response_success: float = Field(ge=0.0, le=1.0)
    minimum_first_attempt_response_success: float = Field(ge=0.0, le=1.0)
    minimum_deterministic_validity: float = Field(ge=0.0, le=1.0)
    minimum_public_target_pass_rate: float = Field(ge=0.0, le=1.0)
    maximum_length_exhausted_attempt_rate: float = Field(ge=0.0, le=1.0)
    maximum_mean_successful_budget_utilization: float = Field(ge=0.0, le=1.0)


class GPT56ReferenceContext(BaseModel):
    """Descriptive hosted-agent resource context; never an operating-point gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_gate: Literal[False]
    trial_count: int = Field(ge=1)
    reasoning_effort: str
    max_output_tokens: int = Field(ge=1)
    mean_output_tokens: float = Field(ge=0.0)
    exact_reasoning_tokens_status: Literal["pending_raw_cache_audit"]


class ProposerCalibrationPrerequisite(BaseModel):
    """Required result from an earlier frozen calibration stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    required_status: Literal["fail"]
    required_selected_max_output_tokens: None = None
    required_evaluated_budgets: tuple[int, ...] = Field(min_length=1)


class ProposerTransportCalibrationPlan(BaseModel):
    """Immutable matrix for selecting a GPT-OSS proposer output budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-proposer-transport-calibration-plan-1"]
    status: Literal["frozen_before_proposer_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    request_stage: Literal["round_0_empty_feedback"]
    cells: tuple[ProposerCalibrationCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    model_contract: ProposerCalibrationModelContract
    gates: ProposerCalibrationGates
    selection_rule: Literal["smallest_budget_passing_all_gates"]
    prerequisite: ProposerCalibrationPrerequisite | None = None
    gpt_5_6_reference_context: GPT56ReferenceContext
    test_data_opened: Literal[False]
    scientific_judge_called: Literal[False]
    parameter_fitting_performed: Literal[False]

    @model_validator(mode="after")
    def matrix_is_unique(self) -> ProposerTransportCalibrationPlan:
        """Require unique cells and repetitions before any calls are made."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("calibration cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            value < 0 for value in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        return self


class ProposerCalibrationTask(BaseModel):
    """One array task that evaluates every token budget for one matched request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0)
    benchmark_id: str
    tier: Literal["easy", "medium", "hard"]
    repetition: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProposerCalibrationResult(BaseModel):
    """Bounded result for one matched request under one output-token budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-proposer-transport-calibration-result-1"] = (
        "phase-b-proposer-transport-calibration-result-1"
    )
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_index: int = Field(ge=0)
    benchmark_id: str
    tier: Literal["easy", "medium", "hard"]
    repetition: int = Field(ge=0)
    model: str
    reasoning_effort: Literal["low", "medium", "high"]
    max_output_tokens: int = Field(ge=128)
    request_hash: str | None = None
    cache_hit: bool = False
    response_success: bool
    first_attempt_response_success: bool
    provider_attempt_count: int = Field(ge=0)
    provider_input_tokens: int | None = Field(default=None, ge=0)
    provider_output_tokens: int | None = Field(default=None, ge=0)
    provider_total_tokens: int | None = Field(default=None, ge=0)
    successful_attempt_input_tokens: int | None = Field(default=None, ge=0)
    successful_attempt_output_tokens: int | None = Field(default=None, ge=0)
    successful_attempt_total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    length_exhausted_attempt_count: int = Field(ge=0)
    reasoning_character_count: int = Field(ge=0)
    deterministic_valid: bool
    public_target_passed: bool
    candidate_structural_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    deterministic_diagnostics: tuple[dict[str, object], ...] = ()
    error_type: str | None = None
    error: str | None = None
    test_data_opened: Literal[False]
    scientific_judge_called: Literal[False]
    parameter_fitting_performed: Literal[False]


def load_proposer_calibration_plan(path: Path) -> ProposerTransportCalibrationPlan:
    """Load one strict calibration plan from JSON."""
    return ProposerTransportCalibrationPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def build_proposer_calibration_tasks(
    plan: ProposerTransportCalibrationPlan,
) -> tuple[ProposerCalibrationTask, ...]:
    """Build matched cell/repetition tasks in deterministic order."""
    tasks: list[ProposerCalibrationTask] = []
    for cell in plan.cells:
        for repetition in plan.repetitions:
            tasks.append(
                ProposerCalibrationTask(
                    task_index=len(tasks),
                    benchmark_id=cell.benchmark_id,
                    tier=cell.tier,
                    repetition=repetition,
                    public_prompt_sha256=cell.public_prompt_sha256,
                    public_target_contract_sha256=(
                        cell.public_target_contract_sha256
                    ),
                )
            )
    return tuple(tasks)


def freeze_proposer_calibration(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path,
    target_contract_root: Path,
    prerequisite_analysis_path: Path | None = None,
) -> dict[str, object]:
    """Validate public inputs and freeze the matched calibration matrix."""
    plan = load_proposer_calibration_plan(config_path)
    tasks = build_proposer_calibration_tasks(plan)
    public_root = public_data_root.expanduser().resolve()
    contract_root = target_contract_root.expanduser().resolve()
    for cell in plan.cells:
        prompt = (
            public_root
            / "phase_b_v1"
            / cell.benchmark_id
            / "proposer_prompt.txt"
        )
        contract_path = contract_root / "specs" / f"{cell.benchmark_id}.json"
        if not prompt.is_file():
            raise ValueError(f"missing public proposer prompt: {prompt}")
        if not contract_path.is_file():
            raise ValueError(f"missing public target contract: {contract_path}")
        if _sha256(prompt) != cell.public_prompt_sha256:
            raise ValueError(f"public proposer prompt differs: {cell.benchmark_id}")
        if _sha256(contract_path) != cell.public_target_contract_sha256:
            raise ValueError(f"public target contract differs: {cell.benchmark_id}")
        contract = PublicTargetContract.model_validate_json(
            contract_path.read_text(encoding="utf-8")
        )
        if (contract.benchmark_id, contract.tier) != (
            cell.benchmark_id,
            cell.tier,
        ):
            raise ValueError(f"target contract identity differs: {cell.benchmark_id}")
        if contract.public_prompt_sha256 != cell.public_prompt_sha256:
            raise ValueError(f"target contract prompt differs: {cell.benchmark_id}")

    prerequisite_record = _validate_prerequisite_analysis(
        plan,
        prerequisite_analysis_path,
    )
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_plan = root / "plan.json"
    task_plan = root / "task_plan.jsonl"
    _write_or_validate(frozen_plan, config_path.read_text(encoding="utf-8"))
    _write_or_validate(
        task_plan,
        "".join(item.model_dump_json() + "\n" for item in tasks),
    )
    manifest = {
        "schema_version": "phase-b-proposer-transport-calibration-freeze-1",
        "status": "frozen_before_proposer_calls",
        "plan_sha256": _sha256(frozen_plan),
        "task_plan_sha256": _sha256(task_plan),
        "matched_request_count": len(tasks),
        "token_budget_count": len(plan.model_contract.max_output_token_budgets),
        "planned_result_count": (
            len(tasks) * len(plan.model_contract.max_output_token_budgets)
        ),
        "test_data_opened": False,
        "scientific_judge_called": False,
        "parameter_fitting_performed": False,
    }
    if prerequisite_record is not None:
        manifest["prerequisite"] = prerequisite_record
    manifest_path = root / "freeze_manifest.json"
    _write_or_validate(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for path in (frozen_plan, task_plan, manifest_path):
        _write_or_validate(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest


def analyze_proposer_calibration(
    plan: ProposerTransportCalibrationPlan,
    results: tuple[ProposerCalibrationResult, ...],
) -> dict[str, object]:
    """Apply the frozen gates and select the least costly passing budget."""
    expected = {
        (task.task_index, budget)
        for task in build_proposer_calibration_tasks(plan)
        for budget in plan.model_contract.max_output_token_budgets
    }
    observed = {(item.task_index, item.max_output_tokens) for item in results}
    if observed != expected or len(observed) != len(results):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"calibration result keys differ; missing={missing}, extra={extra}"
        )

    grouped: dict[int, list[ProposerCalibrationResult]] = defaultdict(list)
    for result in results:
        grouped[result.max_output_tokens].append(result)
    rows: list[dict[str, object]] = []
    for budget in plan.model_contract.max_output_token_budgets:
        items = grouped[budget]
        attempt_count = sum(item.provider_attempt_count for item in items)
        successful_utilization = [
            item.successful_attempt_output_tokens / budget
            for item in items
            if item.response_success
            and item.successful_attempt_output_tokens is not None
        ]
        metrics = {
            "response_success": mean(item.response_success for item in items),
            "first_attempt_response_success": mean(
                item.first_attempt_response_success for item in items
            ),
            "deterministic_validity": mean(
                item.deterministic_valid for item in items
            ),
            "public_target_pass_rate": mean(
                item.public_target_passed for item in items
            ),
            "length_exhausted_attempt_rate": (
                0.0
                if attempt_count == 0
                else sum(item.length_exhausted_attempt_count for item in items)
                / attempt_count
            ),
            "mean_successful_budget_utilization": (
                None
                if not successful_utilization
                else mean(successful_utilization)
            ),
            "mean_latency_ms": _optional_mean(
                item.latency_ms for item in items
            ),
            "mean_provider_output_tokens": _optional_mean(
                item.provider_output_tokens for item in items
            ),
            "provider_attempt_count": attempt_count,
            "requested_result_count": len(items),
        }
        utilization = metrics["mean_successful_budget_utilization"]
        checks = {
            "minimum_response_success": (
                metrics["response_success"] >= plan.gates.minimum_response_success
            ),
            "minimum_first_attempt_response_success": (
                metrics["first_attempt_response_success"]
                >= plan.gates.minimum_first_attempt_response_success
            ),
            "minimum_deterministic_validity": (
                metrics["deterministic_validity"]
                >= plan.gates.minimum_deterministic_validity
            ),
            "minimum_public_target_pass_rate": (
                metrics["public_target_pass_rate"]
                >= plan.gates.minimum_public_target_pass_rate
            ),
            "maximum_length_exhausted_attempt_rate": (
                metrics["length_exhausted_attempt_rate"]
                <= plan.gates.maximum_length_exhausted_attempt_rate
            ),
            "maximum_mean_successful_budget_utilization": (
                utilization is not None
                and utilization
                <= plan.gates.maximum_mean_successful_budget_utilization
            ),
        }
        rows.append(
            {
                "max_output_tokens": budget,
                **metrics,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    selected = next(
        (row["max_output_tokens"] for row in rows if row["passed"]), None
    )
    return {
        "schema_version": "phase-b-proposer-transport-calibration-analysis-1",
        "status": "pass" if selected is not None else "fail",
        "selection_rule": plan.selection_rule,
        "selected_max_output_tokens": selected,
        "selected_reasoning_effort": (
            plan.model_contract.reasoning_effort if selected is not None else None
        ),
        "operating_points": rows,
        "gates": plan.gates.model_dump(mode="json"),
        "gpt_5_6_reference_context": (
            plan.gpt_5_6_reference_context.model_dump(mode="json")
        ),
        "test_data_opened": False,
        "scientific_judge_called": False,
        "parameter_fitting_performed": False,
    }


def prepare_selected_proposer_confirmation(
    source_plan_path: Path,
    source_analysis_path: Path,
    output_config_path: Path,
    *,
    primary_platform: str,
    confirmation_platform: str,
) -> dict[str, object]:
    """Freeze a one-budget cross-cluster confirmation from a passing analysis.

    The confirmation repeats the source plan's public request matrix and every
    model setting except the already selected output budget. Candidate text is
    not required to be identical across accelerator types; both platforms must
    independently pass the same transport and deterministic-validity gates.
    """
    source_plan = load_proposer_calibration_plan(source_plan_path)
    source_analysis = _load_analysis(source_analysis_path)
    selected = source_analysis.get("selected_max_output_tokens")
    if source_analysis.get("status") != "pass" or not isinstance(selected, int):
        raise ValueError(
            "source proposer calibration did not select an operating point"
        )
    if selected not in source_plan.model_contract.max_output_token_budgets:
        raise ValueError("selected output budget is absent from the source plan")
    selected_rows = [
        row
        for row in source_analysis["operating_points"]
        if row.get("max_output_tokens") == selected
    ]
    if len(selected_rows) != 1 or selected_rows[0].get("passed") is not True:
        raise ValueError("selected source operating point did not pass every gate")
    if source_analysis.get("selected_reasoning_effort") != (
        source_plan.model_contract.reasoning_effort
    ):
        raise ValueError("source analysis reasoning effort differs from its plan")
    if not primary_platform or not confirmation_platform:
        raise ValueError("both cluster platform labels must be nonempty")
    if primary_platform == confirmation_platform:
        raise ValueError("confirmation must use a distinct serving platform")

    payload = source_plan.model_dump(mode="json")
    payload["purpose"] = (
        "Confirm the selected GPT-OSS-120B proposer operating point on a "
        "distinct accelerator platform using the unchanged public requests and gates"
    )
    payload["model_contract"]["max_output_token_budgets"] = [selected]
    payload["prerequisite"] = None
    output = output_config_path.expanduser().resolve()
    _write_or_validate(
        output,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    generated = load_proposer_calibration_plan(output)
    if generated.model_contract.max_output_token_budgets != (selected,):
        raise AssertionError("generated confirmation plan has the wrong budget")
    manifest = {
        "schema_version": "phase-b-proposer-cross-cluster-handoff-1",
        "status": "frozen_before_confirmation_calls",
        "primary_platform": primary_platform,
        "confirmation_platform": confirmation_platform,
        "source_plan_sha256": _sha256(source_plan_path.expanduser().resolve()),
        "source_analysis_sha256": _sha256(
            source_analysis_path.expanduser().resolve()
        ),
        "confirmation_plan_sha256": _sha256(output),
        "selected_max_output_tokens": selected,
        "selected_reasoning_effort": source_plan.model_contract.reasoning_effort,
        "model": source_plan.model_contract.model,
        "matched_request_count": len(build_proposer_calibration_tasks(source_plan)),
        "candidate_identity_match_required": False,
        "same_public_requests_and_gates_required": True,
        "test_data_opened": False,
        "scientific_judge_called": False,
        "parameter_fitting_performed": False,
    }
    manifest_path = output.parent / "cross_cluster_handoff.json"
    _write_or_validate(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_or_validate(
        manifest_path.with_name(f"{manifest_path.name}.sha256"),
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
    )
    return manifest


def verify_proposer_cross_cluster_confirmation(
    source_analysis_path: Path,
    confirmation_analysis_path: Path,
    handoff_path: Path,
) -> dict[str, object]:
    """Verify a selected-budget confirmation without comparing generated text."""
    source_analysis = _load_analysis(source_analysis_path)
    confirmation_analysis = _load_analysis(confirmation_analysis_path)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(handoff, dict) or handoff.get("schema_version") != (
        "phase-b-proposer-cross-cluster-handoff-1"
    ):
        raise ValueError("cross-cluster handoff schema differs")
    if _sha256(source_analysis_path.expanduser().resolve()) != handoff.get(
        "source_analysis_sha256"
    ):
        raise ValueError("source analysis differs from the frozen handoff")
    selected = source_analysis.get("selected_max_output_tokens")
    if selected != handoff.get("selected_max_output_tokens"):
        raise ValueError("source selected budget differs from the frozen handoff")
    confirmation_rows = confirmation_analysis.get("operating_points")
    if not isinstance(confirmation_rows, list) or len(confirmation_rows) != 1:
        raise ValueError("confirmation must contain exactly one operating point")
    row = confirmation_rows[0]
    if not isinstance(row, dict) or row.get("max_output_tokens") != selected:
        raise ValueError("confirmation evaluated a different output budget")
    checks = {
        "source_calibration_passed": source_analysis.get("status") == "pass",
        "confirmation_passed": confirmation_analysis.get("status") == "pass",
        "selected_budget_matched": (
            confirmation_analysis.get("selected_max_output_tokens") == selected
        ),
        "reasoning_effort_matched": (
            confirmation_analysis.get("selected_reasoning_effort")
            == source_analysis.get("selected_reasoning_effort")
            == handoff.get("selected_reasoning_effort")
        ),
        "confirmation_gate_row_passed": row.get("passed") is True,
    }
    return {
        "schema_version": "phase-b-proposer-cross-cluster-confirmation-1",
        "status": "pass" if all(checks.values()) else "fail",
        "primary_platform": handoff.get("primary_platform"),
        "confirmation_platform": handoff.get("confirmation_platform"),
        "selected_max_output_tokens": selected,
        "checks": checks,
        "confirmation_metrics": {
            key: row.get(key)
            for key in (
                "response_success",
                "first_attempt_response_success",
                "deterministic_validity",
                "public_target_pass_rate",
                "length_exhausted_attempt_rate",
                "mean_successful_budget_utilization",
                "mean_latency_ms",
            )
        },
        "candidate_identity_match_required": False,
        "test_data_opened": False,
        "scientific_judge_called": False,
        "parameter_fitting_performed": False,
    }


def _load_analysis(path: Path) -> dict[str, object]:
    """Load one strict proposer-calibration analysis object."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"missing proposer calibration analysis: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "phase-b-proposer-transport-calibration-analysis-1"
    ):
        raise ValueError("proposer calibration analysis schema differs")
    rows = payload.get("operating_points")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("proposer calibration operating points are invalid")
    return payload


def _optional_mean(values: object) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else mean(numeric)


def _validate_prerequisite_analysis(
    plan: ProposerTransportCalibrationPlan,
    path: Path | None,
) -> dict[str, object] | None:
    """Bind a continuation plan to its immutable failed predecessor."""
    required = plan.prerequisite
    if required is None:
        if path is not None:
            raise ValueError("plan does not declare a prerequisite analysis")
        return None
    if path is None:
        raise ValueError("plan requires a prerequisite analysis")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"missing prerequisite analysis: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prerequisite analysis must be a JSON object")
    if payload.get("schema_version") != (
        "phase-b-proposer-transport-calibration-analysis-1"
    ):
        raise ValueError("prerequisite analysis schema differs")
    if payload.get("status") != required.required_status:
        raise ValueError("prerequisite analysis status differs")
    if (
        payload.get("selected_max_output_tokens")
        is not required.required_selected_max_output_tokens
    ):
        raise ValueError("prerequisite selected operating point differs")
    rows = payload.get("operating_points")
    if not isinstance(rows, list):
        raise ValueError("prerequisite operating points are missing")
    budgets = tuple(
        row.get("max_output_tokens")
        for row in rows
        if isinstance(row, dict)
    )
    if budgets != required.required_evaluated_budgets:
        raise ValueError("prerequisite evaluated budgets differ")
    return {
        "experiment_id": required.experiment_id,
        "analysis_sha256": _sha256(resolved),
        "status": payload["status"],
        "selected_max_output_tokens": payload["selected_max_output_tokens"],
        "evaluated_budgets": list(budgets),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_or_validate(path: Path, text: str) -> None:
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
