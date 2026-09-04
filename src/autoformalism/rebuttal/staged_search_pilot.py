"""Frozen public-only pilot for feedback-routed staged model search."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StagedSearchCell(BaseModel):
    """One public benchmark and its prompt-derived deterministic contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_target_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mechanism_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StagedSearchModelContract(BaseModel):
    """Pinned local proposer/judge transport settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    proposer_reasoning_effort: Literal["low", "medium"]
    judge_reasoning_effort: Literal["low"]
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=128)
    served_context_tokens: int = Field(ge=32768)
    request_timeout_seconds: float = Field(gt=0.0)


class StagedSearchBudget(BaseModel):
    """Common search and fitting budget for every pilot task."""

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


class StagedSearchPilotPlan(BaseModel):
    """Development-only staged-search evaluation plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-staged-search-pilot-plan-1"]
    status: Literal["frozen_before_search_calls"]
    purpose: str = Field(min_length=1)
    development_only: Literal[True]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]
    weighted_overall_score_defined: Literal[False]
    proposer_construction_mode: Literal["staged_v2"]
    proposer_feedback_mode: Literal["rich_v1"]
    proposal_policy: Literal["incumbent_refinement_v1"]
    apply_postfit_pruning: Literal[False]
    cells: tuple[StagedSearchCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    model_contract: StagedSearchModelContract
    search_budget: StagedSearchBudget
    reported_endpoints: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matrix_is_unique(self) -> StagedSearchPilotPlan:
        """Reject duplicated benchmark cells, seeds, or endpoint names."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("staged search cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("staged search repetitions must be unique and nonnegative")
        if len(self.reported_endpoints) != len(set(self.reported_endpoints)):
            raise ValueError("reported endpoints must be unique")
        if self.search_budget.fit_retry_starts < self.search_budget.fit_starts:
            raise ValueError("fit retry must not reduce the number of starts")
        if self.search_budget.fit_retry_max_nfev < self.search_budget.fit_max_nfev:
            raise ValueError("fit retry must not reduce the evaluation budget")
        if (
            self.search_budget.fit_retry_timeout_seconds
            < self.search_budget.fit_timeout_seconds
        ):
            raise ValueError("fit retry must not reduce the wall-clock budget")
        return self


class StagedSearchTask(BaseModel):
    """One independently runnable cell/seed task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0)
    benchmark_id: str
    tier: str
    repetition: int = Field(ge=0)
    public_prompt_sha256: str
    public_target_contract_sha256: str
    public_mechanism_spec_sha256: str


def load_staged_search_plan(path: Path) -> StagedSearchPilotPlan:
    """Load and strictly validate one staged-search plan."""
    return StagedSearchPilotPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def build_staged_search_tasks(
    plan: StagedSearchPilotPlan,
) -> tuple[StagedSearchTask, ...]:
    """Return a stable cell-major task matrix."""
    return tuple(
        StagedSearchTask(
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


def freeze_staged_search_pilot(
    config_path: Path,
    output_root: Path,
    *,
    public_data_root: Path,
    target_contract_root: Path,
    mechanism_spec_root: Path,
) -> dict[str, object]:
    """Verify every public input and atomically freeze a task ledger."""
    source = config_path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    plan = load_staged_search_plan(source)
    cells: list[dict[str, object]] = []
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
        cells.append(cell.model_dump(mode="json"))

    tasks = build_staged_search_tasks(plan)
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    tasks_path = output / "task_plan.jsonl"
    plan_path.write_bytes(source.read_bytes())
    tasks_path.write_text(
        "".join(
            json.dumps(task.model_dump(mode="json"), sort_keys=True) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "phase-b-staged-search-pilot-freeze-1",
        "status": "frozen_before_search_calls",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "task_count": len(tasks),
        "cell_count": len(plan.cells),
        "repetition_count": len(plan.repetitions),
        "plan_sha256": _sha(plan_path),
        "task_plan_sha256": _sha(tasks_path),
        "public_input_validation": cells,
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (plan_path, tasks_path, manifest_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{_sha(path)}  {path.name}\n",
            encoding="utf-8",
        )
    return manifest


def summarize_staged_search_pilot(
    plan_path: Path,
    task_plan_path: Path,
    search_root: Path,
) -> dict[str, object]:
    """Summarize public search endpoints without opening test data."""
    plan = load_staged_search_plan(plan_path)
    tasks = tuple(
        StagedSearchTask.model_validate_json(line)
        for line in task_plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = build_staged_search_tasks(plan)
    if tasks != expected:
        raise ValueError("staged-search task plan differs from frozen plan")

    rows: list[dict[str, object]] = []
    for task in tasks:
        run_name = f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        run_root = search_root / "runs" / run_name
        summary_path = run_root / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        rounds = sorted((run_root / "checkpoints").glob("round_*.json"))
        round_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in rounds
        ]
        staged = [
            item["staged_proposal"]
            for item in round_payloads
            if isinstance(item.get("staged_proposal"), dict)
        ]
        calls = [
            call
            for item in staged
            for call in (item.get("topology_call"), item.get("functional_call"))
            if isinstance(call, dict)
        ]
        target_evaluations = [
            item["public_target_evaluation"]
            for item in round_payloads
            if isinstance(item.get("public_target_evaluation"), dict)
        ]
        mechanism_evaluations = [
            item["public_mechanism_evaluation"]
            for item in round_payloads
            if isinstance(item.get("public_mechanism_evaluation"), dict)
        ]
        fits = [
            item["fit"]
            for item in round_payloads
            if isinstance(item.get("fit"), dict)
        ]
        successful_fits = [item for item in fits if item.get("success") is True]
        selected_candidate = (
            summary.get("selected_candidate")
            if isinstance(summary, dict)
            and isinstance(summary.get("selected_candidate"), dict)
            else None
        )
        process_time_path = run_root / "search_process_time.json"
        process_time, process_time_status = _read_process_time(
            process_time_path
        )
        failure_classes = Counter(
            str(item["failure_class"])
            for item in round_payloads
            if item.get("failure_class") is not None
        )
        rows.append(
            {
                "task_index": task.task_index,
                "benchmark_id": task.benchmark_id,
                "tier": task.tier,
                "repetition": task.repetition,
                "complete": summary is not None,
                "valid_round_count": sum(
                    bool(item.get("valid")) for item in round_payloads
                ),
                "topology_revision_count": sum(
                    item.get("revision_decision")
                    in {
                        "initial_topology_and_functions",
                        "topology_and_function_revision",
                    }
                    for item in staged
                ),
                "function_only_revision_count": sum(
                    item.get("revision_decision") == "function_only_revision"
                    for item in staged
                ),
                "proposer_logical_calls": sum(
                    int(item.get("logical_calls", 0)) for item in calls
                ),
                "proposer_provider_attempts": sum(
                    int(item.get("provider_attempts", 0)) for item in calls
                ),
                "proposer_input_tokens": sum(
                    int(item.get("input_tokens") or 0) for item in calls
                ),
                "proposer_output_tokens": sum(
                    int(item.get("output_tokens") or 0) for item in calls
                ),
                "public_target_evaluation_count": len(target_evaluations),
                "public_target_pass_count": sum(
                    item.get("passed") is True for item in target_evaluations
                ),
                "mean_graph_mechanism_compliance": _mean_field(
                    mechanism_evaluations,
                    "graph_mechanism_compliance",
                ),
                "mean_mechanism_annotation_compliance": _mean_field(
                    mechanism_evaluations,
                    "mechanism_annotation_compliance",
                ),
                "fit_attempted_round_count": len(fits),
                "fit_successful_round_count": len(successful_fits),
                "fit_retry_activation_count": sum(
                    len(item.get("fit_attempts", ())) > 1
                    for item in round_payloads
                    if isinstance(item.get("fit_attempts"), list)
                ),
                "median_successful_training_normalized_mse": _median_fit_metric(
                    successful_fits,
                    "training_metrics",
                ),
                "median_successful_validation_normalized_mse": _median_fit_metric(
                    successful_fits,
                    "validation_metrics",
                ),
                "selected_state_count": _sequence_length(
                    selected_candidate,
                    "states",
                ),
                "selected_latent_state_count": _latent_state_count(
                    selected_candidate
                ),
                "selected_process_count": _sequence_length(
                    selected_candidate,
                    "processes",
                ),
                "selected_parameter_count": _sequence_length(
                    selected_candidate,
                    "parameters",
                ),
                "failure_class_counts": dict(sorted(failure_classes.items())),
                "process_time_status": process_time_status,
                "process_wall_seconds": _numeric_field(
                    process_time,
                    "elapsed_seconds",
                ),
                "process_cpu_seconds": _process_cpu_seconds(process_time),
                "process_max_rss_kib": _numeric_field(
                    process_time,
                    "max_rss_kib",
                ),
                "validation_normalized_mse": (
                    None
                    if summary is None
                    else summary.get("selection_validation_normalized_mse")
                ),
            }
        )

    completed_nmse = [
        float(item["validation_normalized_mse"])
        for item in rows
        if item["validation_normalized_mse"] is not None
    ]
    target_evaluation_count = sum(
        int(item["public_target_evaluation_count"]) for item in rows
    )
    target_pass_count = sum(int(item["public_target_pass_count"]) for item in rows)
    graph_scores = _present_float_values(
        rows,
        "mean_graph_mechanism_compliance",
    )
    annotation_scores = _present_float_values(
        rows,
        "mean_mechanism_annotation_compliance",
    )
    report = {
        "schema_version": "phase-b-staged-search-pilot-summary-1",
        "status": (
            "complete" if all(item["complete"] for item in rows) else "incomplete"
        ),
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "planned_task_count": len(tasks),
        "completed_task_count": sum(bool(item["complete"]) for item in rows),
        "total_valid_round_count": sum(
            int(item["valid_round_count"]) for item in rows
        ),
        "total_topology_revision_count": sum(
            int(item["topology_revision_count"]) for item in rows
        ),
        "total_function_only_revision_count": sum(
            int(item["function_only_revision_count"]) for item in rows
        ),
        "total_proposer_logical_calls": sum(
            int(item["proposer_logical_calls"]) for item in rows
        ),
        "total_proposer_provider_attempts": sum(
            int(item["proposer_provider_attempts"]) for item in rows
        ),
        "total_proposer_input_tokens": sum(
            int(item["proposer_input_tokens"]) for item in rows
        ),
        "total_proposer_output_tokens": sum(
            int(item["proposer_output_tokens"]) for item in rows
        ),
        "public_target_evaluation_count": target_evaluation_count,
        "public_target_pass_rate": (
            target_pass_count / target_evaluation_count
            if target_evaluation_count
            else None
        ),
        "mean_graph_mechanism_compliance": (
            statistics.fmean(graph_scores) if graph_scores else None
        ),
        "mean_mechanism_annotation_compliance": (
            statistics.fmean(annotation_scores) if annotation_scores else None
        ),
        "total_fit_attempted_round_count": sum(
            int(item["fit_attempted_round_count"]) for item in rows
        ),
        "total_fit_successful_round_count": sum(
            int(item["fit_successful_round_count"]) for item in rows
        ),
        "total_fit_retry_activation_count": sum(
            int(item["fit_retry_activation_count"]) for item in rows
        ),
        "total_process_wall_seconds": sum(
            float(item["process_wall_seconds"] or 0.0) for item in rows
        ),
        "total_process_cpu_seconds": sum(
            float(item["process_cpu_seconds"] or 0.0) for item in rows
        ),
        "process_time_status_counts": dict(
            sorted(
                Counter(str(item["process_time_status"]) for item in rows).items()
            )
        ),
        "median_validation_normalized_mse": (
            statistics.median(completed_nmse) if completed_nmse else None
        ),
        "rows": rows,
    }
    return report


def _read_process_time(path: Path) -> tuple[dict[str, object] | None, str]:
    """Read current JSON or legacy key-value timing without aborting a summary."""
    if not path.is_file():
        return None, "missing"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None, "empty"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _parse_legacy_process_time(text)
        if payload is None:
            return None, "invalid"
        return payload, "legacy_key_value"
    if not isinstance(payload, dict):
        return None, "invalid"
    return payload, "json"


def _parse_legacy_process_time(text: str) -> dict[str, object] | None:
    """Parse the original atomic ``key=value`` resource-timing artifact."""
    payload: dict[str, object] = {}
    for line in text.splitlines():
        if "=" not in line:
            return None
        key, raw_value = line.split("=", 1)
        if not key or not raw_value:
            return None
        if key in {"elapsed_seconds", "user_cpu_seconds", "system_cpu_seconds"}:
            try:
                payload[key] = float(raw_value)
            except ValueError:
                return None
        elif key in {"max_rss_kib", "exit_code"}:
            try:
                payload[key] = int(raw_value)
            except ValueError:
                return None
        else:
            payload[key] = raw_value
    return payload or None


def _mean_field(payloads: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(item[field])
        for item in payloads
        if isinstance(item.get(field), (int, float))
    ]
    return statistics.fmean(values) if values else None


def _median_fit_metric(
    fits: list[dict[str, object]],
    metrics_field: str,
) -> float | None:
    values = [
        float(metrics["normalized_mse"])
        for item in fits
        if isinstance((metrics := item.get(metrics_field)), Mapping)
        and isinstance(metrics.get("normalized_mse"), (int, float))
    ]
    return statistics.median(values) if values else None


def _sequence_length(payload: object, field: str) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field)
    return len(value) if isinstance(value, list) else None


def _latent_state_count(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    states = payload.get("states")
    if not isinstance(states, list):
        return None
    return sum(
        isinstance(item, Mapping) and item.get("kind") == "latent"
        for item in states
    )


def _numeric_field(payload: object, field: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _process_cpu_seconds(payload: object) -> float | None:
    user = _numeric_field(payload, "user_cpu_seconds")
    system = _numeric_field(payload, "system_cpu_seconds")
    if user is None or system is None:
        return None
    return user + system


def _present_float_values(
    rows: list[dict[str, object]],
    field: str,
) -> list[float]:
    return [
        float(item[field])
        for item in rows
        if isinstance(item.get(field), (int, float))
    ]


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = _sha(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: path={path}, expected={expected}, "
            f"actual={actual}"
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
