"""Tests for role-specific reasoning and frozen fit-recovery contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.execution import (
    ExecutionArguments,
    _judge_reasoning_effort,
    _proposer_reasoning_effort,
)
from autoformalism.fitting import (
    EvaluationMetrics,
    FitConfig,
    FitResult,
    OptimizationDiagnostic,
)
from autoformalism.llm import VLLMReasoningEffort
from autoformalism.rebuttal.search_fit_recovery import (
    SearchFitRecoveryPlan,
    load_search_fit_recovery_plan,
    recovery_task_count,
    verify_source_plan,
)
from autoformalism.rebuttal.search_integration_ablation import (
    SearchIntegrationAblationPlan,
    SearchModelContract,
    load_search_integration_plan,
)
from autoformalism.search.controller import _fit_with_retry

RECOVERY_CONFIG = Path("configs/phase_b_search_fit_recovery_v1.json")
V3_CONFIG = Path("configs/phase_b_search_integration_ablation_v3.json")
SEARCH_SLURM = Path(
    "scripts/hpc/phase_b_search_integration_ablation_v2_120b.slurm"
)
V3_SEARCH_SLURM = Path(
    "scripts/hpc/phase_b_search_integration_ablation_v3_120b.slurm"
)
V3_EVALUATION_SLURM = Path(
    "scripts/hpc/phase_b_search_ablation_evaluation_v3.slurm"
)
V3_SUBMIT = Path("scripts/hpc/submit_phase_b_search_integration_ablation_v3.sh")
RECOVERY_SCRIPTS = (
    Path("scripts/hpc/phase_b_search_fit_recovery_v1.slurm"),
    Path("scripts/hpc/phase_b_search_fit_recovery_summary_v1.slurm"),
    Path("scripts/hpc/submit_phase_b_search_fit_recovery_v1.sh"),
)


def _arguments(tmp_path: Path) -> ExecutionArguments:
    return ExecutionArguments(
        data_root=tmp_path,
        benchmark_id="synthetic",
        tier="easy",
        seed=0,
        proposer_model="vllm:openai/gpt-oss-120b",
        judge_model="vllm:openai/gpt-oss-120b",
        iteration_budget=1,
        beam_size=1,
        output_root=tmp_path,
        resume=False,
        dry_run=True,
        mock_llm=False,
        use_clean_observations=False,
        vllm_proposer_reasoning_effort=VLLMReasoningEffort.HIGH,
        vllm_judge_reasoning_effort=VLLMReasoningEffort.LOW,
    )


def test_role_specific_reasoning_overrides_legacy_shared_value(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    assert _proposer_reasoning_effort(arguments) is VLLMReasoningEffort.HIGH
    assert _judge_reasoning_effort(arguments) is VLLMReasoningEffort.LOW


def test_search_model_contract_accepts_split_and_rejects_mixed_effort() -> None:
    common = {
        "proposer_model": "vllm:openai/gpt-oss-120b",
        "judge_model": "vllm:openai/gpt-oss-120b",
        "temperature": 0.2,
        "proposer_max_output_tokens": 4096,
        "request_timeout_seconds": 900,
    }
    contract = SearchModelContract.model_validate(
        {
            **common,
            "proposer_reasoning_effort": "high",
            "judge_reasoning_effort": "low",
        }
    )
    assert contract.effective_proposer_reasoning_effort == "high"
    assert contract.effective_judge_reasoning_effort == "low"
    with pytest.raises(ValidationError, match="cannot be mixed"):
        SearchModelContract.model_validate(
            {
                **common,
                "reasoning_effort": "low",
                "proposer_reasoning_effort": "high",
                "judge_reasoning_effort": "low",
            }
        )


def test_fit_recovery_plan_is_development_only_and_source_bound(
    tmp_path: Path,
) -> None:
    plan = load_search_fit_recovery_plan(RECOVERY_CONFIG)
    assert recovery_task_count(plan) == 3
    assert plan.new_llm_calls is False
    assert plan.test_data_opened is False
    assert [item.profile_id for item in plan.profiles_for("screening")] == [
        "screen_50x1_300s",
        "screen_150x2_600s",
    ]

    source = tmp_path / "frozen" / "plan.json"
    source.parent.mkdir()
    source.write_text("frozen\n", encoding="utf-8")
    payload = plan.model_dump(mode="json")
    payload["source_plan_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    rebound = SearchFitRecoveryPlan.model_validate(payload)
    assert verify_source_plan(rebound, tmp_path) == source
    source.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify_source_plan(rebound, tmp_path)


def test_delta_launchers_are_cpu_only_and_reasoning_is_role_specific() -> None:
    for path in (*RECOVERY_SCRIPTS, SEARCH_SLURM):
        subprocess.run(["bash", "-n", str(path)], check=True)
    for path in RECOVERY_SCRIPTS[:2]:
        text = path.read_text(encoding="utf-8")
        assert "bibo-delta-cpu" in text
        assert "--partition=cpu" in text
        assert "gpu" not in text.lower()

    search = SEARCH_SLURM.read_text(encoding="utf-8")
    assert "--vllm-proposer-reasoning-effort" in search
    assert "--vllm-judge-reasoning-effort" in search
    assert "// .model_contract.reasoning_effort" in search


def test_recovery_config_is_stable_json() -> None:
    payload = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    plan = load_search_fit_recovery_plan(RECOVERY_CONFIG)
    assert payload == plan.model_dump(mode="json")


def _fit_result(*, success: bool, status: int) -> FitResult:
    metrics = EvaluationMetrics(1.0, {"target": 1.0})
    return FitResult(
        success=success,
        global_parameters={},
        global_initial_conditions={},
        training_trajectory_initial_conditions={},
        validation_trajectory_initial_conditions={},
        training_metrics=metrics,
        validation_metrics=metrics,
        diagnostics=(
            OptimizationDiagnostic(
                start_index=0,
                success=success,
                status=status,
                message="fixture",
                cost=1.0,
                function_evaluations=1,
                integration_failures=0,
            ),
        ),
        best_start_index=0,
        target_scales={"target": 1.0},
        message=None if success else "fixture failure",
    )


def test_fit_retry_reuses_the_same_problem_and_records_both_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("autoformalism.search.controller")
    primary = FitConfig(
        integration_backend="fixed_rk4",
        number_of_starts=1,
        maximum_function_evaluations=50,
        maximum_wall_time_seconds=300,
    )
    retry = primary.model_copy(
        update={
            "number_of_starts": 2,
            "maximum_function_evaluations": 150,
            "maximum_wall_time_seconds": 600,
        }
    )
    answers = iter(
        (_fit_result(success=False, status=-2), _fit_result(success=True, status=1))
    )
    observed: list[FitConfig] = []

    def fake_fit(*args, **kwargs):
        observed.append(args[3])
        assert kwargs["initial_global_parameters"] == {"p": 1.0}
        return next(answers)

    monkeypatch.setattr(module, "fit_candidate", fake_fit)
    result, attempts = _fit_with_retry(
        object(),
        object(),
        object(),
        primary_config=primary,
        retry_config=retry,
        initial_global_parameters={"p": 1.0},
    )

    assert result.success is True
    assert observed == [primary, retry]
    assert [item["success"] for item in attempts] == [False, True]
    assert attempts[1]["fit_config"]["maximum_function_evaluations"] == 150


def test_v3_plan_requires_high_proposer_low_judge_and_fit_retry() -> None:
    plan = load_search_integration_plan(V3_CONFIG)

    assert plan.schema_version.endswith("-3")
    assert plan.model_contract.effective_proposer_reasoning_effort == "high"
    assert plan.model_contract.effective_judge_reasoning_effort == "low"
    assert plan.search_budget.fit_retry_starts == 2
    assert plan.search_budget.fit_retry_max_nfev == 150
    assert plan.search_budget.fit_retry_timeout_seconds == 600.0

    incomplete = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    for key in (
        "fit_retry_starts",
        "fit_retry_max_nfev",
        "fit_retry_timeout_seconds",
    ):
        incomplete["search_budget"].pop(key)
    with pytest.raises(ValidationError, match="requires a deterministic"):
        SearchIntegrationAblationPlan.model_validate(incomplete)


def test_v3_launchers_delegate_to_frozen_plan_and_arm_endpoint_report() -> None:
    for path in (V3_SEARCH_SLURM, V3_EVALUATION_SLURM, V3_SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert "phase_b_search_integration_ablation_v3.json" in (
        V3_SEARCH_SLURM.read_text(encoding="utf-8")
    )
    assert "AF_WRITE_ARM_ENDPOINT_REPORT=true" in (
        V3_EVALUATION_SLURM.read_text(encoding="utf-8")
    )
    assert "phase_b_search_integration_ablation_v3_120b.slurm" in (
        V3_SUBMIT.read_text(encoding="utf-8")
    )
