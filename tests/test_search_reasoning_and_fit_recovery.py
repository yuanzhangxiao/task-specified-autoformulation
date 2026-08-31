"""Tests for role-specific reasoning and frozen fit-recovery contracts."""

from __future__ import annotations

import hashlib
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
from autoformalism.llm import VLLMReasoningEffort
from autoformalism.rebuttal.search_fit_recovery import (
    SearchFitRecoveryPlan,
    load_search_fit_recovery_plan,
    recovery_task_count,
    verify_source_plan,
)
from autoformalism.rebuttal.search_integration_ablation import SearchModelContract

RECOVERY_CONFIG = Path("configs/phase_b_search_fit_recovery_v1.json")
SEARCH_SLURM = Path(
    "scripts/hpc/phase_b_search_integration_ablation_v2_120b.slurm"
)
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
