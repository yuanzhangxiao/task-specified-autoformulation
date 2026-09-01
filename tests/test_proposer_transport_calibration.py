"""Tests for the frozen GPT-OSS proposer output-budget calibration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.execution import ExecutionArguments, proposer_system_prompt
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.proposer_transport_calibration import (
    ProposerCalibrationResult,
    ProposerTransportCalibrationPlan,
    analyze_proposer_calibration,
    build_proposer_calibration_tasks,
    freeze_proposer_calibration,
    load_proposer_calibration_plan,
)
from scripts.run_phase_b_proposer_transport_calibration import (
    _attempt_accounting,
)

CONFIG = Path("configs/phase_b_proposer_transport_calibration_v1.json")
GPU_JOB = Path(
    "scripts/hpc/phase_b_proposer_transport_calibration_120b.slurm"
)
ANALYSIS_JOB = Path(
    "scripts/hpc/phase_b_proposer_transport_calibration_analysis.slurm"
)
SUBMIT = Path("scripts/hpc/submit_phase_b_proposer_transport_calibration.sh")


def test_calibration_plan_is_matched_and_budget_ordered() -> None:
    plan = load_proposer_calibration_plan(CONFIG)
    tasks = build_proposer_calibration_tasks(plan)

    assert len(tasks) == 6
    assert plan.model_contract.reasoning_effort == "high"
    assert plan.model_contract.max_output_token_budgets == (4096, 8192, 12288)
    assert plan.scientific_judge_called is False
    assert plan.parameter_fitting_performed is False
    assert plan.test_data_opened is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["model_contract"]["max_output_token_budgets"] = [4096, 4096]
    with pytest.raises(ValidationError, match="unique and increasing"):
        ProposerTransportCalibrationPlan.model_validate(payload)


def test_freeze_validates_prompt_and_target_contract(tmp_path: Path) -> None:
    prompt = "Public calibration prompt.\n"
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    public = tmp_path / "public"
    benchmark = "fixture_benchmark"
    prompt_path = public / "phase_b_v1" / benchmark / "proposer_prompt.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    contract_root = tmp_path / "contracts"
    contract_path = contract_root / "specs" / f"{benchmark}.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "public-target-contract-1",
                "benchmark_id": benchmark,
                "tier": "easy",
                "public_prompt_sha256": prompt_sha,
                "source": "public_prompt",
                "targets": [
                    {
                        "target_channel": "y",
                        "public_requirement": "generate y",
                        "required_dependencies": [],
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["cells"] = [
        {
            "benchmark_id": benchmark,
            "tier": "easy",
            "public_prompt_sha256": prompt_sha,
            "public_target_contract_sha256": hashlib.sha256(
                contract_path.read_bytes()
            ).hexdigest(),
        }
    ]
    payload["repetitions"] = [0]
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest = freeze_proposer_calibration(
        config,
        tmp_path / "frozen",
        public_data_root=public,
        target_contract_root=contract_root,
    )
    assert manifest["matched_request_count"] == 1
    assert manifest["planned_result_count"] == 3
    assert manifest["test_data_opened"] is False
    assert freeze_proposer_calibration(
        config,
        tmp_path / "frozen",
        public_data_root=public,
        target_contract_root=contract_root,
    ) == manifest

    prompt_path.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt differs"):
        freeze_proposer_calibration(
            config,
            tmp_path / "other",
            public_data_root=public,
            target_contract_root=contract_root,
        )


def _result(
    *,
    task_index: int,
    budget: int,
    response: bool,
    valid: bool,
    target: bool,
    length: int = 0,
) -> ProposerCalibrationResult:
    return ProposerCalibrationResult(
        plan_sha256="0" * 64,
        task_index=task_index,
        benchmark_id=(
            "phase_b_dalla_man_t2_canonical_named_easy"
            if task_index < 3
            else "phase_b_anonymous_system_task_canonical_opaque_hard"
        ),
        tier="easy" if task_index < 3 else "hard",
        repetition=task_index % 3,
        model="openai/gpt-oss-120b",
        reasoning_effort="high",
        max_output_tokens=budget,
        response_success=response,
        first_attempt_response_success=response,
        provider_attempt_count=1 if response else 3,
        provider_input_tokens=1000,
        provider_output_tokens=(budget // 2 if response else budget * 3),
        provider_total_tokens=(1000 + budget // 2 if response else 3000 + budget * 3),
        latency_ms=1000.0,
        length_exhausted_attempt_count=length,
        reasoning_character_count=100,
        deterministic_valid=valid,
        public_target_passed=target,
        test_data_opened=False,
        scientific_judge_called=False,
        parameter_fitting_performed=False,
    )


def test_analysis_selects_smallest_budget_passing_all_gates() -> None:
    plan = load_proposer_calibration_plan(CONFIG)
    results: list[ProposerCalibrationResult] = []
    for task in build_proposer_calibration_tasks(plan):
        results.append(
            _result(
                task_index=task.task_index,
                budget=4096,
                response=False,
                valid=False,
                target=False,
                length=3,
            )
        )
        acceptable = task.task_index != 5
        results.append(
            _result(
                task_index=task.task_index,
                budget=8192,
                response=True,
                valid=acceptable,
                target=acceptable,
            )
        )
        results.append(
            _result(
                task_index=task.task_index,
                budget=12288,
                response=True,
                valid=True,
                target=True,
            )
        )

    analysis = analyze_proposer_calibration(plan, tuple(results))
    assert analysis["status"] == "pass"
    assert analysis["selected_max_output_tokens"] == 8192
    assert analysis["operating_points"][0]["passed"] is False
    assert analysis["operating_points"][1]["passed"] is True


def test_attempt_accounting_detects_length_and_reasoning() -> None:
    raw = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "abc"},
            }
        ],
    }
    accounting = _attempt_accounting(
        [
            {
                "event": "llm_failure",
                "request_hash": "request",
                "attempt": 1,
                "raw_response": raw,
            }
        ],
        request_hash="request",
    )
    assert accounting == {
        "attempt_count": 1,
        "input_tokens": 10,
        "output_tokens": 4096,
        "total_tokens": 4106,
        "length_exhausted": 1,
        "reasoning_characters": 3,
    }


def test_calibration_uses_the_production_proposer_prompt_builder(
    tmp_path: Path,
) -> None:
    arguments = ExecutionArguments(
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
        dry_run=False,
        mock_llm=False,
        use_clean_observations=False,
    )
    rendered = proposer_system_prompt(
        arguments,
        public_prompt="Generate y.",
        context=ValidationContext(targets=("y",)),
    )
    assert rendered.startswith(
        "Configured proposer model: vllm:openai/gpt-oss-120b"
    )
    assert "Generate y." in rendered
    assert "Controller requirements:" in rendered


def test_delta_launchers_are_valid_and_keep_calibration_sealed() -> None:
    for path in (GPU_JOB, ANALYSIS_JOB, SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "API_KEY" not in text
    gpu = GPU_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-node=4" in gpu
    assert "#SBATCH --time=02:00:00" in gpu
    assert "run_phase_b_proposer_transport_calibration.py" in gpu
    assert "--max-model-len" in gpu
    analysis = ANALYSIS_JOB.read_text(encoding="utf-8")
    assert "bibo-delta-cpu" in analysis
    assert "--partition=cpu" in analysis
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "--array=0-5%2" in submit
    assert '--dependency="afterany:${calibration_job_id}"' in submit
