"""Tests for the public-only baseline pilot and cluster routing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.rebuttal.baseline_pilot import (
    BaselinePilotPlan,
    BaselinePilotTask,
    build_baseline_pilot_tasks,
    freeze_baseline_llm_operating_point,
    freeze_baseline_pilot,
    load_baseline_pilot_plan,
)

CONFIG = Path("configs/phase_b_public_baseline_pilot_v1.json")
CPU_JOB = Path("scripts/hpc/phase_b_public_baseline_pilot_aces_cpu.slurm")
CPU_SUBMIT = Path("scripts/hpc/submit_phase_b_public_baseline_pilot_aces_cpu.sh")
D3_JOB = Path("scripts/hpc/phase_b_public_baseline_pilot_aces_d3.slurm")
D3_SUBMIT = Path("scripts/hpc/submit_phase_b_public_baseline_pilot_aces_d3.sh")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_baseline_pilot_is_two_cells_three_seeds() -> None:
    plan = load_baseline_pilot_plan(CONFIG)
    tasks = build_baseline_pilot_tasks(plan)

    assert len(tasks) == 18
    assert [method.method for method in plan.methods] == [
        "sindy",
        "pysr",
        "d3_native_no_tools",
    ]
    assert {task.platform for task in tasks[:12]} == {"aces_cpu"}
    assert {task.platform for task in tasks[12:]} == {"aces_h100x2"}
    assert {task.maximum_llm_calls for task in tasks[12:]} == {5}
    assert plan.test_data_opened is False
    assert plan.weighted_overall_score_defined is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["methods"][2]["d3_generations"] = 6
    with pytest.raises(ValueError, match="inconsistent LLM/compute"):
        BaselinePilotPlan.model_validate(payload)


def test_freeze_validates_public_inputs_and_resource_ledger(tmp_path: Path) -> None:
    prompt = "Public baseline prompt.\n"
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    public = tmp_path / "public"
    benchmark = "fixture_baseline"
    prompt_path = public / "phase_b_v1" / benchmark / "proposer_prompt.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    contracts = tmp_path / "contracts"
    contract_path = contracts / "specs" / f"{benchmark}.json"
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
    (contracts / "manifest.json").write_text("{}\n", encoding="utf-8")
    overlay = tmp_path / "overlay.json"
    overlay.write_text("{}\n", encoding="utf-8")
    proposer = tmp_path / "proposer.json"
    proposer.write_text(
        Path("configs/phase_b_proposer_transport_calibration_v2.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["cells"] = [
        {
            "benchmark_id": benchmark,
            "tier": "easy",
            "public_prompt_sha256": prompt_sha,
            "public_target_contract_sha256": _sha(contract_path),
        }
    ]
    payload["repetitions"] = [0]
    payload["prompt_overlay_config_sha256"] = _sha(overlay)
    payload["target_contract_manifest_sha256"] = _sha(
        contracts / "manifest.json"
    )
    payload["proposer_transport_plan_sha256"] = _sha(proposer)
    config = tmp_path / "baseline_plan.json"
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest = freeze_baseline_pilot(
        config,
        tmp_path / "frozen",
        public_data_root=public,
        target_contract_root=contracts,
        prompt_overlay_config_path=overlay,
        proposer_transport_plan_path=proposer,
    )
    assert manifest["task_count"] == 3
    assert manifest["cpu_task_count"] == 2
    assert manifest["d3_task_count"] == 1
    resources = (
        tmp_path / "frozen" / "planned_resource_ledger.jsonl"
    ).read_text(encoding="utf-8")
    assert '"logical_llm_tokens": null' in resources
    assert '"test_data_opened"' not in resources

    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "schema_version": (
                    "phase-b-proposer-transport-calibration-analysis-1"
                ),
                "status": "pass",
                "selected_max_output_tokens": 24576,
                "selected_reasoning_effort": "high",
                "operating_points": [
                    {"max_output_tokens": 24576, "passed": True}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    operating_point = freeze_baseline_llm_operating_point(
        config,
        proposer,
        analysis,
        tmp_path / "frozen" / "d3_llm_operating_point.json",
    )
    assert operating_point["max_output_tokens"] == 24576
    assert operating_point["maximum_llm_calls_per_trial"] == 5


def test_aces_baseline_launchers_are_public_only() -> None:
    for path in (CPU_JOB, CPU_SUBMIT, D3_JOB, D3_SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "API_KEY" not in text
        assert "/private" not in text.lower()
        assert "--development-only" in text or "submit" in path.name

    assert "#SBATCH --partition=cpu" in CPU_JOB.read_text(encoding="utf-8")
    assert (
        'module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"'
        in CPU_JOB.read_text(encoding="utf-8")
    )
    assert "AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE}" in CPU_SUBMIT.read_text(
        encoding="utf-8"
    )
    assert "AF_PYTHON_MODULE=${AF_PYTHON_MODULE}" in CPU_SUBMIT.read_text(
        encoding="utf-8"
    )
    d3 = D3_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h100:2" in d3
    assert 'module load "${AF_GCCCORE_MODULE}" "${AF_PYTHON_MODULE}"' in d3
    assert "d3_llm_operating_point.json" in d3
    submit = D3_SUBMIT.read_text(encoding="utf-8")
    assert "AF_GCCCORE_MODULE=${AF_GCCCORE_MODULE}" in submit
    assert "AF_PYTHON_MODULE=${AF_PYTHON_MODULE}" in submit
    assert "freeze_phase_b_baseline_llm_operating_point.py" in submit
    assert 'dependency="afterok:${image_job_id}"' in submit


def test_public_baseline_summary_and_handoff_exclude_test(tmp_path: Path) -> None:
    task = BaselinePilotTask(
        task_index=0,
        method="sindy",
        comparison_role="classical_partial_observability_control",
        platform="aces_cpu",
        benchmark_id="fixture",
        tier="easy",
        repetition=0,
        cpus_per_task=16,
        gpu_type="none",
        gpu_count=0,
        wall_timeout_seconds=60.0,
        maximum_llm_calls=0,
    )
    task_plan = tmp_path / "task_plan.jsonl"
    task_plan.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    run = tmp_path / "runs" / "sindy" / "fixture_easy_seed0"
    run.mkdir(parents=True)
    result = BaselineDevelopmentResult(
        method="sindy",
        benchmark_id="fixture",
        tier="easy",
        seed=0,
        equations={"y": "-0.5 * y"},
        selected_hyperparameters={"threshold": 0.01},
        training_normalized_mse=0.1,
        validation_normalized_mse=0.2,
        elapsed_wall_seconds=30.0,
        wall_timeout_seconds=60.0,
    )
    (run / "result.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (run / "run_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_wall_seconds": 30.0,
                "wall_timeout_seconds": 60.0,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_output = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_phase_b_public_baseline_pilot.py",
            "--task-plan",
            str(task_plan),
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(summary_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(
        (summary_output / "baseline_development_summary.json").read_text()
    )
    assert summary["status"] == "complete"
    assert summary["groups"][0]["cpu_core_hours"] == pytest.approx(2 / 15)

    handoff = tmp_path / "handoff"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_phase_b_public_baseline_handoff.py",
            "--task-plan",
            str(task_plan),
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(handoff),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((handoff / "handoff_manifest.json").read_text())
    assert manifest["selection_count"] == 1
    assert manifest["test_data_included"] is False
    copied = json.loads((handoff / "tasks" / "task_000.json").read_text())
    assert copied["test_data_opened"] is False
    assert "test_normalized_mse" not in copied
