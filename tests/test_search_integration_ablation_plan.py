"""Tests for the frozen paired-judge/no-judge search plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.rebuttal.search_integration_ablation import (
    SearchIntegrationAblationPlan,
    build_search_integration_tasks,
    freeze_search_ablation_sources,
    freeze_search_integration_plan,
    load_search_integration_plan,
)
from scripts.summarize_phase_b_search_integration_ablation import (
    collect_search_audit,
)

CONFIG = Path("configs/phase_b_search_integration_ablation_v1.json")
SLURM = Path("scripts/hpc/phase_b_search_integration_ablation_120b.slurm")
EVALUATION_SLURM = Path(
    "scripts/hpc/phase_b_search_ablation_evaluation_v1.slurm"
)
SUBMIT_SCRIPT = Path(
    "scripts/hpc/submit_phase_b_search_integration_ablation.sh"
)


def test_search_integration_plan_is_exact_matched_matrix() -> None:
    plan = load_search_integration_plan(CONFIG)
    tasks = build_search_integration_tasks(plan)

    assert len(tasks) == 12
    assert {item.arm_id for item in tasks} == {
        "paired_question_consensus",
        "no_judge",
    }
    assert sum(item.use_judge for item in tasks) == 6
    matched = {
        (item.benchmark_id, item.tier, item.repetition) for item in tasks
    }
    assert len(matched) == 6
    assert all(item.task_index == index for index, item in enumerate(tasks))
    judge = next(item for item in tasks if item.use_judge)
    no_judge = next(item for item in tasks if not item.use_judge)
    assert judge.selection_policy == "incumbent_relative_hybrid"
    assert judge.hybrid_science_weight == 0.5
    assert no_judge.selection_policy == "validation_only"
    assert no_judge.hybrid_science_weight is None


def test_search_integration_freeze_is_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "frozen"
    first = freeze_search_integration_plan(CONFIG, frozen)
    second = freeze_search_integration_plan(CONFIG, frozen)

    assert first == second
    assert first["task_count"] == 12
    assert first["matched_trial_count"] == 6
    assert first["test_data_opened"] is False
    assert len((frozen / "task_plan.jsonl").read_text().splitlines()) == 12

    (frozen / "task_plan.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        freeze_search_integration_plan(CONFIG, frozen)


def test_search_integration_plan_rejects_arm_label_policy_mismatch() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["arms"][1]["use_judge"] = True

    with pytest.raises(ValueError, match="mismatched selection policy"):
        SearchIntegrationAblationPlan.model_validate(payload)


def test_search_ablation_source_freeze_labels_arms_and_retains_missing(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-hidden-subspace-contract-audit-2",
                "status": "pass",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_digest = hashlib.sha256(audit.read_bytes()).hexdigest()
    audit.with_name("audit.json.sha256").write_text(
        f"{audit_digest}  audit.json\n",
        encoding="utf-8",
    )
    plan_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_payload["hidden_contract_audit"]["sha256"] = audit_digest
    test_plan = tmp_path / "plan.json"
    test_plan.write_text(
        json.dumps(plan_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = load_search_integration_plan(test_plan)
    tasks = build_search_integration_tasks(plan)
    search_root = tmp_path / "search"
    first = tasks[0]
    summary = (
        search_root
        / "searches"
        / first.arm_id
        / "runs"
        / f"{first.benchmark_id}_{first.tier}_seed{first.repetition}"
        / "summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_stage": "development_selection_frozen",
                "benchmark_id": first.benchmark_id,
                "tier": first.tier,
                "seed": first.repetition,
                "selection_policy": first.selection_policy,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = freeze_search_ablation_sources(
        test_plan,
        search_root,
        audit,
        tmp_path / "sources",
    )
    requests = [
        json.loads(line)
        for line in (tmp_path / "sources/source_adapter_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["planned_source_count"] == 12
    assert manifest["available_source_count"] == 1
    assert manifest["missing_source_count"] == 11
    assert {item["method_label"] for item in requests} == {
        "autoformalism:paired_question_consensus",
        "autoformalism:no_judge",
    }
    assert manifest["test_data_opened"] is False


def test_search_integration_slurm_is_valid_and_seals_test_data() -> None:
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    source = SLURM.read_text(encoding="utf-8")

    for required in (
        "#SBATCH --array=0-11%2",
        "#SBATCH --account=bibo-delta-gpu",
        "#SBATCH --partition=gpuA40x4",
        "phase_b_search_integration_ablation_v1.json",
        "task_plan.jsonl",
        "--development-only",
        "--selection-policy",
        "--no-judge",
        "--hybrid-science-weight",
        "--llm-cache-root",
        "--resume",
        'vllm serve "${model}"',
    ):
        assert required in source
    assert "API_KEY" not in source
    assert "--evaluate-test" not in source


def test_search_ablation_evaluation_uses_common_postfreeze_pipeline() -> None:
    subprocess.run(["bash", "-n", str(EVALUATION_SLURM)], check=True)
    source = EVALUATION_SLURM.read_text(encoding="utf-8")

    for required in (
        "#SBATCH --account=bibo-delta-cpu",
        "#SBATCH --partition=cpu",
        "freeze_phase_b_search_ablation_sources.py",
        "summarize_phase_b_search_integration_ablation.py",
        "export_phase_b_frozen_subjects.py",
        "evaluate_phase_b_postfreeze.py",
        "evaluate_phase_b_hidden_subspace.py",
        "assemble_phase_b_final_evaluation.py",
        "--hidden-audit",
    ):
        assert required in source
    assert "API_KEY" not in source
    assert "judge" not in source.lower()


def test_submission_orders_judge_cache_population_before_no_judge() -> None:
    subprocess.run(["bash", "-n", str(SUBMIT_SCRIPT)], check=True)
    source = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "--array=0-5%2" in source
    assert "--array=6-11%2" in source
    assert '--dependency="afterany:${judge_job_id}"' in source
    assert '--dependency="afterany:${no_judge_job_id}"' in source
    assert "prepare_phase_b_search_integration_ablation.py" in source


def test_search_audit_verifies_initial_cache_reuse_and_no_judge_calls(
    tmp_path: Path,
) -> None:
    tasks = build_search_integration_tasks(load_search_integration_plan(CONFIG))
    judge = tasks[0]
    no_judge = next(
        item
        for item in tasks
        if item.arm_id == "no_judge"
        and (item.benchmark_id, item.tier, item.repetition)
        == (judge.benchmark_id, judge.tier, judge.repetition)
    )
    for task, cache_hit in ((judge, False), (no_judge, True)):
        run = (
            tmp_path
            / "searches"
            / task.arm_id
            / "runs"
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        (run / "checkpoints").mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "benchmark_id": task.benchmark_id,
                    "tier": task.tier,
                    "seed": task.repetition,
                    "selected_candidate": {"candidate_id": task.arm_id},
                    "selection_validation_normalized_mse": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "proposer_events.jsonl").write_text(
            json.dumps(
                {
                    "event": "llm_response",
                    "role": "proposer",
                    "request_hash": "same-initial-request",
                    "cache_hit": cache_hit,
                    "latency_ms": 100.0,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "task_runtime.json").write_text(
            json.dumps(
                {
                    "task_elapsed_wall_seconds": 60.0,
                    "search_process_elapsed_seconds": 50.0,
                    "allocated_cpus": 16,
                    "allocated_gpus": 4,
                    "allocated_cpu_core_hours": 16 / 60,
                    "allocated_gpu_hours": 4 / 60,
                    "exit_code": 0,
                    "monetary_cost_usd": None,
                    "monetary_cost_status": (
                        "not_priced_local_open_weight_model"
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    judge_run = (
        tmp_path
        / "searches"
        / judge.arm_id
        / "runs"
        / f"{judge.benchmark_id}_{judge.tier}_seed{judge.repetition}"
    )
    (judge_run / "hybrid_pair_events.jsonl").write_text(
        json.dumps(
            {
                "event": "llm_response",
                "role": "atomic_evidence_judge",
                "request_hash": "judge-stage",
                "cache_hit": False,
                "latency_ms": 200.0,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = collect_search_audit(CONFIG, tmp_path)

    assert report["source_completion_count"] == 2
    assert report["matched_source_completion_count"] == 1
    assert report["initial_request_comparable_count"] == 1
    assert report["judge_stage_response_count"] == 1
    resources = report["resource_accounting"]
    assert resources["runtime_record_count"] == 2
    assert resources["logical_response_count"] == 3
    assert resources["provider_attempt_event_count"] == 2
    assert resources["logical_total_tokens"] == 60
    assert resources["provider_total_tokens"] == 45
    assert resources["allocated_gpu_hours"] == pytest.approx(8 / 60)
