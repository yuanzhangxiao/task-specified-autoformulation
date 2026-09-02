"""Tests for the matched feedback-rich refinement pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.rebuttal.proposer_refinement_pilot import (
    build_refinement_pilot_tasks,
    freeze_refinement_pilot,
    load_refinement_pilot_plan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_refinement_plan_is_matched_and_public_only() -> None:
    plan = load_refinement_pilot_plan(
        Path("configs/phase_b_proposer_refinement_pilot_v1.json")
    )
    tasks = build_refinement_pilot_tasks(plan)

    assert len(tasks) == 12
    assert [item.arm_id for item in tasks[:6]] == ["rich_exploratory"] * 6
    assert [item.arm_id for item in tasks[6:]] == [
        "rich_incumbent_refinement"
    ] * 6
    assert all(item.proposer_feedback_mode == "rich_v1" for item in tasks)
    assert plan.search_budget.parameter_fit_strategy == "bounded_nonlinear"
    assert plan.test_data_opened is False
    assert plan.private_reference_opened is False


def test_repository_refinement_smoke_is_one_matched_two_round_trial() -> None:
    plan = load_refinement_pilot_plan(
        Path("configs/phase_b_proposer_refinement_smoke_v1.json")
    )
    tasks = build_refinement_pilot_tasks(plan)

    assert len(tasks) == 2
    assert [item.arm_id for item in tasks] == [
        "rich_exploratory",
        "rich_incumbent_refinement",
    ]
    assert [item.task_index for item in tasks] == [0, 1]
    assert plan.repetitions == (0,)
    assert plan.search_budget.iteration_budget == 2
    assert plan.development_only is True
    assert plan.test_data_opened is False
    assert plan.private_reference_opened is False


def test_matched_arm_tasks_share_worker_endpoint_slots() -> None:
    plan = load_refinement_pilot_plan(
        Path("configs/phase_b_proposer_refinement_pilot_v1.json")
    )
    tasks = build_refinement_pilot_tasks(plan)
    matched_trial_count = len(tasks) // 2

    for exploratory, refinement in zip(
        tasks[:matched_trial_count],
        tasks[matched_trial_count:],
        strict=True,
    ):
        assert (
            exploratory.benchmark_id,
            exploratory.tier,
            exploratory.repetition,
        ) == (
            refinement.benchmark_id,
            refinement.tier,
            refinement.repetition,
        )
        assert (
            exploratory.task_index % matched_trial_count
            == refinement.task_index % matched_trial_count
        )


def test_freeze_validates_public_inputs_and_writes_hash_ledger(
    tmp_path: Path,
) -> None:
    data = tmp_path / "public" / "phase_b_v1" / "cell"
    target_root = tmp_path / "targets" / "specs"
    mechanism_root = tmp_path / "mechanisms" / "specs"
    data.mkdir(parents=True)
    target_root.mkdir(parents=True)
    mechanism_root.mkdir(parents=True)
    prompt = data / "proposer_prompt.txt"
    target = target_root / "cell.json"
    mechanism = mechanism_root / "cell.json"
    judge = tmp_path / "judge.json"
    for path, content in (
        (prompt, "public task\n"),
        (target, "{}\n"),
        (mechanism, "{}\n"),
        (judge, "{}\n"),
    ):
        path.write_text(content, encoding="utf-8")
    source = json.loads(
        Path("configs/phase_b_proposer_refinement_pilot_v1.json").read_text()
    )
    source["cells"] = [
        {
            "benchmark_id": "cell",
            "tier": "easy",
            "public_prompt_sha256": _sha(prompt),
            "public_target_contract_sha256": _sha(target),
            "public_mechanism_spec_sha256": _sha(mechanism),
        }
    ]
    source["repetitions"] = [0]
    source["model_contract"]["judge_protocol_config_sha256"] = _sha(judge)
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(source), encoding="utf-8")

    manifest = freeze_refinement_pilot(
        config,
        tmp_path / "frozen",
        public_data_root=tmp_path / "public",
        target_contract_root=tmp_path / "targets",
        mechanism_spec_root=tmp_path / "mechanisms",
        judge_protocol_path=judge,
    )

    assert manifest["task_count"] == 2
    assert manifest["matched_trial_count"] == 1
    assert manifest["test_data_opened"] is False
    assert (tmp_path / "frozen" / "task_plan.jsonl.sha256").is_file()


def test_freeze_rejects_prompt_drift(tmp_path: Path) -> None:
    plan = Path("configs/phase_b_proposer_refinement_pilot_v1.json")
    with pytest.raises(ValueError, match="missing judge protocol"):
        freeze_refinement_pilot(
            plan,
            tmp_path / "frozen",
            public_data_root=tmp_path,
            target_contract_root=tmp_path,
            mechanism_spec_root=tmp_path,
            judge_protocol_path=tmp_path / "missing.json",
        )


def test_aces_worker_uses_portable_resource_timing() -> None:
    worker = Path(
        "scripts/hpc/phase_b_proposer_refinement_pilot_aces_h100.slurm"
    )
    subprocess.run(["bash", "-n", str(worker)], check=True)
    text = worker.read_text(encoding="utf-8")
    assert "scripts/run_with_resource_timing.py" in text
    assert "/usr/bin/time" not in text


def test_aces_worker_uses_matched_endpoint_identity_across_arms() -> None:
    worker = Path(
        "scripts/hpc/phase_b_proposer_refinement_pilot_aces_h100.slurm"
    )
    text = worker.read_text(encoding="utf-8")

    assert 'matched_trial_count=$((task_count / 2))' in text
    assert 'matched_trial_index=$((AF_TASK_INDEX % matched_trial_count))' in text
    assert 'endpoint_port="$((8100 + matched_trial_index))"' in text
    assert 'endpoint_port="$((8100 + AF_TASK_INDEX))"' not in text


def test_aces_submitter_supports_consistent_single_gpu_smoke() -> None:
    submitter = Path(
        "scripts/hpc/submit_phase_b_proposer_refinement_pilot_aces.sh"
    )
    subprocess.run(["bash", "-n", str(submitter)], check=True)
    text = submitter.read_text(encoding="utf-8")

    assert 'AF_ACES_GRES:=gpu:h100:2' in text
    assert 'AF_TENSOR_PARALLEL_SIZE:=2' in text
    assert '--gres="${AF_ACES_GRES}"' in text
    assert 'AF_TENSOR_PARALLEL_SIZE=${AF_TENSOR_PARALLEL_SIZE}' in text
    assert '${AF_REFINEMENT_DEPENDENCY}:${exploratory_job}' in text
