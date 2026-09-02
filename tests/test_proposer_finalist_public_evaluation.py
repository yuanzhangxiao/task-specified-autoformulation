"""Tests for the frozen repaired-proposer finalist public evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.rebuttal.proposer_finalist_evaluation import (
    ProposerFinalistEvaluationPlan,
    finalist_task_count,
    load_proposer_finalist_evaluation_plan,
    task_identity,
)
from autoformalism.schemas import CandidateModel
from scripts import run_phase_b_proposer_finalist_public_evaluation as runner
from scripts import summarize_phase_b_proposer_finalist_public_evaluation as summary

CONFIG = Path("configs/phase_b_proposer_finalist_public_evaluation_v1.json")
HPC_SCRIPTS = (
    Path("scripts/hpc/phase_b_proposer_finalist_public_evaluation_aces.slurm"),
    Path(
        "scripts/hpc/phase_b_proposer_finalist_public_evaluation_summary_aces.slurm"
    ),
    Path("scripts/hpc/submit_phase_b_proposer_finalist_public_evaluation_aces.sh"),
)


def _candidate(identifier: str = "candidate") -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "states": [
                {
                    "name": "target",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "target state",
                    "mechanisms": [],
                }
            ],
            "state_equations": [
                {"state": "target", "rhs": "-k * target + u"}
            ],
            "processes": [
                {
                    "name": "generated",
                    "expression": "target + u",
                    "unit": "unit",
                    "description": "generated output",
                    "mechanisms": [],
                }
            ],
            "observation_mappings": [
                {
                    "channel": "target",
                    "expression": "generated",
                    "unit": "unit",
                }
            ],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                    "unit": "1/time",
                    "description": "decay",
                }
            ],
            "initial_conditions": [
                {
                    "state": "target",
                    "scope": "global",
                    "expression": "target",
                }
            ],
        }
    )


def test_frozen_plan_has_paired_public_only_matrix() -> None:
    plan = load_proposer_finalist_evaluation_plan(CONFIG)

    assert finalist_task_count(plan) == 12
    assert plan.new_llm_calls_permitted is False
    assert plan.scientific_judge_called is False
    assert plan.test_data_opened is False
    assert plan.private_reference_opened is False
    assert plan.weighted_overall_score_defined is False
    assert plan.automatic_operating_point_selection is False
    assert [item.directory_name for item in plan.conditions] == [
        "low_016384",
        "medium_024576",
    ]
    assert [item.profile_id for item in plan.fit_profiles] == [
        "screen_50x1_300s",
        "screen_150x2_600s",
    ]
    condition, cell, repetition, source_task = task_identity(plan, 11)
    assert condition.directory_name == "medium_024576"
    assert cell.benchmark_id.endswith("canonical_opaque_hard")
    assert repetition == 2
    assert source_task == 5


def test_plan_rejects_matrix_that_differs_from_replay_count() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["repetitions"] = [0, 1]
    with pytest.raises(ValidationError, match="differs from replay result count"):
        ProposerFinalistEvaluationPlan.model_validate(payload)


def test_aces_launchers_are_cpu_only_and_syntactically_valid() -> None:
    for path in HPC_SCRIPTS:
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "gpu" not in text.lower()
    worker = HPC_SCRIPTS[0].read_text(encoding="utf-8")
    assert "#SBATCH --partition=cpu" in worker
    assert "--cpus-per-task=4" in worker
    assert "--time=00:45:00" in worker


def test_source_replay_validation_is_hash_bound(tmp_path: Path) -> None:
    conditions = (("low", 16384), ("medium", 24576))
    root = tmp_path / "replay"
    artifacts: list[dict[str, object]] = []
    rows = []
    for effort, budget in conditions:
        condition = f"{effort}_{budget:06d}"
        candidate = _candidate(f"{effort}_candidate")
        path = root / "finalists" / condition / "task_000.json"
        _write_json(path, candidate.model_dump(mode="json"))
        artifacts.append(_artifact(path, root))
        rows.append(
            {
                "reasoning_effort": effort,
                "max_output_tokens": budget,
                "task_index": 0,
                "benchmark_id": "fixture",
                "tier": "easy",
                "repetition": 0,
                "deterministic_valid": True,
                "public_target_passed": True,
                "candidate_sha256": hashlib.sha256(
                    candidate.model_dump_json().encode()
                ).hexdigest(),
            }
        )
    rows_path = root / "repair_replay_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    artifacts.append(_artifact(rows_path, root))
    ledger_path = root / "artifact_ledger.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in artifacts),
        encoding="utf-8",
    )
    ledger_sha = _sha(ledger_path)
    manifest = {
        "schema_version": "phase-b-proposer-repair-replay-manifest-1",
        "status": "pass",
        "source_plan_sha256": "1" * 64,
        "replay_plan_sha256": "2" * 64,
        "artifact_ledger_sha256": ledger_sha,
        "replay_result_count": 2,
    }
    manifest_path = root / "proposer_repair_replay.json"
    _write_json(manifest_path, manifest)
    (root / "proposer_repair_replay.json.sha256").write_text(
        f"{_sha(manifest_path)}  proposer_repair_replay.json\n",
        encoding="utf-8",
    )
    plan_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_payload["source_replay"].update(
        {
            "source_plan_sha256": "1" * 64,
            "replay_plan_sha256": "2" * 64,
            "artifact_ledger_sha256": ledger_sha,
            "replay_result_count": 2,
        }
    )
    plan_payload["cells"] = [
        {
            "benchmark_id": "fixture",
            "tier": "easy",
            "public_prompt_sha256": "3" * 64,
            "public_target_contract_sha256": "4" * 64,
            "public_mechanism_spec_sha256": "5" * 64,
        }
    ]
    plan_payload["repetitions"] = [0]
    plan = ProposerFinalistEvaluationPlan.model_validate(plan_payload)

    observed, ledger = runner._validate_source_replay(root, plan)
    assert len(observed) == 2
    assert len(ledger) == 3

    rows_path.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact differs"):
        runner._validate_source_replay(root, plan)


def test_summary_preserves_separate_endpoints_and_pairing() -> None:
    plan_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_payload["source_replay"]["replay_result_count"] = 2
    plan_payload["cells"] = [
        {
            "benchmark_id": "fixture",
            "tier": "easy",
            "public_prompt_sha256": "3" * 64,
            "public_target_contract_sha256": "4" * 64,
            "public_mechanism_spec_sha256": "5" * 64,
        }
    ]
    plan_payload["repetitions"] = [0]
    plan = ProposerFinalistEvaluationPlan.model_validate(plan_payload)
    tasks = [
        _task_payload("low", 16384, validation=0.2, cpu=10.0),
        _task_payload("medium", 24576, validation=0.1, cpu=20.0),
    ]

    conditions = summary._condition_summaries(tasks)
    pairs = summary._paired_rows(tasks, plan)

    assert conditions[0]["fit_success_rate"] == 1.0
    assert conditions[0]["mean_public_mechanism_compliance"] == 0.5
    assert conditions[0]["total_process_cpu_seconds"] == 10.0
    assert pairs[0]["fit_outcome"] == "medium_024576"
    assert pairs[0]["validation_nmse_ratio_first_over_second"] == 2.0
    assert "weighted" not in pairs[0]


def test_complexity_counts_all_public_candidate_expressions() -> None:
    complexity = runner._complexity(_candidate())
    assert complexity["state_count"] == 1
    assert complexity["parameter_count"] == 1
    assert complexity["state_equation_additive_term_count"] == 2
    assert complexity["total_expression_ast_node_count"] > 10


def _task_payload(
    effort: str,
    budget: int,
    *,
    validation: float,
    cpu: float,
) -> dict[str, object]:
    return {
        "condition": {
            "reasoning_effort": effort,
            "max_output_tokens": budget,
        },
        "benchmark_id": "fixture",
        "tier": "easy",
        "repetition": 0,
        "fit_success": True,
        "selected_fit_profile": "screen_50x1_300s",
        "selected_fit": {
            "training_normalized_mse": validation / 2,
            "validation_normalized_mse": validation,
        },
        "fit_attempts": [
            {
                "success": True,
                "wall_seconds": 5.0,
                "process_cpu_seconds": cpu,
                "function_evaluations": 4,
                "integration_failures": 0,
            }
        ],
        "allocated_cpus": 4,
        "runtime": {"valid": True},
        "public_target": {"passed": True},
        "public_mechanism": {
            "mechanism_compliance": 0.5,
            "mechanism_compliance_complete": True,
        },
        "complexity": {
            "state_count": 2,
            "latent_state_count": 1,
            "process_count": 1,
            "parameter_count": 3,
            "state_equation_additive_term_count": 4,
            "total_expression_ast_node_count": 20,
        },
    }


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "role": "fixture",
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
