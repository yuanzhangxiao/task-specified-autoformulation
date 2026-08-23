"""Static checks for the matched atomic 20B/120B development experiment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    PairedAbsoluteAssessment,
)
from scripts.run_hybrid_judge import _json

CONFIG = Path("configs/hybrid_judge_atomic_occurrence_v1.json")
COMMON = Path("scripts/hpc/run_vllm_atomic_judge.sh")
SCRIPT_20B = Path("scripts/hpc/phase_b_hybrid_judge_vllm_atomic_20b.slurm")
SCRIPT_120B = Path("scripts/hpc/phase_b_hybrid_judge_vllm_atomic_120b.slurm")
RUNNER = Path("scripts/run_hybrid_judge.py")


def test_atomic_factorial_changes_model_scale_only_within_protocol() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen_matched_development_factorial"
    assert len(payload["selected_pair_ids"]) == 4
    assert payload["selected_mutations"] == [
        "duplicated_gp_flux",
        "wrong_meal_sink",
    ]
    assert payload["protocol"]["semantic_answer_disclosure"] is False
    assert payload["protocol"]["mutation_labels_visible_to_judge"] is False
    assert payload["matched_factors"] == {
        "reasoning_effort": "low",
        "temperature": 0.2,
        "seed_base": 10000,
        "repetitions": 5,
        "candidate_order_policy": "both_orientations",
        "scoring": {
            "partial_tiebreak_weight": 0.05,
            "comparative_weight": 0.25,
            "tie_threshold": 0.05,
        },
    }
    assert payload["models"] == [
        {
            "judge_model": "vllm:openai/gpt-oss-20b",
            "gpus": 1,
            "tensor_parallel_size": 1,
        },
        {
            "judge_model": "vllm:openai/gpt-oss-120b",
            "gpus": 4,
            "tensor_parallel_size": 4,
        },
    ]
    assert payload["planned_per_model"] == {
        "paired_judgments": 40,
        "llm_stages": 80,
    }


def test_atomic_slurm_contracts_are_self_contained_and_syntax_valid() -> None:
    for script in (COMMON, SCRIPT_20B, SCRIPT_120B):
        subprocess.run(["bash", "-n", str(script)], check=True)

    common = COMMON.read_text(encoding="utf-8")
    small = SCRIPT_20B.read_text(encoding="utf-8")
    large = SCRIPT_120B.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "--atomic-signed-occurrences" in common
    assert "--vllm-reasoning-effort low" in common
    assert "--vllm-temperature 0.2" in common
    assert "--tensor-parallel-size" in common
    assert "#SBATCH --gpus-per-node=1" in small
    assert "#SBATCH --array=0-3" in small
    assert "openai/gpt-oss-20b" in small
    assert "#SBATCH --gpus-per-node=4" in large
    assert "openai/gpt-oss-120b" in large
    assert "atomic_evidence_schema_version" in runner
    assert "logical_stages_per_judgment" in runner


def test_atomic_compatibility_assessments_are_csv_json_serializable() -> None:
    assessment = PairedAbsoluteAssessment(
        criterion=AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
        subject_id="candidate",
        candidate_a=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.PASS,
            evidence="Certified occurrence polarity matches the inferred role.",
        ),
        candidate_b=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.FAIL,
            evidence="Certified occurrence polarity conflicts with the inferred role.",
        ),
    )

    payload = json.loads(_json((assessment,)))

    assert payload == [assessment.model_dump(mode="json")]
