"""Tests for the candidate-specific absolute target-completeness protocol."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm import MockLLMClient
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel, TargetCompletenessJudgeResult
from scripts.analyze_target_completeness_judge import (
    evaluate_target_completeness,
)
from scripts.run_target_completeness_judge import (
    target_completeness_request,
    target_completeness_system_prompt,
)

CONFIG = Path("configs/target_completeness_absolute_v7.json")
SLURM = Path(
    "scripts/hpc/phase_b_target_completeness_absolute_v7_120b.slurm"
)
SHARED_LAUNCHER = Path("scripts/hpc/run_vllm_atomic_judge.sh")


def _candidate(identifier: str, total_expression: str) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "change_summary": f"Construction label for {identifier}",
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
                {"name": "X", "kind": "latent"},
            ],
            "processes": [
                {
                    "name": "Uid",
                    "expression": "k * X * Gp",
                    "mechanisms": ["insulin_dependent_disposal"],
                },
                {"name": "U", "expression": total_expression},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "EGP - Uii - Uid"},
                {"state": "I", "rhs": "insulin_input - I"},
                {"state": "X", "rhs": "I - X"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
                {"channel": "I", "expression": "I"},
                {"channel": "U", "expression": "U"},
            ],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
            ],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "expression": "Gp"},
                {"state": "I", "scope": "global", "expression": "I"},
                {"state": "X", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


@pytest.fixture
def context() -> ValidationContext:
    return ValidationContext(
        targets=("Gp", "I", "U"),
        auxiliaries=("EGP", "Uii"),
        external_inputs=("insulin_input",),
        lagged_targets=("Gp", "I", "U"),
    )


def _result(u_verdict: str) -> dict[str, object]:
    return {
        "target_assessments": [
            {
                "target_id": "Gp",
                "verdict": "pass",
                "evidence": "Gp maps to the generated observed state Gp.",
            },
            {
                "target_id": "I",
                "verdict": "pass",
                "evidence": "I maps to the generated observed state I.",
            },
            {
                "target_id": "U",
                "verdict": u_verdict,
                "evidence": "The recursively resolved U definition determines this.",
            },
        ]
    }


def test_schema_requires_exact_unique_target_ids() -> None:
    result = TargetCompletenessJudgeResult.model_validate(_result("fail"))

    result.validate_expected_targets({"Gp", "I", "U"})
    assert result.overall_verdict.value == "fail"
    with pytest.raises(ValueError, match="missing_targets"):
        result.validate_expected_targets({"Gp", "I", "U", "extra"})
    duplicate = _result("pass")
    duplicate["target_assessments"].append(  # type: ignore[union-attr]
        duplicate["target_assessments"][0]  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="must be unique"):
        TargetCompletenessJudgeResult.model_validate(duplicate)


def test_mock_client_exposes_one_candidate_absolute_operation() -> None:
    client = MockLLMClient(target_completeness_responses=[_result("pass")])

    result = client.assess_target_completeness(
        system_prompt="Absolute target prompt.",
        user_prompt="One candidate.",
        expected_target_ids={"Gp", "I", "U"},
    )

    assert result.parsed.overall_verdict.value == "pass"
    assert client.calls[0]["role"] == "target_completeness_judge_v1"


def test_request_blinds_lineage_and_contains_no_second_candidate(
    context: ValidationContext,
) -> None:
    candidate = _candidate("valid_total_model", "Uii + Uid")
    request = target_completeness_request(
        candidate,
        public_prompt=(
            "The primary objective is to recover the following task-required "
            "mechanisms:\n- insulin-dependent disposal"
        ),
        context=context,
    )
    prompt = target_completeness_system_prompt(
        "U is total disposal including Uii and Uid.", context, "judge"
    )

    assert request["candidate"]["candidate_id"] == "candidate"  # type: ignore[index]
    assert request["candidate"]["parent_candidate_id"] is None  # type: ignore[index]
    assert request["candidate"]["change_summary"] == "unspecified"  # type: ignore[index]
    assert "candidate_a" not in json.dumps(request)
    assert "candidate_b" not in json.dumps(request)
    assert "exact_repeat_candidates" not in json.dumps(request)
    assert "absolute assessment, not" in prompt


def test_frozen_analyzer_accepts_perfect_candidate_specific_results() -> None:
    valid = _candidate("valid", "Uii + Uid")
    mutated = _candidate("mutated", "Uid")
    pair = AdversarialPair(
        pair_id="pair",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=valid,
        adversarial_candidate=mutated,
    )
    rows = []
    for repetition in range(2):
        for role, verdict in (("baseline", "pass"), ("mutated", "fail")):
            assessments = _result(verdict)["target_assessments"]
            rows.append(
                {
                    "pair_id": "pair",
                    "repetition": str(repetition),
                    "candidate_role": role,
                    "requested_target_ids": json.dumps(["Gp", "I", "U"]),
                    "target_assessments": json.dumps(assessments),
                    "overall_verdict": verdict,
                }
            )
    gates = {
        "minimum_response_success": 0.95,
        "minimum_joint_candidate_coverage": 0.95,
        "minimum_candidate_verdict_accuracy": 0.90,
        "minimum_complete_trial_accuracy": 0.90,
        "minimum_pair_aggregate_accuracy": 1.00,
        "minimum_all_target_unit_accuracy": 0.90,
        "minimum_evaluated_target_accuracy": 0.90,
        "minimum_valid_target_pass_rate": 0.90,
        "minimum_incomplete_target_fail_rate": 0.90,
        "minimum_repeat_modal_consistency": 0.90,
    }

    result = evaluate_target_completeness(
        rows,
        [],
        (pair,),
        repetitions=2,
        evaluated_target_id="U",
        gates=gates,
    )

    assert result["passed"] is True
    assert result["metrics"]["response_success"] == 1.0  # type: ignore[index]
    assert result["metrics"]["evaluated_target_accuracy"] == 1.0  # type: ignore[index]


def test_frozen_analyzer_fails_when_one_candidate_call_is_missing() -> None:
    valid = _candidate("valid", "Uii + Uid")
    mutated = _candidate("mutated", "Uid")
    pair = AdversarialPair(
        pair_id="pair",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=valid,
        adversarial_candidate=mutated,
    )
    row = {
        "pair_id": "pair",
        "repetition": "0",
        "candidate_role": "baseline",
        "requested_target_ids": json.dumps(["Gp", "I", "U"]),
        "target_assessments": json.dumps(_result("pass")["target_assessments"]),
        "overall_verdict": "pass",
    }
    failure = {
        "pair_id": "pair",
        "repetition": 0,
        "candidate_role": "mutated",
        "error_type": "LLMResponseError",
    }
    gates = json.loads(CONFIG.read_text(encoding="utf-8"))["validation_gate"]

    result = evaluate_target_completeness(
        [row],
        [failure],
        (pair,),
        repetitions=1,
        evaluated_target_id="U",
        gates=gates,
    )

    assert result["passed"] is False
    assert result["metrics"]["response_success"] == 0.5  # type: ignore[index]
    assert result["metrics"]["joint_candidate_coverage"] == 0.0  # type: ignore[index]


def test_v7_config_and_launcher_freeze_candidate_specific_protocol() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    launcher = SLURM.read_text(encoding="utf-8")
    shared = SHARED_LAUNCHER.read_text(encoding="utf-8")

    assert config["status"] == "frozen_before_candidate_specific_target_calls"
    protocol = config["protocol"]
    assert protocol["candidate_policy"] == "one_candidate_per_call"
    assert protocol["atomic_stage_enabled"] is False
    assert protocol["comparative_questions_enabled"] is False
    assert protocol["numeric_score_defined"] is False
    assert "AF_JUDGE_ENTRYPOINT:=target_completeness" in launcher
    assert "run_target_completeness_judge.py" in shared
    assert "analyze_target_completeness_judge.py" in launcher
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    subprocess.run(["bash", "-n", str(SHARED_LAUNCHER)], check=True)
