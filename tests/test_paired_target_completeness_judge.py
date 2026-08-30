"""Tests for the V8 paired target-only judging protocol."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.judging import paired_target_question_consensus
from autoformalism.llm import MockLLMClient
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import (
    CandidateModel,
    PairedTargetCompletenessJudgeResult,
)
from scripts.analyze_paired_target_completeness_judge import (
    evaluate_paired_target_completeness,
)
from scripts.run_paired_target_completeness_judge import (
    _seed_for_attempt,
    paired_target_completeness_request,
    paired_target_completeness_system_prompt,
)

CONFIG = Path("configs/target_completeness_paired_v8.json")
SLURM = Path("scripts/hpc/phase_b_target_completeness_paired_v8_120b.slurm")


def _candidate(identifier: str, total_expression: str) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "change_summary": identifier,
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


def _paired_result(a_u: str, b_u: str) -> dict[str, object]:
    return {
        "target_assessments": [
            {
                "target_id": target,
                "candidate_a": {
                    "verdict": a_u if target == "U" else "pass",
                    "evidence": f"Candidate A resolves {target}.",
                },
                "candidate_b": {
                    "verdict": b_u if target == "U" else "pass",
                    "evidence": f"Candidate B resolves {target}.",
                },
            }
            for target in ("Gp", "I", "U")
        ]
    }


def _pair() -> AdversarialPair:
    return AdversarialPair(
        pair_id="pair",
        benchmark_id="benchmark",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=_candidate("valid", "Uii + Uid"),
        adversarial_candidate=_candidate("mutated", "Uid"),
    )


def test_paired_schema_requires_two_answers_for_exact_targets() -> None:
    result = PairedTargetCompletenessJudgeResult.model_validate(
        _paired_result("pass", "fail")
    )

    result.validate_expected_targets({"Gp", "I", "U"})
    assert result.candidate_overall_verdict("a").value == "pass"
    assert result.candidate_overall_verdict("b").value == "fail"
    with pytest.raises(ValueError, match="missing_targets"):
        result.validate_expected_targets({"Gp", "I", "U", "extra"})

    invalid = _paired_result("pass", "not_applicable")
    with pytest.raises(ValueError, match="cannot be marked"):
        PairedTargetCompletenessJudgeResult.model_validate(invalid)


def test_mock_exposes_distinct_paired_target_role() -> None:
    client = MockLLMClient(
        paired_target_completeness_responses=[_paired_result("pass", "fail")]
    )

    result = client.assess_paired_target_completeness(
        system_prompt="Paired target-only prompt.",
        user_prompt="Two candidates.",
        expected_target_ids={"Gp", "I", "U"},
    )

    assert result.parsed.candidate_overall_verdict("b").value == "fail"
    assert client.calls[0]["role"] == "paired_target_completeness_judge_v1"


def test_request_blinds_lineage_but_keeps_both_candidates(
    context: ValidationContext,
) -> None:
    request = paired_target_completeness_request(
        _candidate("valid", "Uii + Uid"),
        _candidate("mutated", "Uid"),
        public_prompt="U is total disposal including Uii and Uid.",
        context=context,
    )
    prompt = paired_target_completeness_system_prompt(
        "U is total disposal including Uii and Uid.", context, "judge"
    )

    assert request["candidate_a"]["candidate_id"] == "candidate_a"  # type: ignore[index]
    assert request["candidate_b"]["candidate_id"] == "candidate_b"  # type: ignore[index]
    assert request["candidate_a"]["parent_candidate_id"] is None  # type: ignore[index]
    assert "mutation_type" not in json.dumps(request)
    assert "paired-target-completeness-judge-1" in prompt
    assert "exact-repeat" in prompt


def test_consensus_normalizes_reverse_and_fails_dominantly() -> None:
    forward = PairedTargetCompletenessJudgeResult.model_validate(
        _paired_result("pass", "pass")
    )
    reverse = PairedTargetCompletenessJudgeResult.model_validate(
        _paired_result("fail", "pass")
    )

    baseline, mutated, disagreements = paired_target_question_consensus(
        forward,
        reverse,
    )

    assert baseline.overall_verdict.value == "pass"
    assert mutated.overall_verdict.value == "fail"
    assert disagreements == ("U:mutated",)


def test_seed_blocks_do_not_overlap_provider_retries() -> None:
    seeds = [
        _seed_for_attempt(
            10000,
            repetition,
            attempt,
            seed_attempts=2,
            provider_attempts=10,
        )
        for repetition in range(2)
        for attempt in range(2)
    ]
    assert seeds == [10000, 10010, 10020, 10030]


def test_analyzer_scores_perfect_retried_consensus() -> None:
    forward = PairedTargetCompletenessJudgeResult.model_validate(
        _paired_result("pass", "fail")
    )
    reverse = PairedTargetCompletenessJudgeResult.model_validate(
        _paired_result("fail", "pass")
    )
    baseline, mutated, disagreements = paired_target_question_consensus(
        forward, reverse
    )
    rows = []
    for repetition in range(2):
        rows.append(
            {
                "pair_id": "pair",
                "repetition": repetition,
                "requested_target_ids": ["Gp", "I", "U"],
                "selected_seed_attempt": 2 if repetition == 0 else 1,
                "prior_seed_failures": (
                    [
                        {
                            "discarded_successful_orientation_count": 1,
                            "orientation_errors": [
                                {"error_type": "LLMResponseError"}
                            ],
                        }
                    ]
                    if repetition == 0
                    else []
                ),
                "forward": {"result": forward.model_dump(mode="json")},
                "reverse": {"result": reverse.model_dump(mode="json")},
                "consensus": {
                    "baseline": baseline.model_dump(mode="json"),
                    "mutated": mutated.model_dump(mode="json"),
                    "baseline_overall_verdict": "pass",
                    "mutated_overall_verdict": "fail",
                    "orientation_disagreements": list(disagreements),
                },
            }
        )
    gates = {
        "minimum_paired_trial_coverage": 0.95,
        "minimum_candidate_verdict_accuracy": 0.90,
        "minimum_complete_trial_accuracy": 0.90,
        "minimum_pair_aggregate_accuracy": 1.00,
        "minimum_all_target_unit_accuracy": 0.90,
        "minimum_evaluated_target_accuracy": 0.90,
        "minimum_valid_target_pass_rate": 0.90,
        "minimum_incomplete_target_fail_rate": 0.90,
        "minimum_orientation_verdict_consistency": 0.80,
        "minimum_repeat_modal_consistency": 0.90,
    }

    result = evaluate_paired_target_completeness(
        rows,
        [],
        (_pair(),),
        repetitions=2,
        evaluated_target_id="U",
        seed_attempt_limit=2,
        gates=gates,
    )

    assert result["passed"] is True
    assert result["metrics"]["paired_trial_coverage"] == 1.0  # type: ignore[index]
    assert result["metrics"]["retry_activation_rate"] == 0.5  # type: ignore[index]


def test_v8_config_and_launcher_freeze_target_only_transaction() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    launcher = SLURM.read_text(encoding="utf-8")

    assert config["status"] == "frozen_before_paired_target_only_calls"
    protocol = config["protocol"]
    assert protocol["candidate_order_policy"] == "both_orientations_same_seed"
    assert protocol["atomic_stage_enabled"] is False
    assert protocol["comparative_questions_enabled"] is False
    assert protocol["max_paired_seed_attempts"] == 2
    assert "AF_TARGET_COMPLETENESS_PROTOCOL:=paired_v8" in launcher
    assert "analyze_paired_target_completeness_judge.py" in launcher
    assert "cmp --silent" in launcher
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
