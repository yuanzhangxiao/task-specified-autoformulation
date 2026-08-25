"""Tests for the search-time symmetric hybrid judge contract."""

from __future__ import annotations

from pathlib import Path

from autoformalism.judging import (
    HybridScoringConfig,
    build_atomic_evidence_plan,
    semantic_absolute_units,
)
from autoformalism.llm import MockLLMClient
from autoformalism.llm.exceptions import LLMResponseError
from autoformalism.schemas import (
    AbsoluteVerdict,
    AtomicJudgeResult,
    CandidateModel,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
)
from autoformalism.search.hybrid_pair import PairedHybridJudge


def _candidate(identifier: str, rhs: str) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "states": [{"name": "Gp", "kind": "observed"}],
            "state_equations": [{"state": "Gp", "rhs": rhs}],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"}
            ],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "fixed_value": 1.0}
            ],
        }
    )


def _atomic(
    candidate_a: CandidateModel,
    candidate_b: CandidateModel,
) -> AtomicJudgeResult:
    plan = build_atomic_evidence_plan(candidate_a, candidate_b)
    return AtomicJudgeResult.model_validate(
        {
            "signed_occurrence_assessments": [
                {
                    "occurrence_id": item.occurrence_id,
                    "expected_direction": "context_dependent",
                    "evidence": "Public evidence does not fix this direction.",
                }
                for item in plan.occurrences
            ],
            "repeated_contribution_assessments": [
                {
                    "repeat_pair_id": item.repeat_pair_id,
                    "relation": "insufficient_public_information",
                    "evidence": "Public evidence does not identify the flux.",
                }
                for item in plan.repeat_candidates
            ],
        }
    )


def _hybrid(requirements: RequirementRegistry) -> HybridJudgeResult:
    return HybridJudgeResult(
        absolute_assessments=tuple(
            PairedAbsoluteAssessment.model_validate(
                {
                    "criterion": criterion.value,
                    "subject_id": subject,
                    "candidate_a": {
                        "verdict": AbsoluteVerdict.PASS.value,
                        "evidence": "Candidate A evidence.",
                    },
                    "candidate_b": {
                        "verdict": AbsoluteVerdict.PASS.value,
                        "evidence": "Candidate B evidence.",
                    },
                }
            )
            for criterion, subject in semantic_absolute_units(
                requirements,
                include_role_consistency=False,
            )
        ),
        comparative_assessments=tuple(
            RelativeAssessment(
                criterion=criterion,
                verdict=RelativeVerdict.TIE,
                evidence="The candidates are tied on this criterion.",
            )
            for criterion in RelativeCriterion
        ),
    )


class _TerminalClient:
    def assess_atomic_evidence(self, **_kwargs):
        raise LLMResponseError("terminal structured response failure")


def test_paired_hybrid_judge_replaces_whole_failed_seed_and_reverses_order(
    tmp_path: Path,
) -> None:
    del tmp_path
    incumbent = _candidate("incumbent", "meal_event_g")
    challenger = _candidate("challenger", "meal_event_g - 0.1 * Gp")
    requirements = RequirementRegistry()
    successful = MockLLMClient(
        atomic_responses=[
            _atomic(incumbent, challenger),
            _atomic(challenger, incumbent),
        ],
        hybrid_responses=[_hybrid(requirements), _hybrid(requirements)],
    )
    judge = PairedHybridJudge(
        seeded_clients=((100, _TerminalClient()), (101, successful)),
        requirements=requirements,
        task_inputs=("meal_event_g",),
        system_prompt="Hybrid prompt.",
        atomic_system_prompt="Atomic prompt.",
        scoring=HybridScoringConfig(),
        identity="test",
    )

    result = judge.compare(incumbent, challenger)

    assert result is not None
    assert result.seed_attempt_index == 1
    assert result.seed == 101
    assert result.preferred == "tie"
    assert result.decision_value_for_incumbent == 0.0
    assert len(result.prior_terminal_failures) == 1
    assert [call["role"] for call in successful.calls] == [
        "atomic_evidence_judge",
        "hybrid_judge_atomic_repair_v1",
        "atomic_evidence_judge",
        "hybrid_judge_atomic_repair_v1",
    ]


def test_paired_hybrid_judge_records_exhausted_symmetric_evidence() -> None:
    incumbent = _candidate("incumbent", "meal_event_g")
    challenger = _candidate("challenger", "meal_event_g - 0.1 * Gp")
    judge = PairedHybridJudge(
        seeded_clients=((100, _TerminalClient()), (101, _TerminalClient())),
        requirements=RequirementRegistry(),
        task_inputs=("meal_event_g",),
        system_prompt="Hybrid prompt.",
        atomic_system_prompt="Atomic prompt.",
        identity="test",
    )

    result = judge.compare(incumbent, challenger)

    assert result.preferred == "indeterminate"
    assert result.decision_value_for_incumbent is None
    assert result.consensus_result is None
    assert result.request_hashes == ()
    assert len(result.prior_terminal_failures) == 2
