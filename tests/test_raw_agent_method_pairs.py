"""Tests for unlabeled raw-agent versus method pair construction."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    ProposerCandidateV2,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
    ScientificRequirement,
    enrich_proposal_v2,
)
from scripts.build_raw_agent_method_pairs import build_method_pairs
from scripts.summarize_raw_agent_method_judge import summarize


def _candidate(identifier: str, rhs: str):
    proposal = ProposerCandidateV2.model_validate(
        {
            "schema_version": "2",
            "candidate_id": identifier,
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "observed_channel": "x",
                    "rhs": rhs,
                }
            ],
            "algebraics": [],
            "parameters": [],
        }
    )
    return enrich_proposal_v2(proposal, ("x",))


def test_build_method_pairs_keeps_truth_unlabeled_and_task_matched() -> None:
    raw = _candidate("raw", "-x")
    reference = _candidate("reference", "-x + u")
    contexts = {
        ("cell", "easy"): ValidationContext(
            targets=("x",),
            external_inputs=("u",),
            forcing_bounds={"u": (0.0, 1.0)},
        )
    }

    pairs = build_method_pairs(
        [(Path("raw.json"), "cell", "easy", 2, raw)],
        {("cell", "easy"): (Path("summary.json"), reference)},
        contexts,
    )

    assert len(pairs) == 1
    assert pairs[0].mutation_type == "raw_agent_vs_autoformalism_unlabeled"
    assert pairs[0].valid_candidate.candidate_id == "reference"
    assert pairs[0].adversarial_candidate.candidate_id == "raw"


def test_build_method_pairs_ignores_tasks_without_reference() -> None:
    pairs = build_method_pairs(
        [(Path("raw.json"), "other", "hard", 0, _candidate("raw", "-x"))],
        {},
        {},
    )

    assert pairs == ()


def _judge_result(reference_is_a: bool) -> HybridJudgeResult:
    reference_verdict = (
        RelativeVerdict.CANDIDATE_A
        if reference_is_a
        else RelativeVerdict.CANDIDATE_B
    )
    return HybridJudgeResult(
        absolute_assessments=tuple(
            PairedAbsoluteAssessment(
                criterion=criterion,
                subject_id="requirement",
                candidate_a=CandidateAbsoluteAssessment(
                    verdict=AbsoluteVerdict.PASS, evidence="A"
                ),
                candidate_b=CandidateAbsoluteAssessment(
                    verdict=AbsoluteVerdict.PASS, evidence="B"
                ),
            )
            for criterion in (
                AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED,
                AbsoluteCriterion.REQUIRED_MECHANISM_CONNECTED,
            )
        ),
        comparative_assessments=tuple(
            RelativeAssessment(
                criterion=criterion,
                verdict=reference_verdict,
                evidence="reference preferred",
            )
            for criterion in RelativeCriterion
        ),
    )


def _judge_row(pair_id: str, order: str) -> dict[str, str]:
    result = _judge_result(reference_is_a=order == "baseline_a")
    requirements = RequirementRegistry(
        requirements=(
            ScientificRequirement(
                requirement_id="requirement",
                text="Generate the target.",
                source="benchmark",
                enforcement="soft",
            ),
        )
    )
    return {
        "pair_id": pair_id,
        "repetition": "0",
        "order": order,
        "requirements": requirements.model_dump_json(),
        "deterministic_assessments": "[]",
        "absolute_assessments": json.dumps(
            [item.model_dump(mode="json") for item in result.absolute_assessments]
        ),
        "comparative_assessments": json.dumps(
            [
                item.model_dump(mode="json")
                for item in result.comparative_assessments
            ]
        ),
    }


def test_method_summary_normalizes_both_candidate_orientations() -> None:
    pair = build_method_pairs(
        [(Path("raw.json"), "cell", "easy", 0, _candidate("raw", "-x"))],
        {
            ("cell", "easy"): (
                Path("summary.json"),
                _candidate("reference", "-x"),
            )
        },
        {("cell", "easy"): ValidationContext(targets=("x",))},
    )[0]

    result = summarize(
        (pair,),
        [
            _judge_row(pair.pair_id, "baseline_a"),
            _judge_row(pair.pair_id, "baseline_b"),
        ],
    )

    assert result["preference_counts"] == {"autoformalism": 1}
    assert result["outcomes"][0]["preferred"] == "autoformalism"
