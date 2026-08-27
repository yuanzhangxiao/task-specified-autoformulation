"""Tests for fit-free scientific auditing of raw-agent structures."""

from __future__ import annotations

import json

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    PairedAbsoluteAssessment,
    ProposerCandidateV2,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
    enrich_proposal_v2,
)
from scripts.build_raw_agent_scientific_audit_pairs import build_self_audit_pairs
from scripts.summarize_raw_agent_scientific_audit import summarize


def _candidate():
    proposal = ProposerCandidateV2.model_validate(
        {
            "candidate_id": "raw",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "observed_channel": "x",
                    "rhs": "-x + u",
                }
            ],
        }
    )
    return enrich_proposal_v2(proposal, ("x",))


def _assessment(
    criterion: AbsoluteCriterion, verdict: AbsoluteVerdict
) -> PairedAbsoluteAssessment:
    candidate = CandidateAbsoluteAssessment(verdict=verdict, evidence="evidence")
    return PairedAbsoluteAssessment(
        criterion=criterion,
        subject_id="candidate",
        candidate_a=candidate,
        candidate_b=candidate,
    )


def _row(pair_id: str, order: str, runtime: AbsoluteVerdict) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "repetition": "0",
        "order": order,
        "requirements": RequirementRegistry().model_dump_json(),
        "deterministic_assessments": json.dumps(
            [
                _assessment(
                    AbsoluteCriterion.TASK_INPUTS_REACH_TARGETS, runtime
                ).model_dump(mode="json")
            ]
        ),
        "absolute_assessments": json.dumps(
            [
                _assessment(
                    AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
                    AbsoluteVerdict.PASS,
                ).model_dump(mode="json")
            ]
        ),
        "comparative_assessments": json.dumps(
            [
                RelativeAssessment(
                    criterion=criterion,
                    verdict=RelativeVerdict.TIE,
                    evidence="Identical candidates.",
                ).model_dump(mode="json")
                for criterion in RelativeCriterion
            ]
        ),
    }


def test_self_audit_pair_duplicates_structure_without_truth_label() -> None:
    candidate = _candidate()
    pairs = build_self_audit_pairs(
        [("cell", "easy", 0, candidate)],
        {
            ("cell", "easy"): ValidationContext(
                targets=("x",), external_inputs=("u",)
            )
        },
    )

    assert len(pairs) == 1
    assert pairs[0].valid_candidate == pairs[0].adversarial_candidate
    assert pairs[0].mutation_type == "raw_agent_identity_scientific_audit"


def test_scientific_audit_reports_runtime_compliance_without_fitting() -> None:
    pair = build_self_audit_pairs(
        [("cell", "easy", 0, _candidate())],
        {
            ("cell", "easy"): ValidationContext(
                targets=("x",), external_inputs=("u",)
            )
        },
    )[0]

    result = summarize(
        (pair,),
        [
            _row(pair.pair_id, "baseline_a", AbsoluteVerdict.FAIL),
            _row(pair.pair_id, "baseline_b", AbsoluteVerdict.FAIL),
        ],
    )

    assert result["parameter_fitting_used"] is False
    assert result["response_success_rate"] == 1.0
    assert result["neutral_atomic_unit_repair_count"] == 0
    assert result["task_compliance_counts"] == {"fail": 1}
