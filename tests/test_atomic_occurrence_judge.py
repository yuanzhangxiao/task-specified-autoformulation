"""Tests for sign-blinded atomic scientific evidence."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from autoformalism.judging import (
    atomic_candidate_context,
    atomic_findings_payload,
    atomic_role_compatibility_assessments,
    build_atomic_evidence_plan,
    extract_public_requirements,
    merge_atomic_assessments,
    semantic_absolute_units,
)
from autoformalism.llm import MockLLMClient
from autoformalism.schemas import (
    AbsoluteCriterion,
    AtomicJudgeResult,
    CandidateModel,
    HybridJudgeResult,
    RelativeCriterion,
)
from scripts.analyze_atomic_evidence import main as analyze_atomic_main


def _candidate(rhs: str) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [{"name": "Gp", "kind": "observed"}],
            "processes": [
                {
                    "name": "U",
                    "expression": "k * Gp",
                    "mechanisms": ["GlucoseDisposal"],
                }
            ],
            "state_equations": [{"state": "Gp", "rhs": rhs}],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"}
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
                {"state": "Gp", "scope": "global", "fixed_value": 1.0}
            ],
        }
    )


def _algebraic_target_candidate(expression: str) -> CandidateModel:
    payload = _candidate("meal_event_g - U").model_dump(mode="json")
    payload["states"][0]["name"] = "X"
    payload["states"][0]["kind"] = "latent"
    payload["state_equations"] = [{"state": "X", "rhs": "-k * X"}]
    payload["initial_conditions"][0]["state"] = "X"
    payload["processes"][0]["expression"] = "k * X"
    payload["processes"].append(
        {
            "name": "Gp",
            "expression": expression,
            "mechanisms": ["GlucoseBalance"],
            "description": "Generated glucose balance.",
            "unit": "unspecified",
        }
    )
    return CandidateModel.model_validate(payload)


def _prompt() -> str:
    return """A. Task specification

The primary objective is to recover the following task-required mechanism:
- a causal pathway through which meal amount contributes to plasma glucose mass.

B. Available data
- Gp(t): plasma glucose mass.
- meal_event_g(t): declared meal amount pulse.
"""


def _atomic_payload(plan, *, repeated: str | None = None) -> dict[str, object]:
    directions = {}
    for item in plan.occurrences:
        if item.unsigned_expression in {"meal_event_g", "abs(meal_event_g)"}:
            directions[item.occurrence_id] = "positive_contribution"
        elif item.unsigned_expression == "U":
            directions[item.occurrence_id] = "negative_contribution"
        else:
            directions[item.occurrence_id] = "context_dependent"
    return {
        "schema_version": "atomic-judge-1",
        "signed_occurrence_assessments": [
            {
                "occurrence_id": item.occurrence_id,
                "expected_direction": directions[item.occurrence_id],
                "evidence": "Direction inferred from the public scientific task.",
            }
            for item in plan.occurrences
        ],
        "repeated_contribution_assessments": [
            {
                "repeat_pair_id": item.repeat_pair_id,
                "relation": repeated or "distinct_contributions",
                "evidence": "The public interpretation distinguishes the terms.",
            }
            for item in plan.repeat_candidates
        ],
    }


def _hybrid_without_roles() -> HybridJudgeResult:
    registry = extract_public_requirements(_prompt())
    assessments = [
        {
            "criterion": criterion.value,
            "subject_id": subject,
            "candidate_a": {"verdict": "pass", "evidence": "Evidence A."},
            "candidate_b": {"verdict": "pass", "evidence": "Evidence B."},
        }
        for criterion, subject in semantic_absolute_units(
            registry, include_role_consistency=False
        )
    ]
    return HybridJudgeResult.model_validate(
        {
            "schema_version": "hybrid-1",
            "absolute_assessments": assessments,
            "comparative_assessments": [
                {
                    "criterion": criterion.value,
                    "verdict": "tie",
                    "evidence": "Paired evidence.",
                }
                for criterion in RelativeCriterion
            ],
        }
    )


def test_atomic_plan_withholds_outer_polarity_and_state_rhs() -> None:
    plan = build_atomic_evidence_plan(
        _candidate("meal_event_g - U"),
        _candidate("(meal_event_g - U) - abs(meal_event_g)"),
    )
    serialized = json.dumps(plan.prompt_payload(), sort_keys=True)
    context = atomic_candidate_context(
        _candidate("(meal_event_g - U) - abs(meal_event_g)")
    )

    assert "polarity" not in serialized
    assert "actual" not in serialized
    assert "state_equations" not in context
    assert "(meal_event_g - U) - abs(meal_event_g)" not in json.dumps(context)
    assert {item.candidate_side for item in plan.occurrences} == {
        "candidate_a",
        "candidate_b",
    }


def test_atomic_plan_covers_algebraic_target_processes() -> None:
    plan = build_atomic_evidence_plan(
        _algebraic_target_candidate("meal_event_g - U"),
        _algebraic_target_candidate(
            "(meal_event_g - U) - abs(meal_event_g)"
        ),
    )

    gp_occurrences = [
        item
        for item in plan.occurrences
        if item.equation_location == "process:Gp"
    ]
    assert len(gp_occurrences) == 5
    assert any(
        item.unsigned_expression == "abs(meal_event_g)"
        and item.actual_polarity == "negative"
        for item in gp_occurrences
    )
    assert "polarity" not in json.dumps(plan.prompt_payload())


def test_atomic_direction_is_compared_with_private_certified_sign() -> None:
    plan = build_atomic_evidence_plan(
        _candidate("meal_event_g - U"),
        _candidate("(meal_event_g - U) - abs(meal_event_g)"),
    )
    atomic = AtomicJudgeResult.model_validate(_atomic_payload(plan))

    source, sink = atomic_role_compatibility_assessments(atomic, plan)

    assert source.candidate_a.verdict.value == "pass"
    assert source.candidate_b.verdict.value == "fail"
    assert sink.candidate_a.verdict.value == "pass"
    assert sink.candidate_b.verdict.value == "pass"
    assert "occurrence_" in source.candidate_b.evidence


def test_atomic_exact_repeat_failure_overrides_broad_pass() -> None:
    plan = build_atomic_evidence_plan(
        _candidate("meal_event_g - U"),
        _candidate("(meal_event_g - U) + meal_event_g"),
    )
    atomic = AtomicJudgeResult.model_validate(
        _atomic_payload(plan, repeated="same_physical_contribution")
    )
    roles = atomic_role_compatibility_assessments(atomic, plan)

    merged = merge_atomic_assessments(
        _hybrid_without_roles(), atomic, plan, roles
    )
    by_criterion = {
        item.criterion: item for item in merged.absolute_assessments
    }

    duplicated = by_criterion[
        AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED
    ]
    assert duplicated.candidate_a.verdict.value == "pass"
    assert duplicated.candidate_b.verdict.value == "fail"
    assert by_criterion[
        AbsoluteCriterion.SOURCE_ROLES_CONSISTENT
    ].candidate_b.verdict.value == "pass"
    merged.validate_expected_absolute_units(
        set(semantic_absolute_units(extract_public_requirements(_prompt())))
    )


def test_atomic_findings_are_explicit_input_to_second_stage() -> None:
    plan = build_atomic_evidence_plan(
        _candidate("meal_event_g - U"),
        _candidate("(meal_event_g - U) + meal_event_g"),
    )
    atomic = AtomicJudgeResult.model_validate(
        _atomic_payload(plan, repeated="same_physical_contribution")
    )
    roles = atomic_role_compatibility_assessments(atomic, plan)

    payload = atomic_findings_payload(atomic, plan, roles)

    assert payload["runtime_role_compatibility"]
    assert payload["exact_repeat_interpretations"][0]["candidate_side"] == (
        "candidate_b"
    )


def test_mock_atomic_client_requires_exact_runtime_units() -> None:
    plan = build_atomic_evidence_plan(
        _candidate("meal_event_g - U"),
        _candidate("meal_event_g - U"),
    )
    client = MockLLMClient(atomic_responses=[_atomic_payload(plan)])

    result = client.assess_atomic_evidence(
        system_prompt="system",
        user_prompt="unsigned request",
        expected_occurrence_ids=plan.occurrence_ids,
        expected_repeat_pair_ids=plan.repeat_pair_ids,
    )

    assert result.parsed.schema_version == "atomic-judge-1"
    assert client.calls[0]["role"] == "atomic_evidence_judge"
    with pytest.raises(ValueError, match="missing_occurrences"):
        result.parsed.validate_expected_units(
            occurrence_ids={*plan.occurrence_ids, "occurrence_missing"},
            repeat_pair_ids=plan.repeat_pair_ids,
        )


def test_atomic_analyzer_scores_only_mutation_added_units(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for mutation, mutated_rhs, repeated in (
        (
            "wrong_meal_sink",
            "(meal_event_g - U) - abs(meal_event_g)",
            None,
        ),
        (
            "duplicated_gp_flux",
            "(meal_event_g - U) + meal_event_g",
            "same_physical_contribution",
        ),
    ):
        plan = build_atomic_evidence_plan(
            _candidate("meal_event_g - U"),
            _candidate(mutated_rhs),
        )
        rows.append(
            {
                "judge_model": "vllm:openai/gpt-oss-20b",
                "mutation_type": mutation,
                "baseline_position": "A",
                "atomic_evidence_plan": json.dumps(plan.prompt_payload()),
                "atomic_assessments": json.dumps(
                    _atomic_payload(plan, repeated=repeated)
                ),
            }
        )
    score_path = tmp_path / "scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_atomic_evidence.py",
            "--scores",
            str(score_path),
            "--output",
            str(output),
        ],
    )

    analyze_atomic_main()

    metrics = json.loads(output.read_text(encoding="utf-8"))[
        "vllm:openai/gpt-oss-20b"
    ]
    assert metrics["wrong_sink_expected_direction_accuracy"] == 1.0
    assert metrics["duplicate_relation_accuracy"] == 1.0
    assert metrics["combined_atomic_mutation_accuracy"] == 1.0
