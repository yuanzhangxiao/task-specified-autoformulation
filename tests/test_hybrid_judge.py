"""Tests for the calibration-only hybrid scientific judge protocol."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.judging import (
    HybridScoringConfig,
    candidate_claims,
    deterministic_pair_assessments,
    extract_public_requirements,
    score_hybrid_pair,
    semantic_absolute_units,
    structural_facts,
)
from autoformalism.llm import MockLLMClient, OllamaResponseMode
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    ExpectedVerdict,
    mutation_label_contract,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    CandidateModel,
    HybridJudgeResult,
    ProposerClaim,
    RelativeCriterion,
    RequirementRegistry,
    ScientificRequirement,
)
from scripts.analyze_hybrid_judge import main as analyze_hybrid_main
from scripts.build_hybrid_judge_label_template import build_label_template
from scripts.build_hybrid_judge_pairs import augment_pairs
from scripts.merge_hybrid_scores import main as merge_hybrid_main
from scripts.run_hybrid_judge import (
    FAILURE_SCHEMA_VERSION,
    _append_failure,
    _ensure_run_manifest,
    _failed,
    _planned_keys,
    _select_pair_ids,
    _system_prompt,
)


def _candidate(*, disconnected_claim: bool = False) -> CandidateModel:
    processes = [
        {
            "name": "U",
            "expression": "k * X * Gp",
            "mechanisms": ["InsulinDisposal"],
        }
    ]
    if disconnected_claim:
        processes.append(
            {
                "name": "extra",
                "expression": "meal_event_g",
                "mechanisms": ["ExtraMealClaim"],
            }
        )
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {
                    "name": "X",
                    "kind": "latent",
                    "mechanisms": ["DelayedInsulinAction"],
                },
            ],
            "processes": processes,
            "state_equations": [
                {"state": "Gp", "rhs": "meal_event_g - U"},
                {"state": "X", "rhs": "insulin_input - k * X"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
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
                {"state": "Gp", "scope": "global", "fixed_value": 1.0},
                {"state": "X", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _candidate_with_gp_rhs(rhs: str) -> CandidateModel:
    payload = _candidate().model_dump(mode="json")
    equation = next(
        item for item in payload["state_equations"] if item["state"] == "Gp"
    )
    equation["rhs"] = rhs
    return CandidateModel.model_validate(payload)


def _prompt() -> str:
    return """A. Task specification

The primary objective is to recover the following task-required mechanisms:
- a meal-to-target pathway.
- a delayed regulator-dependent removal pathway.

The dimensionality must be inferred.
"""


def test_named_pair_selection_preserves_requested_order_and_fails_closed() -> None:
    first = AdversarialPair(
        pair_id="first",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="wrong_meal_sink",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )
    second = first.model_copy(update={"pair_id": "second"})

    selected = _select_pair_ids((first, second), ["second", "first"])

    assert [pair.pair_id for pair in selected] == ["second", "first"]
    with pytest.raises(ValueError, match="missing"):
        _select_pair_ids((first, second), ["unknown"])
    with pytest.raises(ValueError, match="duplicates"):
        _select_pair_ids((first, second), ["first", "first"])


def _hybrid_payload(
    *,
    requirement_a: str = "pass",
    requirement_b: str = "fail",
    candidate_a: str = "pass",
    candidate_b: str = "pass",
    relative: str = "tie",
) -> dict[str, object]:
    registry = extract_public_requirements(_prompt())
    assessments = []
    for criterion, subject in semantic_absolute_units(registry):
        left = requirement_a if subject != "candidate" else candidate_a
        right = requirement_b if subject != "candidate" else candidate_b
        assessments.append(
            {
                "criterion": criterion.value,
                "subject_id": subject,
                "candidate_a": {"verdict": left, "evidence": "Evidence for A."},
                "candidate_b": {"verdict": right, "evidence": "Evidence for B."},
            }
        )
    return {
        "schema_version": "hybrid-1",
        "absolute_assessments": assessments,
        "comparative_assessments": [
            {
                "criterion": criterion,
                "verdict": relative,
                "evidence": "Direct paired evidence.",
            }
            for criterion in (
                "parsimony_while_task_sufficient",
                "fewer_unsupported_assumptions",
                "mechanistic_interpretability",
            )
        ],
    }


def test_requirements_come_only_from_public_task_bullets() -> None:
    registry = extract_public_requirements(_prompt())

    assert len(registry.requirements) == 2
    assert all(item.source.value == "benchmark" for item in registry.requirements)
    assert all(item.enforcement.value == "hard" for item in registry.requirements)
    assert (
        extract_public_requirements("Predict the supplied targets.").requirements
        == ()
    )


def test_claims_remain_proposer_owned_and_do_not_become_requirements() -> None:
    claims = candidate_claims(_candidate())

    assert {item.mechanism for item in claims} == {
        "InsulinDisposal",
        "DelayedInsulinAction",
    }
    assert all(item.source.value == "proposer" for item in claims)

    with pytest.raises(ValidationError, match="proposer/runtime"):
        RequirementRegistry(
            requirements=(
                ScientificRequirement(
                    requirement_id="invented",
                    text="Candidate-invented requirement.",
                    source="proposer",
                    enforcement="hard",
                ),
            )
        )
    with pytest.raises(ValidationError, match="proposer provenance"):
        ProposerClaim(
            claim_id="claim",
            subject_id="X",
            mechanism="Memory",
            source="benchmark",
        )


def test_structural_facts_certify_paths_without_scientific_verdicts() -> None:
    facts = structural_facts(
        _candidate(disconnected_claim=True),
        task_inputs=("meal_event_g", "insulin_input"),
    )

    assert facts["task_inputs"]["meal_event_g"]["reaches_requested_target"]
    assert facts["components"]["X"]["reaches_requested_target"]
    assert facts["components"]["extra"]["reaches_requested_target"] is False
    assert "scientifically_valid" not in facts["components"]["extra"]


def test_structural_facts_certify_signed_occurrences_and_exact_repeats() -> None:
    baseline = structural_facts(
        _candidate_with_gp_rhs("meal_event_g - U"),
        task_inputs=("meal_event_g", "insulin_input"),
    )
    wrong_sink = structural_facts(
        _candidate_with_gp_rhs("(meal_event_g - U) - abs(meal_event_g)"),
        task_inputs=("meal_event_g", "insulin_input"),
    )
    duplicated = structural_facts(
        _candidate_with_gp_rhs("(meal_event_g - U) + meal_event_g"),
        task_inputs=("meal_event_g", "insulin_input"),
    )

    assert baseline["schema_version"] == "structural-facts-2"
    baseline_equation = baseline["algebraic_expressions"]["equation:Gp"]
    assert baseline_equation["signed_symbol_occurrences"]["meal_event_g"] == {
        "positive_term_ids": ["term_0"],
        "negative_term_ids": [],
        "positive_term_count": 1,
        "negative_term_count": 0,
    }
    wrong_equation = wrong_sink["algebraic_expressions"]["equation:Gp"]
    assert wrong_equation["signed_symbol_occurrences"]["meal_event_g"] == {
        "positive_term_ids": ["term_0"],
        "negative_term_ids": ["term_2"],
        "positive_term_count": 1,
        "negative_term_count": 1,
    }
    assert wrong_equation["exact_repeated_additive_terms"] == []
    duplicate_equation = duplicated["algebraic_expressions"]["equation:Gp"]
    assert duplicate_equation["exact_repeated_additive_terms"] == [
        {
            "polarity": "positive",
            "normalized_expression": "meal_event_g",
            "count": 2,
            "term_ids": ["term_0", "term_2"],
        }
    ]
    assert "scientifically_duplicated" not in duplicate_equation


def test_json_primary_tool_fallback_keeps_general_judge_prompt_unchanged() -> None:
    context = ValidationContext(
        targets=("Gp", "U"),
        external_inputs=("meal_event_g", "insulin_input"),
    )
    ordinary = _system_prompt(
        _prompt(),
        context,
        "ollama:gpt-oss:20b",
        OllamaResponseMode.JSON_SCHEMA,
    )
    fallback = _system_prompt(
        _prompt(),
        context,
        "ollama:gpt-oss:20b",
        OllamaResponseMode.JSON_SCHEMA_TOOL_FALLBACK,
    )
    tool_only = _system_prompt(
        _prompt(),
        context,
        "ollama:gpt-oss:20b",
        OllamaResponseMode.TOOL_CALL,
    )

    assert fallback == ordinary
    assert "Configured transport override" not in fallback
    assert "Configured transport override" in tool_only
    assert "certified signed" in ordinary
    assert "Algebraically redundant terms" in ordinary
    assert "infer that candidates are identical" in ordinary


def test_deterministic_pair_assessments_keep_retained_disconnection() -> None:
    assessments = deterministic_pair_assessments(
        _candidate(),
        _candidate(disconnected_claim=True),
        task_inputs=("meal_event_g", "insulin_input"),
    )
    by_criterion = {item.criterion: item for item in assessments}

    connected = by_criterion[
        AbsoluteCriterion.CLAIMED_COMPONENTS_REACH_TARGETS
    ]
    assert connected.candidate_a.verdict.value == "pass"
    assert connected.candidate_b.verdict.value == "fail"
    assert by_criterion[
        AbsoluteCriterion.LATENT_STATES_HAVE_INCOMING_PATHWAYS
    ].candidate_a.verdict.value == "pass"


def test_hybrid_schema_requires_exact_runtime_requested_units() -> None:
    registry = extract_public_requirements(_prompt())
    result = HybridJudgeResult.model_validate(_hybrid_payload())
    expected = set(semantic_absolute_units(registry))

    result.validate_expected_absolute_units(expected)
    with pytest.raises(ValueError, match="missing"):
        result.validate_expected_absolute_units(
            expected
            | {(AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED, "extra")}
        )


def test_hybrid_schema_discards_only_complete_whitelisted_extra_units() -> None:
    payload = _hybrid_payload()
    result = HybridJudgeResult.model_validate(payload)
    registry = extract_public_requirements(_prompt())
    expected = set(
        semantic_absolute_units(registry, include_role_consistency=False)
    )
    redundant = {
        (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate"),
        (AbsoluteCriterion.SINK_ROLES_CONSISTENT, "candidate"),
    }

    repaired, removed = result.discard_redundant_absolute_units(
        expected=expected,
        redundant=redundant,
    )

    repaired.validate_expected_absolute_units(expected)
    assert set(removed) == redundant
    assert len(repaired.absolute_assessments) == len(expected)

    incomplete_expected = expected | {
        (AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED, "missing")
    }
    unchanged, removed = result.discard_redundant_absolute_units(
        expected=incomplete_expected,
        redundant=redundant,
    )
    assert unchanged is result
    assert removed == ()

    other_extra = (
        AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED,
        "candidate",
    )
    extra_payload = _hybrid_payload()
    extra_payload["absolute_assessments"].append(  # type: ignore[union-attr]
        {
            "criterion": other_extra[0].value,
            "subject_id": other_extra[1],
            "candidate_a": {
                "verdict": "pass",
                "evidence": "Unexpected non-role unit A.",
            },
            "candidate_b": {
                "verdict": "pass",
                "evidence": "Unexpected non-role unit B.",
            },
        }
    )
    unexpected = HybridJudgeResult.model_validate(extra_payload)
    unchanged, removed = unexpected.discard_redundant_absolute_units(
        expected=expected,
        redundant=redundant,
    )
    assert unchanged is unexpected
    assert removed == ()

    one_role_payload = _hybrid_payload()
    one_role_payload["absolute_assessments"] = [  # type: ignore[index]
        item
        for item in one_role_payload["absolute_assessments"]  # type: ignore[union-attr]
        if item["criterion"] != "sink_roles_consistent"
    ]
    one_role = HybridJudgeResult.model_validate(one_role_payload)
    unchanged, removed = one_role.discard_redundant_absolute_units(
        expected=expected,
        redundant=redundant,
    )
    assert unchanged is one_role
    assert removed == ()


def test_mock_atomic_hybrid_logs_redundant_role_unit_repair() -> None:
    payload = _hybrid_payload()
    registry = extract_public_requirements(_prompt())
    expected = set(
        semantic_absolute_units(registry, include_role_consistency=False)
    )
    redundant = {
        (AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, "candidate"),
        (AbsoluteCriterion.SINK_ROLES_CONSISTENT, "candidate"),
    }
    client = MockLLMClient(hybrid_responses=[payload])

    result = client.assess_hybrid(
        system_prompt="system",
        user_prompt="user",
        expected_absolute_units=expected,
        redundant_absolute_units=redundant,
    )

    assert client.calls[0]["role"] == "hybrid_judge_atomic_repair_v1"
    assert len(result.parsed.absolute_assessments) == len(expected)
    repair = result.raw_response["_autoformalism_contract_repair"]
    assert repair == {
        "redundant_absolute_units_removed": [
            "sink_roles_consistent:candidate",
            "source_roles_consistent:candidate",
        ],
        "redundant_absolute_unit_repair_count": 2,
    }


def test_mock_hybrid_client_enforces_requested_units() -> None:
    registry = extract_public_requirements(_prompt())
    payload = _hybrid_payload()
    client = MockLLMClient(hybrid_responses=[payload])

    result = client.assess_hybrid(
        system_prompt="system",
        user_prompt="user",
        expected_absolute_units=set(semantic_absolute_units(registry)),
    )

    assert result.parsed.schema_version == "hybrid-1"
    assert client.calls[0]["role"] == "hybrid_judge"


def test_hybrid_schema_rejects_missing_comparative_criterion() -> None:
    payload = _hybrid_payload()
    payload["comparative_assessments"].pop()  # type: ignore[union-attr]

    with pytest.raises(ValidationError):
        HybridJudgeResult.model_validate(payload)


def test_conjunction_hard_requirement_and_partial_score_are_separate() -> None:
    registry = extract_public_requirements(_prompt())
    result = HybridJudgeResult.model_validate(_hybrid_payload())
    deterministic = deterministic_pair_assessments(
        _candidate(),
        _candidate(),
        task_inputs=("meal_event_g", "insulin_input"),
    )

    score = score_hybrid_pair(result, deterministic, registry)

    assert score.candidate_a.hard_requirement_status is True
    assert score.candidate_b.hard_requirement_status is False
    assert score.preferred == "candidate_a"
    assert score.candidate_b.partial_score is not None
    assert score.candidate_b.partial_score > 0.0


def test_direct_comparison_is_a_separate_configurable_residual() -> None:
    registry = extract_public_requirements(_prompt())
    result = HybridJudgeResult.model_validate(
        _hybrid_payload(
            requirement_a="pass",
            requirement_b="pass",
            relative="candidate_b",
        )
    )
    deterministic = deterministic_pair_assessments(
        _candidate(),
        _candidate(),
        task_inputs=("meal_event_g", "insulin_input"),
    )

    without_relative = score_hybrid_pair(
        result,
        deterministic,
        registry,
        HybridScoringConfig(comparative_weight=0.0),
    )
    with_relative = score_hybrid_pair(
        result,
        deterministic,
        registry,
        HybridScoringConfig(comparative_weight=0.25),
    )

    assert without_relative.preferred == "tie"
    assert with_relative.preferred == "candidate_b"


def test_fixed_denominator_treats_indeterminate_comparisons_as_neutral() -> None:
    payload = _hybrid_payload(
        requirement_a="pass",
        requirement_b="pass",
        relative="indeterminate",
    )
    comparative = payload["comparative_assessments"]
    assert isinstance(comparative, list)
    comparative[0]["verdict"] = "candidate_a"
    result = HybridJudgeResult.model_validate(payload)
    deterministic = deterministic_pair_assessments(
        _candidate(),
        _candidate(),
        task_inputs=("meal_event_g", "insulin_input"),
    )
    registry = extract_public_requirements(_prompt())

    legacy = score_hybrid_pair(
        result,
        deterministic,
        registry,
        HybridScoringConfig(comparative_indeterminate_policy="exclude"),
    )
    fixed = score_hybrid_pair(
        result,
        deterministic,
        registry,
        HybridScoringConfig(
            comparative_indeterminate_policy="neutral_fixed_denominator"
        ),
    )

    assert legacy.relative_preference_for_a == pytest.approx(1.0)
    assert legacy.decision_value == pytest.approx(0.25)
    assert fixed.relative_preference_for_a == pytest.approx(2.0 / 3.0)
    assert fixed.decision_value == pytest.approx(1.0 / 12.0)


def test_label_template_combines_runtime_and_mutation_contract_labels() -> None:
    pair = AdversarialPair(
        pair_id="pair_1",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="retained_disconnected_claimed_mechanism",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )

    labels = build_label_template(
        pair,
        public_prompt=_prompt(),
        task_inputs=("meal_event_g", "insulin_input"),
    )

    runtime = {
        item.criterion: item
        for item in labels.absolute_labels
        if item.label_source == "deterministic_runtime"
    }
    assert runtime[AbsoluteCriterion.CLAIMED_COMPONENTS_REACH_TARGETS].baseline is (
        ExpectedVerdict.PASS
    )
    assert runtime[AbsoluteCriterion.CLAIMED_COMPONENTS_REACH_TARGETS].mutated is (
        ExpectedVerdict.FAIL
    )
    assert labels.schema_version == "hybrid-labels-2"
    assert all(
        "domain_expert" not in item.label_source
        for item in labels.absolute_labels
    )
    relative = {item.criterion: item for item in labels.comparative_labels}
    assert relative[
        RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT
    ].preference is ExpectedPairPreference.BASELINE
    assert relative[
        RelativeCriterion.MECHANISTIC_INTERPRETABILITY
    ].preference is ExpectedPairPreference.UNLABELED


@pytest.mark.parametrize(
    ("mutation_type", "absolute_criterion", "comparative_count"),
    (
        ("wrong_meal_sink", AbsoluteCriterion.SOURCE_ROLES_CONSISTENT, 1),
        (
            "duplicated_gp_flux",
            AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
            2,
        ),
        (
            "unjustified_one_sided_accumulator",
            AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED,
            2,
        ),
        ("retained_disconnected_claimed_mechanism", None, 2),
    ),
)
def test_mutation_contracts_label_only_guaranteed_questions(
    mutation_type: str,
    absolute_criterion: AbsoluteCriterion | None,
    comparative_count: int,
) -> None:
    contract = mutation_label_contract(mutation_type)

    assert contract.overall_preference is ExpectedPairPreference.BASELINE
    assert len(contract.comparative) == comparative_count
    assert (
        contract.absolute[0].criterion if contract.absolute else None
    ) is absolute_criterion
    assert all(
        item.mutated is ExpectedVerdict.FAIL for item in contract.absolute
    )


def test_unknown_mutation_has_no_inferred_scientific_labels() -> None:
    with pytest.raises(ValueError, match="no certified hybrid-label contract"):
        mutation_label_contract("new_unreviewed_mutation")


def test_hybrid_analysis_uses_certified_question_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = AdversarialPair(
        pair_id="pair_1",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="wrong_meal_sink",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )
    labels = build_label_template(
        pair,
        public_prompt=_prompt(),
        task_inputs=("meal_event_g", "insulin_input"),
    )
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(labels.model_dump_json() + "\n", encoding="utf-8")
    result = HybridJudgeResult.model_validate(
        _hybrid_payload(candidate_b="fail", relative="candidate_a")
    )
    deterministic = deterministic_pair_assessments(
        pair.valid_candidate,
        pair.adversarial_candidate,
        task_inputs=("meal_event_g", "insulin_input"),
    )
    score_path = tmp_path / "scores.csv"
    fields = (
        "pair_id",
        "judge_model",
        "repetition",
        "order",
        "baseline_position",
        "baseline_preference",
        "baseline_decision_value",
        "baseline_relative_preference",
        "candidate_a_score",
        "candidate_b_score",
        "deterministic_assessments",
        "absolute_assessments",
        "comparative_assessments",
    )
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "pair_1",
                "judge_model": "ollama:test",
                "repetition": 0,
                "order": "baseline_a",
                "baseline_position": "A",
                "baseline_preference": "baseline",
                "baseline_decision_value": 0.5,
                "baseline_relative_preference": 0.5,
                "candidate_a_score": 0.8,
                "candidate_b_score": 0.3,
                "deterministic_assessments": json.dumps(
                    [item.model_dump(mode="json") for item in deterministic]
                ),
                "absolute_assessments": json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in result.absolute_assessments
                    ]
                ),
                "comparative_assessments": json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in result.comparative_assessments
                    ]
                ),
            }
        )
    output = tmp_path / "metrics.json"
    failures_path = tmp_path / "failures.jsonl"
    _append_failure(
        failures_path,
        {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "pair_id": "pair_1",
            "benchmark_id": "phase_b_test",
            "tier": "easy",
            "mutation_type": "wrong_meal_sink",
            "judge_model": "ollama:test",
            "repetition": 1,
            "order": "baseline_b",
            "baseline_position": "B",
            "error_type": "LLMResponseError",
            "error": "empty final content",
            "failure_category": "repairable_contract",
            "provider_attempt_limit": 10,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--scores",
            str(score_path),
            "--failures",
            str(failures_path),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
        ],
    )

    analyze_hybrid_main()

    metrics = json.loads(output.read_text(encoding="utf-8"))["ollama:test"]
    assert metrics["combined_preference_accuracy"] == 1.0
    assert metrics["structured_response_success_rate"] == 0.5
    assert metrics["combined_preference_accuracy_conditional_on_response"] == 1.0
    assert metrics["combined_preference_accuracy_including_failures"] == 0.5
    assert metrics["failed_comparison_count"] == 1
    assert metrics["failures_by_error_type"] == {"LLMResponseError": 1}
    assert metrics["responses_by_transport"] == {"unrecorded": 1}
    assert metrics["runtime_certification_accuracy"] == 1.0
    assert metrics["absolute_verdict_accuracy"] == 1.0
    assert metrics["gold_label_scope"] == "runtime_and_mutation_contract_only"
    assert metrics["expert_review_used"] is False
    assert metrics["scored_absolute_label_fraction"] < 1.0


def test_canonicalization_turns_unreferenced_stress_pair_into_tie() -> None:
    pair = AdversarialPair(
        pair_id="pair_1",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="disconnected_claimed_mechanism",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )
    context = ValidationContext(
        targets=("Gp", "U"),
        external_inputs=("meal_event_g", "insulin_input"),
    )

    labels = build_label_template(
        pair,
        public_prompt=_prompt(),
        task_inputs=context.external_inputs,
        validation_context=context,
    )

    assert labels.overall_preference.value == "tie"


def test_augmented_disconnection_survives_canonicalization() -> None:
    pair = AdversarialPair(
        pair_id="pair_1",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="disconnected_claimed_mechanism",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )
    retained = augment_pairs((pair,))[1]
    context = ValidationContext(
        targets=("Gp", "U"),
        external_inputs=("meal_event_g", "insulin_input"),
    )

    canonical, repairs = repair_protected_declarations(
        retained.adversarial_candidate, context
    )
    facts = structural_facts(
        canonical,
        task_inputs=context.external_inputs,
    )

    assert not any("claimed_pathway" in repair for repair in repairs)
    assert facts["components"]["claimed_pathway"][
        "reaches_requested_target"
    ] is False
    assert facts["components"]["claimed_pathway_memory"][
        "reaches_requested_target"
    ] is False


def test_hybrid_merge_requires_complete_unique_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "shard.csv"
    with shard.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_id",
                "judge_model",
                "repetition",
                "order",
                "baseline_preference",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "pair_1",
                "judge_model": "ollama:test",
                "repetition": 0,
                "order": "baseline_a",
                "baseline_preference": "baseline",
            }
        )
    output = tmp_path / "merged.csv"
    failure_shard = tmp_path / "failures.jsonl"
    _append_failure(
        failure_shard,
        {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "pair_id": "pair_1",
            "benchmark_id": "phase_b_test",
            "tier": "easy",
            "mutation_type": "wrong_meal_sink",
            "judge_model": "ollama:test",
            "repetition": 0,
            "order": "baseline_b",
            "baseline_position": "B",
            "error_type": "LLMResponseError",
            "error": "empty final content",
            "failure_category": "repairable_contract",
            "provider_attempt_limit": 10,
        },
    )
    merged_failures = tmp_path / "merged_failures.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--inputs",
            str(shard),
            "--output",
            str(output),
            "--failure-inputs",
            str(failure_shard),
            "--failure-output",
            str(merged_failures),
            "--expected",
            "2",
        ],
    )

    merge_hybrid_main()

    assert output.is_file()
    assert len(merged_failures.read_text(encoding="utf-8").splitlines()) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--inputs",
            str(shard),
            "--output",
            str(output),
            "--failure-inputs",
            str(failure_shard),
            "--expected",
            "3",
        ],
    )
    with pytest.raises(SystemExit, match="incomplete hybrid outcomes"):
        merge_hybrid_main()


def test_failure_ledger_is_resumable_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failures.jsonl"
    row = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "pair_id": "pair_1",
        "benchmark_id": "phase_b_test",
        "tier": "easy",
        "mutation_type": "wrong_meal_sink",
        "judge_model": "ollama:test",
        "repetition": 2,
        "order": "baseline_a",
        "baseline_position": "A",
        "error_type": "LLMResponseError",
        "error": "empty final content",
        "failure_category": "repairable_contract",
        "provider_attempt_limit": 10,
    }

    _append_failure(path, row)

    assert _failed(path) == {("pair_1", "ollama:test", 2, "baseline_a")}
    _append_failure(path, row)
    with pytest.raises(ValueError, match="duplicate persistent-failure key"):
        _failed(path)


def test_hybrid_resume_manifest_and_planned_keys_reject_configuration_drift(
    tmp_path: Path,
) -> None:
    pair = AdversarialPair(
        pair_id="pair_1",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="wrong_meal_sink",
        valid_candidate=_candidate(),
        adversarial_candidate=_candidate(disconnected_claim=True),
    )
    keys = _planned_keys(
        (pair,), judge_models=["ollama:test"], repetitions=2
    )
    path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "hybrid-judge-run-1",
        "pairs_sha256": "abc",
        "selected_pair_ids": ["pair_1"],
    }

    assert len(keys) == 4
    assert {key[3] for key in keys} == {"baseline_a", "baseline_b"}
    _ensure_run_manifest(path, manifest)
    _ensure_run_manifest(path, manifest)
    with pytest.raises(ValueError, match="configuration differs"):
        _ensure_run_manifest(path, {**manifest, "pairs_sha256": "changed"})
