"""Tests for versioned target-mapping and initialization judge validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedAbsoluteLabel,
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
    mutation_label_contract,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    CandidateModel,
    PairedAbsoluteAssessment,
)
from scripts.analyze_hybrid_model_semantics_validation import (
    evaluate_model_semantics,
)
from scripts.analyze_hybrid_symmetric_aggregation import (
    RULE_QUESTION_CONSENSUS,
)
from scripts.build_hybrid_judge_model_semantics_pairs import (
    PAIR_TYPES,
    build_model_semantics_pairs,
    select_unseen_semantic_baselines,
)

CONFIG = Path("configs/hybrid_judge_model_semantics_validation_v1.json")
SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_model_semantics_120b.slurm"
)


def _candidate(parameter: str) -> CandidateModel:
    utilization = {"k": "k * X * Gp", "q": "q * X * Gp**2"}[parameter]
    return CandidateModel.model_validate(
        {
            "candidate_id": f"candidate_{parameter}",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
                {"name": "X", "kind": "latent"},
            ],
            "processes": [
                {
                    "name": "U",
                    "expression": utilization,
                    "mechanisms": ["insulin_dependent_disposal"],
                }
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "meal_event_g + EGP - Uii - U"},
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
                    "name": parameter,
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


def _source_pair(pair_id: str, candidate: CandidateModel) -> AdversarialPair:
    return AdversarialPair(
        pair_id=pair_id,
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="wrong_meal_sink",
        valid_candidate=candidate,
        adversarial_candidate=candidate,
    )


@pytest.fixture
def context() -> ValidationContext:
    return ValidationContext(
        targets=("Gp", "I", "U"),
        auxiliaries=("EGP", "Uii"),
        external_inputs=("meal_event_g", "insulin_input"),
    )


def test_builder_isolates_mapping_and_initialization_mutations(
    context: ValidationContext,
) -> None:
    first = _source_pair("first", _candidate("k"))
    second = _source_pair("second", _candidate("q"))
    contexts = {("phase_b_test", "easy"): context}

    selected, excluded_count = select_unseen_semantic_baselines(
        (first, second),
        (first,),
        baseline_count=1,
        contexts=contexts,
        target_channel="U",
        target_component="U",
        observed_state="I",
    )
    pairs = build_model_semantics_pairs(
        selected,
        contexts=contexts,
        target_channel="U",
        target_component="U",
        complete_target_expression="Uii + U",
        observed_state="I",
    )

    assert excluded_count == 1
    assert [pair.mutation_type for pair in pairs] == list(PAIR_TYPES)
    mapping_pair, initialization_pair = pairs
    baseline_mapping = next(
        item
        for item in mapping_pair.valid_candidate.observation_mappings
        if item.channel == "U"
    )
    omitted_mapping = next(
        item
        for item in mapping_pair.adversarial_candidate.observation_mappings
        if item.channel == "U"
    )
    assert baseline_mapping.expression == "Uii + U"
    assert omitted_mapping.expression == "U"
    baseline_initial = next(
        item
        for item in initialization_pair.valid_candidate.initial_conditions
        if item.state == "I"
    )
    zero_initial = next(
        item
        for item in initialization_pair.adversarial_candidate.initial_conditions
        if item.state == "I"
    )
    assert baseline_initial.expression == "I"
    assert zero_initial.fixed_value == 0.0
    assert zero_initial.expression is None


@pytest.mark.parametrize(
    ("mutation_type", "criterion"),
    (
        (
            "omitted_target_component",
            AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT,
        ),
        (
            "unjustified_zero_observed_initialization",
            AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT,
        ),
    ),
)
def test_mutation_contract_targets_new_absolute_question(
    mutation_type: str,
    criterion: AbsoluteCriterion,
) -> None:
    contract = mutation_label_contract(mutation_type)

    assert contract.overall_preference is ExpectedPairPreference.BASELINE
    assert len(contract.absolute) == 1
    assert contract.absolute[0].criterion is criterion
    assert contract.absolute[0].baseline is ExpectedVerdict.PASS
    assert contract.absolute[0].mutated is ExpectedVerdict.FAIL


def _labels(
    pair_id: str,
    mutation_type: str,
    criterion: AbsoluteCriterion,
) -> HybridCalibrationLabels:
    return HybridCalibrationLabels(
        pair_id=pair_id,
        overall_preference=ExpectedPairPreference.BASELINE,
        absolute_labels=(
            ExpectedAbsoluteLabel(
                criterion=criterion,
                subject_id="candidate",
                baseline=ExpectedVerdict.PASS,
                mutated=ExpectedVerdict.FAIL,
                rationale="Controlled mutation contract.",
                label_source=f"mutation_contract:{mutation_type}",
            ),
        ),
        comparative_labels=(),
    )


def _trial(
    pair_id: str,
    mutation_type: str,
    criterion: AbsoluteCriterion,
) -> dict[str, object]:
    absolute = PairedAbsoluteAssessment(
        criterion=criterion,
        subject_id="candidate",
        candidate_a=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.PASS,
            evidence="Baseline follows the public contract.",
        ),
        candidate_b=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.FAIL,
            evidence="Mutation violates the public contract.",
        ),
    ).model_dump(mode="json")
    return {
        "pair_id": pair_id,
        "mutation_type": mutation_type,
        "orientation_half_gap": 0.0,
        "comparative_disagreements": [],
        "consensus_absolute_assessments": [absolute],
        "rules": {
            RULE_QUESTION_CONSENSUS: {
                "decision": 0.25,
                "preference": "baseline",
            }
        },
    }


def test_frozen_gate_accepts_correct_consensus_answers() -> None:
    labels = {
        "mapping": _labels(
            "mapping",
            "omitted_target_component",
            AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT,
        ),
        "initial": _labels(
            "initial",
            "unjustified_zero_observed_initialization",
            AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT,
        ),
    }
    trials = [
        _trial(
            "mapping",
            "omitted_target_component",
            AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT,
        ),
        _trial(
            "initial",
            "unjustified_zero_observed_initialization",
            AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT,
        ),
    ]
    symmetric = {
        "trials": trials,
        "paired_response_coverage": 1.0,
        "rules": {
            RULE_QUESTION_CONSENSUS: {
                "labeled_decision_coverage": 1.0,
                "labeled_accuracy_conditional": 1.0,
                "mean_pair_modal_preference_consistency": 1.0,
                "mean_repeat_decision_sd": 0.0,
            }
        },
    }
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    result = evaluate_model_semantics(
        symmetric,
        labels,
        response_success=1.0,
        tie_threshold=0.05,
        gates=config["validation_gate"],
    )

    assert result["passed"] is True
    assert all(item["passed"] for item in result["checks"].values())


def test_launcher_and_config_freeze_versioned_protocol() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    launcher = SLURM.read_text(encoding="utf-8")

    assert config["status"] == "frozen_before_model_semantics_validation_calls"
    assert config["pair_construction"]["mutation_labels_visible_to_judge"] is False
    assert config["pair_construction"]["hidden_generator_visible_to_judge"] is False
    assert config["protocol"]["scoring"]["comparative_indeterminate_policy"] == (
        "neutral_fixed_denominator"
    )
    assert config["protocol"]["scoring"]["include_model_semantics"] is True
    assert "AF_MODEL_SEMANTIC_CONTRACT:=true" in launcher
    assert "AF_COMPARATIVE_INDETERMINATE_POLICY:=neutral_fixed_denominator" in (
        launcher
    )
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
