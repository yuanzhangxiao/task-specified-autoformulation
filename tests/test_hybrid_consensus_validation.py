"""Tests for the frozen fresh-structure question-consensus validation."""

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
from scripts.analyze_hybrid_consensus_validation import evaluate_validation
from scripts.analyze_hybrid_symmetric_aggregation import (
    RULE_QUESTION_CONSENSUS,
)
from scripts.build_hybrid_judge_consensus_validation_pairs import (
    KNOWN_DOMINANCE_TYPES,
    PAIR_TYPES,
    build_consensus_validation_pairs,
    select_unseen_baselines,
)
from scripts.build_hybrid_judge_heldout_pairs import (
    candidate_structure_fingerprint,
)

CONFIG = Path("configs/hybrid_judge_consensus_validation_v1.json")
SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_consensus_validation_120b.slurm"
)


def _candidate(parameter: str) -> CandidateModel:
    utilization = {"k": "k * X * Gp", "q": "q * X * Gp**2"}[parameter]
    return CandidateModel.model_validate(
        {
            "candidate_id": f"candidate_{parameter}",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "X", "kind": "latent"},
            ],
            "processes": [
                {
                    "name": "U",
                    "expression": utilization,
                    "mechanisms": ["InsulinDisposal"],
                }
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "meal_event_g - U - EGP"},
                {"state": "X", "rhs": f"insulin_input - {parameter} * X"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
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
                {"state": "Gp", "scope": "global", "fixed_value": 1.0},
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
        targets=("Gp", "U"),
        auxiliaries=("EGP",),
        external_inputs=("meal_event_g", "insulin_input"),
    )


def test_builder_selects_unseen_structures_and_creates_all_pair_types(
    context: ValidationContext,
) -> None:
    first = _source_pair("first", _candidate("k"))
    second = _source_pair("second", _candidate("q"))
    contexts = {("phase_b_test", "easy"): context}

    selected, excluded_count = select_unseen_baselines(
        (first, second),
        (first,),
        baseline_count=1,
        contexts=contexts,
    )
    pairs = build_consensus_validation_pairs(selected, contexts=contexts)

    assert excluded_count == 1
    assert selected[0][0] == candidate_structure_fingerprint(
        second.valid_candidate,
        context,
    )
    assert [pair.mutation_type for pair in pairs] == list(PAIR_TYPES)
    assert len({pair.pair_id for pair in pairs}) == len(PAIR_TYPES)
    wrong_plus = next(
        pair
        for pair in pairs
        if pair.mutation_type == "additional_accumulator_on_wrong_sink"
    )
    assert len(wrong_plus.adversarial_candidate.states) == (
        len(wrong_plus.valid_candidate.states) + 1
    )
    assert "abs(meal_event_g)" in (
        wrong_plus.valid_candidate.state_equations[0].rhs
    )
    assert "abs(meal_event_g)" in (
        wrong_plus.adversarial_candidate.state_equations[0].rhs
    )


def test_builder_refuses_to_reuse_opened_structure(
    context: ValidationContext,
) -> None:
    pair = _source_pair("opened", _candidate("k"))

    with pytest.raises(ValueError, match="found 0"):
        select_unseen_baselines(
            (pair,),
            (pair,),
            baseline_count=1,
            contexts={("phase_b_test", "easy"): context},
        )


@pytest.mark.parametrize(
    "mutation_type",
    (
        "additional_accumulator_on_wrong_sink",
        "additional_accumulator_on_duplicate",
    ),
)
def test_monotonic_defect_contracts_label_only_the_added_defect(
    mutation_type: str,
) -> None:
    contract = mutation_label_contract(mutation_type)

    assert contract.overall_preference is ExpectedPairPreference.BASELINE
    assert [item.criterion for item in contract.absolute] == [
        AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED
    ]
    assert contract.absolute[0].baseline is ExpectedVerdict.UNLABELED
    assert contract.absolute[0].mutated is ExpectedVerdict.FAIL
    assert len(contract.comparative) == 2
    assert all(
        item.preference is ExpectedPairPreference.BASELINE
        for item in contract.comparative
    )


def _labels(pair_id: str, mutation_type: str) -> HybridCalibrationLabels:
    contract = mutation_label_contract(mutation_type)
    return HybridCalibrationLabels(
        pair_id=pair_id,
        overall_preference=contract.overall_preference,
        absolute_labels=tuple(
            ExpectedAbsoluteLabel(
                criterion=item.criterion,
                subject_id=item.subject_id,
                baseline=item.baseline,
                mutated=item.mutated,
                rationale=item.rationale,
                label_source=f"mutation_contract:{mutation_type}",
            )
            for item in contract.absolute
        ),
        comparative_labels=(),
    )


def _absolute(
    criterion: AbsoluteCriterion,
) -> dict[str, object]:
    return PairedAbsoluteAssessment(
        criterion=criterion,
        subject_id="candidate",
        candidate_a=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.PASS,
            evidence="Baseline passes the controlled atomic question.",
        ),
        candidate_b=CandidateAbsoluteAssessment(
            verdict=AbsoluteVerdict.FAIL,
            evidence="Mutation fails the controlled atomic question.",
        ),
    ).model_dump(mode="json")


def _trial(
    pair_id: str,
    mutation_type: str,
    preference: ExpectedPairPreference,
    criterion: AbsoluteCriterion | None = None,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "mutation_type": mutation_type,
        "orientation_half_gap": 0.0,
        "comparative_disagreements": [],
        "consensus_absolute_assessments": (
            [] if criterion is None else [_absolute(criterion)]
        ),
        "rules": {
            RULE_QUESTION_CONSENSUS: {
                "decision": {
                    ExpectedPairPreference.BASELINE: 0.25,
                    ExpectedPairPreference.MUTATED: -0.25,
                    ExpectedPairPreference.TIE: 0.0,
                    ExpectedPairPreference.UNLABELED: 0.0,
                }[preference],
                "preference": preference.value,
            }
        },
    }


def test_validation_analyzer_applies_gates_without_scoring_tradeoff() -> None:
    specifications = (
        ("tie", "algebraic_reordering_equivalent", None),
        ("sink", "wrong_meal_sink", AbsoluteCriterion.SOURCE_ROLES_CONSISTENT),
        (
            "duplicate",
            "duplicated_gp_flux",
            AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
        ),
        (
            "accumulator",
            "unjustified_one_sided_accumulator",
            AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED,
        ),
        ("tradeoff", "tradeoff_wrong_sink_vs_duplicate", None),
    )
    labels = {
        pair_id: _labels(pair_id, mutation_type)
        for pair_id, mutation_type, _criterion in specifications
    }
    trials = [
        _trial(
            pair_id,
            mutation_type,
            (
                ExpectedPairPreference.BASELINE
                if labels[pair_id].overall_preference
                is ExpectedPairPreference.UNLABELED
                else labels[pair_id].overall_preference
            ),
            criterion,
        )
        for pair_id, mutation_type, criterion in specifications
    ]
    symmetric = {
        "paired_response_coverage": 1.0,
        "rules": {
            RULE_QUESTION_CONSENSUS: {
                "labeled_decision_coverage": 1.0,
                "labeled_accuracy_conditional": 1.0,
                "equivalence_tie_rate": 1.0,
                "mean_pair_modal_preference_consistency": 1.0,
                "mean_repeat_decision_sd": 0.0,
            }
        },
        "trials": trials,
    }
    gates = {
        "minimum_response_success": 1.0,
        "minimum_paired_response_coverage": 1.0,
        "minimum_labeled_decision_coverage": 1.0,
        "minimum_labeled_accuracy": 1.0,
        "minimum_equivalence_tie_accuracy": 1.0,
        "minimum_known_dominance_accuracy": 1.0,
        "minimum_known_dominance_pair_accuracy": 1.0,
        "minimum_wrong_sink_absolute_accuracy": 1.0,
        "minimum_duplicate_absolute_accuracy": 1.0,
        "minimum_accumulator_absolute_accuracy": 1.0,
        "minimum_modal_preference_consistency": 1.0,
        "maximum_mean_repeat_decision_sd": 0.0,
        "maximum_mean_orientation_half_gap": 0.0,
        "maximum_equivalence_mean_orientation_half_gap": 0.0,
        "maximum_comparative_disagreement_rate": 0.0,
    }

    result = evaluate_validation(
        symmetric,
        labels,
        response_success=1.0,
        tie_threshold=0.05,
        gates=gates,
    )

    assert result["passed"] is True
    assert result["known_dominance_trials"] == 3
    assert result["unlabeled_tradeoff_trials"] == 1
    assert result["unlabeled_tradeoff_preference_counts"] == {"baseline": 1}


def test_frozen_config_and_slurm_preserve_consensus_contract() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen_before_consensus_validation_calls"
    assert payload["judge_model"] == "vllm:openai/gpt-oss-120b"
    assert payload["aggregation"]["primary_rule"] == (
        RULE_QUESTION_CONSENSUS
    )
    assert payload["aggregation"]["automatic_uncertainty_abstention"] is False
    assert payload["pair_construction"]["known_dominance_types"] == list(
        KNOWN_DOMINANCE_TYPES
    )
    assert payload["planned"] == {
        "baseline_structures": 2,
        "pairs": 14,
        "paired_judgments": 140,
        "llm_stages": 280,
        "gpus": 4,
        "tensor_parallel_size": 4,
    }
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    script = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-node=4" in script
    assert "openai/gpt-oss-120b" in script
    assert ": \"${AF_PAIR_IDS:=}\"" in script
