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
from scripts.build_hybrid_judge_target_mapping_clean_names import (
    build_clean_target_mapping_pairs,
    certify_clean_pair,
)
from scripts.build_hybrid_judge_target_mapping_pairs import (
    build_target_mapping_pairs,
    certify_pair,
    select_baselines,
)

CONFIG = Path("configs/hybrid_judge_model_semantics_validation_v1.json")
SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_model_semantics_120b.slurm"
)
V2_CONFIG = Path("configs/hybrid_judge_target_mapping_validation_v2.json")
V2_SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v2_120b.slurm"
)
V4_CONFIG = Path("configs/hybrid_judge_target_mapping_clean_names_v4.json")
V4_SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_target_mapping_v4_clean_names_120b.slurm"
)
V1_ADJUDICATION = Path(
    "configs/hybrid_judge_model_semantics_validation_v1_adjudication.json"
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


def test_builder_can_record_previously_opened_development_baseline(
    context: ValidationContext,
) -> None:
    opened = _source_pair("opened", _candidate("k"))
    contexts = {("phase_b_test", "easy"): context}

    with pytest.raises(ValueError, match="found 0"):
        select_unseen_semantic_baselines(
            (opened,),
            (opened,),
            baseline_count=1,
            contexts=contexts,
            target_channel="U",
            target_component="U",
            observed_state="I",
        )
    selected, excluded_count = select_unseen_semantic_baselines(
        (opened,),
        (opened,),
        baseline_count=1,
        contexts=contexts,
        target_channel="U",
        target_component="U",
        observed_state="I",
        allow_opened_baselines=True,
    )

    assert len(selected) == 1
    assert excluded_count == 1


def test_v2_builder_certifies_target_process_before_mapping_edit(
    context: ValidationContext,
) -> None:
    source = _source_pair("source", _candidate("k"))
    contexts = {("phase_b_test", "easy"): context}
    selected, excluded = select_baselines(
        (source,),
        (source,),
        baseline_count=1,
        contexts=contexts,
        target_channel="U",
        target_component="U",
        supplied_component="Uii",
        allow_opened_baselines=True,
    )
    pairs, certifications = build_target_mapping_pairs(
        selected,
        contexts=contexts,
        target_channel="U",
        target_component="U",
        supplied_component="Uii",
    )

    assert len(excluded) == 1
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.mutation_type == "omitted_target_component"
    assert certifications[pair.pair_id] == certify_pair(
        pair.valid_candidate,
        pair.adversarial_candidate,
        target_channel="U",
        target_component="U",
        supplied_component="Uii",
    )
    assert certifications[pair.pair_id][
        "target_process_excludes_supplied_component"
    ]


def test_v2_builder_rejects_process_that_already_contains_supplied_component(
    context: ValidationContext,
) -> None:
    payload = _candidate("k").model_dump(mode="json")
    payload["processes"][0]["expression"] = "Uii + k * X * Gp"
    invalid = _source_pair("invalid", CandidateModel.model_validate(payload))
    contexts = {("phase_b_test", "easy"): context}

    with pytest.raises(ValueError, match="found 0"):
        select_baselines(
            (invalid,),
            (),
            baseline_count=1,
            contexts=contexts,
            target_channel="U",
            target_component="U",
            supplied_component="Uii",
            allow_opened_baselines=True,
        )


def test_v4_builder_separates_total_and_dependent_process_names(
    context: ValidationContext,
) -> None:
    source = AdversarialPair(
        pair_id="source",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=_candidate("k"),
        adversarial_candidate=_candidate("k"),
    )
    pairs, certifications = build_clean_target_mapping_pairs(
        (source,),
        contexts={("phase_b_test", "easy"): context},
    )

    assert len(pairs) == 1
    pair = pairs[0]
    baseline_processes = {
        item.name: item for item in pair.valid_candidate.processes
    }
    mutated_processes = {
        item.name: item for item in pair.adversarial_candidate.processes
    }
    assert set(baseline_processes) == {"U", "Uid"}
    assert baseline_processes["U"].expression == "Uii + Uid"
    assert mutated_processes["U"].expression == "Uid"
    assert baseline_processes["Uid"].expression == "k * X * Gp"
    assert baseline_processes["Uid"].mechanisms == (
        "insulin_dependent_disposal",
    )
    assert next(
        item.expression
        for item in pair.valid_candidate.observation_mappings
        if item.channel == "U"
    ) == "U"
    assert next(
        item.rhs
        for item in pair.valid_candidate.state_equations
        if item.state == "Gp"
    ) == "meal_event_g + EGP - Uii - Uid"
    assert certifications[pair.pair_id] == certify_clean_pair(
        pair.valid_candidate,
        pair.adversarial_candidate,
        target_channel="U",
        target_process="U",
        dependent_process="Uid",
        supplied_component="Uii",
    )


def test_v4_certification_rejects_a_second_pair_difference(
    context: ValidationContext,
) -> None:
    source = AdversarialPair(
        pair_id="source",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=_candidate("k"),
        adversarial_candidate=_candidate("k"),
    )
    pairs, _ = build_clean_target_mapping_pairs(
        (source,),
        contexts={("phase_b_test", "easy"): context},
    )
    pair = pairs[0]
    payload = pair.adversarial_candidate.model_dump(mode="json")
    next(
        item for item in payload["state_equations"] if item["state"] == "Gp"
    )["rhs"] = "meal_event_g + EGP - Uii"
    changed = CandidateModel.model_validate(payload)

    with pytest.raises(ValueError, match="outside the controlled total expression"):
        certify_clean_pair(
            pair.valid_candidate,
            changed,
            target_channel="U",
            target_process="U",
            dependent_process="Uid",
            supplied_component="Uii",
        )


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
    assert config["pair_construction"]["baseline_structure_novelty_required"] is (
        False
    )
    assert config["protocol"]["scoring"]["comparative_indeterminate_policy"] == (
        "neutral_fixed_denominator"
    )
    assert config["protocol"]["scoring"]["include_model_semantics"] is True
    assert "AF_MODEL_SEMANTIC_CONTRACT:=true" in launcher
    assert "AF_COMPARATIVE_INDETERMINATE_POLICY:=neutral_fixed_denominator" in (
        launcher
    )
    subprocess.run(["bash", "-n", str(SLURM)], check=True)


def test_v2_config_launcher_and_v1_adjudication_are_explicit() -> None:
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    launcher = V2_SLURM.read_text(encoding="utf-8")
    adjudication = json.loads(V1_ADJUDICATION.read_text(encoding="utf-8"))

    assert config["status"] == "frozen_before_target_mapping_validation_calls"
    scoring = config["protocol"]["scoring"]
    assert scoring["include_target_mapping_semantics"] is True
    assert scoring["include_initialization_semantics"] is False
    assert "AF_TARGET_MAPPING_SEMANTIC_CONTRACT:=true" in launcher
    assert adjudication["status"] == "invalid_pair_construction"
    assert adjudication["result_use"] == (
        "exclude_from_protocol_accuracy_and_paper_claims"
    )
    subprocess.run(["bash", "-n", str(V2_SLURM)], check=True)


def test_v4_config_and_launcher_freeze_clean_name_experiment() -> None:
    config = json.loads(V4_CONFIG.read_text(encoding="utf-8"))
    launcher = V4_SLURM.read_text(encoding="utf-8")

    assert config["status"] == (
        "frozen_before_target_mapping_clean_names_calls"
    )
    construction = config["pair_construction"]
    assert construction["total_process"] == "U"
    assert construction["dependent_process"] == "Uid"
    assert construction["valid_total_expression"] == "Uii + Uid"
    assert construction["mutated_total_expression"] == "Uid"
    assert config["protocol"]["scoring"][
        "comparative_indeterminate_policy"
    ] == "neutral_fixed_denominator"
    assert "AF_PROJECT:=/projects/bibo/${af_user}" in launcher
    assert "build_hybrid_judge_target_mapping_clean_names.py" in launcher
    assert "AF_TARGET_MAPPING_SEMANTIC_CONTRACT:=true" in launcher
    subprocess.run(["bash", "-n", str(V4_SLURM)], check=True)
