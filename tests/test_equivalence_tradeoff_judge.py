"""Tests for equivalence/tie and non-ordered tradeoff judge development."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedAbsoluteLabel,
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
    mutation_label_contract,
)
from autoformalism.schemas import CandidateModel
from scripts.analyze_equivalence_tradeoff_judge import analyze
from scripts.build_hybrid_judge_equivalence_tradeoff_pairs import (
    PAIR_TYPES,
    build_equivalence_tradeoff_pairs,
)
from scripts.build_hybrid_judge_label_template import build_label_template

CONFIG = Path("configs/hybrid_judge_equivalence_tradeoff_v1.json")
SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_equivalence_tradeoff_120b.slurm"
)


def _candidate(parameter: str) -> CandidateModel:
    utilization = {
        "k": "k * X * Gp",
        "q": "q * X * Gp**2",
    }[parameter]
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


def _source_pair(parameter: str) -> AdversarialPair:
    candidate = _candidate(parameter)
    return AdversarialPair(
        pair_id=f"source_{parameter}",
        benchmark_id="phase_b_test",
        tier="easy",
        mutation_type="wrong_meal_sink",
        valid_candidate=candidate,
        adversarial_candidate=candidate,
    )


def test_pair_builder_creates_equivalence_and_unordered_tradeoffs() -> None:
    context = ValidationContext(
        targets=("Gp", "U"),
        auxiliaries=("EGP",),
        external_inputs=("meal_event_g", "insulin_input"),
    )

    pairs, fingerprints = build_equivalence_tradeoff_pairs(
        (_source_pair("k"), _source_pair("q")),
        contexts={("phase_b_test", "easy"): context},
        baseline_count=2,
    )

    assert len(fingerprints) == 2
    assert len(pairs) == 8
    assert [pair.mutation_type for pair in pairs[:4]] == list(PAIR_TYPES)
    assert len({pair.pair_id for pair in pairs}) == 8
    equivalent = pairs[0]
    assert equivalent.valid_candidate.state_equations[0].rhs != (
        equivalent.adversarial_candidate.state_equations[0].rhs
    )

    public_prompt = """Requested mechanisms:
- Meal input contributes to glucose balance.
- Insulin-dependent disposal contributes to glucose balance.
"""
    equivalent_labels = build_label_template(
        equivalent,
        public_prompt=public_prompt,
        task_inputs=context.external_inputs,
        validation_context=context,
    )
    tradeoff_labels = build_label_template(
        pairs[1],
        public_prompt=public_prompt,
        task_inputs=context.external_inputs,
        validation_context=context,
    )
    assert equivalent_labels.overall_preference is ExpectedPairPreference.TIE
    assert all(
        item.preference is ExpectedPairPreference.TIE
        for item in equivalent_labels.comparative_labels
    )
    assert (
        tradeoff_labels.overall_preference
        is ExpectedPairPreference.UNLABELED
    )


def test_new_label_contracts_do_not_invent_tradeoff_winners() -> None:
    equivalent = mutation_label_contract("algebraic_reordering_equivalent")
    assert equivalent.overall_preference is ExpectedPairPreference.TIE
    assert all(
        item.preference is ExpectedPairPreference.TIE
        for item in equivalent.comparative
    )

    for pair_type in PAIR_TYPES[1:]:
        contract = mutation_label_contract(pair_type)
        assert contract.overall_preference is ExpectedPairPreference.UNLABELED
        assert contract.comparative == ()
        assert len(contract.absolute) == 2
        assert sum(
            verdict is ExpectedVerdict.FAIL
            for item in contract.absolute
            for verdict in (item.baseline, item.mutated)
        ) == 2


def _label(pair_id: str, pair_type: str) -> HybridCalibrationLabels:
    contract = mutation_label_contract(pair_type)
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
                label_source=f"mutation_contract:{pair_type}",
            )
            for item in contract.absolute
        ),
        comparative_labels=(),
    )


def _row(
    *,
    pair_id: str,
    pair_type: str,
    label: HybridCalibrationLabels,
    order: str,
    repetition: int,
) -> dict[str, str]:
    baseline_position = "A" if order == "baseline_a" else "B"
    assessments = []
    for item in label.absolute_labels:
        first = (
            item.baseline.value
            if item.baseline is not ExpectedVerdict.UNLABELED
            else "pass"
        )
        second = (
            item.mutated.value
            if item.mutated is not ExpectedVerdict.UNLABELED
            else "pass"
        )
        candidate_a, candidate_b = (
            (first, second) if baseline_position == "A" else (second, first)
        )
        assessments.append(
            {
                "criterion": item.criterion.value,
                "subject_id": item.subject_id,
                "candidate_a": {
                    "verdict": candidate_a,
                    "evidence": "Certified test evidence A.",
                },
                "candidate_b": {
                    "verdict": candidate_b,
                    "evidence": "Certified test evidence B.",
                },
            }
        )
    comparative = [
        {
            "criterion": criterion,
            "verdict": "tie",
            "evidence": "Equivalent or balanced test evidence.",
        }
        for criterion in (
            "parsimony_while_task_sufficient",
            "fewer_unsupported_assumptions",
            "mechanistic_interpretability",
        )
    ]
    return {
        "pair_id": pair_id,
        "mutation_type": pair_type,
        "repetition": str(repetition),
        "order": order,
        "baseline_position": baseline_position,
        "baseline_preference": "tie",
        "baseline_decision_value": "0.0",
        "absolute_assessments": json.dumps(assessments),
        "comparative_assessments": json.dumps(comparative),
        "redundant_absolute_unit_repairs": "0",
    }


def test_analyzer_scores_equivalence_and_atomic_facts_but_not_tradeoff_winner() -> None:
    equivalence = _label("equivalent", "algebraic_reordering_equivalent")
    tradeoff_type = "tradeoff_wrong_sink_vs_unjustified_accumulator"
    tradeoff = _label("tradeoff", tradeoff_type)
    rows = [
        _row(
            pair_id=pair_id,
            pair_type=pair_type,
            label=label,
            order=order,
            repetition=repetition,
        )
        for pair_id, pair_type, label in (
            ("equivalent", "algebraic_reordering_equivalent", equivalence),
            ("tradeoff", tradeoff_type, tradeoff),
        )
        for repetition in range(2)
        for order in ("baseline_a", "baseline_b")
    ]
    gate = {
        "minimum_response_success": 1.0,
        "minimum_equivalence_call_tie_accuracy": 1.0,
        "minimum_equivalence_pair_tie_accuracy": 1.0,
        "minimum_equivalence_comparative_tie_accuracy": 1.0,
        "minimum_tradeoff_targeted_absolute_accuracy": 1.0,
        "minimum_overall_order_consistency": 1.0,
        "maximum_tradeoff_orientation_bias": 0.0,
        "maximum_mean_repeat_decision_sd": 0.0,
    }

    result = analyze(
        rows,
        [],
        {"equivalent": equivalence, "tradeoff": tradeoff},
        tie_threshold=0.05,
        gate=gate,
    )

    assert result["passed"] is True
    assert result["tradeoffs"]["overall_preference_truth"] == "unlabeled"
    assert result["tradeoffs"]["targeted_absolute_accuracy"] == 1.0


def test_frozen_config_and_slurm_preserve_selected_120b_protocol() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_equivalence_tradeoff_calls"
    assert payload["judge_model"] == "vllm:openai/gpt-oss-120b"
    assert payload["pair_design"]["tradeoff_overall_truth"] == "unlabeled"
    assert payload["planned"] == {
        "pairs": 8,
        "paired_judgments": 80,
        "llm_stages": 160,
        "gpus": 4,
        "tensor_parallel_size": 4,
    }
    assert payload["protocol"]["contract_repair_version"] == (
        "atomic-redundant-role-unit-repair-1"
    )
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    script = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-node=4" in script
    assert "openai/gpt-oss-120b" in script
    assert ": \"${AF_PAIR_IDS:=}\"" in script
