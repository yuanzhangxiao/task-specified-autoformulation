"""Tests for the frozen unseen-structure judge confirmation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel
from scripts.analyze_hybrid_judge_confirmation import evaluate_confirmation
from scripts.build_hybrid_judge_confirmation_pairs import (
    CONFIRMATION_MUTATIONS,
    build_confirmation_pairs,
    deduplicate_candidate_pairs,
)

CONFIG = Path("configs/hybrid_judge_atomic_confirmation_v1.json")
COMMON = Path("scripts/hpc/run_vllm_atomic_judge.sh")
SLURM = Path(
    "scripts/hpc/phase_b_hybrid_judge_vllm_atomic_confirmation_120b.slurm"
)

ALL_MUTATIONS = (
    "wrong_meal_sink",
    "duplicated_gp_flux",
    "disconnected_claimed_mechanism",
    "unjustified_one_sided_accumulator",
)


def _candidate(parameter: str, *, disconnected: bool = False) -> CandidateModel:
    utilization = {
        "k": "k * X * Gp",
        "q": "q * X * Gp**2",
        "r": "r * X",
        "s": "s * Gp",
    }[parameter]
    payload = {
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
                {"state": "Gp", "rhs": "meal_event_g - U"},
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
    if disconnected:
        payload["processes"].append(
            {
                "name": "claimed_meal_pathway",
                "expression": "meal_event_g",
                "mechanisms": ["MealPathway"],
            }
        )
    return CandidateModel.model_validate(payload)


def _group(parameter: str, *, prefix: str = "source") -> tuple[AdversarialPair, ...]:
    baseline = _candidate(parameter)
    return tuple(
        AdversarialPair(
            pair_id=f"{prefix}_{parameter}_{mutation}",
            benchmark_id="phase_b_test",
            tier="easy",
            mutation_type=mutation,
            valid_candidate=baseline,
            adversarial_candidate=_candidate(
                parameter,
                disconnected=mutation == "disconnected_claimed_mechanism",
            ),
        )
        for mutation in ALL_MUTATIONS
    )


def test_confirmation_excludes_all_opened_structures_and_targets_two_mutations(
) -> None:
    contexts = {("phase_b_test", "easy"): None}
    exclusions = (*_group("k"), *_group("q"))
    candidates = (
        *_group("k"),
        *_group("r"),
        *_group("r", prefix="duplicate-root"),
        *_group("s"),
    )

    selected, fingerprints = build_confirmation_pairs(
        candidates,
        exclusions,
        baseline_count=2,
        contexts=contexts,
    )

    assert len(fingerprints) == 2
    assert len(selected) == 4
    assert {pair.mutation_type for pair in selected} == set(
        CONFIRMATION_MUTATIONS
    )
    assert all(pair.pair_id.startswith("confirmation_") for pair in selected)
    assert {pair.valid_candidate.parameters[0].name for pair in selected} == {
        "r",
        "s",
    }


def test_candidate_inventory_deduplicates_repeated_run_roots() -> None:
    contexts = {("phase_b_test", "easy"): None}
    group = _group("r")

    deduplicated = deduplicate_candidate_pairs(
        (*group, *group),
        contexts=contexts,
    )

    assert len(deduplicated) == len(ALL_MUTATIONS)


def test_frozen_confirmation_config_and_slurm_preserve_selected_protocol() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen_before_unseen_structure_confirmation"
    assert payload["judge_model"] == "vllm:openai/gpt-oss-120b"
    assert payload["pair_construction"]["selected_mutations"] == [
        "duplicated_gp_flux",
        "wrong_meal_sink",
    ]
    assert payload["protocol"] == {
        "hybrid_judge_protocol_version": (
            "hybrid-judge-protocol-3-atomic-occurrence"
        ),
        "atomic_evidence_schema_version": "atomic-evidence-plan-1",
        "structural_facts_schema_version": "structural-facts-2",
        "semantic_answer_disclosure": False,
        "candidate_order_policy": "both_orientations",
        "reasoning_effort": "low",
        "temperature": 0.2,
        "seed_base": 10000,
        "repetitions": 5,
        "max_provider_attempts": 10,
        "scoring": {
            "partial_tiebreak_weight": 0.05,
            "comparative_weight": 0.25,
            "tie_threshold": 0.05,
        },
    }
    assert payload["planned"]["paired_judgments"] == 40
    assert payload["planned"]["llm_stages"] == 80

    for script in (COMMON, SLURM):
        subprocess.run(["bash", "-n", str(script)], check=True)
    common = COMMON.read_text(encoding="utf-8")
    wrapper = SLURM.read_text(encoding="utf-8")
    assert ': "${AF_PAIR_IDS=heldout_' in common
    assert ': "${AF_PAIR_IDS:=heldout_' not in common
    assert '"${pair_id_args[@]}"' in common
    assert ": \"${AF_PAIR_IDS:=}\"" in wrapper
    assert "#SBATCH --gpus-per-node=4" in wrapper
    assert "openai/gpt-oss-120b" in wrapper


def test_confirmation_gate_requires_every_predeclared_threshold() -> None:
    gate = {
        "minimum_response_success": 0.95,
        "minimum_pair_aggregate_accuracy": 1.0,
        "minimum_order_consistency": 0.8,
        "minimum_wrong_sink_atomic_accuracy": 0.75,
        "minimum_duplicate_atomic_accuracy": 0.75,
        "minimum_targeted_comparative_accuracy": 0.6,
    }
    hybrid = {
        "structured_response_success_rate": 1.0,
        "pair_aggregated_accuracy": 1.0,
        "order_consistency_rate": 0.85,
        "comparative_question_accuracy": 0.9,
    }
    atomic = {
        "wrong_sink_expected_direction_accuracy": 0.95,
        "duplicate_relation_accuracy": 1.0,
    }

    passed = evaluate_confirmation(hybrid, atomic, gate)
    failed = evaluate_confirmation(
        {**hybrid, "order_consistency_rate": 0.79},
        atomic,
        gate,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["minimum_order_consistency"]["passed"] is False
