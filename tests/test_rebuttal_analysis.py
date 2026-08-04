"""Focused tests for rebuttal artifact and deterministic metric utilities."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.rebuttal.adversarial import MutationRecipe, mutate_candidate
from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.hidden import hidden_mechanism_nmse
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.objectives import compare_ratio_and_weighted_sum
from autoformalism.rebuttal.structure import pairwise_similarities
from autoformalism.schemas import CandidateModel


def _candidate(identifier: str = "candidate") -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "states": [
                {
                    "name": "memory",
                    "kind": "latent",
                    "mechanisms": ["input_memory"],
                },
                {"name": "target", "kind": "observed"},
            ],
            "processes": [],
            "state_equations": [
                {"state": "memory", "rhs": "input_u - memory"},
                {"state": "target", "rhs": "memory - target"},
            ],
            "observation_mappings": [
                {"channel": "target", "expression": "target"}
            ],
            "initial_conditions": [
                {"state": "memory", "scope": "global", "fixed_value": 0.0},
                {"state": "target", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _artifact(identifier: str, loss: float, score: float) -> CandidateArtifact:
    return CandidateArtifact(
        artifact_id=identifier,
        source_checkpoint=f"/{identifier}.json",
        run_directory=f"/{identifier}",
        benchmark_id="synthetic",
        tier="hard",
        seed=0,
        round_index=0,
        structural_hash=identifier,
        candidate=_candidate(identifier),
        validation_mse=loss,
        training_mse=loss,
        judge_score=score,
        judge_category_scores={},
        state_count=2,
        latent_state_count=1,
        process_count=0,
        parameter_count=0,
        term_count=4,
        use_judge=True,
    )


def test_ratio_and_weighted_sum_comparison_is_development_only() -> None:
    result = compare_ratio_and_weighted_sum(
        (_artifact("a", 0.01, 0.9), _artifact("b", 0.02, 0.2)),
        lambda_multiplier=1.0,
    )

    assert result.candidate_count == 2
    assert result.ratio_selected_artifact_id == "a"
    assert result.lambda_value == 0.015


def test_mechanism_coverage_requires_driver_memory_and_target_path() -> None:
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(_candidate(), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == 1.0
    assert result.manual_review_required is False


def test_mechanism_tags_ignore_case_and_separator_variation() -> None:
    candidate = _candidate().model_copy(
        update={
            "states": (
                _candidate().states[0].model_copy(
                    update={"mechanisms": ("Input-Memory",)}
                ),
                _candidate().states[1],
            )
        }
    )
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    assert evaluate_mechanisms(candidate, spec).structural_validity == 1.0


def test_dynamic_memory_does_not_count_observed_target_state() -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"] = [
        state for state in payload["states"] if state["name"] == "target"
    ]
    payload["states"][0]["mechanisms"] = ["input_memory"]
    payload["state_equations"] = [
        {"state": "target", "rhs": "input_u - target"}
    ]
    payload["initial_conditions"] = [
        initial
        for initial in payload["initial_conditions"]
        if initial["state"] == "target"
    ]
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(CandidateModel.model_validate(payload), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == pytest.approx(2 / 3)


def test_process_inheriting_latent_state_counts_as_dynamic_memory() -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"][0]["mechanisms"] = []
    payload["processes"] = [
        {
            "name": "memory_effect",
            "expression": "memory",
            "mechanisms": ["input_memory"],
        }
    ]
    payload["state_equations"][1]["rhs"] = "memory_effect - target"
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(CandidateModel.model_validate(payload), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == 1.0


def test_hidden_metric_is_invariant_to_positive_affine_coordinate() -> None:
    train_reference = np.asarray([0.0, 1.0, 2.0, 3.0])
    test_reference = np.asarray([4.0, 5.0])
    train_candidate = (train_reference - 7.0) / 2.5
    test_candidate = (test_reference - 7.0) / 2.5

    result = hidden_mechanism_nmse(
        train_candidate,
        train_reference,
        test_candidate,
        test_reference,
    )

    assert result.scale == pytest.approx(2.5)
    assert result.offset == pytest.approx(7.0)
    assert result.test_nmse == pytest.approx(0.0, abs=1e-20)


def test_structure_similarity_ignores_alpha_renaming() -> None:
    first = _candidate("first")
    payload = first.model_dump(mode="json")
    payload["candidate_id"] = "second"
    replacements = {"memory": "buffer", "target": "output"}
    for state in payload["states"]:
        state["name"] = replacements[state["name"]]
    for equation in payload["state_equations"]:
        equation["state"] = replacements[equation["state"]]
        for old, new in replacements.items():
            equation["rhs"] = equation["rhs"].replace(old, new)
    payload["observation_mappings"][0]["expression"] = "output"
    for initial in payload["initial_conditions"]:
        initial["state"] = replacements[initial["state"]]
    second = CandidateModel.model_validate(payload)

    result = pairwise_similarities((("first", first), ("second", second)))[0]

    assert result.edge_jaccard == 1.0
    assert result.term_jaccard == 1.0


def test_adversarial_symbol_replacement_is_ast_scoped() -> None:
    changed = mutate_candidate(
        _candidate(),
        MutationRecipe(
            mutation_type="replace_symbol",
            component="memory",
            old_symbol="input_u",
            new_symbol="wrong_input",
        ),
    )

    equations = {item.state: item.rhs for item in changed.state_equations}
    assert "wrong_input" in equations["memory"]
    assert "input_u" not in equations["memory"]


def test_adversarial_memory_replacement_removes_dynamic_state() -> None:
    changed = mutate_candidate(
        _candidate(),
        MutationRecipe(
            mutation_type="replace_state_with_algebraic",
            component="memory",
            replacement_expression="input_u",
        ),
    )

    assert {item.name for item in changed.states} == {"target"}
    assert {item.name for item in changed.processes} == {"memory"}
