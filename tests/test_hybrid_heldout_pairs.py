"""Tests for baseline-structure-held-out hybrid judge pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel
from scripts.build_hybrid_judge_heldout_pairs import (
    candidate_structure_fingerprint,
    select_heldout_pairs,
)

MUTATIONS = (
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
    }[parameter]
    processes = [
        {
            "name": "U",
            "expression": utilization,
            "mechanisms": ["InsulinDisposal"],
        }
    ]
    if disconnected:
        processes.append(
            {
                "name": "claimed_meal_pathway",
                "expression": "meal_event_g",
                "mechanisms": ["MealPathway"],
            }
        )
    return CandidateModel.model_validate(
        {
            "candidate_id": f"candidate_{parameter}",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "X", "kind": "latent"},
            ],
            "processes": processes,
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
    )


def _group(parameter: str) -> tuple[AdversarialPair, ...]:
    baseline = _candidate(parameter)
    return tuple(
        AdversarialPair(
            pair_id=f"{parameter}_{mutation}",
            benchmark_id="phase_b_test",
            tier="easy",
            mutation_type=mutation,
            valid_candidate=baseline,
            adversarial_candidate=_candidate(
                parameter,
                disconnected=mutation == "disconnected_claimed_mechanism",
            ),
        )
        for mutation in MUTATIONS
    )


def test_select_heldout_pairs_rejects_calibration_structures() -> None:
    calibration = _group("k")
    candidates = (*calibration, *_group("q"), *_group("r"))

    selected, fingerprints = select_heldout_pairs(
        candidates,
        calibration,
        baseline_count=1,
    )

    assert len(selected) == 5
    assert {pair.mutation_type for pair in selected} == {
        *MUTATIONS,
        "retained_disconnected_claimed_mechanism",
    }
    assert fingerprints == (candidate_structure_fingerprint(_candidate("q")),)
    assert fingerprints[0] != candidate_structure_fingerprint(_candidate("k"))
    assert all(pair.pair_id.startswith(("heldout_", "hybrid_")) for pair in selected)


def test_unseen_structure_is_rekeyed_when_source_pair_ids_collide() -> None:
    calibration = _group("k")
    unseen = tuple(
        pair.model_copy(update={"pair_id": calibration[index].pair_id})
        for index, pair in enumerate(_group("q"))
    )

    selected, _ = select_heldout_pairs(
        unseen,
        calibration,
        baseline_count=1,
    )

    selected_ids = {pair.pair_id for pair in selected}
    calibration_ids = {pair.pair_id for pair in calibration}
    assert not (selected_ids & calibration_ids)


def test_select_heldout_pairs_fails_when_unseen_supply_is_insufficient() -> None:
    calibration = _group("k")

    with pytest.raises(ValueError, match=r"requested 2.*found 1"):
        select_heldout_pairs(
            (*calibration, *_group("q")),
            calibration,
            baseline_count=2,
        )


def test_frozen_protocol_is_json_symmetric_and_abstaining() -> None:
    payload = json.loads(
        Path("configs/hybrid_judge_protocol_v1.json").read_text(encoding="utf-8")
    )

    assert payload["transport"] == "json_schema"
    assert payload["tool_fallback"] is False
    assert payload["candidate_order_policy"] == "both_orientations"
    assert payload["retry_policy"] == {
        "trigger": "missing_orientation_only",
        "max_distinct_seeds_per_orientation": 5,
        "require_both_orientations": True,
        "terminal_behavior": "abstain",
    }


def test_native_retry_pilot_is_frozen_as_separate_transport() -> None:
    payload = json.loads(
        Path("configs/hybrid_judge_native_retry_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["transport"] == "json_schema_native_retry"
    assert payload["planned_logical_calls"] == 40
    assert len(payload["selected_pair_ids"]) == 4
    assert payload["provider_retry"] == {
        "max_attempts": 10,
        "endpoint": "/api/chat",
        "thinking": "low",
        "schema": "unchanged_on_retry",
        "repair_feedback": "contract_only",
        "seed_policy": "configured_seed_plus_attempt_minus_one",
    }
    assert payload["matched_control"]["transport"] == "json_schema"


def test_openai_thinking_retry_pilot_is_endpoint_ablation() -> None:
    payload = json.loads(
        Path("configs/hybrid_judge_openai_thinking_retry_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["transport"] == "json_schema_openai_thinking_retry"
    assert payload["planned_logical_calls"] == 10
    assert payload["selected_pair_ids"] == ["heldout_cca8883e6ae1b33f"]
    assert payload["provider_retry"]["first_endpoint"] == "/api/chat"
    assert payload["provider_retry"]["retry_endpoint"] == "/v1/chat/completions"
    assert payload["provider_retry"]["reasoning_effort"] == "low"
