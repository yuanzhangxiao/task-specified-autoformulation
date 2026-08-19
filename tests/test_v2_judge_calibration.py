"""Tests for prospective scientific-judge calibration pair construction."""

import csv
import json
import sys
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.schemas import CandidateModel
from scripts.analyze_adversarial_judge import main as analyze_main
from scripts.analyze_comparative_judge import main as analyze_comparative_main
from scripts.build_v2_judge_calibration_pairs import _mutations
from scripts.merge_comparative_scores import main as merge_comparative_main
from scripts.run_adversarial_judge import _select_shard
from scripts.run_comparative_judge import _reachability_facts


def _baseline() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "baseline",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
            ],
            "processes": [
                {"name": "U", "expression": "k * I * Gp"},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "meal_event_g - U"},
                {"state": "I", "rhs": "insulin_input - k * I"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
                {"channel": "I", "expression": "I"},
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
                {"state": "I", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def test_v2_mutations_are_distinct_and_runtime_valid() -> None:
    baseline = _baseline()
    context = ValidationContext(
        targets=("Gp", "I", "U"),
        auxiliaries=("meal_event_g", "insulin_input"),
    )

    mutations = _mutations(baseline)

    assert {name for name, _ in mutations} == {
        "wrong_meal_sink",
        "duplicated_gp_flux",
        "disconnected_claimed_mechanism",
        "unjustified_one_sided_accumulator",
    }
    for _, candidate in mutations:
        compile_candidate(candidate, context)
        assert candidate.candidate_id != baseline.candidate_id
        assert "wrong" not in candidate.candidate_id
        assert "duplicated" not in candidate.candidate_id
        assert candidate.parent_candidate_id is None
        assert candidate.change_summary == (
            "Candidate submitted for scientific assessment."
        )


def test_v2_mutations_encode_the_claimed_scientific_defects() -> None:
    mutations = dict(_mutations(_baseline()))

    wrong_sink_rhs = next(
        item.rhs
        for item in mutations["wrong_meal_sink"].state_equations
        if item.state == "Gp"
    )
    duplicate_rhs = next(
        item.rhs
        for item in mutations["duplicated_gp_flux"].state_equations
        if item.state == "Gp"
    )
    assert "- abs(meal_event_g)" in wrong_sink_rhs
    assert duplicate_rhs.count("meal_event_g") == 2
    assert any(
        item.name.startswith("claimed_meal_pathway")
        for item in mutations["disconnected_claimed_mechanism"].processes
    )
    assert any(
        item.name.startswith("unjustified_accumulator")
        for item in mutations["unjustified_one_sided_accumulator"].states
    )


def test_reachability_facts_expose_disconnected_process_without_a_verdict() -> None:
    disconnected = dict(_mutations(_baseline()))[
        "disconnected_claimed_mechanism"
    ]

    facts = _reachability_facts(disconnected)["components"]

    disconnected_name = next(
        name for name in facts if name.startswith("claimed_meal_pathway")
    )
    assert facts[disconnected_name] == {
        "component_kind": "process",
        "reaches_requested_target": False,
        "reachable_targets": [],
    }
    assert facts["U"]["reaches_requested_target"] is True


def test_analysis_accepts_baseline_mutated_labels(
    tmp_path: Path, monkeypatch
) -> None:
    scores = tmp_path / "scores.csv"
    categories = json.dumps({"mechanistic_coherence": 0.8})
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_id",
                "mutation_type",
                "known_label",
                "judge_model",
                "repetition",
                "aggregate_score",
                "category_scores",
            ),
        )
        writer.writeheader()
        for repetition in range(2):
            writer.writerow(
                {
                    "pair_id": "pair_1",
                    "mutation_type": "wrong_meal_sink",
                    "known_label": "baseline",
                    "judge_model": "ollama:gpt-oss:20b",
                    "repetition": repetition,
                    "aggregate_score": 0.8,
                    "category_scores": categories,
                }
            )
            writer.writerow(
                {
                    "pair_id": "pair_1",
                    "mutation_type": "wrong_meal_sink",
                    "known_label": "mutated",
                    "judge_model": "ollama:gpt-oss:20b",
                    "repetition": repetition,
                    "aggregate_score": 0.2,
                    "category_scores": json.dumps(
                        {"mechanistic_coherence": 0.2}
                    ),
                }
            )
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze", "--scores", str(scores), "--output", str(output)],
    )

    analyze_main()

    metrics = json.loads(output.read_text(encoding="utf-8"))
    model = metrics["ollama:gpt-oss:20b"]
    assert model["paired_preference_accuracy"] == 1.0
    assert model["mean_score_margin"] == pytest.approx(0.6)
    assert model["repeat_icc_1_1"] == 1.0


def test_contiguous_shards_keep_each_baseline_mutation_block_together() -> None:
    pairs = tuple(range(20))

    shards = [
        _select_shard(
            pairs,
            shard_index=index,
            shard_count=5,
            strategy="contiguous",
        )
        for index in range(5)
    ]

    assert shards == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
        (16, 17, 18, 19),
    ]


def test_comparative_analysis_normalizes_order_and_aggregates_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    scores = tmp_path / "comparative.csv"
    fields = (
        "pair_id",
        "mutation_type",
        "judge_model",
        "repetition",
        "order",
        "baseline_position",
        "baseline_preference",
        "indeterminate_rate",
        "not_applicable_rate",
        "answers",
    )
    answers_a = json.dumps(
        {
            "fluxes_not_duplicated": {
                "verdict": "candidate_a",
                "evidence": "A counts the flux once.",
            }
        }
    )
    answers_b = json.dumps(
        {
            "fluxes_not_duplicated": {
                "verdict": "candidate_b",
                "evidence": "B counts the flux once.",
            }
        }
    )
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repetition in range(2):
            writer.writerow(
                {
                    "pair_id": "pair_1",
                    "mutation_type": "duplicated_gp_flux",
                    "judge_model": "ollama:gpt-oss:120b",
                    "repetition": repetition,
                    "order": "baseline_a",
                    "baseline_position": "A",
                    "baseline_preference": 1.0,
                    "indeterminate_rate": 0.0,
                    "not_applicable_rate": 0.0,
                    "answers": answers_a,
                }
            )
            writer.writerow(
                {
                    "pair_id": "pair_1",
                    "mutation_type": "duplicated_gp_flux",
                    "judge_model": "ollama:gpt-oss:120b",
                    "repetition": repetition,
                    "order": "baseline_b",
                    "baseline_position": "B",
                    "baseline_preference": 1.0,
                    "indeterminate_rate": 0.0,
                    "not_applicable_rate": 0.0,
                    "answers": answers_b,
                }
            )
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze", "--scores", str(scores), "--output", str(output)],
    )

    analyze_comparative_main()

    metrics = json.loads(output.read_text(encoding="utf-8"))[
        "ollama:gpt-oss:120b"
    ]
    assert metrics["atomic_preference_accuracy"] == 1.0
    assert metrics["pair_aggregated_accuracy"] == 1.0
    assert metrics["order_consistency_rate"] == 1.0
    assert metrics["order_exact_reversal_rate"] == 0.0
    assert metrics["order_transition_counts"] == {
        "baseline_a:baseline|baseline_b:baseline": 2
    }
    assert metrics["by_question"]["fluxes_not_duplicated"][
        "baseline_preference_accuracy"
    ] == 1.0


def test_comparative_merge_requires_explicit_conflict_policy(
    tmp_path: Path, monkeypatch
) -> None:
    shard = tmp_path / "shard.csv"
    fields = (
        "pair_id",
        "judge_model",
        "repetition",
        "order",
        "baseline_preference",
    )
    with shard.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for preference in (0.75, 0.25):
            writer.writerow(
                {
                    "pair_id": "pair_1",
                    "judge_model": "ollama:test",
                    "repetition": 0,
                    "order": "baseline_a",
                    "baseline_preference": preference,
                }
            )
    output = tmp_path / "merged.csv"
    base_args = [
        "merge",
        "--inputs",
        str(shard),
        "--output",
        str(output),
        "--expected",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", base_args)
    with pytest.raises(SystemExit, match="conflicting duplicate"):
        merge_comparative_main()

    monkeypatch.setattr(
        sys, "argv", [*base_args, "--duplicate-policy", "first"]
    )
    merge_comparative_main()

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["baseline_preference"]) == 0.75
