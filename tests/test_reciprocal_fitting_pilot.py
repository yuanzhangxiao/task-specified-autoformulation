"""Tests for the frozen reciprocal-coordinate fitting pilot matrix."""

import json
import shutil
from pathlib import Path

from autoformalism.rebuttal.reciprocal_fitting_pilot import (
    canonical_reciprocal_fitting_plan_sha256,
    load_reciprocal_fitting_pilot_plan,
    reciprocal_fitting_task_count,
    reciprocal_fitting_task_identity,
)
from scripts.summarize_phase_b_reciprocal_fitting_pilot import summarize

CONFIG = Path("configs/phase_b_reciprocal_fitting_pilot_v1.json")
RANGE_CONFIG = Path("configs/phase_b_parameter_range_ownership_pilot_v1.json")


def test_reciprocal_fitting_pilot_is_two_cells_by_three_seeds_by_three_arms() -> None:
    plan = load_reciprocal_fitting_pilot_plan(CONFIG)

    assert reciprocal_fitting_task_count(plan) == 18
    assert len(canonical_reciprocal_fitting_plan_sha256(plan)) == 64
    assert [item.condition_id for item in plan.fit_conditions] == [
        "bounded_rollout",
        "profiled_original_coordinate",
        "profiled_certified_reciprocal",
    ]
    assert plan.repetitions == (0, 1, 2)
    assert plan.test_data_opened is False
    assert plan.exact_training_observed_derivatives_supplied is True
    assert plan.latent_values_supplied is False
    assert plan.latent_derivatives_supplied is False


def test_reciprocal_fitting_pilot_reuses_candidate_identity_across_arms() -> None:
    plan = load_reciprocal_fitting_pilot_plan(CONFIG)

    identities = [
        reciprocal_fitting_task_identity(plan, index)
        for index in range(reciprocal_fitting_task_count(plan))
    ]
    for candidate_index in range(6):
        matched = [item for item in identities if item[3] == candidate_index]
        assert len(matched) == 3
        assert len({item[1].benchmark_id for item in matched}) == 1
        assert len({item[2] for item in matched}) == 1
        assert {item[0].condition_id for item in matched} == {
            "bounded_rollout",
            "profiled_original_coordinate",
            "profiled_certified_reciprocal",
        }


def test_parameter_range_ownership_pilot_is_two_cells_by_three_seeds() -> None:
    plan = load_reciprocal_fitting_pilot_plan(RANGE_CONFIG)

    assert reciprocal_fitting_task_count(plan) == 18
    assert plan.schema_version == "phase-b-parameter-range-ownership-pilot-1"
    assert {item.condition_id for item in plan.fit_conditions} == {
        "legacy_profiled_suggestions",
        "range_free_profiled",
        "range_free_rollout",
    }
    assert all(
        item.remove_legacy_parameter_ranges
        for item in plan.fit_conditions
        if item.condition_id.startswith("range_free_")
    )


def test_reciprocal_fitting_summary_requires_and_ledgers_complete_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    frozen = root / "frozen"
    tasks = root / "tasks"
    frozen.mkdir(parents=True)
    tasks.mkdir()
    shutil.copyfile(CONFIG, frozen / "plan.json")
    plan = load_reciprocal_fitting_pilot_plan(CONFIG)
    plan_sha256 = canonical_reciprocal_fitting_plan_sha256(plan)
    for task_index in range(reciprocal_fitting_task_count(plan)):
        condition, cell, repetition, candidate_index = (
            reciprocal_fitting_task_identity(plan, task_index)
        )
        reciprocal = condition.condition_id == "profiled_certified_reciprocal"
        row = {
            "schema_version": "phase-b-reciprocal-fitting-pilot-task-1",
            "status": "complete",
            "task_index": task_index,
            "plan_sha256": plan_sha256,
            "condition": condition.model_dump(mode="json"),
            "benchmark_id": cell.benchmark_id,
            "repetition": repetition,
            "candidate_sha256": f"{candidate_index:064x}",
            "fit_contract_compatible": True,
            "fit_success": True,
            "training_normalized_mse": 1.0,
            "validation_normalized_mse": 2.0,
            "function_evaluations": 3,
            "integration_failures": 0,
            "fit_wall_seconds": 4.0,
            "fit_process_cpu_seconds": 3.0,
            "certified_reciprocal_transformations": (
                [{"parameter_name": "tau"}] if reciprocal else []
            ),
            "error_type": None,
        }
        (tasks / f"task_{task_index:03d}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )

    result = summarize(root)

    assert result["status"] == "complete"
    assert len(result["groups"]) == 3
    assert len(result["matched_profiled_coordinate_trials"]) == 6
    assert (
        root / "summary" / "task_artifact_ledger.jsonl"
    ).read_text().count("\n") == 18


def test_parameter_range_summary_matches_scientific_identity(tmp_path: Path) -> None:
    root = tmp_path / "range-pilot"
    frozen = root / "frozen"
    tasks = root / "tasks"
    frozen.mkdir(parents=True)
    tasks.mkdir()
    shutil.copyfile(RANGE_CONFIG, frozen / "plan.json")
    plan = load_reciprocal_fitting_pilot_plan(RANGE_CONFIG)
    plan_sha256 = canonical_reciprocal_fitting_plan_sha256(plan)
    for task_index in range(reciprocal_fitting_task_count(plan)):
        condition, cell, repetition, candidate_index = (
            reciprocal_fitting_task_identity(plan, task_index)
        )
        range_free = condition.condition_id.startswith("range_free_")
        row = {
            "schema_version": "phase-b-parameter-range-ownership-pilot-task-1",
            "status": "complete",
            "task_index": task_index,
            "plan_sha256": plan_sha256,
            "condition": condition.model_dump(mode="json"),
            "benchmark_id": cell.benchmark_id,
            "repetition": repetition,
            "candidate_sha256": f"{candidate_index:064x}",
            "fit_candidate_identity": {
                "topology_sha256": "1" * 64,
                "functional_sha256": "2" * 64,
                "executable_sha256": ("3" if range_free else "4") * 64,
            },
            "legacy_parameter_range_field_count_removed": 4 if range_free else 0,
            "fit_contract_compatible": True,
            "fit_success": True,
            "training_normalized_mse": 1.0,
            "validation_normalized_mse": 2.0,
            "function_evaluations": 3,
            "integration_failures": 0,
            "fit_wall_seconds": 4.0,
            "fit_process_cpu_seconds": 3.0,
            "certified_reciprocal_transformations": [],
            "error_type": None,
        }
        (tasks / f"task_{task_index:03d}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )

    result = summarize(root)

    assert result["status"] == "complete"
    assert len(result["matched_parameter_range_trials"]) == 6
    assert all(
        item["same_scientific_structure"]
        for item in result["matched_parameter_range_trials"]
    )
