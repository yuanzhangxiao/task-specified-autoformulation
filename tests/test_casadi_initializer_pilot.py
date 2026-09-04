"""Tests for the frozen matched CasADi initializer pilot."""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.rebuttal.casadi_initializer_pilot import (
    CasadiInitializerPilotPlan,
    canonical_casadi_initializer_plan_sha256,
    casadi_initializer_task_count,
    casadi_initializer_task_identity,
    load_casadi_initializer_pilot_plan,
)
from scripts import prepare_phase_b_casadi_initializer_pilot as prepare_script
from scripts.summarize_phase_b_casadi_initializer_pilot import summarize

CONFIG = Path("configs/phase_b_casadi_initializer_pilot_v1.json")


def test_prepare_cli_maps_config_option_to_function_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(prepare_script, "prepare_pilot", fake_prepare)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_phase_b_casadi_initializer_pilot.py",
            "--config",
            str(tmp_path / "plan.json"),
            "--source-replay-root",
            str(tmp_path / "replay"),
            "--public-data-root",
            str(tmp_path / "public"),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )

    prepare_script.main()

    assert captured["config_path"] == tmp_path / "plan.json"
    assert "config" not in captured


def test_casadi_initializer_pilot_is_two_cells_by_three_seeds_by_two_arms() -> None:
    plan = load_casadi_initializer_pilot_plan(CONFIG)

    assert casadi_initializer_task_count(plan) == 12
    assert len(canonical_casadi_initializer_plan_sha256(plan)) == 64
    assert plan.repetitions == (0, 1, 2)
    assert {item.condition_id for item in plan.fit_conditions} == {
        "runtime_owned_start",
        "casadi_multiple_shooting_start",
    }
    assert {
        item.total_wall_time_budget_seconds for item in plan.fit_conditions
    } == {900.0}
    assert plan.observed_derivatives_supplied is False
    assert plan.latent_values_supplied is False
    assert plan.latent_derivatives_supplied is False
    assert plan.test_data_opened is False


def test_casadi_initializer_pilot_reuses_candidate_across_arms() -> None:
    plan = load_casadi_initializer_pilot_plan(CONFIG)
    identities = [
        casadi_initializer_task_identity(plan, index)
        for index in range(casadi_initializer_task_count(plan))
    ]

    for candidate_index in range(6):
        matched = [item for item in identities if item[3] == candidate_index]
        assert len(matched) == 2
        assert len({item[1].benchmark_id for item in matched}) == 1
        assert len({item[2] for item in matched}) == 1
        assert {item[0].condition_id for item in matched} == {
            "runtime_owned_start",
            "casadi_multiple_shooting_start",
        }


def test_casadi_initializer_pilot_rejects_unmatched_total_budget() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["fit_conditions"][1]["core_fit_wall_time_seconds"] = 700.0

    with pytest.raises(ValidationError, match="equal total wall-time budgets"):
        CasadiInitializerPilotPlan.model_validate(payload)


def test_casadi_initializer_summary_requires_and_ledgers_complete_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    frozen = root / "frozen"
    tasks = root / "tasks"
    frozen.mkdir(parents=True)
    tasks.mkdir()
    shutil.copyfile(CONFIG, frozen / "plan.json")
    plan = load_casadi_initializer_pilot_plan(CONFIG)
    plan_sha256 = canonical_casadi_initializer_plan_sha256(plan)
    for task_index in range(casadi_initializer_task_count(plan)):
        condition, cell, repetition, candidate_index = (
            casadi_initializer_task_identity(plan, task_index)
        )
        uses_casadi = condition.nonlinear_initializer == "casadi_multiple_shooting"
        row = {
            "schema_version": "phase-b-casadi-initializer-pilot-task-1",
            "status": "complete",
            "task_index": task_index,
            "plan_sha256": plan_sha256,
            "condition": condition.model_dump(mode="json"),
            "benchmark_id": cell.benchmark_id,
            "repetition": repetition,
            "candidate_sha256": f"{candidate_index:064x}",
            "fit_candidate_identity": {
                "schema_version": "candidate-identity-1",
                "topology_sha256": "1" * 64,
                "functional_sha256": "2" * 64,
                "executable_sha256": "3" * 64,
            },
            "fit_contract_compatible": True,
            "fit_success": True,
            "training_normalized_mse": 1.0,
            "validation_normalized_mse": 2.0 - float(uses_casadi),
            "function_evaluations": 3,
            "integration_failures": 0,
            "fit_wall_seconds": 4.0,
            "fit_process_cpu_seconds": 3.0,
            "initialization_diagnostics": (
                [
                    {
                        "backend": "casadi_multiple_shooting",
                        "success": True,
                        "status": "complete",
                        "message": "ok",
                        "objective": 1.0,
                        "iterations": 2,
                        "wall_seconds": 0.5,
                        "parameter_estimates": [],
                    }
                ]
                if uses_casadi
                else []
            ),
            "error_type": None,
        }
        (tasks / f"task_{task_index:03d}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )

    result = summarize(root)

    assert result["status"] == "complete"
    assert len(result["groups"]) == 2
    assert len(result["matched_trials"]) == 6
    assert all(item["both_fits_successful"] for item in result["matched_trials"])
    assert all(
        item["validation_nmse_casadi_minus_runtime_start"] == -1.0
        for item in result["matched_trials"]
    )
    assert (
        root / "summary" / "task_artifact_ledger.jsonl"
    ).read_text().count("\n") == 12
