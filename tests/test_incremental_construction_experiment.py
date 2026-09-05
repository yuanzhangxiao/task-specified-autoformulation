"""Frozen-plan tests for the incremental-construction pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal.incremental_construction_experiment import (
    build_incremental_construction_tasks,
    freeze_incremental_construction_experiment,
    load_incremental_construction_plan,
    summarize_incremental_construction_experiment,
)


def test_repository_incremental_construction_plan_has_six_tasks() -> None:
    plan = load_incremental_construction_plan(
        Path("configs/phase_b_incremental_construction_pilot_v1.json")
    )
    tasks = build_incremental_construction_tasks(plan)

    assert len(tasks) == 6
    assert tasks[0].benchmark_id == ("phase_b_dalla_man_t2_canonical_named_easy")
    assert tasks[-1].benchmark_id == (
        "phase_b_anonymous_system_task_canonical_opaque_hard"
    )
    assert {item.repetition for item in tasks} == {0, 1, 2}
    assert plan.model_contract.max_output_tokens == 8192


def test_repository_v2_plan_uses_phased_runtime_agenda() -> None:
    plan = load_incremental_construction_plan(
        Path("configs/phase_b_incremental_construction_pilot_v2.json")
    )

    assert plan.construction_protocol == "phased_runtime_agenda_v2"
    assert plan.construction_budget.maximum_topology_action_steps == 6
    assert len(build_incremental_construction_tasks(plan)) == 6


def test_freeze_and_summary_keep_test_and_private_data_closed(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "public"
    target_root = tmp_path / "targets"
    mechanism_root = tmp_path / "mechanisms"
    benchmark_id = "fixture"
    prompt = public_root / "phase_b_v1" / benchmark_id / "proposer_prompt.txt"
    target = target_root / "specs" / f"{benchmark_id}.json"
    mechanism = mechanism_root / "specs" / f"{benchmark_id}.json"
    for path, text in (
        (prompt, "public prompt"),
        (target, "{}"),
        (mechanism, "{}"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-incremental-construction-plan-1",
                "status": "frozen_before_proposer_calls",
                "purpose": "fixture",
                "development_only": True,
                "test_data_opened": False,
                "private_reference_opened": False,
                "parameter_fitting_performed": False,
                "scientific_judge_called": False,
                "cells": [
                    {
                        "benchmark_id": benchmark_id,
                        "tier": "easy",
                        "public_prompt_sha256": _sha(prompt),
                        "public_target_contract_sha256": _sha(target),
                        "public_mechanism_spec_sha256": _sha(mechanism),
                    }
                ],
                "repetitions": [0],
                "model_contract": {
                    "model": "model",
                    "reasoning_effort": "low",
                    "temperature": 0.0,
                    "max_output_tokens": 1024,
                    "served_context_tokens": 32768,
                    "request_timeout_seconds": 10,
                    "maximum_provider_attempts": 1,
                },
                "construction_budget": {
                    "topology_branch_count": 1,
                    "function_children_per_topology": 1,
                    "maximum_topology_action_steps": 2,
                    "maximum_functional_action_steps": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen"
    manifest = freeze_incremental_construction_experiment(
        config,
        frozen,
        public_data_root=public_root,
        target_contract_root=target_root,
        mechanism_spec_root=mechanism_root,
    )
    summary = summarize_incremental_construction_experiment(
        frozen / "plan.json",
        frozen / "task_plan.jsonl",
        tmp_path / "results",
    )

    assert manifest["task_count"] == 1
    assert manifest["test_data_opened"] is False
    assert summary["status"] == "incomplete"
    assert summary["parameter_fitting_performed"] is False
    assert summary["scientific_judge_called"] is False
    assert summary["private_reference_opened"] is False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
