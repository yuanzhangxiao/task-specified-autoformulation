"""Inventory preserves fit provenance and never evaluates or selects models."""

import json
import sys
from copy import deepcopy

import pytest

from scripts import collect_dalla_fitted_models as collect

CELL = "phase_b_dalla_man_t2_canonical_named_easy"


def sealed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**value, "artifact_sha256": collect.digest(value)}))


def campaign(root):
    task = {"task_id": "cell00_seed0_full", "cell": CELL, "arm": "full", "seed": 0}
    plan = {
        "protocol": "review-deadline-5",
        "config": {"rounds": 2},
        "continuation": {"source_round": 12},
        "tasks": [task],
        "cells": {
            CELL: {"brief": {"task": "public"}, "training": "DO NOT EXPORT ARRAYS"}
        },
    }
    endpoint = {
        "request": {
            "base_candidate": {"state_equations": []},
            "context": {},
            "initialization_plan": {},
            "extra": "DO NOT EXPORT",
        },
        "fit": {
            "parameters": {"k": 0.1},
            "lowered_candidate_sha256": "model",
            "training": {"normalized_mse": 0.1},
            "validation": {"normalized_mse": 0.2},
            "test": "DO NOT EXPORT",
            "status": "complete",
        },
        "certificate": {"runtime_valid": True},
        "origin_round": 10,
        "packet": {"residuals": "DO NOT EXPORT"},
    }
    result = {
        "task": task,
        "round": 0,
        "status": "complete",
        "selected": endpoint,
        "trial": deepcopy(endpoint),
    }
    sealed(root / "plan.json", plan)
    path = root / "results" / task["task_id"] / "round_00/result.json"
    sealed(path, result)
    # Malformed evaluation files must never be read.
    (root / "summary.json").write_text("not json")
    (root / "test.json").write_text("not json")
    return plan, result, path


def test_collects_both_roles_deduplicates_and_exposes_missing_round(tmp_path):
    campaign(tmp_path)
    value = collect.collect([tmp_path, tmp_path])
    assert value["status"] == "complete_for_supported_roots"
    assert len(value["models"]) == 1
    assert len(value["campaigns"]) == 1
    assert len(value["coverage"]) == 64
    first, second = value["occurrences"]
    assert first["selected"]["model_id"] == first["trial"]["model_id"]
    assert first["global_round"] == 12
    assert first["selected"]["origin_round"] == 10
    assert second["status"] == "checkpoint_missing"
    assert "DO NOT EXPORT" not in json.dumps(value)
    assert collect.collect([tmp_path]) == value
    row = next(r for r in value["coverage"] if r["cell"] == CELL and r["arm"] == "full")
    assert row["unique_retained_fits"] == row["unique_trial_fits"] == 1
    assert row["missing_checkpoints"] == 1


def test_unretained_trial_is_not_promoted(tmp_path):
    _, result, path = campaign(tmp_path)
    result["trial"]["fit"]["parameters"]["k"] = 0.7
    result["trial"]["fit"]["validation"]["normalized_mse"] = 0.001
    sealed(path, result)
    value = collect.collect([tmp_path])
    assert len(value["models"]) == 2
    row = value["occurrences"][0]
    assert row["selected"]["fit"]["parameters"] == {"k": 0.1}
    assert row["trial"]["fit"]["parameters"] == {"k": 0.7}


@pytest.mark.parametrize("fault", ["digest", "task", "round", "parameters"])
def test_invalid_checkpoint_is_explicit_error(tmp_path, fault):
    _, result, path = campaign(tmp_path)
    if fault == "task":
        result["task"]["cell"] = CELL.replace("t2", "t3")
    elif fault == "round":
        result["round"] = 1
    elif fault == "parameters":
        result["selected"]["fit"]["parameters"]["k"] = float("inf")
    sealed(path, result)
    if fault == "digest":
        result = json.loads(path.read_text())
        result["status"] = "tampered"
        path.write_text(json.dumps(result))
    value = collect.collect([tmp_path])
    assert value["status"] == "partial"
    assert value["errors"]
    assert not value["models"]


def test_excludes_external_baseline_and_non_dalla_arms(tmp_path):
    plan, _, _ = campaign(tmp_path)
    plan["tasks"][0]["arm"] = "raw_data_agent"
    sealed(tmp_path / "plan.json", plan)
    assert not collect.collect([tmp_path])["models"]
    plan["tasks"][0]["arm"] = "full"
    plan["tasks"][0]["cell"] = (
        "phase_b_alien_device_unknown_device_mechanism_canonical_opaque_easy"
    )
    sealed(tmp_path / "plan.json", plan)
    assert not collect.collect([tmp_path])["occurrences"]


def test_unsupported_protocol_and_missing_root_are_not_silent(tmp_path):
    plan, _, _ = campaign(tmp_path)
    plan["protocol"] = "future-campaign-1"
    sealed(tmp_path / "plan.json", plan)
    value = collect.collect([tmp_path, tmp_path / "missing"])
    assert value["campaigns"][0]["status"] == "unsupported_protocol"
    assert value["campaigns"][0]["declared_dalla_tasks"][0]["cell"] == CELL
    assert value["status"] == "partial"
    assert not value["models"]


def test_cli_reuses_identical_snapshot_and_refuses_changed_input(tmp_path, monkeypatch):
    _, result, path = campaign(tmp_path)
    output = tmp_path / "download.json"
    monkeypatch.setattr(
        sys, "argv", ["collector", "--root", str(tmp_path), "--out", str(output)]
    )
    collect.main()
    original = output.read_bytes()
    collect.main()
    assert output.read_bytes() == original
    result["selected"]["fit"]["parameters"]["k"] = 0.4
    sealed(path, result)
    with pytest.raises(SystemExit) as exc:
        collect.main()
    assert exc.value.code == 2
    assert output.read_bytes() == original


def test_nonobject_plan_is_reported(tmp_path):
    (tmp_path / "plan.json").write_text("[]")
    value = collect.collect([tmp_path])
    assert value["status"] == "partial"
    assert value["campaigns"][0]["status"] == "inventory_error"
