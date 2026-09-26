"""Model/initializer preservation and report-only mean/SD aggregation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import export_component_mechanisms as export
from scripts import submit_component_mechanisms as submitter
from scripts import summarize_functional_mechanisms as summary
from scripts.smoke_public_fitting import control


def fixture(source):
    request, train, val = control("general-rollout-v1")
    lowered, guesses, _ = public._lower(request)
    candidate = lowered.validated.candidate.model_dump(mode="json")
    task = {
        "task_id": "toy_seed0_full_c1v0s1",
        "cell": "toy",
        "arm": "full",
        "critic": True,
        "scientific_verifier": False,
        "shared_processes": True,
        "seed": 0,
    }
    cell = {
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
        "assets": {"proposer_prompt.txt": "a" * 64},
        "target_contract": {
            "benchmark_id": "toy",
            "tier": "easy",
            "public_prompt_sha256": "a" * 64,
            "targets": [
                {"target_channel": "v01", "public_requirement": "Generate v01"}
            ],
        },
        "mechanism_spec": {
            "benchmark_id": "toy",
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "input_response",
                    "required_drivers": ["u01"],
                    "required_targets": ["v01"],
                }
            ],
        },
    }
    plan = sealed_write(
        source / "plan.json",
        {
            "protocol": "final-component-campaign-1",
            "config": {"rounds": 3},
            "tasks": [task],
            "cells": {"toy": cell},
        },
    )
    fit = {
        "status": "complete",
        "parameters": guesses,
        "request_sha256": public.content_sha256(request.model_dump(mode="json")),
        "lowered_candidate_sha256": public.content_sha256(candidate),
        "training_content_sha256": public.content_sha256(cell["training"]),
        "validation_content_sha256": public.content_sha256(cell["validation"]),
        "validation": {"normalized_mse": 0.2},
    }
    directory = source / "results" / task["task_id"] / "round_00"
    (directory / "fit").mkdir(parents=True)
    receipt = {"result": fit, "sha256": public.content_sha256(fit)}
    (directory / "fit/result.json").write_text(json.dumps(receipt))
    selected = {
        "request": request.model_dump(mode="json"),
        "fit": fit,
        "bundle": {"candidate": candidate},
        "origin_task": task["task_id"],
        "origin_round": 0,
        "fit_result_sha256": public.content_sha256(receipt),
    }
    result = sealed_write(
        directory / "result.json",
        {
            "round": 0,
            "task": task,
            "selected": selected,
            "status": "complete",
            "test_data_opened": False,
        },
    )
    return plan, result


def test_preserve_fit_initial_map_and_rescore_verifier_off(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    plan, result = fixture(source)
    value = export.export(source, output)
    row = value["rows"][0]
    assert row["parameters"] == result["selected"]["fit"]["parameters"]
    assert row["candidate"] == result["selected"]["bundle"]["candidate"]
    assert row["parameters"]["init_z_scale"] == 0.5
    assert row["initials"] == {}
    assert row["legacy_certificate"]["scientific_verifier_enabled"] is True
    assert not row["task"]["scientific_verifier"]
    assert row["status"] == "ready"
    assert row["selected_origin_round"] == 0
    assert value["source_plan_sha256"] == plan["artifact_sha256"]
    assert not value["test_data_opened"] and value["solver_rollouts"] == 0


def test_round_selection_no_fallback_and_frozen_resume(tmp_path):
    source = tmp_path / "source"
    plan, result = fixture(source)
    old = export.export(source, tmp_path / "old")
    missing = export.export(source, tmp_path / "missing", 1)
    assert missing["rows"][0]["status"] == "unavailable"
    assert missing["rows"][0]["source_terminal_status"] is None
    empty = {k: v for k, v in result.items() if k != "artifact_sha256"}
    empty.update(round=1, selected=None, status="worker_interrupted")
    sealed_write(
        source / "results" / plan["tasks"][0]["task_id"] / "round_01/result.json", empty
    )
    assert export.export(source, tmp_path / "old") == old
    new = export.export(source, tmp_path / "new")
    assert new["rows"][0]["status"] == "ready"
    assert new["rows"][0]["source_round"] == 0
    assert new["rows"][0]["last_recorded_round"] == 1
    with pytest.raises(ValueError, match="different inputs"):
        export.export(source, tmp_path / "old", 0)


@pytest.mark.parametrize("change", ["task", "receipt", "parameters"])
def test_corrupted_bindings_rejected(tmp_path, change):
    source = tmp_path / "source"
    plan, result = fixture(source)
    directory = source / "results" / plan["tasks"][0]["task_id"] / "round_00"
    if change == "receipt":
        path = directory / "fit/result.json"
        value = json.loads(path.read_text())
        value["result"]["parameters"]["a"] = 99
        path.write_text(json.dumps(value))
    else:
        value = {
            k: copy.deepcopy(v) for k, v in result.items() if k != "artifact_sha256"
        }
        if change == "task":
            value["task"]["seed"] = 1
        else:
            value["selected"]["fit"]["parameters"].pop("init_z_scale")
        (directory / "result.json").unlink()
        sealed_write(directory / "result.json", value)
    with pytest.raises(ValueError):
        export.export(source, tmp_path / "output")


def test_equal_case_weight_sample_sd_and_unresolved_denominator():
    rows = [
        {
            "method": "m",
            "benchmark_id": cell,
            "repetition": rep,
            "mechanisms": [{"status": s} for s in outcomes],
        }
        for cell, rep, outcomes in [
            ("a", 0, ["pass"]),
            ("a", 1, ["pass"]),
            ("b", 0, ["fail", "unresolved"]),
        ]
    ]
    result = summary.aggregate(rows)[0]
    assert result["confirmed"]["mean"] == 0.5
    assert result["confirmed"]["sd"] == pytest.approx(2**-0.5)
    assert result["possible"]["mean"] == 0.75
    assert result["counts"] == {"pass": 2, "fail": 1, "unresolved": 1}
    assert result["cases"][1]["confirmed"]["sd"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summary.aggregate([*rows, rows[0]])
    rows[0]["mechanisms"][0]["status"] = "missing"
    with pytest.raises(ValueError, match="outcomes"):
        summary.aggregate(rows)


def test_report_cli_sealed_results_no_data_or_rollouts(tmp_path):
    root, output = tmp_path / "source", tmp_path / "mean-sd"
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "fitted-public-mechanism-tests-1",
            "rows": [
                {
                    "index": 0,
                    "method": "m",
                    "benchmark_id": "a",
                    "repetition": 0,
                    "mechanisms": [{"id": "response"}],
                }
            ],
            "code_identity": {"deliberately": "old runtime"},
        },
    )
    with pytest.raises(FileNotFoundError):
        summary.report(root, output)
    sealed_write(
        root / "results/0000/result.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "index": 0,
            "status": "assessed",
            "mechanisms": [{"id": "response", "status": "unresolved"}],
        },
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_functional_mechanisms.py",
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
    )
    value = sealed_read(output / "summary.json")
    assert value["methods"][0]["confirmed"]["mean"] == 0
    assert value["methods"][0]["possible"]["mean"] == 1
    assert summary.report(root, output) == value
    assert value["solver_rollouts"] == 0


def test_export_cli_help():
    subprocess.run(
        [sys.executable, "scripts/export_component_mechanisms.py", "--help"],
        check=True,
        capture_output=True,
    )


def test_submit_cpu_only_idempotent_and_uncertain_receipts(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "assessment"
    fixture(source)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.delenv("AF_COMMIT", raising=False)
    monkeypatch.setattr(submitter, "source_commit", lambda *_: "b" * 40)
    # Isolate submitter's exported environment from other tests in this worker.
    monkeypatch.setattr(submitter.os, "environ", dict(submitter.os.environ))
    calls = []

    def queue(directory, stage, options, worker, *args):
        calls.append((stage, options))
        assert "--partition=cpu" in options
        assert not any("gpu" in v for v in options)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    value = submitter.submit(source, output)
    assert len(calls) == 3 and value["models"] == 1
    assert "--dependency=afterok:101" in calls[1][1]
    assert "--dependency=afterany:102" in calls[2][1]
    assert submitter.submit(source, output) == value and len(calls) == 3
    with pytest.raises(ValueError, match="identity differs"):
        submitter.submit(source, output, 0)
    interrupted = tmp_path / "interrupted"
    (interrupted / "submission-intent").mkdir(parents=True)
    with pytest.raises(ValueError, match="partial submission"):
        submitter.submit(source, interrupted)
