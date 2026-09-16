"""Validation replay cannot fit, reset targets, or consume test trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.rebuttal import baseline_validation as replay
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
)
from autoformalism.rebuttal.prefit_replay import sealed_read
from tests.test_baseline_development_adapter import _payload
from tests.test_postfreeze_evaluation import _subject, _trajectory


def splits():
    train = DatasetSplit(SplitName.TRAIN, (_trajectory("train", (1.0, 2.0)),), "train")
    val = DatasetSplit(
        SplitName.VALIDATION, (_trajectory("val", (1.0, 2.0, 3.0)),), "val"
    )
    return train, val


def test_open_rollout_uses_train_scales_and_never_resets_targets():
    train, val = splits()
    subject = _subject()
    subject = subject.model_copy(
        update={
            "validation_context": subject.validation_context.model_copy(
                update={"lagged_targets": ("x",)}
            )
        }
    )
    result = replay.evaluate_validation(subject, train, val, replay.ReplaySettings())
    assert result["status"] == "complete"
    assert result["normalized_mse"] == pytest.approx(20 / 3)
    assert result["normalization_scales"] == {"x": 0.5}
    assert result["reset_observed_states"] is False
    assert result["validation_initials_fitted"] is False
    assert result["test_data_opened"] is False


@pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.TEST])
def test_reject_non_validation_split(split):
    train, val = splits()
    wrong = DatasetSplit(split, val.trajectories, val.fingerprint)
    with pytest.raises(ValueError, match="never TEST"):
        replay.evaluate_validation(_subject(), train, wrong, replay.ReplaySettings())


def test_incomplete_parameters_cannot_be_fitted():
    train, val = splits()
    with pytest.raises(ValueError, match="complete fitted"):
        replay.evaluate_validation(
            _subject(missing_parameter=True), train, val, replay.ReplaySettings()
        )


def test_failed_trajectory_keeps_null_aggregate(monkeypatch):
    from types import SimpleNamespace

    train, val = splits()
    monkeypatch.setattr(
        replay,
        "simulate_trajectory",
        lambda *a, **k: SimpleNamespace(
            success=False, message="deadline", predictions={}
        ),
    )
    result = replay.evaluate_validation(_subject(), train, val, replay.ReplaySettings())
    assert result["status"] == "rollout_failed"
    assert result["normalized_mse"] is None
    assert len(result["trajectories"]) == 1


def test_nonfinite_residual_is_failure(monkeypatch):
    from types import SimpleNamespace

    train, val = splits()
    monkeypatch.setattr(
        replay,
        "simulate_trajectory",
        lambda *a, **k: SimpleNamespace(
            success=True, message="ok", predictions={"x": np.array([1.0, np.nan, 3.0])}
        ),
    )
    result = replay.evaluate_validation(_subject(), train, val, replay.ReplaySettings())
    assert result["status"] == "rollout_failed"
    assert result["normalized_mse"] is None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_plan(tmp_path, monkeypatch, *, duplicate=False):
    from types import SimpleNamespace

    train, val = splits()
    subject = _subject()
    data = SimpleNamespace(train=train, validation=val)
    identity = {"train": "train", "validation": "val", "prompt": "p"}
    monkeypatch.setattr(
        replay, "load_public", lambda *a: (data, subject.validation_context, identity)
    )
    source = tmp_path / "source" / "result.json"
    payload = {**_payload("sindy"), "equations": {"x": "0.0"}}
    write(source, payload)
    if duplicate:
        write(tmp_path / "other" / "result.json", {**payload, "equations": {"x": "-x"}})
    roster = tmp_path / "roster.json"
    write(
        roster,
        {
            "protocol": replay.PROTOCOL,
            "cells": [{"benchmark_id": "fixture", "tier": "easy"}],
            "repetitions": [0],
        },
    )
    scan = tmp_path / "inventory.json"
    replay.inventory([tmp_path], scan)
    output = tmp_path / "output"
    plan = replay.prepare(
        scan, roster, tmp_path, tmp_path, output, replay.ReplaySettings()
    )
    return output, plan, source


def test_inventory_prepare_run_resume_and_denominator(tmp_path, monkeypatch):
    output, plan, _ = fixture_plan(tmp_path, monkeypatch)
    assert len(plan["rows"]) == 4
    assert [r["status"] for r in plan["rows"]].count("ready") == 1
    first = replay.run(output)
    assert first["status"] == "complete"
    assert first["groups"][0]["counts"] == {"source_unavailable": 1}
    ready = next(r for r in first["rows"] if r["source_kind"] == "sindy")
    assert ready["normalized_mse"] == pytest.approx(20 / 3)
    monkeypatch.setattr(
        replay, "evaluate_validation", lambda *a: pytest.fail("resimulated")
    )
    assert replay.run(output) == first
    original = (output / "results" / "0000.json").read_text()
    (output / "results" / "0000.json").write_text(
        original.replace("complete", "failed")
    )
    with pytest.raises(ValueError, match="digest"):
        replay.report(output)


def test_source_drift_blocks_evaluation(tmp_path, monkeypatch):
    output, _, source = fixture_plan(tmp_path, monkeypatch)
    source.write_text(source.read_text() + "\n")
    result = replay.run(output)
    row = next(r for r in result["rows"] if r["source_kind"] == "sindy")
    assert row["status"] == "evaluation_failed"
    assert "changed after freezing" in row["error"]


def test_public_data_drift_blocks_evaluation(tmp_path, monkeypatch):
    output, _, _ = fixture_plan(tmp_path, monkeypatch)
    original = replay.load_public

    def changed(*args):
        data, context, identity = original(*args)
        return data, context, {**identity, "validation": "changed"}

    monkeypatch.setattr(replay, "load_public", changed)
    row = next(r for r in replay.run(output)["rows"] if r["source_kind"] == "sindy")
    assert row["status"] == "evaluation_failed"
    assert "data/context changed" in row["error"]


def test_duplicate_sources_do_not_select_by_score(tmp_path, monkeypatch):
    _, plan, _ = fixture_plan(tmp_path, monkeypatch, duplicate=True)
    row = next(r for r in plan["rows"] if r["source_kind"] == "sindy")
    assert row["status"] == "unavailable"
    assert "found 2" in row["error"]


def test_inventory_never_reads_csv_or_cache(tmp_path, monkeypatch):
    write(tmp_path / "run" / "result.json", _payload("sindy"))
    write(tmp_path / "llm_cache" / "result.json", _payload("sindy"))
    (tmp_path / "test.csv").write_text("MUST NOT OPEN")
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.suffix != ".csv" and "llm_cache" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    value = replay.inventory(
        [tmp_path, tmp_path / "missing"], tmp_path / "inventory.json"
    )
    assert len(value["rows"]) == 1
    assert len(value["missing_roots"]) == 1


def test_raw_agent_requires_matching_prompt_and_data(tmp_path):
    identity = {"train": "a", "validation": "b", "prompt": "c"}
    config = {
        "parameter_refit_applied": False,
        "public_input_hashes": {"proposer_prompt.txt": "c"},
        "split_fingerprints": {"train": "a", "validation": "b"},
    }
    write(tmp_path / "run_config.json", config)
    write(
        tmp_path / "evaluation.json",
        {"schema_version": "raw-data-agent-fitted-evaluation-1"},
    )
    request = SourceAdapterRequest(
        request_id="raw", source_kind="raw_data_agent", source_path=tmp_path
    )
    assert (
        replay.audit_source(request, identity)["source_data_provenance"] == "verified"
    )
    with pytest.raises(ValueError, match="prompt differs"):
        replay.audit_source(request, {**identity, "prompt": "other"})
    with pytest.raises(ValueError, match="fingerprints differ"):
        replay.audit_source(request, {**identity, "train": "other"})
    write(tmp_path / "run_config.json", {**config, "parameter_refit_applied": True})
    with pytest.raises(ValueError, match="no parameter refit"):
        replay.audit_source(request, identity)


def test_train_plus_validation_sindy_cannot_enter(tmp_path):
    source = tmp_path / "result.json"
    write(
        source,
        {
            **_payload("sindy"),
            "finalization_protocol": (
                "selected_threshold_refit_on_train_plus_validation"
            ),
        },
    )
    request = SourceAdapterRequest(
        request_id="sindy", source_kind="sindy", source_path=source
    )
    with pytest.raises(ValueError):
        replay.audit_source(request, {})


@pytest.mark.parametrize("development", [False, True])
def test_d3_reuses_selected_candidate_values_only(tmp_path, development):
    subject = _subject()
    payload = {
        **_payload("d3_native_no_tools"),
        "equations": {"x": "0.0"},
        "selected_hyperparameters": {"selected_generation": 7},
    }
    if not development:
        payload.pop("schema_version")
        payload.pop("test_data_opened")
        payload.pop("selection_payload")
        payload.update(
            status="complete",
            test_normalized_mse=999,
            test_per_target_normalized_mse={"x": 999},
        )
    write(tmp_path / "result.json", payload)
    checkpoint = {
        "records": [
            {
                "generation": 7,
                "candidate": subject.candidate.model_dump(mode="json"),
                "parameters": {},
            }
        ]
    }
    write(tmp_path / "d3_checkpoint.json", checkpoint)
    request = SourceAdapterRequest(
        request_id="d3", source_kind="d3", source_path=tmp_path / "result.json"
    )
    adapted = adapt_source(request, subject.validation_context)
    assert adapted.candidate == subject.candidate
    assert adapted.target_prediction.status == "missing"
    checkpoint["records"] *= 2
    write(tmp_path / "d3_checkpoint.json", checkpoint)
    with pytest.raises(ValueError, match="selected generation"):
        adapt_source(request, subject.validation_context)


def test_runtime_drift_refuses_resume(tmp_path, monkeypatch):
    output, _, _ = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "_runtime", lambda: {})
    with pytest.raises(ValueError, match="versions differ"):
        replay.run(output)
    assert sealed_read(output / "plan.json")["live_llm_calls"] == 0


def test_worker_invokes_only_validation_cli():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    for filename in (
        "run_baseline_validation_delta.sh",
        "submit_baseline_validation_delta.sh",
    ):
        subprocess.run(["bash", "-n", str(root / "scripts/hpc" / filename)], check=True)
    worker = (root / "scripts/hpc/run_baseline_validation_delta.sh").read_text()
    assert "baseline_validation.py" in worker
    assert "--shards 8" in worker
    assert "postfreeze" not in worker


def test_historical_source_precedence_comes_from_existing_rebuttal_rosters():
    root = Path(__file__).resolve().parents[1]
    roster = replay.read_json(root / "configs/baseline_validation_v1.json")
    for name in ("phase_a3_dalla_cohort_v1", "phase_a3_cohort_v1"):
        source = replay.read_json(root / "configs/interventions" / f"{name}.json")
        for benchmark, patterns in source["methods"]["D3"]["patterns"].items():
            expected = [
                next(p for p in Path(pattern).parts if p.startswith("d3-native-"))
                for pattern in patterns
            ]
            assert roster["historical_d3_source_precedence"][benchmark] == expected


def test_d3_result_checkpoint_mismatch_rejected(tmp_path):
    source = tmp_path / "result.json"
    payload = {
        "selected_hyperparameters": {"selected_generation": 0},
        "equations": {"x": "0"},
    }
    write(
        source.with_name("d3_checkpoint.json"),
        {
            "records": [
                {
                    "generation": 0,
                    "candidate": {"state_equations": [{"state": "x", "rhs": "-x"}]},
                    "parameters": {},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="equations differ"):
        replay._check_d3_selection(source, payload)


@pytest.mark.parametrize("failure", [False, True])
def test_submission_receipts_and_duplicate_guard(tmp_path, failure):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/bash\nif [[ "$*" == *rev-parse* ]]; then echo abc123; fi\n')
    git.chmod(0o755)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import os,sys,json\nfrom pathlib import Path\n"
        + 'p=Path(os.environ["CALL_LOG"])\n'
        + "rows=json.loads(p.read_text()) if p.exists() else []\n"
        + "rows.append(sys.argv[1:]);p.write_text(json.dumps(rows))\n"
        + "print(100+len(rows))\n"
        + ("sys.exit(1)\n" if failure else "")
    )
    sbatch.chmod(0o755)
    calls = tmp_path / "calls.json"
    output = tmp_path / "out"
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(tmp_path),
        "AF_PYTHON": sys.executable,
        "AF_OUTPUT_ROOT": str(output),
        "CALL_LOG": str(calls),
    }
    command = ["bash", str(root / "scripts/hpc/submit_baseline_validation_delta.sh")]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert (first.returncode != 0) == failure
    before = calls.read_bytes()
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert (second.returncode != 0) == failure
    assert calls.read_bytes() == before
    assert len(json.loads(before)) == (1 if failure else 3)
    assert (output / "submission/prepare.receipt").is_file()


def test_pre_native_d3_uses_training_checkpoint_not_later_refit(tmp_path):
    payload = {
        "method": "d3_no_tools",
        "equations": {"x": "-k*x"},
        "selected_hyperparameters": {
            "selected_generation": 0,
            "adaptation": "restricted_schema",
            "selected_parameters": '{"k": 999}',
        },
    }
    source = tmp_path / "result.json"
    write(source, payload)
    write(
        source.with_name("d3_checkpoint.json"),
        {
            "records": [
                {
                    "generation": 0,
                    "candidate": {"state_equations": [{"state": "x", "rhs": "-k*x"}]},
                    "parameters": {"k": 0.5},
                }
            ]
        },
    )
    assert replay._check_d3_selection(source, payload) is True
    payload["method"] = "d3_native_no_tools"
    with pytest.raises(ValueError, match="parameters differ"):
        replay._check_d3_selection(source, payload)
