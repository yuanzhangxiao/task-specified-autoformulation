"""Numerical recovery, public split isolation, provenance and restart regressions."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.data.exceptions import DataAlignmentError, DataFileNotFoundError
from autoformalism.rebuttal import fitter_diagnostic as diagnostic
from autoformalism.rebuttal.fitter_diagnostic import (
    ARMS,
    DiagnosticPlan,
    execute_task,
    prepare_diagnostic,
    read_json,
    sha256,
    summarize_diagnostic,
    verify_freeze,
    write_json,
)
from autoformalism.schemas import CandidateModel
from scripts import run_phase_b_fitter_diagnostic as runner

PLAN_PATH = Path("configs/phase_b_fitter_diagnostic_v1.json")


def _candidate(targets: tuple[str, ...]) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "decay",
            "parent_candidate_id": None,
            "change_summary": "Synthetic decay recovery fixture.",
            "states": [
                {
                    "name": f"s{i}",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed state.",
                }
                for i in range(len(targets))
            ],
            "state_equations": [
                {"state": f"s{i}", "rhs": f"-decay*s{i}"} for i in range(len(targets))
            ],
            "observation_mappings": [
                {"channel": channel, "expression": f"s{i}", "unit": "unit"}
                for i, channel in enumerate(targets)
            ],
            "parameters": [{"name": "decay", "scope": "global", "role": "rate"}],
            "initial_conditions": [
                {"state": f"s{i}", "scope": "global", "fixed_value": 1}
                for i in range(len(targets))
            ],
        }
    )


@pytest.fixture
def bundle(tmp_path: Path):
    """Create synthetic Phase-B-shaped public data with no test.csv anywhere."""
    production = DiagnosticPlan.model_validate(read_json(PLAN_PATH))
    cell = production.cells[1].model_copy(
        update={
            "public_prompt_sha256": hashlib.sha256(
                b"Synthetic public task\n"
            ).hexdigest()
        }
    )
    plan = DiagnosticPlan(
        cells=(cell,),
        repetitions=(0,),
        total_max_nfev=60,
        fit_seconds=20,
        replay_seconds=5,
        supervisor_grace_seconds=5,
    )
    public = tmp_path / "public"
    root = public / "phase_b_v1" / cell.benchmark_id
    root.mkdir(parents=True)
    (root / "proposer_prompt.txt").write_bytes(b"Synthetic public task\n")
    spec = BenchmarkRegistry().get(cell.benchmark_id)
    roles = spec.tier_roles[cell.tier]
    columns = (
        "trajectory_id",
        "t",
        *roles.targets,
        *roles.auxiliaries,
        *spec.external_inputs,
    )
    for split in ("train", "validation"):
        with (root / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index in range(21):
                time = index * spec.sampling_interval
                writer.writerow(
                    {
                        "trajectory_id": split + "_0",
                        "t": time,
                        **{
                            channel: float((i + 1) * np.exp(-0.72 * time))
                            for i, channel in enumerate(roles.targets)
                        },
                        **dict.fromkeys(
                            (*roles.auxiliaries, *spec.external_inputs), 1.0
                        ),
                    }
                )
    write_json(
        root / "manifest.json",
        {
            "schema_version": "phase_b_public_release_v1",
            "status": "production_registered",
            "test_sealed": True,
            "benchmark_id": cell.benchmark_id,
            "tier": cell.tier,
            "channels": [
                {"public_name": channel, "role": role}
                for role, channels in (
                    ("target", roles.targets),
                    ("auxiliary", roles.auxiliaries),
                    ("external_input", spec.external_inputs),
                )
                for channel in channels
            ],
            "splits": {
                "train": sha256(root / "train.csv"),
                "validation": sha256(root / "validation.csv"),
                "test": "0" * 64,
            },
        },
    )
    run = (
        tmp_path
        / "historical"
        / (f"openai_gpt-5-6-sol_{cell.benchmark_id}_{cell.tier}_rep0")
    )
    run.mkdir(parents=True)
    write_json(
        run / "run_config.json",
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "benchmark_id": cell.benchmark_id,
            "tier": cell.tier,
            "repetition": 0,
            "parameter_refit_applied": False,
            "test_data_opened": False,
            "agent_config": {"output_contract": "fitted_model"},
            "public_input_hashes": {
                name: sha256(root / name) for name in diagnostic.PUBLIC_FILES[1:]
            },
        },
    )
    write_json(
        run / "candidate.json", _candidate(roles.targets).model_dump(mode="json")
    )
    write_json(
        run / "evaluation.json",
        {
            "schema_version": "raw-data-agent-fitted-evaluation-1",
            "parameter_refit_applied": False,
            "test_data_opened": False,
            "fitted_parameter_values": {"decay": 0.2},
            # Historical scores must not be trusted/reused in the new evaluator.
            "validation_metrics": {"normalized_mse": -1000},
        },
    )
    kwargs = {
        "public_root": public,
        "historical_root": tmp_path / "historical",
        "refresh_root": tmp_path / "refresh",
        "output_root": tmp_path / "output",
    }
    return plan, kwargs, run


def test_production_plan_has_six_sources_and_equal_fit_budgets() -> None:
    plan = DiagnosticPlan.model_validate(read_json(PLAN_PATH))
    assert len(plan.cells) * len(plan.repetitions) * len(ARMS) == 24
    configs = [plan.fit_config(arm, 2) for arm in ARMS[1:]]
    assert {c.number_of_starts * c.maximum_function_evaluations for c in configs} == {
        300
    }
    assert {c.maximum_wall_time_seconds for c in configs} == {900}
    assert len({c.random_seed for c in configs}) == 1
    assert all(not c.allow_derivative_regression for c in configs)
    with pytest.raises(ValueError, match="divisible"):
        DiagnosticPlan(cells=plan.cells, total_max_nfev=301)


def test_development_manifest_never_reads_test_but_full_validation_still_does(
    bundle, monkeypatch
):
    plan, kwargs, _ = bundle
    cell = plan.cells[0]
    original = Path.read_bytes

    def guard(path):
        assert path.name != "test.csv", "development manifest accessed test data"
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guard)
        frozen = prepare_diagnostic(plan, **kwargs)
        assert frozen["sources"][0]["status"] == "ready"
        assert verify_freeze(kwargs["output_root"]) == frozen
    loader = BenchmarkLoader()
    spec = BenchmarkRegistry().get(cell.benchmark_id)
    roles = spec.tier_roles[cell.tier]
    with pytest.raises(DataFileNotFoundError, match=r"test\.csv"):
        loader._validate_manifest(kwargs["public_root"], spec, cell.tier, roles)
    # A present but corrupted sealed test file must still fail the full check.
    test_path = kwargs["public_root"] / spec.relative_root / "test.csv"
    test_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(DataAlignmentError, match=r"test\.csv"):
        loader._validate_manifest(kwargs["public_root"], spec, cell.tier, roles)
    dataset = loader.load_development(
        DataConfig(
            benchmark_id=cell.benchmark_id, tier=cell.tier, root=kwargs["public_root"]
        )
    )
    assert dataset.train.trajectories


@pytest.mark.parametrize("arm_index", [1, 2, 3])
def test_actual_fit_recovers_train_parameters_with_unchanged_structure(
    bundle, arm_index
):
    plan, kwargs, run = bundle
    original_candidate = (run / "candidate.json").read_bytes()
    prepare_diagnostic(plan, **kwargs)
    root = kwargs["output_root"]
    baseline = execute_task(root, 0)
    fitted = execute_task(root, arm_index)
    assert baseline["parameters"] == {"decay": 0.2}
    assert baseline["replay"]["validation"]["normalized_mse"] > 0.1
    assert fitted["status"] == "complete"
    assert fitted["parameters"]["decay"] == pytest.approx(0.72, abs=1e-5)
    assert fitted["replay"]["validation"]["normalized_mse"] < 1e-10
    assert (run / "candidate.json").read_bytes() == original_candidate
    assert fitted["fit"]["diagnostics"][0]["backend"] == "rollout_least_squares"


def test_replay_never_optimizes_and_fitted_checkpoint_resumes(bundle, monkeypatch):
    plan, kwargs, _ = bundle
    prepare_diagnostic(plan, **kwargs)
    root = kwargs["output_root"]
    fitted = execute_task(root, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected optimizer call")

    monkeypatch.setattr(diagnostic, "fit_candidate", forbidden)
    assert execute_task(root, 0)["parameters"] == {"decay": 0.2}
    assert execute_task(root, 1)["parameters"] == fitted["parameters"]


def test_validation_targets_do_not_change_fitted_parameters(bundle):
    plan, kwargs, run = bundle
    root = kwargs["public_root"] / "phase_b_v1" / plan.cells[0].benchmark_id
    path = root / "validation.csv"
    lines = list(csv.DictReader(path.open()))
    for line in lines:
        line["v01"] = str(float(np.exp(-3.0 * float(line["t"]))))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=lines[0])
        writer.writeheader()
        writer.writerows(lines)
    manifest = read_json(root / "manifest.json")
    manifest["splits"]["validation"] = sha256(path)
    write_json(root / "manifest.json", manifest)
    source = read_json(run / "run_config.json")
    source["public_input_hashes"]["validation.csv"] = sha256(path)
    write_json(run / "run_config.json", source)
    prepare_diagnostic(plan, **kwargs)
    result = execute_task(kwargs["output_root"], 1)
    assert result["parameters"]["decay"] == pytest.approx(0.72, abs=1e-5)
    assert result["replay"]["validation"]["normalized_mse"] > 0.1


@pytest.mark.parametrize(
    "mutation", ["missing", "bad_parameter", "hash", "unsafe_expression"]
)
def test_invalid_sources_remain_in_denominator(bundle, mutation):
    plan, kwargs, run = bundle
    if mutation == "missing":
        (run / "candidate.json").unlink()
    elif mutation == "bad_parameter":
        value = read_json(run / "evaluation.json")
        value["fitted_parameter_values"] = {"wrong": 0.2}
        write_json(run / "evaluation.json", value)
    elif mutation == "hash":
        value = read_json(run / "run_config.json")
        value["public_input_hashes"]["train.csv"] = "0" * 64
        write_json(run / "run_config.json", value)
    else:
        value = read_json(run / "candidate.json")
        value["state_equations"][0]["rhs"] = "__import__('os').system('false')"
        write_json(run / "candidate.json", value)
    frozen = prepare_diagnostic(plan, **kwargs)
    assert frozen["sources"][0]["status"] == (
        "source_missing" if mutation == "missing" else "source_invalid"
    )
    for index in range(4):
        result = execute_task(kwargs["output_root"], index)
        write_json(kwargs["output_root"] / "results" / f"{index:03d}.json", result)
    summary = summarize_diagnostic(kwargs["output_root"])
    assert summary["expected_models"] == 1
    assert summary["complete_by_arm"] == dict.fromkeys(ARMS, 0)


def test_changed_inputs_or_runtime_reject_even_completed_checkpoints(
    bundle, monkeypatch
):
    plan, kwargs, _ = bundle
    root = kwargs["output_root"]
    prepare_diagnostic(plan, **kwargs)
    write_json(root / "results" / "000.json", execute_task(root, 0))
    with monkeypatch.context() as patch:
        patch.setattr(diagnostic, "runtime_identity", lambda: {"changed": True})
        with pytest.raises(ValueError, match="runtime differs"):
            runner.run_supervised(root, 0)
    frozen_candidate = root / "candidates" / "000.json"
    frozen_candidate.write_text("{}")
    with pytest.raises(ValueError, match="input hash differs"):
        runner.run_supervised(root, 0)


def test_preparation_is_idempotent_and_rejects_changed_sources(bundle):
    plan, kwargs, run = bundle
    original = prepare_diagnostic(plan, **kwargs)
    assert prepare_diagnostic(plan, **kwargs) == original
    value = read_json(run / "evaluation.json")
    value["fitted_parameter_values"]["decay"] = 0.3
    write_json(run / "evaluation.json", value)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        prepare_diagnostic(plan, **kwargs)


def test_failed_rollout_has_no_numeric_score(bundle, monkeypatch):
    plan, kwargs, _ = bundle
    prepare_diagnostic(plan, **kwargs)
    original = diagnostic.simulate_trajectory

    def fail(*args, **kwargs):
        return original(*args, **{**kwargs, "deadline": 0.0})

    monkeypatch.setattr(diagnostic, "simulate_trajectory", fail)
    result = execute_task(kwargs["output_root"], 0)
    assert result["status"] == "rollout_failed"
    assert result["replay"]["validation"]["normalized_mse"] is None
    assert result["replay"]["validation"]["failures"]


def test_supervisor_records_hard_timeout_and_kills_process_group(bundle, monkeypatch):
    plan, kwargs, _ = bundle
    prepare_diagnostic(plan, **kwargs)

    class HungProcess:
        pid = 123456

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"]

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    killed = []
    monkeypatch.setattr(runner.subprocess, "Popen", HungProcess)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append(pid))
    result = runner.run_supervised(kwargs["output_root"], 1)
    assert result["status"] == "timeout"
    assert killed == [123456]
    assert runner.run_supervised(kwargs["output_root"], 1) == result


def test_cli_end_to_end_and_completed_tasks_resume_without_rewrite(bundle):
    plan, kwargs, _ = bundle
    root = kwargs["output_root"]
    config = root.parent / "plan.json"
    write_json(config, plan.model_dump(mode="json"))
    command = [sys.executable, "scripts/run_phase_b_fitter_diagnostic.py"]
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}

    def call(*args):
        return subprocess.run(
            [*command, *args, "--output-root", str(root)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    call(
        "prepare",
        "--config",
        str(config),
        "--public-data-root",
        str(kwargs["public_root"]),
        "--historical-root",
        str(kwargs["historical_root"]),
        "--refresh-root",
        str(kwargs["refresh_root"]),
    )
    for index in range(4):
        call("run", "--task-index", str(index))
    checkpoint = root / "results" / "001.json"
    before = checkpoint.stat().st_mtime_ns
    call("run", "--task-index", "1")
    assert checkpoint.stat().st_mtime_ns == before
    call("summarize")
    summary = read_json(root / "summary.json")
    assert summary["complete_by_arm"] == dict.fromkeys(ARMS, 1)
    assert (root / "summary.md").is_file()
    assert not list(root.rglob("test.csv"))


def test_summary_serializes_extreme_finite_scores_without_nan(bundle):
    plan, kwargs, _ = bundle
    manifest = prepare_diagnostic(plan, **kwargs)
    root = kwargs["output_root"]
    for task, score in zip(manifest["tasks"][:2], (1e-300, 1e100), strict=True):
        result = {
            **task,
            "freeze_sha256": sha256(root / "freeze.json"),
            "status": "complete",
            "replay": {
                split: {"normalized_mse": score} for split in ("train", "validation")
            },
        }
        write_json(root / "results" / f"{task['index']:03d}.json", result)
    summary = summarize_diagnostic(root)
    assert summary["rows"][0]["arms"]["warm_1"]["validation_ratio_to_agent"] is None
    write_json(root / "summary.json", summary)
    assert summary["complete_by_arm"] == {
        "agent_replay": 1,
        "warm_1": 1,
        "cold_1": 0,
        "cold_3": 0,
    }
