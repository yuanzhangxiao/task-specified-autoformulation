"""Scientific-source binding, split isolation and checkpointed numerical handoff."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.rebuttal import staged_fit_probe as probe
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.staged_topology import content_hash
from scripts import run_staged_fit_probe as runner
from tests.test_fitter_diagnostic import bundle as diagnostic_bundle  # noqa: F401


@pytest.fixture
def staged_bundle(diagnostic_bundle, tmp_path: Path):  # noqa: F811
    """Reuse a synthetic decay data fixture with no test table present."""
    old_plan, old_paths, run = diagnostic_bundle
    cell = old_plan.cells[0]
    public = old_paths["public_root"] / "phase_b_v1" / cell.benchmark_id
    (public / "proposer_prompt.txt").write_text(
        "A. Synthetic decay task\n\nF. Required response\nSynthetic contract\n"
    )
    task = {
        "task_id": "synthetic_functions_seed0",
        "kind": "benchmark",
        "brief": {"scientific_context": "A. Synthetic decay task"},
        "context": {"targets": ["v01"]},
    }
    parent = {"tasks": [task]}
    parent["plan_sha256"] = content_hash(parent)
    source = {
        "complete_model": True,
        "candidate": read_json(run / "candidate.json"),
        "test_data_opened": False,
        "private_reference_opened": False,
        "parameter_fitting_performed": False,
    }
    source_root = tmp_path / "functions" / task["task_id"]
    source_root.mkdir(parents=True)
    write_json(source_root / "result.json", source)
    write_json(
        source_root / "terminal.json",
        {
            "result": source,
            "identity": content_hash([parent["plan_sha256"], task]),
        },
    )
    write_json(tmp_path / "function_plan.json", parent)
    config = read_json(Path("configs/staged_fit_probe_v1.json"))
    config.update(
        function_plan_sha256=parent["plan_sha256"],
        function_task_id=task["task_id"],
        function_result_sha256=content_hash(source),
        public_prompt_sha256=sha256(public / "proposer_prompt.txt"),
        screen_seconds=5,
        replay_seconds=5,
        supervisor_grace_seconds=5,
    )
    config["fit_config"].update(
        maximum_wall_time_seconds=15,
        maximum_function_evaluations=30,
        number_of_starts=1,
    )
    plan = probe.StagedFitPlan.model_validate(config)
    paths = {
        "function_plan": tmp_path / "function_plan.json",
        "function_results": source_root.parent,
        "public_root": old_paths["public_root"],
        "output": tmp_path / "probe",
    }
    return plan, paths


def test_probe_recovers_decay_and_reuses_completed_fit(staged_bundle, monkeypatch):
    plan, paths = staged_bundle
    frozen = probe.prepare_probe(plan, **paths)
    root = paths["output"]
    original_fit = probe.fit_candidate
    preferred = []

    def record_start(*args, **kwargs):
        preferred.append(kwargs["initial_global_parameters"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(probe, "fit_candidate", record_start)
    assert not list(root.rglob("test.csv"))
    assert set(frozen["assets"]) == {
        "candidate.json",
        "source_function.json",
        *(
            f"public/phase_b_v1/{plan.benchmark_id}/{name}"
            for name in (
                "manifest.json",
                "proposer_prompt.txt",
                "train.csv",
                "validation.csv",
            )
        ),
    }
    first = probe.execute_probe(root)
    assert first["status"] == "complete"
    assert first["fit"]["global_parameters"]["decay"] == pytest.approx(0.72, abs=1e-4)
    assert first["replay"]["validation"]["normalized_mse"] < 1e-8
    assert first["default_replay"]["replay"]["train"]["normalized_mse"] > 1e-3
    assert first["fit"]["global_initial_conditions"] == {}
    assert first["test_data_opened"] is False
    assert first["private_reference_opened"] is False
    assert preferred == [{"decay": 1.0}]
    assert (root / "final_replay.json").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fit must not run again")

    monkeypatch.setattr(probe, "fit_candidate", forbidden)
    monkeypatch.setattr(probe, "replay_parameters", forbidden)
    assert probe.execute_probe(root) == first
    (root / "result.json").unlink()
    assert probe.execute_probe(root) == first


def test_failed_initial_screen_never_calls_fitter(staged_bundle, monkeypatch):
    plan, paths = staged_bundle
    probe.prepare_probe(plan, **paths)
    monkeypatch.setattr(
        probe,
        "replay_parameters",
        lambda *a, **k: {
            "train": {"normalized_mse": 1.0},
            "validation": {"normalized_mse": None, "failures": ["integration failed"]},
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("screen failure must not invoke fitter")

    monkeypatch.setattr(probe, "fit_candidate", forbidden)
    result = probe.execute_probe(paths["output"])
    assert result["status"] == "screen_failed"
    assert (paths["output"] / "default_replay.json").exists()
    assert not (paths["output"] / "fit.json").exists()
    assert probe.execute_probe(paths["output"]) == result


@pytest.mark.parametrize("artifact", ["plan", "result", "terminal", "prompt"])
def test_probe_rejects_tampered_generation_or_prompt(staged_bundle, artifact):
    plan, paths = staged_bundle
    if artifact == "plan":
        path = paths["function_plan"]
        value = read_json(path)
        value["tasks"][0]["brief"]["scientific_context"] = "changed"
    elif artifact == "prompt":
        path = (
            paths["public_root"]
            / "phase_b_v1"
            / plan.benchmark_id
            / "proposer_prompt.txt"
        )
        path.write_text("changed")
        with pytest.raises(ValueError, match="prompt differs"):
            probe.prepare_probe(plan, **paths)
        return
    else:
        path = paths["function_results"] / plan.function_task_id / f"{artifact}.json"
        value = read_json(path)
        if artifact == "result":
            value["candidate"]["parameters"][0]["name"] = "changed"
        else:
            value["identity"] = "0" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="differs"):
        probe.prepare_probe(plan, **paths)


def test_probe_rejects_nonfixed_initializer_without_repair(staged_bundle):
    plan, paths = staged_bundle
    source_root = paths["function_results"] / plan.function_task_id
    source = read_json(source_root / "result.json")
    source["candidate"]["initial_conditions"][0].update(
        fixed_value=None,
        expression="u01",
    )
    terminal = read_json(source_root / "terminal.json")
    terminal["result"] = source
    write_json(source_root / "result.json", source)
    write_json(source_root / "terminal.json", terminal)
    changed = plan.model_copy(update={"function_result_sha256": content_hash(source)})
    with pytest.raises(ValueError, match="explicitly fixed"):
        probe.prepare_probe(changed, **paths)


@pytest.mark.parametrize("target", ["asset", "runtime", "launcher", "checkpoint"])
def test_resume_verifies_inputs_runtime_and_checkpoint(
    staged_bundle, monkeypatch, target
):
    plan, paths = staged_bundle
    probe.prepare_probe(plan, **paths)
    root = paths["output"]
    if target == "asset":
        (root / "candidate.json").write_text("{}")
    elif target == "runtime":
        monkeypatch.setattr(probe, "runtime_identity", lambda: {})
    elif target == "launcher":
        monkeypatch.setattr(probe, "launcher_hash", lambda: "0" * 64)
    else:
        write_json(root / "result.json", {"freeze_sha256": "0" * 64})
    with pytest.raises(ValueError):
        probe.execute_probe(root)


def test_supervisor_timeout_is_terminal_and_preserves_completed_phases(
    staged_bundle, monkeypatch
):
    plan, paths = staged_bundle
    probe.prepare_probe(plan, **paths)
    root = paths["output"]
    prior = {
        "freeze_sha256": sha256(root / "freeze.json"),
        "fit": {"marker": "retained"},
    }
    write_json(root / "fit.json", prior)
    killed = []

    class HungWorker:
        pid = 99999

        def wait(self, timeout=None):
            if timeout is not None:
                assert timeout == plan.worker_seconds
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: HungWorker())
    monkeypatch.setattr(runner.os, "killpg", lambda *args: killed.append(args))
    result = runner.run_supervised(root)
    assert result["status"] == "timeout"
    assert len(killed) == 1
    assert read_json(root / "fit.json") == prior
    assert runner.run_supervised(root) == result
    assert len(killed) == 1


def test_probe_plan_enforces_bounded_rollout_only():
    config = read_json(Path("configs/staged_fit_probe_v1.json"))
    plan = probe.StagedFitPlan.model_validate(config)
    assert plan.worker_seconds == 450
    assert plan.fit_config.number_of_starts == 3
    for key, value in (
        ("maximum_wall_time_seconds", None),
        ("allow_derivative_regression", True),
        ("nonlinear_initializer", "casadi_multiple_shooting"),
    ):
        bad = copy.deepcopy(config)
        bad["fit_config"][key] = value
        with pytest.raises(ValueError, match="bounded solve_ivp"):
            probe.StagedFitPlan.model_validate(bad)


def test_cli_freezes_runs_and_resumes_without_test_data(staged_bundle, tmp_path):
    plan, paths = staged_bundle
    config = tmp_path / "config.json"
    write_json(config, plan.model_dump(mode="json"))
    repository = Path(__file__).resolve().parents[1]
    base = [sys.executable, str(repository / "scripts/run_staged_fit_probe.py")]
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    subprocess.run(
        [
            *base,
            "prepare",
            "--config",
            str(config),
            "--function-plan",
            str(paths["function_plan"]),
            "--function-results",
            str(paths["function_results"]),
            "--public-root",
            str(paths["public_root"]),
            "--output",
            str(paths["output"]),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    command = [*base, "run", "--output", str(paths["output"])]
    subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=40,
    )
    first = read_json(paths["output"] / "result.json")
    assert first["status"] == "complete"
    subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert read_json(paths["output"] / "result.json") == first
