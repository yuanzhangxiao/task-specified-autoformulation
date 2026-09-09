"""Independent reference checks, active recovery and safe campaign resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import simulate_trajectory
from autoformalism.rebuttal import fitter_recovery as campaign
from autoformalism.rebuttal.fitter_diagnostic import read_json
from autoformalism.staged_topology import content_hash
from scripts import run_fitter_recovery as runner

CONFIG = Path(__file__).resolve().parents[1] / "configs/fitter_active_recovery_v1.json"


def small_plan() -> campaign.RecoveryPlan:
    """Short, nonconstant fixture, without benchmark tables or latent-state labels."""
    payload = read_json(CONFIG)
    payload["cases"] = payload["cases"][:1]
    payload.update(
        fit_seconds=40,
        guard_seconds=120,
        replay_seconds=10,
        grace_seconds=5,
        maximum_nfev=80,
        sample_step=0.4,
    )
    payload["inputs"] = [
        {
            "name": "train_pulse",
            "split": "train",
            "times": [0, 0.8, 1.2, 2.8, 3.2, 5.6, 6, 8],
            "values": [0, 0, 1, 1, -1, -1, 0, 0],
        },
        {
            "name": "train_reverse",
            "split": "train",
            "times": [0, 0.8, 1.2, 2.8, 3.2, 5.6, 6, 8],
            "values": [0, 0, -1, -1, 1.5, 1.5, 0, 0],
        },
        {
            "name": "validation_later",
            "split": "validation",
            "times": [0, 1.6, 2, 4.4, 4.8, 6.8, 7.2, 8],
            "values": [0, 0, 0.5, 0.5, -0.8, -0.8, 0, 0],
        },
    ]
    return campaign.RecoveryPlan.model_validate(payload)


def test_reference_agrees_with_closed_form_memories_and_compiled_model():
    plan = small_plan()
    truth = plan.cases[0].truth.model_dump()
    forcing = campaign.RecoveryInput(
        name="constant", split="train", times=(0, 8), values=(1, 1)
    )
    time = np.arange(21) * 0.4
    first = campaign.reference_rollout(forcing, truth, time, "Radau", monotonic() + 15)
    second = campaign.reference_rollout(
        forcing, truth, time, "DOP853", monotonic() + 15
    )
    analytic_m = truth["k_u"] * truth["tau"] * (1 - np.exp(-time / truth["tau"]))
    analytic_p = (
        truth["k_u"]
        * truth["tau"]
        * (
            truth["tau_p"] * (1 - np.exp(-time / truth["tau_p"]))
            - (np.exp(-time / truth["tau"]) - np.exp(-time / truth["tau_p"]))
            / (1 / truth["tau_p"] - 1 / truth["tau"])
        )
    )
    assert np.max(np.abs(first - second)) < 1e-8
    assert np.max(np.abs(first[:, 3] - analytic_m)) < 1e-9
    assert np.max(np.abs(first[:, 4] - analytic_p)) < 1e-9
    assert (
        np.max(np.abs(campaign.exact_memories(forcing, truth, time) - first[:, 3:5]))
        < 1e-9
    )
    trajectory = Trajectory(
        "constant",
        time.copy(),
        {"v01": np.zeros(len(time))},
        {},
        {"u01": np.ones(len(time))},
        {},
        {},
    )
    model = compile_candidate(campaign.recovery_candidate(), campaign.CONTEXT)
    simulated = simulate_trajectory(
        model, trajectory, truth, {}, plan.settings(), reset_observed_states=False
    )
    assert simulated.success
    assert np.max(np.abs(simulated.predictions["v01"] - first[:, 2])) < 1e-6


def test_reference_zero_gains_reduce_to_signed_constant():
    truth = {**small_plan().cases[0].truth.model_dump(), "k": 0.0, "k_u": 0.0}
    values = campaign.reference_rollout(
        small_plan().inputs[0], truth, np.arange(21) * 0.4, "Radau", monotonic() + 5
    )
    assert np.all(values[:, 2] == truth["c"])
    assert np.all(values[:, 3:] == 0)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "off_grid", "missing_validation", "nonpositive_truth"]
)
def test_invalid_synthetic_recipe_is_rejected(mutation):
    payload = read_json(CONFIG)
    if mutation == "duplicate":
        payload["inputs"][1]["name"] = payload["inputs"][0]["name"]
    elif mutation == "off_grid":
        payload["inputs"][0]["times"][1] = 1.01
    elif mutation == "missing_validation":
        payload["inputs"] = [i for i in payload["inputs"] if i["split"] == "train"]
    else:
        payload["cases"][0]["truth"]["k_u"] = 0
    with pytest.raises(ValueError):
        campaign.RecoveryPlan.model_validate(payload)


def test_production_freeze_is_complete_and_deterministic(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare must not integrate or optimize")

    monkeypatch.setattr(campaign, "reference_rollout", forbidden)
    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    plan = campaign.RecoveryPlan.model_validate(read_json(CONFIG))
    first = campaign.prepare_recovery(plan, tmp_path)
    assert first == campaign.prepare_recovery(plan, tmp_path)
    assert campaign.verify_recovery(tmp_path) == first
    assert len(first["tasks"]) == 8
    assert (
        first["starts"]["moderate"]["all_ones"]
        == first["starts"]["separated"]["all_ones"]
    )
    assert first["starts"]["moderate"]["broad"] == first["starts"]["separated"]["broad"]
    assert first["plan"] == plan.model_dump(mode="json")
    assert not (tmp_path / "results").exists()


@pytest.fixture
def guarded(tmp_path):
    plan = small_plan()
    frozen = campaign.prepare_recovery(plan, tmp_path)
    result = campaign.execute_recovery(tmp_path, 0)
    assert result["status"] == "complete", result
    return plan, tmp_path, frozen, result


def test_active_truth_and_local_trajectory_recovery(guarded, monkeypatch):
    plan, output, _, guard = guarded
    assert guard["reference_pass"] and guard["activity_pass"] and guard["truth_pass"]
    assert (
        min(guard["component_sd_over_target_sd"].values())
        >= plan.minimum_component_sd_fraction
    )
    dataset = campaign._dataset(
        output / "results/guard_moderate/reference", guard["records"], guard["identity"]
    )
    assert not hasattr(dataset, "test")
    assert not dataset.train.trajectories[0].derivatives
    assert not dataset.train.trajectories[0].auxiliaries
    assert set(dataset.train.trajectories[0].targets) == {"v01"}
    result = campaign.execute_recovery(output, 3)
    assert result["oracle_assisted_start"]
    assert result["trajectory_recovered"], result
    assert not result["response_collapsed"]
    assert result["scores"]["validation"] <= plan.recovery_nmse
    assert result["fit"]["actual_residual_calls"] > 7

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fit was repeated")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    assert campaign.execute_recovery(output, 3) == result
    (output / "results/fit_moderate_near_truth_control/result.json").unlink()
    rebuilt = campaign.execute_recovery(output, 3)
    assert rebuilt["fit"] == result["fit"]
    assert rebuilt["replays"] == result["replays"]
    assert campaign.summarize_recovery(output)["rows"][1]["status"] == "missing"


def test_modified_reference_array_is_rejected_on_completed_resume(guarded, monkeypatch):
    _, output, _, guard = guarded
    path = output / "results/guard_moderate/reference" / guard["records"][0]["array"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        campaign.execute_recovery(output, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("changed reference reached optimizer")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    result = campaign.execute_recovery(output, 1)
    assert result["status"] == "failed"
    assert "digest" in result["error"]


def test_failed_guard_cannot_start_optimizer(tmp_path, monkeypatch):
    campaign.prepare_recovery(small_plan(), tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("controlled reference timeout")

    monkeypatch.setattr(campaign, "reference_rollout", fail)
    guard = campaign.execute_recovery(tmp_path, 0)
    assert guard["status"] == "failed"

    def forbidden(*args, **kwargs):
        raise AssertionError("failed case reached optimizer")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    fit = campaign.execute_recovery(tmp_path, 1)
    assert fit["status"] == "truth_guard_failed"
    assert campaign.execute_recovery(tmp_path, 0) == guard


def test_freeze_candidate_tampering_fails_closed(tmp_path):
    campaign.prepare_recovery(small_plan(), tmp_path)
    (tmp_path / "candidate.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        campaign.verify_recovery(tmp_path)


def test_cli_prepare_summary_and_invalid_index(tmp_path):
    repo = CONFIG.parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    command = [sys.executable, "scripts/run_fitter_recovery.py"]

    def run(*args):
        return subprocess.run(
            [*command, *args, "--output", str(tmp_path)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    assert "tasks 8" in run("prepare", "--config", str(CONFIG)).stdout
    assert "summary.md" in run("summarize").stdout
    assert run("run", "--task-index", "-1").returncode != 0
    assert len(read_json(tmp_path / "summary.json")["rows"]) == 8


@pytest.mark.parametrize("failure_at", [0, 1, 2])
def test_submission_records_success_and_uncertainty_without_duplicates(
    tmp_path, failure_at
):
    repo = CONFIG.parents[1]
    bins = tmp_path / "bin"
    bins.mkdir()
    git = bins / "git"
    git.write_text('#!/bin/sh\nif [ "$1" = rev-parse ]; then echo fakecommit; fi\n')
    git.chmod(0o755)
    sbatch = bins / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
p=Path(os.environ['FAKE_LOG'])
rows=p.read_text().splitlines() if p.exists() else []
with p.open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
if len(rows)+1==int(os.environ['FAILURE_AT']): sys.exit(42)
print(10000+len(rows))
"""
    )
    sbatch.chmod(0o755)
    output = tmp_path / "output"
    log = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": sys.executable,
        "AF_OUTPUT_ROOT": str(output),
        "AF_CONFIG": str(CONFIG),
        "FAKE_LOG": str(log),
        "FAILURE_AT": str(failure_at),
    }

    def run():
        return subprocess.run(
            ["bash", "scripts/hpc/submit_fitter_recovery_delta.sh"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )

    first = run()
    assert first.returncode == (42 if failure_at else 0), first.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--array=0,1%2" in rows[0]
    if failure_at != 1:
        assert "--array=2,3,4,5,6,7%2" in rows[1]
        assert "--dependency=afterany:10000" in rows[1]
    assert (output / "submission.intent").is_dir()
    run()
    assert len(log.read_text().splitlines()) == len(rows)


def test_supervisor_hard_timeout_is_terminal(tmp_path, monkeypatch):
    frozen = campaign.prepare_recovery(small_plan(), tmp_path)

    class Worker:
        pid = 12345

        def __init__(self):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

    worker = Worker()
    signals = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: worker)
    monkeypatch.setattr(runner.os, "killpg", lambda *args: signals.append(args))
    result = runner.run_supervised(tmp_path, 0)
    assert result["status"] == "timeout"
    assert signals
    assert result["identity"] == content_hash(
        [frozen["freeze_sha256"], frozen["tasks"][0]]
    )
    assert runner.run_supervised(tmp_path, 0) == result
