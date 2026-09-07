"""Frozen numerical factorial, public-data isolation and independent CLI resume."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.rebuttal import fitter_stagnation as campaign
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.staged_fit_probe import prepare_probe
from scripts import run_fitter_stagnation as runner
from tests.test_fitter_diagnostic import bundle as diagnostic_bundle  # noqa: F401
from tests.test_staged_fit_probe import staged_bundle  # noqa: F401


@pytest.fixture
def diagnosis_bundle(staged_bundle, tmp_path):  # noqa: F811
    parent_plan, paths = staged_bundle
    prepare_probe(parent_plan, **paths)
    source = paths["output"]
    write_json(
        source / "result.json",
        {
            "freeze_sha256": sha256(source / "freeze.json"),
            "default_replay": {"parameters": {"decay": 1.0}},
            "fit": {"global_parameters": {"decay": 0.72}},
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )
    plan = campaign.StagnationPlan(
        source_freeze_sha256=sha256(source / "freeze.json"),
        source_result_sha256=sha256(source / "result.json"),
        fit_seconds=10,
        profile_seconds=20,
        replay_seconds=5,
        grace_seconds=5,
        maximum_nfev=30,
        profile_trajectories=1,
        profile_steps=(1e-4, 1e-3),
    )
    return plan, source, tmp_path / "diagnosis"


def test_all_arms_hold_model_fixed_and_recover_synthetic_parameter(diagnosis_bundle):
    plan, source, output = diagnosis_bundle
    original_hash = sha256(source / "candidate.json")
    frozen = campaign.prepare_diagnosis(plan, source, output)
    assert len(frozen["tasks"]) == 5
    assert not list(output.rglob("test.csv"))
    assert sha256(output / "candidate.json") == original_hash
    results = [campaign.execute_diagnosis(output, i) for i in range(5)]
    assert all(r["status"] == "complete" for r in results)
    assert len(results[0]["reports"]) == 4
    for row in results[1:]:
        assert row["fit"]["parameters"]["decay"] == pytest.approx(0.72, abs=1e-4)
        assert row["fit"]["initial_parameters"] == {"decay": 1.0}
        assert row["fit"]["actual_residual_calls"] > row["fit"]["nfev"]
        assert row["replay"]["validation"]["normalized_mse"] < 1e-7
        assert row["test_data_opened"] is False
    assert results[1]["diff_step"] is None
    assert results[2]["diff_step"] == plan.large_diff_step
    assert results[1]["settings"]["relative_tolerance"] == 1e-7
    assert results[3]["settings"]["relative_tolerance"] == plan.tight_rtol
    assert sha256(source / "candidate.json") == original_hash
    assert sha256(output / "candidate.json") == original_hash
    summary = runner.write_summary(output)
    assert len(summary["rows"]) == 5
    assert not summary["model_selection_performed"]
    assert (output / "summary.md").exists()
    report = (output / "summary.md").read_text()
    assert "current/tight residual difference norm=" in report
    assert "Central Jacobian relative difference" in report


def test_completed_optimizer_and_profile_calls_are_not_repeated(
    diagnosis_bundle, monkeypatch
):
    plan, source, output = diagnosis_bundle
    campaign.prepare_diagnosis(plan, source, output)
    profile = campaign.execute_diagnosis(output, 0)
    fit = campaign.execute_diagnosis(output, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed numerical phase was repeated")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    monkeypatch.setattr(campaign, "replay_parameters", forbidden)
    monkeypatch.setattr(campaign.RolloutOracle, "__call__", forbidden)
    assert campaign.execute_diagnosis(output, 0) == profile
    assert campaign.execute_diagnosis(output, 1) == fit
    (output / "results/current_default/result.json").unlink()
    assert campaign.execute_diagnosis(output, 1) == fit
    (output / "results/step_profile/result.json").unlink()
    assert campaign.execute_diagnosis(output, 0) == profile


@pytest.mark.parametrize("name", ["freeze.json", "result.json", "candidate.json"])
def test_source_tampering_rejected_before_numerical_work(diagnosis_bundle, name):
    plan, source, output = diagnosis_bundle
    (source / name).write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        campaign.prepare_diagnosis(plan, source, output)


@pytest.mark.parametrize("kind", ["asset", "runtime", "launcher", "checkpoint"])
def test_frozen_resume_checks_all_identities(diagnosis_bundle, monkeypatch, kind):
    plan, source, output = diagnosis_bundle
    campaign.prepare_diagnosis(plan, source, output)
    if kind == "asset":
        (output / "candidate.json").write_text("{}")
    elif kind == "runtime":
        monkeypatch.setattr(campaign, "runtime_identity", lambda: {})
    elif kind == "launcher":
        monkeypatch.setattr(campaign, "launch_identity", lambda: "0" * 64)
    else:
        write_json(
            output / "results/current_default/result.json", {"identity": "wrong"}
        )
    with pytest.raises(ValueError, match="differs"):
        campaign.execute_diagnosis(output, 1)


def test_dependency_drift_from_original_fit_is_not_a_hidden_treatment(
    diagnosis_bundle, monkeypatch
):
    plan, source, output = diagnosis_bundle
    current = campaign.runtime_identity()
    monkeypatch.setattr(
        campaign, "runtime_identity", lambda: {**current, "python": "changed"}
    )
    with pytest.raises(ValueError, match="dependency versions differ"):
        campaign.prepare_diagnosis(plan, source, output)


def test_supervisor_timeout_preserves_partial_traces_and_records_terminal_failure(
    diagnosis_bundle, monkeypatch
):
    plan, source, output = diagnosis_bundle
    campaign.prepare_diagnosis(plan, source, output)
    root = output / "results/current_default"
    write_json(root / "attempt-0/000001.json", {"marker": "partial"})
    killed = []

    class Worker:
        pid = 4242

        def wait(self, timeout=None):
            if timeout is not None:
                assert timeout == plan.worker_seconds(1)
                raise subprocess.TimeoutExpired("test", timeout)
            return -9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Worker())
    monkeypatch.setattr(runner.os, "killpg", lambda *a: killed.append(a))
    result = runner.run_supervised(output, 1)
    assert result["status"] == "timeout"
    assert len(killed) == 1
    assert read_json(root / "attempt-0/000001.json") == {"marker": "partial"}
    assert runner.run_supervised(output, 1) == result
    assert len(killed) == 1
    assert campaign.summarize_diagnosis(output)["rows"][2]["status"] == "missing"


@pytest.mark.parametrize("steps", [(), (0.0, 1e-4), (1e-4, 1e-4), (1e-4, 1e-6)])
def test_invalid_step_scan_is_rejected(diagnosis_bundle, steps):
    plan, _, _ = diagnosis_bundle
    with pytest.raises(ValueError):
        campaign.StagnationPlan.model_validate(
            {**plan.model_dump(), "profile_steps": steps}
        )


def test_cli_prepare_supervised_fit_summary_and_resume(diagnosis_bundle, tmp_path):
    plan, source, output = diagnosis_bundle
    config = tmp_path / "diagnostic_config.json"
    write_json(config, plan.model_dump(mode="json"))
    repository = Path(__file__).resolve().parents[1]
    base = [sys.executable, str(repository / "scripts/run_fitter_stagnation.py")]
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}

    def call(*args):
        subprocess.run(
            [*base, *args, "--output", str(output)],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=40,
        )

    call("prepare", "--source", str(source), "--config", str(config))
    call("run", "--task-index", "1")
    first = read_json(output / "results/current_default/result.json")
    assert first["status"] == "complete"
    call("run", "--task-index", "1")
    assert read_json(output / "results/current_default/result.json") == first
    call("summarize")
    assert read_json(output / "summary.json")["rows"][0]["status"] == "missing"
