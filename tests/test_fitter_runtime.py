"""End-to-end runtime campaign, exact provenance, numerical gates and resume."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import runtime_probe
from autoformalism.rebuttal import fitter_runtime as campaign
from autoformalism.rebuttal import fitter_stagnation as previous
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.staged_topology import content_hash
from scripts import run_fitter_runtime as runner
from tests.test_fitter_diagnostic import bundle as diagnostic_bundle  # noqa: F401
from tests.test_fitter_stagnation import diagnosis_bundle  # noqa: F401
from tests.test_staged_fit_probe import staged_bundle  # noqa: F401


@pytest.fixture
def runtime_bundle(diagnosis_bundle, tmp_path):  # noqa: F811
    old_plan, original, source = diagnosis_bundle
    frozen = previous.prepare_diagnosis(old_plan, original, source)
    result = previous.execute_diagnosis(source, 2)
    fit = result["fit"]
    assert len(fit["iterations"]) >= 2
    anchors = {
        "all_ones": fit["initial_parameters"],
        "intermediate": fit["iterations"][-2]["parameters"],
        "late": fit["iterations"][-1]["parameters"],
    }
    plan = campaign.RuntimePlan(
        source_code_sha256=frozen["runtime"]["source_sha256"],
        source_plan_sha256=content_hash(old_plan.model_dump(mode="json")),
        anchor_hashes={k: content_hash(v) for k, v in anchors.items()},
        profile_seconds=30,
        call_seconds=5,
        fit_seconds=10,
        replay_seconds=5,
        grace_seconds=5,
        maximum_nfev=30,
        profile_trajectories=1,
    )
    return plan, source, tmp_path / "runtime"


def test_all_nine_tasks_hold_model_fixed_and_recover_known_parameter(runtime_bundle):
    plan, source, output = runtime_bundle
    digest = sha256(source / "candidate.json")
    frozen = campaign.prepare_runtime(plan, source, output)
    assert len(frozen["tasks"]) == 9
    assert not list(output.rglob("test.csv"))
    results = [campaign.execute_runtime(output, i) for i in range(9)]
    assert all(r["status"] == "complete" for r in results)
    for profile in results[:3]:
        assert all(profile["method_gates"].values())
        assert len(profile["cases"]) == 14
        assert all(d["status"] == "complete" for d in profile["derivatives"])
        assert profile["interpreter"]["timing_comparison_eligible"] is False
    for result in results[3:]:
        assert result["fit"]["initial_parameters"] == {"decay": 1.0}
        assert result["fit"]["parameters"]["decay"] == pytest.approx(0.72, abs=1e-4)
        assert result["replays"]["DOP853"]["validation"]["normalized_mse"] < 1e-7
        assert result["replay_score_difference"]["validation"] < 1e-7
        assert (
            result["settings"]["finite_difference_policy"] == result["task"]["policy"]
        )
    assert (
        sha256(source / "candidate.json") == sha256(output / "candidate.json") == digest
    )
    summary = runner.write_summary(output)
    assert len(summary["rows"]) == 9
    assert not summary["model_selection_performed"]
    report = (output / "summary.md").read_text()
    assert "Observation expression s" in report
    assert "Difference from smaller scaled step" in report


def test_completed_profiles_fits_and_replays_are_reused(runtime_bundle, monkeypatch):
    plan, source, output = runtime_bundle
    campaign.prepare_runtime(plan, source, output)
    profiles = [campaign.execute_runtime(output, i) for i in range(3)]
    fit = campaign.execute_runtime(output, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed numerical work repeated")

    monkeypatch.setattr(runtime_probe, "measured_residual", forbidden)
    monkeypatch.setattr(runtime_probe, "simulate_trajectory", forbidden)
    monkeypatch.setattr(runtime_probe.RolloutOracle, "__call__", forbidden)
    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    monkeypatch.setattr(campaign, "replay_parameters", forbidden)
    (output / "results/profile_all_ones/result.json").unlink()
    (output / "results/fit_RK45_relative/result.json").unlink()
    assert campaign.execute_runtime(output, 0) == profiles[0]
    assert campaign.execute_runtime(output, 3) == fit


@pytest.mark.parametrize(
    "field", ["source_code_sha256", "source_plan_sha256", "anchor_hashes"]
)
def test_reviewed_code_plan_and_vectors_are_pinned(runtime_bundle, field):
    plan, source, output = runtime_bundle
    updates = {
        field: dict.fromkeys(campaign.ANCHORS, "0" * 64)
        if field == "anchor_hashes"
        else "0" * 64
    }
    with pytest.raises(ValueError, match="differ"):
        campaign.prepare_runtime(plan.model_copy(update=updates), source, output)


def test_modified_candidate_cannot_be_rebound_to_previous_manifest(runtime_bundle):
    plan, source, output = runtime_bundle
    value = read_json(source / "candidate.json")
    value["change_summary"] = "Changed after the reviewed run"
    write_json(source / "candidate.json", value)
    frozen = read_json(source / "freeze.json")
    frozen["assets"]["candidate.json"] = sha256(source / "candidate.json")
    frozen["freeze_sha256"] = content_hash(
        {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    )
    write_json(source / "freeze.json", frozen)
    with pytest.raises(ValueError, match="original snapshot"):
        campaign.prepare_runtime(plan, source, output)


@pytest.mark.parametrize("kind", ["asset", "runtime", "launcher", "checkpoint"])
def test_resume_rejects_changed_inputs_and_code(runtime_bundle, monkeypatch, kind):
    plan, source, output = runtime_bundle
    campaign.prepare_runtime(plan, source, output)
    if kind == "asset":
        (output / "candidate.json").write_text("{}")
    elif kind == "runtime":
        monkeypatch.setattr(campaign, "runtime_identity", lambda: {})
    elif kind == "launcher":
        monkeypatch.setattr(campaign, "launch_identity", lambda: "0" * 64)
    else:
        write_json(
            output / "results/profile_all_ones/result.json", {"identity": "wrong"}
        )
    with pytest.raises(ValueError, match="differ"):
        campaign.execute_runtime(output, 0)


def test_missing_accuracy_evidence_blocks_fit_without_hiding_task(
    runtime_bundle, monkeypatch
):
    plan, source, output = runtime_bundle
    campaign.prepare_runtime(plan, source, output)

    def forbidden(*args, **kwargs):
        raise AssertionError("unverified backend was fitted")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    result = campaign.execute_runtime(output, 3)
    assert result["status"] == "accuracy_guard_failed"
    assert not result["guard"]["pass"]
    assert len(campaign.summarize_runtime(output)["rows"]) == 9


def test_failed_rollout_case_is_terminal_and_preserves_partial_measurements(
    runtime_bundle, monkeypatch
):
    plan, source, output = runtime_bundle
    frozen = campaign.prepare_runtime(plan, source, output)
    parent, dataset, model, scales = campaign.load_problem(output, frozen)

    def failed(*args):
        args[-1].append({"trajectory_id": "partial", "rhs_calls": 7})
        raise TimeoutError("deadline")

    monkeypatch.setattr(runtime_probe, "measured_residual", failed)
    from time import monotonic

    arguments = (
        output / "case",
        "case",
        model,
        dataset.train,
        {"decay": 1.0},
        scales,
        parent.fit_config,
        monotonic() + 10,
        5,
    )
    record = runtime_probe.rollout_case(*arguments)
    assert record["status"] == "timeout"
    assert record["trajectories"][0]["rhs_calls"] == 7
    assert runtime_probe.rollout_case(*arguments) == record


def test_cli_profile_supervision_and_summary(runtime_bundle, tmp_path):
    plan, source, output = runtime_bundle
    config = tmp_path / "runtime_config.json"
    write_json(config, plan.model_dump(mode="json"))
    repository = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repository / "scripts/run_fitter_runtime.py")]

    def call(*args):
        subprocess.run(
            [*command, *args, "--output", str(output)],
            check=True,
            cwd=repository,
            env={**os.environ, "PYTHONPATH": str(repository / "src")},
            capture_output=True,
            text=True,
            timeout=60,
        )

    call("prepare", "--source", str(source), "--config", str(config))
    call("run", "--task-index", "0")
    first = read_json(output / "results/profile_all_ones/result.json")
    assert first["status"] == "complete"
    call("run", "--task-index", "0")
    assert read_json(output / "results/profile_all_ones/result.json") == first
    call("summarize")
    assert read_json(output / "summary.json")["rows"][1]["status"] == "missing"


@pytest.mark.parametrize("failure", ["inaccurate_rk45", "failed_radau"])
def test_inaccurate_backend_is_blocked_and_reference_fallback_is_explicit(
    runtime_bundle, monkeypatch, failure
):
    plan, source, output = runtime_bundle
    campaign.prepare_runtime(plan, source, output)
    original = runtime_probe.measured_residual

    def altered(model, training, parameters, scales, settings, deadline, profiles):
        if failure == "failed_radau" and settings.integration_method == "Radau":
            raise ValueError("synthetic reference integration failure")
        values = original(
            model, training, parameters, scales, settings, deadline, profiles
        )
        if failure == "inaccurate_rk45" and settings.integration_method == "RK45":
            values = values + 0.1
        return values

    monkeypatch.setattr(runtime_probe, "measured_residual", altered)
    result = campaign.execute_runtime(output, 0)
    assert result["method_gates"]["DOP853"]
    if failure == "inaccurate_rk45":
        assert not result["method_gates"]["RK45"]
        assert result["reference"] == "Radau_reference"
    else:
        assert not result["method_gates"]["Radau"]
        assert result["reference"] == "DOP853_reference"


def test_configured_step_and_floor_are_used_in_derivative_profile(runtime_bundle):
    plan, source, output = runtime_bundle
    plan = plan.model_copy(update={"step": 2e-4, "scale_floor": 2.0})
    campaign.prepare_runtime(plan, source, output)
    result = campaign.execute_runtime(output, 0)
    records = {d["name"]: d for d in result["derivatives"]}
    assert records["relative"]["actual_steps"][0] == pytest.approx(2e-4)
    assert records["scaled"]["actual_steps"][0] == pytest.approx(4e-4)
    assert records["scaled_smaller"]["actual_steps"][0] == pytest.approx(4e-5)


def test_supervisor_bounds_fit_worker_and_does_not_repeat_terminal_failure(
    runtime_bundle, monkeypatch
):
    plan, source, output = runtime_bundle
    campaign.prepare_runtime(plan, source, output)
    killed = []

    class Worker:
        pid = 4242

        def wait(self, timeout=None):
            if timeout is not None:
                assert (
                    timeout
                    == plan.fit_seconds + 2 * plan.replay_seconds + plan.grace_seconds
                )
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Worker())
    monkeypatch.setattr(runner.os, "killpg", lambda *a: killed.append(a))
    first = runner.run_supervised(output, 3)
    assert first["status"] == "timeout"
    assert runner.run_supervised(output, 3) == first
    assert len(killed) == 1


@pytest.mark.parametrize("fail_submission", [False, True])
def test_submission_dependencies_and_partial_manifest_prevent_duplicate_jobs(
    runtime_bundle, tmp_path, fail_submission
):
    import json

    plan, source, output = runtime_bundle
    config = tmp_path / "submit_config.json"
    write_json(config, plan.model_dump(mode="json"))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = "rev-parse" ]; then '
        'echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; fi\n'
    )
    git.chmod(0o755)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
path=Path(os.environ["FAKE_SUBMISSIONS"])
rows=path.read_text().splitlines() if path.exists() else []
with path.open("a") as stream:
    stream.write(json.dumps(sys.argv[1:])+"\\n")
if os.environ.get("FAKE_FAIL_SECOND") == "1" and len(rows) == 1:
    sys.exit(42)
print(10000+len(rows))
"""
    )
    sbatch.chmod(0o755)
    repository = Path(__file__).resolve().parents[1]
    log = tmp_path / "submissions.jsonl"
    environment = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repository),
        "AF_PYTHON": sys.executable,
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(output),
        "AF_CONFIG": str(config),
        "FAKE_SUBMISSIONS": str(log),
        "FAKE_FAIL_SECOND": "1" if fail_submission else "0",
    }

    def run():
        return subprocess.run(
            ["bash", "scripts/hpc/submit_fitter_runtime_delta.sh"],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    first = run()
    assert first.returncode == (42 if fail_submission else 0), first.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--array=0-2%2" in rows[0]
    assert "--dependency=afterany:10000" in rows[1]
    assert "--array=3-8%2" in rows[1]
    manifest = read_json(output / "submission.json")
    assert manifest["profile_job_id"] == "10000"
    if fail_submission:
        assert "fit_job_id" not in manifest
    else:
        assert "--dependency=afterany:10001" in rows[2]
        assert manifest["summary_job_id"] == "10002"
    assert run().returncode == 0
    assert len(log.read_text().splitlines()) == len(rows)
