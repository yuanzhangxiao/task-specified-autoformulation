"""Scheduler recovery and portable inspection for the detention refit control."""

import json
import sys
import tarfile

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import public_fitting as public
from scripts import run_llm_sr_detention_refit as launcher


def ready(root, monkeypatch):
    """Isolate launcher tests from the separately tested numerical campaign."""
    plan = {"source_sha256": "a" * 64, "tasks": list(range(10))}
    monkeypatch.setattr(launcher.campaign, "verify", lambda root: (plan, None, None))
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/campaign")
    seal(root / "plan.json", plan)
    seal(root / "inputs.json", {"protocol": "portable-training-history"})


def test_submission_has_matched_cpu_limits_and_no_duplicate_retry(
    tmp_path, monkeypatch
):
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(stage)
        job = str(101 + len(calls))
        argv = ["sbatch", "--parsable", *options, str(worker), stage, str(index)]
        public._write(directory / f"{key}.intent.json", {"argv": argv})
        public._write(
            directory / f"{key}.reply.json",
            {"returncode": 0, "stdout": job + "\n", "stderr": ""},
        )
        (directory / f"{key}.id").write_text(job + "\n")
        return job

    monkeypatch.setattr(launcher, "submit_job", submit)
    result = launcher.submit(tmp_path)
    assert launcher.submit(tmp_path) == result
    assert calls == ["run", "report"]
    assert result["tasks"] == 10 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    directory = tmp_path / launcher.SUBMISSIONS
    run = json.loads((directory / "run.intent.json").read_text())["argv"]
    report = json.loads((directory / "report.intent.json").read_text())["argv"]
    assert "--array=0-9%2" in run
    assert "--time=00:30:00" in run and "--cpus-per-task=1" in run
    assert "--time=00:15:00" in report and "--dependency=afterany:102" in report
    import os

    assert os.environ["AF_LLMSR_DETENTION_REFIT_ROOT"] == str(tmp_path)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/campaign"


def test_uncertain_scheduler_response_blocks_duplicate_submission(
    tmp_path, monkeypatch
):
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(stage)
        public._write(directory / f"{key}.intent.json", {"argv": []})
        raise ValueError("uncertain scheduler reply")

    monkeypatch.setattr(launcher, "submit_job", submit)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    assert calls == ["run"]


def test_execution_identity_rejects_changed_dependencies(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    monkeypatch.setenv("AF_CASADI_ROOT", "/different/dependencies")
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_inspection_includes_portable_inputs_and_read_only_progress(
    tmp_path, monkeypatch
):
    root = tmp_path / "refit"
    ready(root, monkeypatch)
    summary = {"status": "incomplete", "expected": 10, "recorded": 0}
    monkeypatch.setattr(launcher.campaign, "report", lambda root: summary)
    seal(root / "tasks/0/operations/optimization/started.json", {"reserved": 600})
    result = launcher.inspect(root)
    assert result["status"] == "incomplete"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "refit/inputs.json" in names
    assert "refit/tasks/0/operations/optimization/started.json" in names
    assert "refit/report/summary.json" in names
