"""Durable scheduler intent, exact execution identity, and portable inspection."""

import json
import sys
import tarfile

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_instantaneous_campaign as campaign
from scripts import run_llm_sr_instantaneous_controls as launcher
from tests.test_llm_sr_instantaneous_campaign import inputs, policy


def ready(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, inputs(monkeypatch), policy())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/historical/campaign")


def test_confirmed_submission_resumes_without_duplicate_jobs(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(stage)
        job = str(100 + len(calls))
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
    assert calls == ["run", "report"]
    assert launcher.submit(tmp_path) == result
    assert calls == ["run", "report"]
    identity = launcher.verify_execution(tmp_path)
    assert identity["cpus_per_task"] == 1 and identity["memory"] == "8G"
    assert identity["wall_time"] == "00:15:00"
    directory = tmp_path / launcher.SUBMISSIONS
    argv = json.loads((directory / "report.intent.json").read_text())["argv"]
    assert "--dependency=afterany:101" in argv
    assert (
        "--array=0-3%2"
        in json.loads((directory / "run.intent.json").read_text())["argv"]
    )
    import os

    assert os.environ["AF_LLMSR_INSTANTANEOUS_ROOT"] == str(tmp_path)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/historical/campaign"


def test_ambiguous_submission_cannot_be_retried(tmp_path, monkeypatch):
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


def test_changed_runtime_or_worker_fails_before_execution(tmp_path, monkeypatch):
    from autoformalism.benchmarks.audited_release import seal

    ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    monkeypatch.setenv("AF_CASADI_ROOT", "/different/dependency/overlay")
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_inspection_contains_four_rows_and_operation_receipts(tmp_path, monkeypatch):
    root = tmp_path / "control"
    ready(root, monkeypatch)
    campaign.run_task(root, 0)
    campaign.run_task(root, 1)
    campaign.run_task(root, 2)
    campaign.run_task(root, 3)
    result = launcher.inspect(root)
    assert result["status"] == "complete"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "control/plan.json" in names
    assert "control/tasks/0/operations/primary/result.json" in names
    assert "control/tasks/1/result.json" in names
    assert "control/report/summary.json" in names
