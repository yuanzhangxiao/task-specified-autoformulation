"""Scheduler recovery and portable inspection for active-block continuations."""

import json
import sys
import tarfile
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import public_fitting as public
from scripts import run_llm_sr_detention_lazy_controls as launcher


def ready(root, monkeypatch):
    """Isolate launcher tests from the separately tested numerical campaign."""
    plan = {"source_sha256": "a" * 64, "tasks": list(range(27))}
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
    assert calls == ["qualify", "run", "report"]
    assert result["tasks"] == 27 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    directory = tmp_path / launcher.SUBMISSIONS
    run = json.loads((directory / "run.intent.json").read_text())["argv"]
    report = json.loads((directory / "report.intent.json").read_text())["argv"]
    qualify = json.loads((directory / "qualify.intent.json").read_text())["argv"]
    assert "--array=0-5%2" in qualify and "--time=02:00:00" in qualify
    assert "--array=6-26%2" in run and "--dependency=afterany:102" in run
    assert "--time=04:00:00" in run and "--cpus-per-task=1" in run
    assert "--time=00:30:00" in report and "--dependency=afterany:103" in report
    import os

    assert os.environ["AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT"] == str(tmp_path)
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
    assert calls == ["qualify"]


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
    summary = {"status": "incomplete", "expected": 5, "recorded": 0}
    monkeypatch.setattr(launcher.campaign, "report", lambda root: summary)
    seal(root / "tasks/0/operations/direction/started.json", {"reserved": 600})
    result = launcher.inspect(root)
    assert result["status"] == "incomplete"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "refit/inputs.json" in names
    assert "refit/tasks/0/operations/direction/started.json" in names
    assert "refit/report/summary.json" in names


def test_compact_status_keeps_phase_progress_without_large_receipts():
    summary = {
        "protocol": "llm-sr-detention-lazy-controls-1",
        "status": "incomplete",
        "expected": 5,
        "recorded": 1,
        "tasks": [
            {
                "identity": {
                    "task": {"index": 0, "kind": "fit", "start_id": "generic_1"}
                },
                "status": "incomplete",
                "retained": {"training_nmse": 0.2},
                "phases_completed": 1,
                "active_names": ["C"],
                "held_names": ["eps_s"],
                "operations": {"large_saved_arrays": [0.0] * 1000},
            }
        ],
    }
    output = launcher.compact_report(summary)
    assert output["tasks"][0]["training_nmse"] == 0.2
    assert output["tasks"][0]["held_names"] == ["eps_s"]
    assert "operations" not in output["tasks"][0]


def test_worker_shares_one_authenticated_context(tmp_path, monkeypatch, capsys):
    """Execution pinning must not repeat expensive history authentication."""
    context = SimpleNamespace(plan={"source_sha256": "b" * 64})
    calls = []

    def load(root):
        assert root == tmp_path
        calls.append("load")
        return context

    def verify(root, *, verified_plan):
        assert root == tmp_path and verified_plan is context.plan
        calls.append("execution")

    def run(root, index, *, context):
        assert root == tmp_path and index == 6
        calls.append("run")
        return {"status": "complete", "retained": {"training_nmse": 0.1}}

    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher, "verify_execution", verify)
    monkeypatch.setattr(launcher.campaign, "run_task", run)
    launcher.main(["run", "--root", str(tmp_path), "--index", "6"])
    assert calls == ["load", "execution", "run"]
    assert json.loads(capsys.readouterr().out)["training_nmse"] == 0.1
