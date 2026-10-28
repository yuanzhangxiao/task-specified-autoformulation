"""Portable six-task launcher receipts, progress and read-only inspection."""

import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import public_fitting as public
from scripts import run_llm_sr_detention_probe_recovery as launcher


def ready(root, monkeypatch):
    """Separate scheduler contract tests from numerical campaign execution."""
    plan = {
        "protocol": "llm-sr-detention-probe-recovery-1",
        "source_sha256": "a" * 64,
        "tasks": [
            {"index": index, "kind": "continue" if index < 2 else "diagnostic"}
            for index in range(6)
        ],
    }
    context = SimpleNamespace(plan=plan)
    monkeypatch.setattr(launcher.campaign, "load_run", lambda root: context)
    monkeypatch.setattr(launcher.campaign, "verify", lambda root: (plan, None, None))
    monkeypatch.setattr(launcher.public, "_runtime", lambda: {"python": "test"})
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/campaign")
    seal(root / "plan.json", plan)
    seal(root / "inputs.json", {"protocol": "portable-training-endpoints"})
    return context


def record_submission(directory, key, options, worker, stage, index):
    """Write the same durable evidence as a successful mocked sbatch reply."""
    job = "101" if stage == "run" else "102"
    argv = ["sbatch", "--parsable", *options, str(worker), stage, str(index)]
    public._write(directory / f"{key}.intent.json", {"argv": argv})
    public._write(
        directory / f"{key}.reply.json",
        {"returncode": 0, "stdout": job + "\n", "stderr": ""},
    )
    (directory / f"{key}.id").write_text(job + "\n")
    return job


def test_submission_limits_deduplication_and_immediate_progress(
    tmp_path, monkeypatch, capsys
):
    context = ready(tmp_path, monkeypatch)
    calls = []
    loads = []

    def load(root):
        loads.append(root)
        return context

    def submit(*args):
        calls.append(args[1])
        return record_submission(*args)

    def forbidden_verify(*args, **kwargs):
        raise AssertionError("already authenticated campaign context")

    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher.campaign, "verify", forbidden_verify)
    monkeypatch.setattr(launcher, "submit_job", submit)
    result = launcher.submit(tmp_path)
    messages = capsys.readouterr().err
    assert messages.index("Input authentication complete") < messages.index(
        "Submitting run"
    )
    assert messages.index("Slurm accepted run: job 101") < messages.index(
        "Submitting report"
    )
    assert launcher.submit(tmp_path) == result
    assert calls == ["run", "report"]
    assert loads == [tmp_path, tmp_path]
    assert result["tasks"] == 6 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    directory = tmp_path / launcher.SUBMISSIONS
    run = json.loads((directory / "run.intent.json").read_text())["argv"]
    report = json.loads((directory / "report.intent.json").read_text())["argv"]
    assert "--array=0-5%2" in run and "--time=06:00:00" in run
    assert "--cpus-per-task=1" in run and "--mem=8G" in run
    assert "--time=00:30:00" in report and "--dependency=afterany:101" in report
    assert os.environ["AF_LLMSR_DETENTION_PROBE_RECOVERY_ROOT"] == str(tmp_path)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/campaign"


@pytest.mark.parametrize("uncertain_stage", ["run", "report"])
def test_uncertain_intent_never_resubmits_array_or_report(
    tmp_path, monkeypatch, uncertain_stage
):
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(stage)
        if stage == uncertain_stage:
            public._write(directory / f"{key}.intent.json", {"argv": []})
            raise ValueError("uncertain scheduler reply")
        return record_submission(directory, key, options, worker, stage, index)

    monkeypatch.setattr(launcher, "submit_job", submit)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    assert calls == (["run"] if uncertain_stage == "run" else ["run", "report"])


def test_execution_identity_rejects_changed_dependencies(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    monkeypatch.setenv("AF_CASADI_ROOT", "/different/dependencies")
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_worker_shares_one_authenticated_context(tmp_path, monkeypatch, capsys):
    context = ready(tmp_path, monkeypatch)
    calls = []

    def load(root):
        assert root == tmp_path
        calls.append("load")
        return context

    def verify(root, *, verified_plan):
        assert root == tmp_path and verified_plan is context.plan
        calls.append("execution")

    def run(root, index, *, context):
        assert root == tmp_path and index == 5
        assert context is shared_context
        calls.append("run")
        return {"status": "complete", "retained": {"training_nmse": 0.1}}

    shared_context = context
    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher, "verify_execution", verify)
    monkeypatch.setattr(launcher.campaign, "run_task", run)
    launcher.main(["run", "--root", str(tmp_path), "--index", "5"])
    assert calls == ["load", "execution", "run"]
    captured = capsys.readouterr()
    assert json.loads(captured.out)["training_nmse"] == 0.1
    assert "Task 5: authentication complete" in captured.err


def test_lightweight_status_does_not_load_inputs_or_recompute_report(
    tmp_path, monkeypatch
):
    context = ready(tmp_path, monkeypatch)
    plan = context.plan
    receipt = {
        "identity": {
            "plan_sha256": public.content_sha256(plan),
            "task": plan["tasks"][0],
        },
        "status": "complete",
        "diagnostic_status": "passed",
        "retained": {"training_nmse": 0.01},
    }
    seal(tmp_path / "tasks/0/result.json", receipt)
    seal(tmp_path / "tasks/1/operations/initial/started.json", {"reserved": 360})
    (tmp_path / "inputs.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("lightweight status must not replay/authenticate history")

    for name in ("report", "verify", "load_run", "run_task"):
        monkeypatch.setattr(launcher.campaign, name, forbidden)
    result = launcher.status(tmp_path)
    assert result["status"] == "incomplete" and result["recorded"] == 1
    assert result["tasks"][0]["training_nmse"] == 0.01
    assert result["tasks"][1]["started_operations"] == 1
    assert result["fitting_performed"] is False
    assert result["submission_performed"] is False


@pytest.mark.parametrize("changed_field", ["plan_sha256", "task"])
def test_status_rejects_receipt_for_another_frozen_task(
    tmp_path, monkeypatch, changed_field
):
    plan = ready(tmp_path, monkeypatch).plan
    identity = {
        "plan_sha256": public.content_sha256(plan),
        "task": plan["tasks"][0],
    }
    identity[changed_field] = "wrong"
    seal(tmp_path / "tasks/0/result.json", {"identity": identity, "status": "complete"})
    with pytest.raises(ValueError, match="receipt differs"):
        launcher.status(tmp_path)


def test_inspect_includes_probe_caches_and_operation_receipts(tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    ready(root, monkeypatch)
    summary = {"status": "incomplete", "expected": 6, "recorded": 0}
    monkeypatch.setattr(launcher.campaign, "report", lambda root: summary)
    seal(root / "tasks/0/operations/direction/started.json", {"reserved": 360})
    seal(root / "tasks/0/probe-cache/endpoint/base.json", {"residual": [1.0]})
    result = launcher.inspect(root)
    assert result["status"] == "incomplete"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "recovery/inputs.json" in names
    assert "recovery/tasks/0/operations/direction/started.json" in names
    assert "recovery/tasks/0/probe-cache/endpoint/base.json" in names
    assert "recovery/report/summary.json" in names


@pytest.mark.parametrize("index, expected_code", [("3", 0), ("6", 2)])
def test_shell_worker_selects_array_index_and_rejects_out_of_roster(
    tmp_path, index, expected_code
):
    """Exercise actual shell routing without importing numerical dependencies."""
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "AF_REPO_ROOT": str(launcher.REPO),
        "AF_LLMSR_DETENTION_PROBE_RECOVERY_ROOT": str(tmp_path / "campaign"),
        "AF_PYTHON": str(fake_python),
        "SLURM_ARRAY_TASK_ID": index,
    }
    result = subprocess.run(
        [
            "bash",
            str(
                launcher.REPO
                / "scripts/hpc/run_llm_sr_detention_probe_recovery_delta.sh"
            ),
            "run",
            "0",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_code
    if expected_code == 0:
        assert result.stdout.splitlines()[-2:] == ["--index", "3"]
        assert "Detention probe recovery: stage=run" in result.stderr
    else:
        assert "Expected task index 0 through 5" in result.stderr
        assert not result.stdout
