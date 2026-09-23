"""Scheduler-only recovery preserves accepted work and frozen scientific inputs."""

import getpass
import json
import shlex
import subprocess

import pytest

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from scripts import recover_review_continuation_submission as recovery
from scripts.smoke_review_multi import fixture
from tests.test_review_continuation import ROOT, _submit_module


def control(argv, *, state="RUNNING", job="2157971", user=None):
    name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
    return (
        f"JobId={job} JobName={name} UserId={user or getpass.getuser()}(123) "
        f"JobState={state} ExitCode=0:0 "
        f"SubmitLine={shlex.join(['/usr/bin/sbatch', *argv[1:]])} "
        "WorkDir=/home/example"
    )


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "continued"
    fixture(source, fitted_only=True)
    plan = continuation.prepare(source, root, visits=3, protocol=io.INTEGRITY_PROTOCOL)
    original = _submit_module()
    monkeypatch.setenv("AF_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        path = tmp_path / key
        path.write_text("fixture")
        monkeypatch.setenv(key, str(path))
    for key in ("AF_HF_HOME", "AF_COMPUTE_CACHE_ROOT", "AF_IPC_TMP_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / key))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda argv, **kw: "" if "status" in argv else "a" * 40,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv,
            0,
            "\n",
            "sbatch: error: Batch job submission failed: "
            "Socket timed out on send/recv operation\n",
        ),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        original.submit(root, 1)
    directory = root / "submission-intent/round-1"
    argv = json.loads((directory / "prepare-0.intent.json").read_text())["argv"]
    calls = []

    def scheduler(command, **kw):
        if command[0] == "scontrol":
            return subprocess.CompletedProcess(command, 0, control(argv), "")
        assert command[0] == "sbatch"
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(3000000 + len(calls)), "")

    monkeypatch.setattr(subprocess, "run", scheduler)
    monkeypatch.setattr(recovery, "load_submitter", lambda repo: original)
    return root, directory, argv, calls, plan, scheduler, original


def test_adopt_prepare_and_resume_original_dispatch(campaign):
    root, directory, _argv, calls, plan, _, original = campaign
    paths = [
        root / "plan.json",
        directory / "prepare-0.reply.json",
        directory / "prepare-0.intent.json",
    ]
    paths += list((root / "results").rglob("result.json"))
    before = {p: p.read_bytes() for p in paths}
    result = recovery.recover(root, "2157971")
    assert result["jobs"]["prepare-0"] == "2157971"
    assert result["plan_sha256"] == plan["artifact_sha256"]
    assert result["commit"] == "a" * 40
    assert [call[-2:] for call in calls] == [
        ["propose", "1"],
        ["fit", "1"],
        ["finish", "1"],
        ["dispatch", "2"],
    ]
    assert "--dependency=afterok:2157971" in calls[0]
    assert "--dependency=afterok:3000001" in calls[1]
    assert "--dependency=afterany:3000002" in calls[2]
    assert "--dependency=afterok:3000003" in calls[3]
    assert {p: p.read_bytes() for p in paths} == before
    assert recovery.recover(root, "2157971") == result and len(calls) == 4
    assert original.submit(root, 1) == result and len(calls) == 4
    with pytest.raises(ValueError, match="identity differs"):
        recovery.recover(root, "999")


@pytest.mark.parametrize("change", ["owner", "command", "failed", "job"])
def test_reject_wrong_or_failed_prepare(campaign, monkeypatch, change):
    root, directory, argv, calls, *_ = campaign
    text = control(argv)
    if change == "owner":
        text = control(argv, user="somebody_else")
    elif change == "command":
        text = text.replace(str(root), str(root) + "-other")
    elif change == "failed":
        text = control(argv, state="FAILED")
    else:
        text = control(argv, job="999")

    def scheduler(command, **kw):
        return subprocess.CompletedProcess(
            command, 0, text if command[0] == "scontrol" else "", ""
        )

    monkeypatch.setattr(subprocess, "run", scheduler)
    with pytest.raises(ValueError):
        recovery.recover(root, "2157971")
    assert not calls and not (directory / "prepare-0.id").exists()


def test_retired_successful_prepare_uses_accounting_without_stale_dependency(
    campaign, monkeypatch
):
    root, directory, argv, calls, _, scheduler, _ = campaign

    def completed(command, **kw):
        if command[0] == "scontrol":
            return subprocess.CompletedProcess(command, 1, "", "Invalid job id")
        if command[0] == "sacct":
            line = (
                f"2157971|{getpass.getuser()}|review-v8-prepare-0|COMPLETED|0:0|"
                f"{shlex.join(argv)}\n"
            )
            return subprocess.CompletedProcess(command, 0, line, "")
        return scheduler(command, **kw)

    monkeypatch.setattr(subprocess, "run", completed)
    recovery.recover(root, "2157971")
    assert not any(a.startswith("--dependency") for a in calls[0])
    assert (
        json.loads((directory / "prepare-0.adoption.json").read_text())["scheduler"][
            "source"
        ]
        == "sacct"
    )


def test_another_timeout_is_not_retried(campaign, monkeypatch):
    root, directory, _, calls, _, scheduler, _ = campaign

    def fail(command, **kw):
        if command[0] == "sbatch" and command[-2] == "fit":
            return subprocess.CompletedProcess(command, 0, "", "Socket timed out")
        return scheduler(command, **kw)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="unconfirmed"):
        recovery.recover(root, "2157971")
    assert len(calls) == 1
    receipt = (directory / "fit-1.reply.json").read_bytes()
    with pytest.raises(ValueError, match="uncertain fit-1"):
        recovery.recover(root, "2157971")
    assert len(calls) == 1 and (directory / "fit-1.reply.json").read_bytes() == receipt


def test_corrupt_prerequisites_stop_before_submission(campaign):
    root, directory, _, calls, *_ = campaign
    (directory / "prerequisites.json").write_text("{}")
    with pytest.raises(ValueError, match="prerequisites differ"):
        recovery.recover(root, "2157971")
    assert not calls


def test_load_original_submitter():
    assert recovery.load_submitter(ROOT).io.REPO == ROOT


def test_completed_nonzero_exit_is_not_adopted(campaign, monkeypatch):
    root, directory, argv, calls, *_ = campaign
    text = control(argv, state="COMPLETED").replace("ExitCode=0:0", "ExitCode=1:0")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 0, text, ""),
    )
    with pytest.raises(ValueError, match="exit code"):
        recovery.recover(root, "2157971")
    assert not calls and not (directory / "prepare-0.id").exists()
