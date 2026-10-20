"""Lost scheduler acknowledgements never require another GPU submission."""

import fcntl
import getpass
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import recover_construction_functions_submission as recovery


def write(path, value, *, sealed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        value = {**value, "artifact_sha256": recovery.content_hash(value)}
    path.write_text(json.dumps(value, indent=2))
    return value


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A sealed tiny campaign with the actual launcher shape and timeout receipt."""
    repo, root = tmp_path / "pinned", tmp_path / "campaign"
    for name in [*recovery.ENTRYPOINTS, "src/autoformalism/__init__.py"]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source\n")
    archive = root / "historical/source.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"frozen source archive")
    value = write(root / "inputs/case-0.json", {"case_id": "case-0"}, sealed=True)
    hashes = {name: recovery.file_hash(repo / name) for name in recovery.ENTRYPOINTS}
    plan = write(
        root / "plan.json",
        {
            "protocol": recovery.PROTOCOL,
            "cases": [{"case_id": "case-0", "input_sha256": value["artifact_sha256"]}],
            "settings": {
                "model": recovery.MODEL,
                "model_revision": recovery.REVISION,
                "max_output_tokens": 8192,
                "maximum_requests": 96,
                "maximum_total_tokens": 2097152,
            },
            "budgets": {
                "maximum_calls_per_case": 96,
                "maximum_charged_tokens_per_case": 2097152,
                "maximum_output_tokens": 8192,
                "attempts_per_step": 2,
            },
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
            "entrypoints": hashes,
            "runtime_source_sha256": recovery.runtime_source_hash(repo),
            "source_archive_sha256": recovery.file_hash(archive),
        },
        sealed=True,
    )
    execution = write(
        root / "execution.json",
        {
            "protocol": recovery.EXECUTION_PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "entrypoint_sha256": hashes,
            "model": recovery.MODEL,
            "model_revision": recovery.REVISION,
            "platform": "aces-h100x1",
            "maximum_cases": 1,
            "maximum_generation_requests": 96,
            "maximum_charged_tokens": 2097152,
            "worker_seconds": 5400,
            "scheduler_walltime_seconds": 7200,
            "cpus_per_task": 8,
            "memory_gib": 64,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
            "model_promotion": False,
        },
        sealed=True,
    )
    directory = root / "submissions/functions-1"
    argv = [
        "sbatch",
        "--parsable",
        "--kill-on-invalid-dep=yes",
        "--nodes=1",
        "--ntasks=1",
        "--export=ALL",
        "--account=156264627414",
        "--partition=gpu",
        "--gres=gpu:h100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=02:00:00",
        "--signal=B:TERM@300",
        "--exclude=ac042",
        "--job-name=phasec-functions",
        f"--output={root}/logs/functions-%j.out",
        f"--error={root}/logs/functions-%j.err",
        str(repo / "scripts/hpc/run_construction_functions_aces.sh"),
        "propose",
        "0",
    ]
    write(directory / "propose.intent.json", {"argv": argv})
    write(
        directory / "propose.reply.json",
        {
            "returncode": 0,
            "stdout": "\n",
            "stderr": "sbatch: error: Socket timed out on send/recv operation\n",
        },
    )
    stamp = 1792497600
    for name, time in [("intent", stamp), ("reply", stamp + 60)]:
        os.utime(directory / f"propose.{name}.json", (time, time))
    write(root / "results/case-0/cost.json", {"physical_requests": 3})
    record = {
        "JobId": "2236653",
        "JobName": "phasec-functions",
        "UserId": f"{getpass.getuser()}(123)",
        "JobState": "PENDING",
        "ExitCode": "0:0",
        "SubmitTime": datetime.fromtimestamp(stamp + 10, UTC).isoformat(),
        "SubmitLine": shlex.join(["/usr/bin/sbatch", *argv[1:]]),
        "WorkDir": "/home/example",
    }
    calls = []

    def scheduler(command, **kwargs):
        assert command[0] == "scontrol", "recovery must never submit or cancel work"
        assert kwargs["env"]["TZ"] == "UTC"
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, " ".join(f"{k}={v}" for k, v in record.items()), ""
        )

    monkeypatch.setattr(recovery.subprocess, "run", scheduler)
    return root, repo, directory, argv, record, calls, execution


def test_adopt_preserves_inputs_results_and_original_receipts(campaign, monkeypatch):
    root, repo, directory, _, record, calls, execution = campaign
    before = {
        p: p.read_bytes()
        for base in (repo, root)
        for p in base.rglob("*")
        if p.is_file()
    }
    result = recovery.recover(root, "2236653")
    assert result == {
        "execution_sha256": execution["artifact_sha256"],
        "wave": "functions-1",
        "job": "2236653",
        "automatic_followup": False,
        "maximum_cases": 1,
        "maximum_generation_requests": 96,
        "maximum_charged_tokens": 2097152,
    }
    assert (directory / "propose.id").read_text() == "2236653\n"
    assert {p: p.read_bytes() for p in before} == before
    audit = recovery.sealed_read(directory / "propose.adoption.json")
    assert audit["scheduler"]["submit_line"] == record["SubmitLine"]
    assert audit["automatic_retry"] is False
    assert recovery.recover(root, "2236653") == result
    assert len(calls) == 1
    # A partial metadata write can be recovered even after Slurm retires the job.
    (directory / "manifest.json").unlink()
    (directory / "propose.id").unlink()
    monkeypatch.setattr(
        recovery, "scheduler_record", lambda *a, **kw: pytest.fail("already audited")
    )
    assert recovery.recover(root, "2236653") == result


@pytest.mark.parametrize(
    "key,value",
    [
        ("JobId", "999"),
        ("UserId", "another-user(123)"),
        ("JobName", "other-job"),
        ("SubmitTime", "2026-01-01T00:00:00Z"),
        ("SubmitTime", "2027-01-01T00:00:00Z"),
        ("SubmitTime", "Unknown"),
        ("SubmitLine", "sbatch --parsable wrong-worker.sh"),
    ],
)
def test_wrong_scheduler_identity_never_adopted(campaign, key, value):
    root, _, directory, _, record, *_ = campaign
    record[key] = value
    with pytest.raises(ValueError):
        recovery.recover(root, "2236653")
    assert not (directory / "propose.adoption.json").exists()
    assert not (directory / "propose.id").exists()
    assert not (directory / "manifest.json").exists()


@pytest.mark.parametrize("target", ["source", "plan", "execution", "input", "archive"])
def test_changed_campaign_refused_before_scheduler(campaign, target):
    root, repo, directory, _, _, calls, _ = campaign
    path = {
        "source": repo / "src/autoformalism/__init__.py",
        "plan": root / "plan.json",
        "execution": root / "execution.json",
        "input": root / "inputs/case-0.json",
        "archive": root / "historical/source.tar.gz",
    }[target]
    path.write_text("{}")
    with pytest.raises(ValueError):
        recovery.recover(root, "2236653")
    assert not calls
    assert not (directory / "manifest.json").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("propose.id", "999\n"),
        ("manifest.json", '{"job":"999"}'),
    ],
)
def test_conflicting_metadata_is_preserved(campaign, name, value):
    root, _, directory, _, _, calls, _ = campaign
    path = directory / name
    path.write_text(value)
    with pytest.raises(ValueError, match="differs"):
        recovery.recover(root, "2236653")
    assert path.read_text() == value and not calls


def test_partial_publication_resumes_from_sealed_audit(campaign, monkeypatch):
    root, _, directory, _, _, calls, _ = campaign
    original = recovery.atomic_write

    def interruption(path, text):
        if path.name == "propose.id":
            raise OSError("simulated interrupted write")
        original(path, text)

    monkeypatch.setattr(recovery, "atomic_write", interruption)
    with pytest.raises(OSError, match="interrupted"):
        recovery.recover(root, "2236653")
    assert (directory / "propose.adoption.json").exists()
    assert not (directory / "propose.id").exists()
    monkeypatch.setattr(recovery, "atomic_write", original)
    assert recovery.recover(root, "2236653")["job"] == "2236653"
    assert len(calls) == 1


def test_failed_retired_job_is_recorded_without_retry(campaign, monkeypatch):
    root, _, directory, argv, record, calls, _ = campaign

    def accounting(command, **kwargs):
        assert command[0] in {"scontrol", "sacct"}
        calls.append(command)
        if command[0] == "scontrol":
            return subprocess.CompletedProcess(command, 1, "", "Invalid job id")
        line = (
            f"2236653|{getpass.getuser()}|phasec-functions|FAILED|1:0|"
            f"{record['SubmitTime']}|{shlex.join(argv)}\n"
        )
        return subprocess.CompletedProcess(command, 0, line, "")

    monkeypatch.setattr(recovery.subprocess, "run", accounting)
    assert recovery.recover(root, "2236653")["job"] == "2236653"
    audit = recovery.sealed_read(directory / "propose.adoption.json")
    assert audit["scheduler"]["source"] == "sacct"
    assert audit["scheduler"]["state"] == "FAILED"
    assert [c[0] for c in calls] == ["scontrol", "sacct"]


def test_scheduler_lock_prevents_racing_submission(campaign):
    root, _, _, _, _, calls, _ = campaign
    (root / "scheduler").mkdir()
    with (root / "scheduler/.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="in use"):
            recovery.recover(root, "2236653")
    assert not calls


def test_confirmed_receipt_for_another_job_is_not_overridden(campaign):
    root, _, directory, _, _, calls, _ = campaign
    write(
        directory / "propose.reply.json",
        {
            "returncode": 0,
            "stdout": "999;cluster\n",
            "stderr": "",
        },
    )
    with pytest.raises(ValueError, match="different job"):
        recovery.recover(root, "2236653")
    assert not calls and not (directory / "manifest.json").exists()


def test_altered_original_receipt_invalidates_recovery_audit(campaign):
    root, _, directory, *_ = campaign
    recovery.recover(root, "2236653")
    path = directory / "propose.reply.json"
    original = json.loads(path.read_text())
    write(path, {**original, "stderr": "different receipt"})
    with pytest.raises(ValueError, match="adoption identity"):
        recovery.recover(root, "2236653")


def test_standalone_help_needs_no_project_or_site_packages():
    reply = subprocess.run(
        [sys.executable, "-I", "-S", str(Path(recovery.__file__).resolve()), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--job-id" in reply.stdout
