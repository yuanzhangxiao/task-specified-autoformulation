"""Recover expired Slurm dependencies without changing scientific code or receipts."""

import getpass
import json
import shlex
import subprocess

import pytest

from scripts import recover_dalla_sign_submission as recovery
from scripts import submit_dalla_sign_repair as submitter
from scripts.smoke_dalla_sign_repair import fixture

REJECTION = "sbatch: error: Batch job submission failed: Job dependency problem\n"
TIMEOUT = "sbatch: error: Socket timed out on send/recv operation\n"


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, config, root = fixture(tmp_path, protocol="dalla-sign-repair-2")
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        path = tmp_path / key
        path.touch()
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("AF_HF_HOME", str(tmp_path))
    monkeypatch.setenv("AF_REPO_ROOT", str(submitter.io.REPO))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda repo: "a" * 40)
    monkeypatch.setattr(recovery, "load_submitter", lambda repo: submitter)
    jobs, calls = {}, []

    def initial(argv, **kwargs):
        if argv[-2] == "prepare":
            jobs["2162020"] = (argv, "COMPLETED", "0:0")
            return subprocess.CompletedProcess(argv, 0, "2162020\n", "")
        assert argv[-2] == "review"
        return subprocess.CompletedProcess(argv, 0, "\n", REJECTION)

    monkeypatch.setattr(subprocess, "run", initial)
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(source, root, config=config)

    def scheduler(argv, **kwargs):
        if argv[0] == "sbatch":
            calls.append(argv)
            job = str(3000000 + len(calls))
            jobs[job] = (argv, "PENDING", "0:0")
            return subprocess.CompletedProcess(argv, 0, job + "\n", "")
        if argv[0] == "scontrol":
            job = argv[3]
            if job == "2162020":
                return subprocess.CompletedProcess(argv, 1, "", "Invalid job id")
            command, state, code = jobs[job]
            name = next(
                a.split("=", 1)[1] for a in command if a.startswith("--job-name=")
            )
            text = (
                f"JobId={job} JobName={name} UserId={getpass.getuser()}(123) "
                f"JobState={state} ExitCode={code} "
                f"SubmitLine={shlex.join(command)} WorkDir=/home/example"
            )
            return subprocess.CompletedProcess(argv, 0, text, "")
        assert argv[0] == "sacct"
        lines = []
        requested = next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--name=")), None
        )
        requested_job = argv[argv.index("-j") + 1] if "-j" in argv else None
        for job, (command, state, code) in jobs.items():
            name = next(
                a.split("=", 1)[1] for a in command if a.startswith("--job-name=")
            )
            if (requested and requested != name) or (
                requested_job and requested_job != job
            ):
                continue
            raw = job + "_0" if any(a.startswith("--array=") for a in command) else job
            lines.append(
                f"{raw}|{getpass.getuser()}|{name}|{state}|{code}|"
                + shlex.join(command)
            )
        return subprocess.CompletedProcess(argv, 0, "\n".join(lines), "")

    monkeypatch.setattr(subprocess, "run", scheduler)
    return root, jobs, calls, scheduler


def test_retired_prepare_is_verified_and_never_referenced_again(campaign):
    root, _, calls, _ = campaign
    originals = [root / "plan.json", *(root / "submission-intent").glob("*")]
    before = {p: p.read_bytes() for p in originals}
    result = recovery.recover(root, "2162020")
    assert [c[-2] for c in calls] == ["review", "fit", "report"]
    assert not any(a.startswith("--dependency=") for a in calls[0])
    assert "--dependency=afterok:3000001" in calls[1]
    assert "--dependency=afterany:3000002" in calls[2]
    assert not any("2162020" in a for c in calls for a in c)
    assert result["maximum_llm_calls"] == 6 and result["jobs"]["prepare"] == "2162020"
    assert {p: p.read_bytes() for p in originals} == before
    assert recovery.recover(root, "2162020") == result and len(calls) == 3
    assert (
        submitter.submit(
            root.parent / "source.json", root, config=root.parent / "config.json"
        )
        == result
    )


@pytest.mark.parametrize("stage", ["review", "fit", "report"])
def test_timeout_accepted_jobs_are_adopted_including_arrays(
    campaign, monkeypatch, stage
):
    root, _, calls, scheduler = campaign

    def timeout(argv, **kwargs):
        result = scheduler(argv, **kwargs)
        if argv[0] == "sbatch" and argv[-2] == stage:
            return subprocess.CompletedProcess(argv, 0, "\n", TIMEOUT)
        return result

    monkeypatch.setattr(subprocess, "run", timeout)
    result = recovery.recover(root, "2162020")
    assert len(calls) == 3 and result["jobs"][stage].isdecimal()
    directory = root / "submission-recovery-1"
    assert (directory / f"{stage}-0.adoption.json").is_file()
    assert (
        json.loads((directory / f"{stage}-0.reply.json").read_text())["stderr"]
        == TIMEOUT
    )


def test_unknown_timeout_is_not_resent_and_later_accounting_resumes(
    campaign, monkeypatch
):
    root, jobs, calls, scheduler = campaign
    hidden = []

    def timeout(argv, **kwargs):
        result = scheduler(argv, **kwargs)
        if argv[0] == "sbatch" and argv[-2] == "review":
            hidden.append(jobs.pop("3000001"))
            return subprocess.CompletedProcess(argv, 0, "\n", TIMEOUT)
        return result

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="remains unconfirmed"):
        recovery.recover(root, "2162020")
    with pytest.raises(ValueError, match="remains unconfirmed"):
        recovery.recover(root, "2162020")
    assert len(calls) == 1
    jobs["3000001"] = hidden[0]
    assert recovery.recover(root, "2162020")["jobs"]["review"] == "3000001"
    assert len(calls) == 3


def test_completed_review_and_fit_do_not_leave_stale_dependencies(
    campaign, monkeypatch
):
    root, jobs, calls, scheduler = campaign

    def immediate(argv, **kwargs):
        result = scheduler(argv, **kwargs)
        if argv[0] == "sbatch":
            jobs[result.stdout.strip()] = (argv, "COMPLETED", "0:0")
        return result

    monkeypatch.setattr(subprocess, "run", immediate)
    recovery.recover(root, "2162020")
    assert not any(a.startswith("--dependency=") for c in calls for a in c)


def test_dependency_race_retries_with_new_receipt_only_after_completion(
    campaign, monkeypatch
):
    root, jobs, calls, scheduler = campaign

    def race(argv, **kwargs):
        result = scheduler(argv, **kwargs)
        if argv[0] == "sbatch" and argv[-2] == "fit" and len(calls) == 2:
            jobs.pop("3000002")
            command, _, _ = jobs["3000001"]
            jobs["3000001"] = (command, "COMPLETED", "0:0")
            return subprocess.CompletedProcess(argv, 0, "", REJECTION)
        return result

    monkeypatch.setattr(subprocess, "run", race)
    result = recovery.recover(root, "2162020")
    assert result["jobs"]["fit"] == "3000003"
    assert len(calls) == 4
    assert "--dependency=afterok:3000001" in calls[1]
    assert not any(a.startswith("--dependency=") for a in calls[2])
    directory = root / "submission-recovery-1"
    assert (
        json.loads((directory / "fit-0.reply.json").read_text())["stderr"] == REJECTION
    )
    assert (directory / "fit-1.id").read_text().strip() == "3000003"


def test_failed_prepare_cannot_release_review(campaign):
    root, jobs, calls, _ = campaign
    argv, _, _ = jobs["2162020"]
    jobs["2162020"] = (argv, "FAILED", "1:0")
    with pytest.raises(ValueError, match="not viable"):
        recovery.recover(root, "2162020")
    assert calls == []


def test_original_timeout_is_not_treated_as_rejection(campaign):
    root, _, calls, _ = campaign
    path = root / "submission-intent/review.reply.json"
    path.write_text(json.dumps({"stdout": "", "stderr": TIMEOUT}))
    with pytest.raises(ValueError, match="not a confirmed dependency rejection"):
        recovery.recover(root, "2162020")
    assert calls == []


def test_original_matching_review_prevents_duplicate(campaign):
    root, jobs, calls, _ = campaign
    argv = json.loads((root / "submission-intent/review.intent.json").read_text())[
        "argv"
    ]
    jobs["999"] = (argv, "RUNNING", "0:0")
    with pytest.raises(ValueError, match="original review job exists"):
        recovery.recover(root, "2162020")
    assert calls == []


def test_fit_completion_requires_every_array_element():
    records = [{"job": "123", "raw_job": "123_0", "state": "COMPLETED"}]
    assert recovery.fit_finished(records, "123", 1)
    assert not recovery.fit_finished(records, "123", 2)
    assert not recovery.fit_finished(records, "124", 1)
    records.append({"job": "123", "raw_job": "123_1", "state": "CANCELLED by 100"})
    assert recovery.fit_finished(records, "123", 2)
