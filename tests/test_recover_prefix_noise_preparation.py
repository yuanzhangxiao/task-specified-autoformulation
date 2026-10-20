"""M47 scheduler recovery preserves frozen inputs and cannot duplicate a request."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recover_prefix_noise_preparation as recovery

JOBS = {"prepare": "23060456", "run": "23060458", "report": "23060459"}


def sealed(path: Path, value: dict) -> None:
    """Write a valid original-format seal in a synthetic campaign."""
    recovery.write(path, {"sha256": recovery.digest(value), "value": value})


def accounting(array_rows: list[str] | None = None) -> str:
    """Represent the confirmed original timeout, cancelled array and report."""
    if array_rows is None:
        array_rows = [array_row("[0-35%2]")]
    return "\n".join([
        f"{JOBS['prepare']}|prefix-noise-reliability-prepare|TIMEOUT|600|2026-10-20T01:00:00",
        *array_rows,
        f"{JOBS['report']}|prefix-noise-reliability-report|COMPLETED|1|2026-10-20T01:10:01",
    ])


def array_row(
    suffix: str, *, state: str = "CANCELLED", elapsed: str = "0",
    start: str = "Unknown", name: str = "prefix-noise-reliability-run",
) -> str:
    """Build an allocation row without relying on Slurm being installed."""
    return f"{JOBS['run']}_{suffix}|{name}|{state}|{elapsed}|{start}"


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Create a sealed standalone bundle and fake every external command."""
    root, repo, casadi = (tmp_path / name for name in ("campaign", "bundle", "deps"))
    (root / "submission").mkdir(parents=True)
    (root / "logs").mkdir()
    (repo / recovery.WORKER).parent.mkdir(parents=True)
    casadi.mkdir()
    worker = repo / recovery.WORKER
    worker.write_text("#!/bin/bash\nexit 0\n")
    (repo / "SOURCE_COMMIT").write_text("original-commit\n")
    (repo / "SHA256SUMS").write_text(
        f"{recovery.file_digest(worker)}  {recovery.WORKER}\n"
    )
    data = {"records": {"frozen": [1.0, 2.0]}}
    plan = {
        "protocol": "phase-c-prefix-noise-reliability-1",
        "tasks": [{"task_id": f"task-{i}"} for i in range(36)],
        "inputs_sha256": recovery.digest(data),
    }
    identity = {
        "plan_sha256": recovery.digest(plan), "commit": "original-commit",
        "account": "bibo-delta-cpu", "concurrency": 2,
        "python": sys.executable, "repo": str(repo),
        "worker_sha256": recovery.file_digest(worker),
        "prepare_time": "00:10:00", "run_time": "00:45:00",
    }
    manifest = {"identity": identity, "jobs": JOBS.copy(), "array_tasks": 36}
    sealed(root / "plan.json", plan)
    sealed(root / "inputs.json", data)
    sealed(root / "submission/identity.json", identity)
    recovery.write(root / "submission_manifest.json", manifest)
    recovery.write(root / "summary.json", {
        "status": "incomplete", "expected": 36, "recorded": 0,
        "status_counts": {"missing": 36}, "cost": {"operations": 0},
    })
    for stage, job in JOBS.items():
        recovery.write(root / "submission" / f"{stage}.intent.json", {
            "argv": recovery.command(root, identity, JOBS, stage, recovery=False),
        })
        recovery.write(root / "submission" / f"{stage}.reply.json", {
            "returncode": 0, "stdout": job, "stderr": "",
        })
        (root / "submission" / f"{stage}.id").write_text(job + "\n")
    state = SimpleNamespace(
        root=root, repo=repo, casadi=casadi, identity=identity, plan=plan,
        data=data, manifest=manifest, commands=[], sbatch=[], envs=[],
        accounting=accounting(), queue="", failure=None, checksum_failure=False,
    )

    def run(argv, **kwargs):
        state.commands.append((argv, kwargs))
        if argv[0] == "sha256sum":
            assert kwargs["cwd"] == repo and kwargs["check"]
            if state.checksum_failure:
                raise subprocess.CalledProcessError(1, argv, stderr="checksum differs")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] in ("sacct", "squeue"):
            assert kwargs["check"]
            output = state.accounting if argv[0] == "sacct" else state.queue
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        assert argv[0] == "sbatch"
        stage = argv[-1]
        intent = root / recovery.RECOVERY / f"{stage}.intent.json"
        assert recovery.read(intent) == {"argv": argv}
        state.sbatch.append(argv)
        state.envs.append(kwargs["env"])
        if state.failure and state.failure[0] == stage:
            failure = state.failure[1]
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 45)
            if failure == "oserror":
                raise OSError("scheduler connection failed")
            return SimpleNamespace(
                returncode=int(failure == "nonzero"),
                stdout="Submitted batch job 401" if failure == "malformed" else "",
                stderr="scheduler response unconfirmed",
            )
        return SimpleNamespace(
            returncode=0, stdout=f"{301 + recovery.STAGES.index(stage)};delta\n",
            stderr="",
        )

    monkeypatch.setattr(recovery.subprocess, "run", run)
    return state


@pytest.mark.parametrize("rows", [
    [array_row("[0-35]")],
    [array_row("[0-35%2]", state="CANCELLED by 12345")],
    [array_row(str(i)) for i in range(36)],
    [array_row("[0-9,11-19,21-35%2]"), array_row("10"), array_row("20")],
])
def test_accounting_accepts_exact_compressed_explicit_and_mixed_rosters(rows):
    recovery.check_accounting(accounting(rows), JOBS)


@pytest.mark.parametrize("start", ["Unknown", "None", "N/A", ""])
def test_accounting_accepts_only_unstarted_timestamp_markers(start):
    recovery.check_accounting(accounting([array_row("[0-35]", start=start)]), JOBS)


@pytest.mark.parametrize("suffix", [
    "[0-35]+", "[0-", "[0-36]", "[-1-35]", "[35-0]", "[0-35:2]",
    "[0-35,0]", "[0-20,20-35]", "[0--35]", "[]", "[0-35%]", "36",
])
def test_array_ids_reject_truncation_unknown_ranges_and_overlap(suffix):
    with pytest.raises(ValueError):
        recovery.array_indices(f"{JOBS['run']}_{suffix}", JOBS["run"])


@pytest.mark.parametrize("rows", [
    [],
    [array_row("[0-34]")],
    [array_row("[0-35]"), array_row("20")],
    [array_row("[0-19]"), array_row("[19-35]")],
    [array_row("[0-35]", state="PENDING")],
    [array_row("[0-35]", state="RUNNING")],
    [array_row("[0-35]", state="COMPLETED")],
    [array_row("[0-35]", elapsed="1")],
    [array_row("[0-35]", start="2026-10-20T01:10:00")],
    [array_row("[0-35]", name="some-other-campaign")],
    [f"{JOBS['run']}|prefix-noise-reliability-run|CANCELLED|0|Unknown"],
])
def test_accounting_rejects_parent_partial_overlapping_active_or_started_arrays(rows):
    with pytest.raises(ValueError):
        recovery.check_accounting(accounting(rows), JOBS)


@pytest.mark.parametrize("changed", [
    lambda output: output.replace("|TIMEOUT|", "|COMPLETED|"),
    lambda output: output.replace("|COMPLETED|", "|RUNNING|"),
    lambda output: "\n".join(output.splitlines()[1:]),
    lambda output: "\n".join(output.splitlines()[:-1]),
    lambda output: output + "\n" + output.splitlines()[0],
    lambda output: output + "\nmalformed|record",
])
def test_accounting_requires_unique_terminal_prepare_and_report(changed):
    with pytest.raises(ValueError):
        recovery.check_accounting(changed(accounting()), JOBS)


def test_accounting_ignores_only_recognized_job_steps():
    output = accounting() + "\n" + "\n".join([
        f"{JOBS['prepare']}.batch|batch|CANCELLED|600|2026-10-20T01:00:00",
        f"{JOBS['report']}.extern|extern|COMPLETED|1|2026-10-20T01:10:01",
    ])
    recovery.check_accounting(output, JOBS)
    with pytest.raises(ValueError):
        recovery.check_accounting(output + "\n12345.0|step|COMPLETED|1|Unknown", JOBS)


@pytest.mark.parametrize("queued", [
    JOBS["prepare"], JOBS["report"], JOBS["run"] + "_[0-35%2]",
    JOBS["run"] + "_12",
])
def test_still_queued_original_jobs_block_recovery(campaign, queued):
    campaign.queue = queued + "\n"
    with pytest.raises(ValueError, match="still queued"):
        recovery.recover(campaign.root, campaign.casadi)
    assert not campaign.sbatch


def test_incomplete_accounting_blocks_before_any_submission(campaign):
    campaign.accounting = accounting([array_row("[0-34]")])
    with pytest.raises(ValueError, match="incomplete original job accounting"):
        recovery.recover(campaign.root, campaign.casadi)
    assert not campaign.sbatch


def test_recovery_preserves_original_files_and_frozen_execution_contract(campaign):
    c = campaign
    original = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for parent in (c.root, c.repo) for path in parent.rglob("*") if path.is_file()
    }
    result = recovery.recover(c.root, c.casadi)
    assert result["jobs"] == {"prepare": "301", "run": "302", "report": "303"}
    assert result["original_jobs"] == JOBS and result["array_tasks"] == 36
    assert result["gpus"] == result["live_llm_calls"] == 0
    assert result["test_data_opened"] is False
    assert original == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original
    }
    assert [argv[-1] for argv in c.sbatch] == list(recovery.STAGES)
    for argv, duration in zip(
        c.sbatch, ("00:30:00", "00:45:00", "00:05:00"), strict=True,
    ):
        assert f"--time={duration}" in argv
        assert "--partition=cpu" in argv and "--cpus-per-task=1" in argv
        assert "--mem=16G" in argv and "--kill-on-invalid-dep=yes" in argv
        assert argv[-2] == str(c.repo / recovery.WORKER)
    assert "--dependency=afterok:301" in c.sbatch[1]
    assert "--array=0-35%2" in c.sbatch[1]
    assert "--dependency=afterany:301:302" in c.sbatch[2]
    for env in c.envs:
        assert env["AF_REPO_ROOT"] == str(c.repo)
        assert env["AF_OUTPUT_ROOT"] == str(c.root)
        assert env["AF_PYTHON"] == sys.executable
        assert env["AF_COMMIT"] == c.identity["commit"]
        assert env["AF_CASADI_ROOT"] == str(c.casadi)
        assert all(env[key] == "1" for key in (
            "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        ))
    directory = c.root / recovery.RECOVERY
    assert recovery.read(directory / "original-summary.json") == recovery.read(
        c.root / "summary.json"
    )
    evidence = recovery.read(directory / "original-accounting.json")
    assert evidence["sacct"] == c.accounting
    assert recovery.read(Path(result["artifact"])) == result


def test_completed_recovery_is_idempotent_after_results_appear(campaign):
    c = campaign
    first = recovery.recover(c.root, c.casadi)
    results = c.root / "results/task-0"
    results.mkdir(parents=True)
    (results / "result.json").write_text("completed scientific work")
    recovery.write(c.root / "summary.json", {"status": "complete", "recorded": 36})
    directory = c.root / recovery.RECOVERY
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*") if path.is_file()
    }
    c.accounting = "must not be queried again"
    assert recovery.recover(c.root, c.casadi) == first
    assert len(c.sbatch) == 3
    assert sum(argv[0] == "sacct" for argv, _ in c.commands) == 1
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    }


@pytest.mark.parametrize("next_stage", ["run", "report"])
def test_partial_resume_keeps_confirmed_jobs_even_when_results_appear(
    campaign, monkeypatch, next_stage,
):
    c = campaign
    submit_once = recovery.submit_once

    def interrupted(directory, stage, argv, env):
        if stage == next_stage:
            raise RuntimeError("process stopped before unattempted stage")
        return submit_once(directory, stage, argv, env)

    monkeypatch.setattr(recovery, "submit_once", interrupted)
    with pytest.raises(RuntimeError, match="process stopped"):
        recovery.recover(c.root, c.casadi)
    assert len(c.sbatch) == recovery.STAGES.index(next_stage)
    folder = c.root / "results/task-0"
    folder.mkdir(parents=True)
    (folder / ".lock").touch()
    monkeypatch.setattr(recovery, "submit_once", submit_once)
    c.accounting = "must not be queried on resume"
    result = recovery.recover(c.root, c.casadi)
    assert result["jobs"] == {"prepare": "301", "run": "302", "report": "303"}
    assert [argv[-1] for argv in c.sbatch] == list(recovery.STAGES)


@pytest.mark.parametrize("stage", recovery.STAGES)
@pytest.mark.parametrize(
    "failure", ["timeout", "oserror", "nonzero", "empty", "malformed"],
)
def test_uncertain_submission_never_retries_any_stage(campaign, stage, failure):
    c = campaign
    c.failure = (stage, failure)
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        recovery.recover(c.root, c.casadi)
    submitted = len(c.sbatch)
    assert submitted == recovery.STAGES.index(stage) + 1
    directory = c.root / recovery.RECOVERY
    assert (directory / f"{stage}.intent.json").is_file()
    assert (directory / f"{stage}.reply.json").is_file()
    assert not (directory / f"{stage}.accepted.json").exists()
    c.failure = None
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        recovery.recover(c.root, c.casadi)
    assert len(c.sbatch) == submitted


@pytest.mark.parametrize("artifact", ["intent.json", "reply.json"])
def test_orphan_recovery_receipt_never_submits(campaign, artifact):
    directory = campaign.root / recovery.RECOVERY
    directory.mkdir()
    recovery.write(directory / f"prepare.{artifact}", {})
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        recovery.recover(campaign.root, campaign.casadi)
    assert not campaign.sbatch


def test_started_scientific_work_blocks_first_recovery(campaign):
    (campaign.root / "results").mkdir()
    (campaign.root / "results/task-0").mkdir()
    with pytest.raises(ValueError, match="results already exist"):
        recovery.recover(campaign.root, campaign.casadi)
    assert not campaign.sbatch
    assert not any(argv[0] == "sacct" for argv, _ in campaign.commands)


@pytest.mark.parametrize(
    "name", ["plan.json", "inputs.json", "submission/identity.json"],
)
def test_invalid_original_seals_are_rejected(campaign, name):
    path = campaign.root / name
    value = recovery.read(path)
    value["sha256"] = "0" * 64
    recovery.write(path, value)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        recovery.recover(campaign.root, campaign.casadi)
    assert not campaign.sbatch


@pytest.mark.parametrize("change", [
    "source_commit", "worker", "plan", "inputs", "manifest", "original_id",
    "original_intent", "python", "checksums",
])
def test_changed_original_identity_or_source_blocks_submission(campaign, change):
    c = campaign
    if change == "source_commit":
        (c.repo / "SOURCE_COMMIT").write_text("another-commit")
    elif change == "worker":
        (c.repo / recovery.WORKER).write_text("modified worker")
    elif change == "plan":
        sealed(c.root / "plan.json", c.plan | {"extra": "changed"})
    elif change == "inputs":
        sealed(c.root / "inputs.json", {"records": {"different": []}})
    elif change == "manifest":
        recovery.write(
            c.root / "submission_manifest.json", c.manifest | {"array_tasks": 35},
        )
    elif change == "original_id":
        (c.root / "submission/run.id").write_text("999")
    elif change == "original_intent":
        recovery.write(
            c.root / "submission/run.intent.json", {"argv": ["sbatch", "different"]},
        )
    elif change == "python":
        identity = c.identity | {"python": str(c.root / "nonexistent-python")}
        sealed(c.root / "submission/identity.json", identity)
        recovery.write(
            c.root / "submission_manifest.json", c.manifest | {"identity": identity},
        )
    else:
        c.checksum_failure = True
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        recovery.recover(c.root, c.casadi)
    assert not c.sbatch


@pytest.mark.parametrize("change", ["casadi", "bundle_manifest", "accepted", "intent"])
def test_changed_recovery_identity_or_receipts_never_resubmit(campaign, change):
    c = campaign
    recovery.recover(c.root, c.casadi)
    directory = c.root / recovery.RECOVERY
    casadi = c.casadi
    if change == "casadi":
        casadi = c.root / "other-deps"
        casadi.mkdir()
    elif change == "bundle_manifest":
        (c.repo / "SHA256SUMS").write_text("different manifest")
    elif change == "accepted":
        recovery.write(directory / "run.accepted.json", {"job": "not-a-job"})
    else:
        recovery.write(directory / "run.intent.json", {"argv": ["changed"]})
    with pytest.raises(ValueError, match="identity differs"):
        recovery.recover(c.root, casadi)
    assert len(c.sbatch) == 3


def test_recovery_helper_uses_only_standard_library_imports():
    tree = ast.parse(Path(recovery.__file__).read_text())
    modules = {
        name.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for name in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules <= sys.stdlib_module_names


def test_standalone_help_works_outside_the_checkout(tmp_path):
    helper = tmp_path / "recover.py"
    helper.write_bytes(Path(recovery.__file__).read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", str(helper), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=10,
    )
    assert "--root" in result.stdout and "--casadi-root" in result.stdout
    assert not result.stderr
