"""Report-only M47 recovery preserves completed work and cannot duplicate jobs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recover_prefix_noise_report as recovery
from tests import test_recover_prefix_noise_preparation as preparation_tests

previous = recovery.previous
preparation_campaign = preparation_tests.campaign
JOBS = {"prepare": "301", "run": "302", "report": "303"}


def array_row(
    suffix: str, *, state: str = "COMPLETED", exit_code: str = "0:0"
) -> str:
    """Represent a recovered array allocation without contacting Slurm."""
    return (
        f"{JOBS['run']}_{suffix}|prefix-noise-reliability-recovery-run|"
        f"{state}|{exit_code}"
    )


def accounting(rows: list[str] | None = None) -> str:
    """Keep the successful preparation and timed-out report around all runs."""
    return "\n".join([
        f"{JOBS['prepare']}|prefix-noise-reliability-recovery-prepare|COMPLETED|0:0",
        *(rows if rows is not None else [array_row(str(i)) for i in range(36)]),
        f"{JOBS['report']}|prefix-noise-reliability-recovery-report|TIMEOUT|0:15",
    ])


@pytest.fixture
def campaign(preparation_campaign, monkeypatch):
    """Complete the original preparation recovery before faking report failure."""
    state = preparation_campaign
    result = previous.recover(state.root, state.casadi)
    assert result["jobs"] == JOBS
    for task in state.plan["tasks"]:
        folder = state.root / "results" / task["task_id"]
        folder.mkdir(parents=True)
        previous.write(folder / "result.json", {"status": "complete"})
    state.accounting = accounting()
    state.commands.clear()
    state.sbatch.clear()
    state.envs.clear()
    original_run = previous.subprocess.run

    def run(argv, **kwargs):
        if argv[0] != "sbatch":
            return original_run(argv, **kwargs)
        state.commands.append((argv, kwargs))
        intent = state.root / recovery.RECOVERY / "report.intent.json"
        assert previous.read(intent) == {"argv": argv}
        state.sbatch.append(argv)
        state.envs.append(kwargs["env"])
        if state.failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 45)
        if state.failure == "oserror":
            raise OSError("scheduler connection failed")
        if state.failure:
            return SimpleNamespace(
                returncode=int(state.failure == "nonzero"),
                stdout=(
                    "Submitted batch job 900" if state.failure == "malformed" else ""
                ),
                stderr="unconfirmed scheduler response",
            )
        return SimpleNamespace(returncode=0, stdout="900;delta\n", stderr="")

    monkeypatch.setattr(recovery.subprocess, "run", run)
    return state


def snapshot(paths) -> dict:
    """Retain bytes and timestamps to detect writes to historical evidence."""
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


@pytest.mark.parametrize("rows", [
    [array_row("[0-35]")],
    [array_row(str(i)) for i in range(36)],
    [array_row("[0-9,11-35%2]"), array_row("10")],
])
def test_accounting_accepts_complete_compressed_explicit_and_mixed_arrays(rows):
    recovery.check_accounting(accounting(rows), JOBS)


def test_report_retry_preserves_all_prior_artifacts_and_uses_original_worker(campaign):
    c = campaign
    before = snapshot(
        path for parent in (c.root, c.repo)
        for path in parent.rglob("*") if path.is_file()
    )
    result = recovery.recover(c.root)
    assert result["jobs"] == {"report": "900"}
    assert result["previous_jobs"] == JOBS
    assert result["gpus"] == result["live_llm_calls"] == 0
    assert result["fitting_submitted"] is False
    assert snapshot(before) == before
    assert len(c.sbatch) == 1
    argv = c.sbatch[0]
    assert argv[-2:] == [str(c.repo / previous.WORKER), "report"]
    assert {
        "--time=00:30:00", "--partition=cpu", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", "--mem=16G",
    } <= set(argv)
    assert not any(arg.startswith(("--array", "--dependency", "--gpus", "--gres"))
                   for arg in argv)
    env = c.envs[0]
    assert env["AF_REPO_ROOT"] == str(c.repo)
    assert env["AF_OUTPUT_ROOT"] == str(c.root)
    assert env["AF_PYTHON"] == sys.executable
    assert env["AF_COMMIT"] == c.identity["commit"]
    assert env["AF_CASADI_ROOT"] == str(c.casadi)
    assert all(env[key] == "1" for key in (
        "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    ))
    folder = c.root / recovery.RECOVERY
    assert previous.read(folder / "prior-summary.json") == previous.read(
        c.root / "summary.json"
    )
    assert previous.read(folder / "prior-accounting.json") == {
        "sacct": c.accounting, "squeue": "", "result_files": 36,
    }
    assert previous.read(Path(result["artifact"])) == result


def test_confirmed_retry_is_idempotent_after_new_summary(campaign):
    c = campaign
    first = recovery.recover(c.root)
    previous.write(c.root / "summary.json", {"status": "complete", "recorded": 36})
    before = snapshot(path for path in c.root.rglob("*") if path.is_file())
    c.accounting = "must not query old accounting again"
    assert recovery.recover(c.root) == first
    assert len(c.sbatch) == 1
    assert sum(argv[0] == "sacct" for argv, _ in c.commands) == 1
    assert snapshot(before) == before


@pytest.mark.parametrize("changed", [
    lambda: accounting([array_row("[0-34]")]),
    lambda: accounting([array_row("[0-35]", state="FAILED")]),
    lambda: accounting([array_row("[0-35]", state="RUNNING")]),
    lambda: accounting([array_row("[0-35]", exit_code="1:0")]),
    lambda: accounting([array_row("[0-35]"), array_row("35")]),
    lambda: accounting().replace("prepare|COMPLETED|0:0", "prepare|COMPLETED|1:0"),
    lambda: accounting().replace("report|TIMEOUT", "report|RUNNING"),
    lambda: "\n".join(accounting().splitlines()[:-1]),
    lambda: accounting() + "\nmalformed|row",
])
def test_incomplete_failed_or_live_accounting_submits_nothing(campaign, changed):
    campaign.accounting = changed()
    with pytest.raises(ValueError):
        recovery.recover(campaign.root)
    assert not campaign.sbatch
    assert not (campaign.root / recovery.RECOVERY / "identity.json").exists()


@pytest.mark.parametrize("job", ["301", "302_[0-35%2]", "302_35", "303"])
def test_live_queue_blocks_report_retry(campaign, job):
    campaign.queue = job + "\n"
    with pytest.raises(ValueError, match="still queued"):
        recovery.recover(campaign.root)
    assert not campaign.sbatch


def test_missing_terminal_result_blocks_report_retry(campaign):
    (campaign.root / "results/task-35/result.json").unlink()
    with pytest.raises(ValueError, match="not all 36"):
        recovery.recover(campaign.root)
    assert not campaign.sbatch


@pytest.mark.parametrize("change", [
    "manifest_identity", "original_jobs", "accepted", "intent",
    "helper_hash", "bundle_manifest",
])
def test_changed_preparation_chain_blocks_report_retry(campaign, change):
    c = campaign
    folder = c.root / previous.RECOVERY
    manifest = previous.read(folder / "manifest.json")
    if change == "manifest_identity":
        manifest["identity"] = manifest["identity"] | {"unexpected": True}
        previous.write(folder / "manifest.json", manifest)
    elif change == "original_jobs":
        manifest["original_jobs"]["run"] = "999"
        previous.write(folder / "manifest.json", manifest)
    elif change == "accepted":
        previous.write(folder / "run.accepted.json", {"job": "999"})
    elif change == "intent":
        previous.write(folder / "report.intent.json", {"argv": ["changed"]})
    elif change == "helper_hash":
        identity = manifest["identity"] | {"helper_sha256": "0" * 64}
        previous.write(folder / "identity.json", identity)
        previous.write(folder / "manifest.json", manifest | {"identity": identity})
    else:
        (c.repo / "SHA256SUMS").write_text("changed bundle manifest")
    with pytest.raises(ValueError, match=r"identity differs|receipt differs"):
        recovery.recover(c.root)
    assert not c.sbatch


@pytest.mark.parametrize("change", ["helper_hash", "accepted", "intent", "manifest"])
def test_changed_retry_identity_or_receipts_never_resubmit(campaign, change):
    c = campaign
    recovery.recover(c.root)
    folder = c.root / recovery.RECOVERY
    if change == "helper_hash":
        identity = previous.read(folder / "identity.json")
        previous.write(
            folder / "identity.json", identity | {"helper_sha256": "changed"}
        )
    elif change == "accepted":
        previous.write(folder / "report.accepted.json", {"job": "invalid"})
    elif change == "intent":
        previous.write(folder / "report.intent.json", {"argv": ["changed"]})
    else:
        previous.write(folder / "manifest.json", {"changed": True})
    with pytest.raises(ValueError, match=r"identity differs|manifest differs"):
        recovery.recover(c.root)
    assert len(c.sbatch) == 1


@pytest.mark.parametrize(
    "failure", ["timeout", "oserror", "nonzero", "empty", "malformed"]
)
def test_uncertain_report_submission_never_retries(campaign, failure):
    c = campaign
    c.failure = failure
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        recovery.recover(c.root)
    folder = c.root / recovery.RECOVERY
    assert (folder / "report.intent.json").is_file()
    assert (folder / "report.reply.json").is_file()
    assert not (folder / "report.accepted.json").exists()
    c.failure = None
    with pytest.raises(ValueError, match="Unconfirmed report"):
        recovery.recover(c.root)
    assert len(c.sbatch) == 1


def test_two_script_delivery_works_with_isolated_python(tmp_path):
    for module in (recovery, previous):
        path = Path(module.__file__)
        (tmp_path / path.name).write_bytes(path.read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", str(tmp_path / Path(recovery.__file__).name), "--help"],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=10,
    )
    assert "--root" in result.stdout
    assert not result.stderr
