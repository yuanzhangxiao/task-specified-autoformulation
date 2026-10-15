"""M39 CPU submission receipts, immutable resume and standalone entry points."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.fitting import practical_uncertainty_campaign as campaign
from scripts import submit_phase_c_practical_uncertainty as submitter

CONFIG = Path("configs/phase_c_practical_uncertainty_v1.json")


@pytest.fixture(autouse=True)
def stable_source_identity(monkeypatch):
    """Keep receipt unit tests independent of concurrent checkout edits."""
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture
def scheduler(monkeypatch):
    """Use fake Slurm receipts while retaining real plan and submission seals."""
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(200 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    return calls


def test_submission_idempotent_cpu_array_and_afterany_report(tmp_path, scheduler):
    root = tmp_path / "campaign"
    first = submitter.submit(root, CONFIG, account="test", concurrency=2)
    assert first["array_tasks"] == 150 and first["gpus"] == 0
    assert first["live_llm_calls"] == 0 and not first["test_data_opened"]
    assert "--array=0-149%2" in scheduler[1]
    assert "--dependency=afterok:201" in scheduler[1]
    assert "--dependency=afterany:201:202" in scheduler[2]
    for argv in scheduler:
        assert "--cpus-per-task=1" in argv and "--partition=cpu" in argv
        assert argv[-2].endswith("run_phase_c_practical_uncertainty_delta.sh")
        assert not any("--gres" in item or "--gpus" in item for item in argv)
    assert "--time=00:10:00" in scheduler[1]
    assert submitter.submit(root, CONFIG, account="test", concurrency=2) == first
    assert len(scheduler) == 3
    plan, _ = campaign.verify(root)
    assert plan["protocol"] == "phase-c-practical-uncertainty-fitting-1"


@pytest.mark.parametrize(
    "stage_index,stage", [(0, "prepare"), (1, "run"), (2, "report")]
)
@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_receipt_never_resubmits(
    tmp_path, scheduler, monkeypatch, stage_index, stage, timeout
):
    calls = []

    def uncertain(argv, **kwargs):
        calls.append(argv)
        if len(calls) - 1 == stage_index:
            if timeout:
                raise subprocess.TimeoutExpired(argv, 45)
            return SimpleNamespace(returncode=0, stdout="", stderr="uncertain")
        return SimpleNamespace(returncode=0, stdout=str(300 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", uncertain)
    root = tmp_path / "uncertain"
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        submitter.submit(root, CONFIG, account="test")
    assert len(calls) == stage_index + 1
    assert (root / "submission" / f"{stage}.intent.json").is_file()
    assert (root / "submission" / f"{stage}.reply.json").is_file()
    assert not (root / "submission" / f"{stage}.id").exists()
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("resubmitted an unconfirmed scheduler intent"),
    )
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(root, CONFIG, account="test")


def test_changed_submission_identity_rejected_before_sbatch(tmp_path, scheduler):
    root = tmp_path / "campaign"
    submitter.submit(root, CONFIG, account="test", concurrency=2)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(root, CONFIG, account="test", concurrency=3)
    assert len(scheduler) == 3


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_rejected_before_sbatch(tmp_path, scheduler, concurrency):
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(
            tmp_path / "campaign", CONFIG, account="test", concurrency=concurrency
        )
    assert scheduler == []


def test_hpc_shell_syntax_and_cli_help():
    for kind in ("submit", "run", "inspect"):
        subprocess.run(
            [
                "bash", "-n",
                f"scripts/hpc/{kind}_phase_c_practical_uncertainty_delta.sh",
            ],
            check=True,
        )
    for script in (
        "phase_c_practical_uncertainty.py", "submit_phase_c_practical_uncertainty.py"
    ):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--root" in result.stdout or "prepare,verify,run,report" in result.stdout


def test_worker_prepare_gate_references_existing_qualification_tests():
    worker = Path("scripts/hpc/run_phase_c_practical_uncertainty_delta.sh").read_text()
    tests = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", worker))
    assert tests == {
        "tests/test_practical_profile.py",
        "tests/test_practical_uncertainty_fixtures.py",
        "tests/test_practical_uncertainty_campaign.py",
        "tests/test_practical_uncertainty_launchers.py",
    }
    assert all(Path(name).is_file() for name in tests)
