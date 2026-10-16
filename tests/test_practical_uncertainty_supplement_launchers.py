"""M40 CPU-only submission, immutable receipt recovery and standalone CLIs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import submit_phase_c_practical_uncertainty_supplement as submitter

CONFIG = Path("configs/phase_c_practical_uncertainty_supplement_v1.json")


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Isolate scheduler effects; real campaign source validation has its own tests."""
    source = tmp_path / "old-m39"
    source.mkdir()
    root = tmp_path / "supplement"
    plan = {"tasks": [{"task_id": str(i)} for i in range(9)]}

    def prepare(output, saved, policy):
        assert saved == source
        assert policy.maximum_seconds == 90
        output.mkdir(exist_ok=True)

    monkeypatch.setattr(submitter.campaign, "prepare", prepare)
    monkeypatch.setattr(submitter.campaign, "verify", lambda *_: (plan, {}))
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    return root, source, plan, calls


@pytest.mark.parametrize("platform", ["delta", "aces"])
def test_cpu_only_idempotent_nine_task_supplement(scheduler, platform):
    root, source, _, calls = scheduler
    first = submitter.submit(root, source, CONFIG, account="test", platform=platform)
    assert first["array_tasks"] == 9
    assert first["gpus"] == first["live_llm_calls"] == 0
    assert first["test_data_opened"] is False
    assert "--array=0-8%2" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101:102" in calls[2]
    for argv in calls:
        assert "--cpus-per-task=1" in argv and "--partition=cpu" in argv
        assert not any("--gres" in item or "--gpus" in item for item in argv)
        assert argv[-2].endswith(f"practical_uncertainty_supplement_{platform}.sh")
    assert (
        submitter.submit(root, source, CONFIG, account="test", platform=platform)
        == first
    )
    assert len(calls) == 3


def test_platform_change_cannot_reuse_delta_receipts(scheduler):
    root, source, _, calls = scheduler
    submitter.submit(root, source, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(root, source, CONFIG, account="test", platform="aces")
    assert len(calls) == 3


def test_invalid_platform_submits_nothing(scheduler):
    root, source, _, calls = scheduler
    with pytest.raises(ValueError, match="platform must be"):
        submitter.submit(root, source, CONFIG, account="test", platform="unknown")
    assert calls == []


def test_submission_progress_is_visible_on_stderr(scheduler, capsys):
    root, source, _, _ = scheduler
    submitter.submit(root, source, CONFIG, account="test")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "auditing saved fits" in captured.err
    assert "snapshot prepared" in captured.err
    assert "9 diagnostics selected" in captured.err


def test_aces_shell_selects_local_runtime_source_and_worker(tmp_path):
    """Execute the shell boundary without modules, remote storage or scheduler calls."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text("{}")
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "module").write_text("#!/bin/bash\nexit 0\n")
    (bins / "module").chmod(0o755)
    fake_python = bins / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if '-c' not in sys.argv:\n"
        "    print(json.dumps({'argv': sys.argv[1:], "
        "'path': os.environ['PYTHONPATH']}))\n"
    )
    fake_python.chmod(0o755)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("AF_")
    }
    environment.update(
        PATH=f"{bins}{os.pathsep}{os.environ['PATH']}",
        AF_PYTHON=str(fake_python),
        AF_GROUP=str(tmp_path / "group"),
        AF_PROFILE_SOURCE=str(source),
    )
    result = subprocess.run(
        ["bash", "scripts/hpc/submit_phase_c_practical_uncertainty_supplement_aces.sh"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    invocation = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    args = invocation["argv"]
    assert args[args.index("--platform") + 1] == "aces"
    assert args[args.index("--source") + 1] == str(source)
    assert args[args.index("--account") + 1] == "156264627414"
    assert args[args.index("--root") + 1].endswith(
        "practical-uncertainty-supplement-aces-v1"
    )
    assert "/projects/bibo" not in invocation["path"]
    assert "checking numerical imports" in result.stdout


@pytest.mark.parametrize(
    "stage_index,stage", [(0, "prepare"), (1, "run"), (2, "report")]
)
@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_receipt_never_resubmits(
    scheduler, monkeypatch, stage_index, stage, timeout
):
    root, source, _, calls = scheduler

    def uncertain(argv, **kwargs):
        calls.append(argv)
        if len(calls) - 1 == stage_index:
            if timeout:
                raise subprocess.TimeoutExpired(argv, 45)
            return SimpleNamespace(returncode=0, stdout="", stderr="Socket timed out")
        return SimpleNamespace(returncode=0, stdout=str(200 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", uncertain)
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        submitter.submit(root, source, CONFIG, account="test")
    assert len(calls) == stage_index + 1
    assert (root / "submission" / f"{stage}.reply.json").exists()
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(root, source, CONFIG, account="test")
    assert len(calls) == stage_index + 1


def test_empty_roster_submits_nothing(scheduler):
    root, source, plan, calls = scheduler
    plan["tasks"] = []
    result = submitter.submit(root, source, CONFIG, account="test")
    assert result["status"] == "no_pending_diagnostics" and calls == []


def test_changed_receipt_identity_blocked(scheduler):
    root, source, _, calls = scheduler
    submitter.submit(root, source, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(root, source, CONFIG, account="test", concurrency=3)
    assert len(calls) == 3


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_submits_nothing(scheduler, concurrency):
    root, source, _, calls = scheduler
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(root, source, CONFIG, account="test", concurrency=concurrency)
    assert calls == []


def test_shell_syntax_help_and_prepare_gate():
    stem = "phase_c_practical_uncertainty_supplement"
    for stage in ("run", "submit", "inspect"):
        for platform in ("delta", "aces"):
            subprocess.run(
                ["bash", "-n", f"scripts/hpc/{stage}_{stem}_{platform}.sh"], check=True
            )
    for script in (stem, f"submit_{stem}"):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--root" in result.stdout or "prepare,verify,run,report" in result.stdout
    worker = Path(f"scripts/hpc/run_{stem}_delta.sh").read_text()
    tests = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", worker))
    assert tests == {
        "tests/test_practical_profile.py",
        "tests/test_practical_uncertainty_supplement.py",
        "tests/test_practical_uncertainty_supplement_launchers.py",
    }
    assert all(Path(path).is_file() for path in tests)
