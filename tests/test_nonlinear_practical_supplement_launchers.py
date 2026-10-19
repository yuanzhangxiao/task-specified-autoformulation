"""M42 CPU-only receipts, fail-closed resume, visible preflight and CLI checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import submit_phase_c_nonlinear_practical_supplement as submitter

CONFIG = Path("configs/phase_c_nonlinear_practical_supplement_v1.json")
STEM = "phase_c_nonlinear_practical_supplement"


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Isolate scheduling; campaign tests validate immutable source artifacts."""
    source = tmp_path / "m41"
    source.mkdir()
    root = tmp_path / "supplement"
    plan = {"tasks": [{"task_id": str(i)} for i in range(2)]}

    def prepare(output, saved, policy):
        assert saved == source
        assert policy.selection == "incomplete_profiles"
        assert policy.diagnostic.reference_boundary_cells is True
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


def test_cpu_only_idempotent_two_task_supplement(scheduler):
    root, source, _, calls = scheduler
    first = submitter.submit(root, source, CONFIG, account="test")
    assert first["array_tasks"] == 2
    assert first["gpus"] == first["live_llm_calls"] == 0
    assert first["test_data_opened"] is False
    assert "--array=0-1%2" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101:102" in calls[2]
    for argv in calls:
        assert "--cpus-per-task=1" in argv and "--partition=cpu" in argv
        assert not any("--gres" in item or "--gpus" in item for item in argv)
        assert argv[-2].endswith(f"{STEM}_delta.sh")
    assert "--time=00:10:00" in calls[1]
    assert submitter.submit(root, source, CONFIG, account="test") == first
    assert len(calls) == 3


def test_submission_progress_is_visible_on_stderr(scheduler, capsys):
    root, source, _, _ = scheduler
    submitter.submit(root, source, CONFIG, account="test")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reading saved M41 diagnostics" in captured.err
    assert "source frozen" in captured.err
    assert "2 diagnostics selected" in captured.err


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
    assert (root / "submission" / f"{stage}.intent.json").exists()
    assert (root / "submission" / f"{stage}.reply.json").exists()
    assert not (root / "submission" / f"{stage}.id").exists()
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(root, source, CONFIG, account="test")
    assert len(calls) == stage_index + 1


def test_empty_roster_submits_nothing(scheduler):
    root, source, plan, calls = scheduler
    plan["tasks"] = []
    result = submitter.submit(root, source, CONFIG, account="test")
    assert result["status"] == "no_pending_diagnostics" and calls == []
    assert result["parameter_refitting_performed"] is False


def test_changed_receipt_identity_blocked(scheduler):
    root, source, _, calls = scheduler
    submitter.submit(root, source, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(root, source, CONFIG, account="test", concurrency=3)
    assert len(calls) == 3


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_does_not_read_source(scheduler, monkeypatch, concurrency):
    root, source, _, calls = scheduler
    monkeypatch.setattr(
        submitter.campaign, "prepare",
        lambda *_: pytest.fail("read source before rejecting invalid scheduler policy"),
    )
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(root, source, CONFIG, account="test", concurrency=concurrency)
    assert calls == []


def test_delta_shell_preflight_and_source_override(tmp_path):
    """Run shell argument handling without accessing Delta or scheduler commands."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text("{}")
    fake_python = tmp_path / "python"
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
        AF_PYTHON=str(fake_python),
        AF_NONLINEAR_SOURCE=str(source),
        AF_NONLINEAR_SUPPLEMENT_ROOT=str(tmp_path / "output"),
    )
    result = subprocess.run(
        ["bash", f"scripts/hpc/submit_{STEM}_delta.sh"],
        env=environment, check=True, capture_output=True, text=True,
    )
    invocation = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    args = invocation["argv"]
    assert args[args.index("--source") + 1] == str(source)
    assert args[args.index("--root") + 1] == str(tmp_path / "output")
    assert args[args.index("--account") + 1] == "bibo-delta-cpu"
    assert args[args.index("--concurrency") + 1] == "2"
    assert "checking source plan" in result.stdout
    assert "checking numerical imports" in result.stdout


def test_shell_syntax_help_and_prepare_gate():
    for stage in ("run", "submit", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{stage}_{STEM}_delta.sh"], check=True
        )
    for script in (STEM, f"submit_{STEM}"):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}.py", "--help"],
            check=True, capture_output=True, text=True,
        )
        assert "--root" in result.stdout or "prepare,verify,run,report" in result.stdout
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh").read_text()
    tests = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", worker))
    assert tests == {
        "tests/test_nonlinear_practical_profile.py",
        "tests/test_nonlinear_practical_supplement.py",
        "tests/test_nonlinear_practical_supplement_launchers.py",
    }
    assert all(Path(path).is_file() for path in tests)
