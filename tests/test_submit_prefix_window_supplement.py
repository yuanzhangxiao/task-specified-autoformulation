"""M45 CPU launchers, source-read preflight and fail-closed scheduler receipts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import phase_c_prefix_window_supplement as cli
from scripts import submit_phase_c_prefix_window_supplement as submitter

CONFIG = Path("configs/phase_c_prefix_window_supplement_v1.json")
STEM = "phase_c_prefix_window_supplement"
TASKS = ["clean_r0_s0", "clean_r0_s1"]


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Use real scheduler receipts with a sealed two-task plan and fake Slurm."""
    source = tmp_path / "m44"
    source.mkdir()
    root = tmp_path / "supplement"
    plan = {"tasks": [{"task_id": name} for name in TASKS]}
    prepared = []
    verified = []

    def prepare(output, saved, policy):
        assert saved in (source, None)
        assert policy.protocol == "phase-c-prefix-window-supplement-policy-1"
        assert policy.selection == "failed_verification_only"
        prepared.append((output, saved, policy))
        output.mkdir(exist_ok=True)
        return {"status": "prepared", "tasks": len(plan["tasks"])}

    def verify(output):
        verified.append(output)
        return plan, {}

    monkeypatch.setattr(submitter.campaign, "prepare", prepare)
    monkeypatch.setattr(submitter.campaign, "verify", verify)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    return SimpleNamespace(
        root=root, source=source, plan=plan, prepared=prepared,
        verified=verified, calls=calls,
    )


def test_cpu_only_two_task_supplement_is_idempotent(scheduler):
    s = scheduler
    first = submitter.submit(s.root, s.source, CONFIG, account="test")
    assert first["array_tasks"] == 2
    assert first["gpus"] == first["live_llm_calls"] == 0
    assert first["test_data_opened"] is False
    assert len(s.prepared) == len(s.verified) == 1
    assert s.prepared[0][1] == s.source
    assert "--array=0-1%2" in s.calls[1]
    assert "--dependency=afterok:101" in s.calls[1]
    assert "--dependency=afterany:101:102" in s.calls[2]
    for argv, duration in zip(
        s.calls, ("00:10:00", "00:20:00", "00:05:00"), strict=True
    ):
        assert "--cpus-per-task=1" in argv and "--mem=16G" in argv
        assert "--partition=cpu" in argv and f"--time={duration}" in argv
        assert not any("--gres" in item or "--gpus" in item for item in argv)
        assert argv[-2].endswith(f"run_{STEM}_delta.sh")
    assert submitter.submit(s.root, None, CONFIG, account="test") == first
    assert s.prepared[1][1] is None
    assert len(s.calls) == 3


def test_progress_is_visible_before_source_read(scheduler, monkeypatch, capsys):
    s = scheduler
    prepare = submitter.campaign.prepare

    def observed(*args):
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "reading saved M44 artifacts" in captured.err
        return prepare(*args)

    monkeypatch.setattr(submitter.campaign, "prepare", observed)
    submitter.submit(s.root, s.source, CONFIG)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "source frozen" in captured.err and "2 incomplete tasks" in captured.err


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_does_not_read_source(
    scheduler, monkeypatch, concurrency
):
    s = scheduler
    monkeypatch.setattr(
        submitter.campaign, "prepare",
        lambda *_: pytest.fail("read source before scheduler preflight"),
    )
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(s.root, s.source, CONFIG, concurrency=concurrency)
    assert s.calls == []


@pytest.mark.parametrize("python", [None, "/no/such/python"])
def test_missing_python_does_not_read_source(scheduler, monkeypatch, python):
    s = scheduler
    if python is None:
        monkeypatch.delenv("AF_PYTHON")
    else:
        monkeypatch.setenv("AF_PYTHON", python)
    monkeypatch.setattr(
        submitter.campaign, "prepare",
        lambda *_: pytest.fail("read source before Python preflight"),
    )
    with pytest.raises(ValueError, match="existing Python"):
        submitter.submit(s.root, s.source, CONFIG)
    assert s.calls == []


@pytest.mark.parametrize("change", [{"account": "other"}, {"concurrency": 3}])
def test_changed_submission_identity_is_rejected(scheduler, change):
    s = scheduler
    submitter.submit(s.root, s.source, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(s.root, s.source, CONFIG, **({"account": "test"} | change))
    assert len(s.calls) == 3


@pytest.mark.parametrize(
    "stage_index,stage", [(0, "prepare"), (1, "run"), (2, "report")]
)
@pytest.mark.parametrize("failure", ["timeout", "empty_reply", "nonzero"])
def test_uncertain_receipt_never_resubmits(
    scheduler, monkeypatch, stage_index, stage, failure
):
    s = scheduler

    def uncertain(argv, **kwargs):
        s.calls.append(argv)
        if len(s.calls) - 1 == stage_index:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 45)
            return SimpleNamespace(
                returncode=int(failure == "nonzero"), stdout="", stderr="unconfirmed"
            )
        return SimpleNamespace(
            returncode=0, stdout=str(200 + len(s.calls)), stderr=""
        )

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", uncertain)
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        submitter.submit(s.root, s.source, CONFIG)
    assert len(s.calls) == stage_index + 1
    assert (s.root / "submission" / f"{stage}.intent.json").is_file()
    assert (s.root / "submission" / f"{stage}.reply.json").is_file()
    assert not (s.root / "submission" / f"{stage}.id").exists()
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(s.root, None, CONFIG)
    assert len(s.calls) == stage_index + 1


def test_empty_roster_submits_nothing(scheduler):
    s = scheduler
    s.plan["tasks"] = []
    result = submitter.submit(s.root, s.source, CONFIG)
    assert result["status"] == "no_numerical_gaps"
    assert result["array_tasks"] == 0 and result["jobs"] == {}
    assert result["gpus"] == result["live_llm_calls"] == 0
    assert result["test_data_opened"] is False and s.calls == []


def test_scheduler_adapter_rejects_changed_root_or_policy(tmp_path):
    policy = cli.campaign.Policy()
    adapter = submitter._PreparedCampaign(tmp_path, policy, {"tasks": []}, {})
    adapter.prepare(tmp_path, tmp_path, policy)
    assert adapter.verify(tmp_path) == ({"tasks": []}, {})
    for root, inputs, changed in (
        (tmp_path / "other", tmp_path, policy),
        (tmp_path, tmp_path / "other", policy),
        (tmp_path, tmp_path, policy.model_copy(update={"selection": "changed"})),
    ):
        with pytest.raises(ValueError, match="prepared campaign identity"):
            adapter.prepare(root, inputs, changed)
    with pytest.raises(ValueError, match="prepared campaign root"):
        adapter.verify(tmp_path / "other")


@pytest.mark.parametrize("command", ["prepare", "verify", "run"])
def test_cli_routes_arguments_to_campaign(tmp_path, monkeypatch, capsys, command):
    observed = []
    monkeypatch.setattr(
        cli.campaign, "prepare",
        lambda *args: observed.append(args) or {"status": "prepared"},
    )
    monkeypatch.setattr(
        cli.campaign, "verify",
        lambda *args: observed.append(args) or ({"tasks": TASKS}, {}),
    )
    monkeypatch.setattr(
        cli.campaign, "run_task",
        lambda *args: observed.append(args)
        or {"task_id": TASKS[1], "status": "complete", "rows": [1, 2]},
    )
    argv = ["cli", command, "--root", str(tmp_path)]
    if command == "prepare":
        argv += ["--source", str(tmp_path / "source"), "--config", str(CONFIG)]
    elif command == "run":
        argv += ["--index", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert observed[0][0] == tmp_path and f"M45: {command}" in captured.err
    if command == "prepare":
        assert observed[0][1] == tmp_path / "source"
        assert isinstance(observed[0][2], cli.campaign.Policy)
    elif command == "run":
        assert observed[0][1] == 1 and "rows" not in receipt
        assert receipt["artifact"].endswith(f"/results/{TASKS[1]}/result.json")
    else:
        assert receipt["verified"] and receipt["tasks"] == 2


def test_cli_prepare_can_resume_snapshot_without_original_source(
    tmp_path, monkeypatch, capsys
):
    observed = []
    monkeypatch.setattr(
        cli.campaign, "prepare",
        lambda *args: observed.append(args) or {"status": "prepared"},
    )
    monkeypatch.setattr(
        sys, "argv", ["cli", "prepare", "--root", str(tmp_path)]
    )
    cli.main()
    assert observed[0][1] is None
    assert "snapshot" in capsys.readouterr().err


def test_report_receipt_is_compact_and_preserves_full_artifact(
    tmp_path, monkeypatch, capsys
):
    full = {
        "status": "complete",
        "expected": 2,
        "recorded": 2,
        "status_counts": {"complete": 2},
        "recovered_forecast_gaps": 3,
        "selections": [{"task_id": name, "duration": 0.1} for name in TASKS],
        "source_cost": {"wall_seconds": 1602.98, "operations": [1] * 1000},
        "supplemental_cost": {"wall_seconds": 12.3, "operations": [2] * 1000},
        "combined_cost": {"wall_seconds": 1615.28, "operations": [3] * 1000},
        "test_data_opened": False,
        "rows": [{"forecasts": [1.0] * 1000} for _ in TASKS],
    }

    def report(root):
        (root / "summary.json").write_text(json.dumps(full))
        return full

    monkeypatch.setattr(cli.campaign, "report", report)
    monkeypatch.setattr(sys, "argv", ["cli", "report", "--root", str(tmp_path)])
    cli.main()
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert "rows" not in receipt
    for name in ("source_cost", "supplemental_cost", "combined_cost"):
        assert receipt[name] == {"wall_seconds": full[name]["wall_seconds"]}
    assert receipt["selections"] == full["selections"]
    assert receipt["recovered_forecast_gaps"] == 3
    assert receipt["artifact"] == str(tmp_path / "summary.json")
    assert json.loads((tmp_path / "summary.json").read_text()) == full
    assert len(captured.out) < 1500


@pytest.mark.parametrize("no_gaps", [False, True])
def test_submission_cli_prints_small_receipt(
    tmp_path, monkeypatch, capsys, no_gaps
):
    monkeypatch.setattr(
        submitter, "submit",
        lambda **kwargs: {
            "status": "no_numerical_gaps" if no_gaps else "submitted",
            "identity": {"details": list(range(1000))},
            "jobs": {} if no_gaps else {"run": "42"},
            "array_tasks": 0 if no_gaps else 2,
            "gpus": 0,
        },
    )
    monkeypatch.setattr(sys, "argv", ["submit", "--root", str(tmp_path)])
    submitter.main()
    receipt = json.loads(capsys.readouterr().out)
    assert "identity" not in receipt
    artifact = "plan.json" if no_gaps else "submission_manifest.json"
    assert receipt["artifact"] == str(tmp_path / artifact)


def test_delta_shell_preflight_and_overrides(tmp_path):
    """Exercise shell argument handling without accessing Delta or Slurm."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text("{}")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if '-c' not in sys.argv:\n"
        "    print(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: os.environ[k] for k in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS', "
        "'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}))\n"
    )
    fake_python.chmod(0o755)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("AF_")
    }
    environment.update(
        AF_PYTHON=str(fake_python),
        AF_PREFIX_WINDOW_SOURCE=str(source),
        AF_PREFIX_WINDOW_SUPPLEMENT_ROOT=str(tmp_path / "output"),
        AF_PREFIX_WINDOW_SUPPLEMENT_CONFIG=str(tmp_path / "config.json"),
        AF_CASADI_ROOT=str(tmp_path / "deps"),
        AF_ACCOUNT="test",
        AF_CONCURRENCY="2",
    )
    result = subprocess.run(
        ["bash", f"scripts/hpc/submit_{STEM}_delta.sh"],
        env=environment, check=True, capture_output=True, text=True,
    )
    invocation = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    argv = invocation["argv"]
    for option, value in (
        ("--source", source),
        ("--root", tmp_path / "output"),
        ("--config", tmp_path / "config.json"),
        ("--account", "test"),
        ("--concurrency", "2"),
    ):
        assert argv[argv.index(option) + 1] == str(value)
    assert "checking source plan" in result.stdout
    assert "checking numerical imports" in result.stdout
    assert invocation["env"]["PYTHONPATH"].endswith(str(tmp_path / "deps"))
    assert all(
        invocation["env"][name] == "1"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )

    (source / "plan.json").unlink()
    failed = subprocess.run(
        ["bash", f"scripts/hpc/submit_{STEM}_delta.sh"],
        env=environment, capture_output=True, text=True,
    )
    assert failed.returncode != 0 and "No M44 plan" in failed.stderr
    assert "checking numerical imports" not in failed.stdout


def test_shell_syntax_help_and_prepare_gate():
    for stage in ("run", "submit"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{stage}_{STEM}_delta.sh"], check=True
        )
    for script in (STEM, f"submit_{STEM}"):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}.py", "--help"],
            check=True, capture_output=True, text=True,
        )
        assert "--root" in result.stdout or "prepare,verify,run,report" in result.stdout
        assert result.stderr == "" and len(result.stdout) < 2000
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh").read_text()
    tests = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", worker))
    assert tests == {
        "tests/test_prefix_verification.py",
        "tests/test_prefix_window_supplement.py",
        "tests/test_submit_prefix_window_supplement.py",
    }
    assert all(Path(path).is_file() for path in tests)
    assert "pip install" not in worker
