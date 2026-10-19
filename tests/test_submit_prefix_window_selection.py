"""M44 CPU launchers, frozen submission receipts and compact standalone CLIs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import phase_c_prefix_window_selection as cli
from scripts import submit_phase_c_prefix_window_selection as submitter

CONFIG = Path("configs/phase_c_prefix_window_selection_v1.json")
TASKS = [
    f"{case}_s{seed}"
    for case in ("clean_r0", "noisy_r0", "noisy_r1")
    for seed in (0, 1)
]


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Exercise real scheduler receipts with a sealed-plan stand-in and no jobs."""
    root = tmp_path / "campaign"
    plan = {"tasks": [{"task_id": name} for name in TASKS]}
    prepared = []

    def prepare(output, inputs, policy):
        assert policy.protocol == "phase-c-prefix-window-policy-1"
        assert list(policy.durations) == [0, 0.1, 0.2, 0.5, 1, 2]
        prepared.append(inputs)
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
    return root, plan, prepared, calls


def test_cpu_only_six_task_receipts_are_idempotent(scheduler):
    root, _, prepared, calls = scheduler
    first = submitter.submit(root, CONFIG, account="test")
    assert first["array_tasks"] == 6
    assert first["gpus"] == first["live_llm_calls"] == 0
    assert first["test_data_opened"] is False
    assert prepared == [None, root / "inputs.json"]
    assert "--array=0-5%2" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101:102" in calls[2]
    for argv, duration in zip(calls, ("00:10:00", "00:30:00", "00:05:00"), strict=True):
        assert "--cpus-per-task=1" in argv and "--mem=16G" in argv
        assert "--partition=cpu" in argv and f"--time={duration}" in argv
        assert not any("--gres" in item or "--gpus" in item for item in argv)
        assert argv[-2].endswith("run_phase_c_prefix_window_selection_delta.sh")
    assert submitter.submit(root, CONFIG, account="test") == first
    assert len(calls) == 3


def test_imported_inputs_only_used_for_initial_freeze(scheduler, tmp_path):
    root, _, prepared, _ = scheduler
    imported = tmp_path / "imported.json"
    submitter.submit(root, CONFIG, account="test", inputs=imported)
    assert prepared == [imported, root / "inputs.json"]


@pytest.mark.parametrize("change", [{"concurrency": 3}, {"account": "different"}])
def test_changed_submission_identity_is_rejected(scheduler, change):
    root, _, _, calls = scheduler
    submitter.submit(root, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(root, CONFIG, **({"account": "test"} | change))
    assert len(calls) == 3


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_submits_nothing(scheduler, concurrency):
    root, _, _, calls = scheduler
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(root, CONFIG, concurrency=concurrency)
    assert calls == []


@pytest.mark.parametrize(
    "stage_index,stage", [(0, "prepare"), (1, "run"), (2, "report")]
)
@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_receipt_never_resubmits(
    scheduler, monkeypatch, stage_index, stage, timeout
):
    root, _, _, calls = scheduler

    def uncertain(argv, **kwargs):
        calls.append(argv)
        if len(calls) - 1 == stage_index:
            if timeout:
                raise subprocess.TimeoutExpired(argv, 45)
            return SimpleNamespace(returncode=0, stdout="", stderr="Socket timed out")
        return SimpleNamespace(returncode=0, stdout=str(200 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", uncertain)
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        submitter.submit(root, CONFIG)
    assert len(calls) == stage_index + 1
    assert (root / "submission" / f"{stage}.intent.json").is_file()
    assert (root / "submission" / f"{stage}.reply.json").is_file()
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(root, CONFIG)
    assert len(calls) == stage_index + 1


def test_empty_roster_submits_nothing(scheduler):
    root, plan, _, calls = scheduler
    plan["tasks"] = []
    assert submitter.submit(root, CONFIG)["status"] == "no_tasks"
    assert calls == []


def test_progress_is_visible_on_stderr(scheduler, capsys):
    root, _, _, _ = scheduler
    submitter.submit(root, CONFIG)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "preparing frozen tasks" in captured.err and "6 tasks" in captured.err


def test_report_stdout_is_compact_and_full_report_remains_durable(
    tmp_path, monkeypatch, capsys
):
    full = {
        "status": "complete",
        "expected": 6,
        "recorded": 6,
        "status_counts": {"complete": 6},
        "selection_counts": {"selected": 6},
        "selections": [{"task_id": name, "duration": 0.2} for name in TASKS],
        "cost": {"wall_seconds": 12.3, "calls": 30, "per_operation": [{"calls": 30}]},
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
    assert "rows" not in receipt and "per_operation" not in receipt["cost"]
    assert receipt["selections"] == full["selections"]
    assert receipt["test_data_opened"] is False
    assert receipt["artifact"] == str(tmp_path / "summary.json")
    assert json.loads((tmp_path / "summary.json").read_text()) == full
    assert len(captured.out) < 2000 and "M44: report" in captured.err


@pytest.mark.parametrize("command", ["prepare", "verify", "run"])
def test_cli_routes_arguments_to_campaign(tmp_path, monkeypatch, capsys, command):
    observed = []
    monkeypatch.setattr(
        cli.campaign,
        "prepare",
        lambda *args: observed.append(args) or {"status": "prepared"},
    )
    monkeypatch.setattr(
        cli.campaign,
        "verify",
        lambda *args: observed.append(args) or ({"tasks": TASKS}, {}),
    )
    monkeypatch.setattr(
        cli.campaign,
        "run_task",
        lambda *args: observed.append(args)
        or {"task_id": TASKS[2], "status": "complete", "rows": [1, 2]},
    )
    argv = ["cli", command, "--root", str(tmp_path)]
    if command == "prepare":
        argv += ["--inputs", str(tmp_path / "imported.json"), "--config", str(CONFIG)]
    elif command == "run":
        argv += ["--index", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    receipt = json.loads(capsys.readouterr().out)
    assert observed[0][0] == tmp_path
    if command == "prepare":
        assert observed[0][1] == tmp_path / "imported.json"
        assert isinstance(observed[0][2], cli.campaign.Policy)
    elif command == "run":
        assert observed[0][1] == 2 and "rows" not in receipt
        assert receipt["artifact"].endswith(f"/results/{TASKS[2]}/result.json")
    else:
        assert receipt["verified"] and receipt["tasks"] == 6


def test_submission_cli_prints_small_receipt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        submitter,
        "submit",
        lambda **kwargs: {
            "identity": {"details": list(range(1000))},
            "jobs": {"run": "42"},
            "array_tasks": 6,
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
        },
    )
    monkeypatch.setattr(sys, "argv", ["submit", "--root", str(tmp_path)])
    submitter.main()
    receipt = json.loads(capsys.readouterr().out)
    assert "identity" not in receipt
    assert receipt["jobs"] == {"run": "42"}
    assert receipt["artifact"] == str(tmp_path / "submission_manifest.json")


def test_delta_shell_honors_local_overrides_without_submitting(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: os.environ[k] for k in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS', "
        "'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}))\n"
    )
    fake_python.chmod(0o755)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    environment.update(
        AF_PYTHON=str(fake_python),
        AF_PREFIX_WINDOW_ROOT=str(tmp_path / "output"),
        AF_PREFIX_WINDOW_CONFIG=str(tmp_path / "config.json"),
        AF_CASADI_ROOT=str(tmp_path / "deps"),
        AF_ACCOUNT="test",
        AF_CONCURRENCY="2",
    )
    result = subprocess.run(
        ["bash", "scripts/hpc/submit_phase_c_prefix_window_selection_delta.sh"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    invocation = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    argv = invocation["argv"]
    assert argv[argv.index("--root") + 1] == str(tmp_path / "output")
    assert argv[argv.index("--config") + 1] == str(tmp_path / "config.json")
    assert argv[argv.index("--account") + 1] == "test"
    assert argv[argv.index("--concurrency") + 1] == "2"
    assert invocation["env"]["PYTHONPATH"].endswith(str(tmp_path / "deps"))
    assert all(
        invocation["env"][name] == "1"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )


def test_shell_syntax_help_and_prepare_gate():
    stem = "phase_c_prefix_window_selection"
    for stage in ("run", "submit"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{stage}_{stem}_delta.sh"], check=True
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
    assert "tests/test_submit_prefix_window_selection.py" in tests
    assert all(Path(path).is_file() for path in tests)
    assert "pip install" not in worker


def test_config_preserves_declared_windows_starts_and_allowances():
    policy = cli.campaign.Policy.model_validate_json(CONFIG.read_text())
    assert list(policy.root_starts) == [0.5, 1.1]
    assert list(policy.durations) == [0, 0.1, 0.2, 0.5, 1, 2]
    assert policy.training_numerical.seconds == 30
    assert policy.training_numerical.verification_seconds == 10
    assert policy.prefix_numerical.seconds == 10
    assert policy.prefix_numerical.verification_seconds == 5
    assert policy.validation_nmse_limit == policy.start_prediction_spread_limit == 0.01
