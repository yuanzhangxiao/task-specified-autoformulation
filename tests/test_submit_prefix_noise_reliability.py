"""M47 CPU launchers, self-contained plans and fail-closed scheduler receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import phase_c_prefix_noise_reliability as cli
from scripts import submit_phase_c_prefix_noise_reliability as submitter

CONFIG = Path("configs/phase_c_prefix_noise_reliability_v1.json")
STEM = "phase_c_prefix_noise_reliability"
TASKS = [
    f"{mode}_{noise}_r{repeat}_s{seed}"
    for mode in ("separated", "near_confounded", "exact_confounded")
    for noise, repeats in (("clean", [99]), ("noisy", range(100, 105)))
    for repeat in repeats
    for seed in (0, 1)
]


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Use real scheduler receipts with a frozen-plan stand-in and fake Slurm."""
    root = tmp_path / "campaign"
    plan = {"tasks": [{"task_id": name} for name in TASKS]}
    prepared = []
    verified = []

    def prepare(output, policy):
        assert isinstance(policy, cli.campaign.Policy)
        prepared.append((output, policy))
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
        root=root, plan=plan, prepared=prepared, verified=verified, calls=calls
    )


def test_cpu_only_submission_is_idempotent_and_uses_frozen_roster(scheduler):
    s = scheduler
    first = submitter.submit(s.root, CONFIG, account="test")
    assert first["array_tasks"] == len(TASKS)
    assert first["gpus"] == first["live_llm_calls"] == 0
    assert first["test_data_opened"] is False
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh")
    assert first["identity"]["worker_sha256"] == hashlib.sha256(
        worker.read_bytes()
    ).hexdigest()
    assert len(s.prepared) == len(s.verified) == 1
    assert f"--array=0-{len(TASKS) - 1}%2" in s.calls[1]
    assert "--dependency=afterok:101" in s.calls[1]
    assert "--dependency=afterany:101:102" in s.calls[2]
    for argv, duration in zip(
        s.calls, ("00:30:00", "00:45:00", "00:05:00"), strict=True
    ):
        assert "--cpus-per-task=1" in argv and "--mem=16G" in argv
        assert "--partition=cpu" in argv and f"--time={duration}" in argv
        assert f"--job-name=prefix-noise-reliability-{argv[-1]}" in argv
        assert not any("--gres" in item or "--gpus" in item for item in argv)
        assert argv[-2].endswith(f"run_{STEM}_delta.sh")
    assert submitter.submit(s.root, CONFIG, account="test") == first
    assert len(s.calls) == 3


@pytest.mark.parametrize("concurrency", [1, 8])
def test_supported_concurrency_boundaries(scheduler, concurrency):
    submitter.submit(scheduler.root, CONFIG, concurrency=concurrency)
    assert f"--array=0-{len(TASKS) - 1}%{concurrency}" in scheduler.calls[1]


def test_progress_precedes_preparation(scheduler, monkeypatch, capsys):
    s = scheduler
    prepare = submitter.campaign.prepare

    def observed(*args):
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "preparing frozen tasks" in captured.err
        return prepare(*args)

    monkeypatch.setattr(submitter.campaign, "prepare", observed)
    submitter.submit(s.root, CONFIG)
    captured = capsys.readouterr()
    assert captured.out == "" and f"{len(TASKS)} frozen tasks" in captured.err


@pytest.mark.parametrize("concurrency", [0, 9])
def test_invalid_concurrency_fails_before_preparation(
    scheduler, monkeypatch, concurrency
):
    s = scheduler
    monkeypatch.setattr(
        submitter.campaign, "prepare",
        lambda *_: pytest.fail("prepared inputs before scheduler preflight"),
    )
    with pytest.raises(ValueError, match=r"concurrency 1\.\.8"):
        submitter.submit(s.root, CONFIG, concurrency=concurrency)
    assert s.calls == []


@pytest.mark.parametrize("python", [None, "/no/such/python"])
def test_missing_python_fails_before_preparation(scheduler, monkeypatch, python):
    s = scheduler
    if python is None:
        monkeypatch.delenv("AF_PYTHON")
    else:
        monkeypatch.setenv("AF_PYTHON", python)
    monkeypatch.setattr(
        submitter.campaign, "prepare",
        lambda *_: pytest.fail("prepared inputs before Python preflight"),
    )
    with pytest.raises(ValueError, match="existing Python"):
        submitter.submit(s.root, CONFIG)
    assert s.calls == []


@pytest.mark.parametrize("change", [{"account": "other"}, {"concurrency": 3}])
def test_changed_submission_identity_is_rejected(scheduler, change):
    s = scheduler
    submitter.submit(s.root, CONFIG, account="test")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(s.root, CONFIG, **({"account": "test"} | change))
    assert len(s.calls) == 3


def test_changed_task_roster_is_rejected(scheduler):
    s = scheduler
    submitter.submit(s.root, CONFIG)
    s.plan["tasks"].pop()
    with pytest.raises(ValueError, match="sealed artifact differs"):
        submitter.submit(s.root, CONFIG)
    assert len(s.calls) == 3


@pytest.mark.parametrize(
    "stage_index,stage", [(0, "prepare"), (1, "run"), (2, "report")]
)
@pytest.mark.parametrize(
    "failure", ["timeout", "empty_reply", "nonzero", "malformed_reply"]
)
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
                returncode=int(failure == "nonzero"),
                stdout="Submitted 100" if failure == "malformed_reply" else "",
                stderr="unconfirmed",
            )
        return SimpleNamespace(
            returncode=0, stdout=str(200 + len(s.calls)), stderr=""
        )

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", uncertain)
    with pytest.raises(ValueError, match=r"unknown|unconfirmed"):
        submitter.submit(s.root, CONFIG)
    assert len(s.calls) == stage_index + 1
    assert (s.root / "submission" / f"{stage}.intent.json").is_file()
    assert (s.root / "submission" / f"{stage}.reply.json").is_file()
    assert not (s.root / "submission" / f"{stage}.id").exists()
    with pytest.raises(ValueError, match=f"Unconfirmed {stage}"):
        submitter.submit(s.root, CONFIG)
    assert len(s.calls) == stage_index + 1


def test_failed_verification_submits_nothing(scheduler, monkeypatch):
    def fail(_):
        raise ValueError("snapshot identity differs")

    monkeypatch.setattr(submitter.campaign, "verify", fail)
    with pytest.raises(ValueError, match="snapshot identity differs"):
        submitter.submit(scheduler.root, CONFIG)
    assert scheduler.calls == []


def test_empty_roster_submits_nothing(scheduler):
    s = scheduler
    s.plan["tasks"] = []
    result = submitter.submit(s.root, CONFIG)
    assert result["status"] == "no_tasks"
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
        (tmp_path, tmp_path, policy.model_copy(update={"protocol": "changed"})),
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
        argv += ["--config", str(CONFIG)]
    elif command == "run":
        argv += ["--index", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert observed[0][0] == tmp_path and f"M47: {command}" in captured.err
    if command == "prepare":
        assert len(observed[0]) == 2
        assert isinstance(observed[0][1], cli.campaign.Policy)
    elif command == "run":
        assert observed[0][1] == 1 and "rows" not in receipt
        assert receipt["artifact"].endswith(f"/results/{TASKS[1]}/result.json")
    else:
        assert receipt["verified"] and receipt["tasks"] == len(TASKS)


def test_real_prepare_and_verify_receipts(tmp_path, monkeypatch, capsys):
    """Exercise preparation's scalar identity and verify its durable plan."""
    monkeypatch.setattr(
        sys, "argv",
        ["cli", "prepare", "--root", str(tmp_path), "--config", str(CONFIG)],
    )
    cli.main()
    prepared = json.loads(capsys.readouterr().out)
    assert re.fullmatch(r"[0-9a-f]{64}", prepared["identity"])
    assert prepared["tasks"] == len(TASKS)
    assert prepared["artifact"] == str(tmp_path / "plan.json")
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "inputs.json").is_file()

    monkeypatch.setattr(sys, "argv", ["cli", "verify", "--root", str(tmp_path)])
    cli.main()
    verified = json.loads(capsys.readouterr().out)
    assert verified == {
        "verified": True, "tasks": len(TASKS), "artifact": str(tmp_path / "plan.json")
    }


def test_report_receipt_preserves_full_artifact(tmp_path, monkeypatch, capsys):
    full = {
        "status": "complete",
        "expected": len(TASKS),
        "recorded": len(TASKS),
        "status_counts": {"complete": len(TASKS)},
        "policies": ["baseline", "noise_aware"],
        "selection_counts": {
            "baseline": {"selected": 36},
            "noise_aware": {"selected": 30, "no_adequate_duration": 6},
        },
        "groups": [
            {
                "scenario": "near_confounded",
                "noise_sigma": 0.01,
                "policy": "noise_aware",
                "independent_acquisitions": 5,
                "training_starts": 2,
                "cost": {"wall_seconds": 2, "operations": [1] * 1000},
                "rows": [1] * 1000,
            }
        ],
        "cost": {"wall_seconds": 12.3, "operations": [1] * 1000},
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
    assert receipt["cost"] == {"wall_seconds": 12.3}
    assert receipt["policies"] == full["policies"]
    assert receipt["selection_counts"] == full["selection_counts"]
    assert receipt["groups"] == [
        {
            key: value
            for key, value in full["groups"][0].items()
            if key not in ("rows", "cost")
        } | {"cost": {"wall_seconds": 2}}
    ]
    assert receipt["artifact"] == str(tmp_path / "summary.json")
    assert json.loads((tmp_path / "summary.json").read_text()) == full
    assert len(captured.out) < 2000


@pytest.mark.parametrize("no_tasks", [False, True])
def test_submission_cli_prints_small_receipt(
    tmp_path, monkeypatch, capsys, no_tasks
):
    monkeypatch.setattr(
        submitter, "submit",
        lambda **kwargs: {
            "status": "no_tasks" if no_tasks else "submitted",
            "identity": {"details": list(range(1000))},
            "jobs": {} if no_tasks else {"run": "42"},
            "array_tasks": 0 if no_tasks else len(TASKS),
            "gpus": 0,
        },
    )
    monkeypatch.setattr(sys, "argv", ["submit", "--root", str(tmp_path)])
    submitter.main()
    receipt = json.loads(capsys.readouterr().out)
    assert "identity" not in receipt
    artifact = "plan.json" if no_tasks else "submission_manifest.json"
    assert receipt["artifact"] == str(tmp_path / artifact)


@pytest.fixture
def shell_environment(tmp_path):
    """Capture launcher invocations without importing dependencies or submitting."""
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if '-c' not in sys.argv:\n"
        "    print(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: os.environ[k] for k in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS', "
        "'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}))\n"
    )
    fake_python.chmod(0o755)
    config = tmp_path / "config.json"
    config.write_text("{}")
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("AF_")
    }
    environment.update(
        AF_PYTHON=str(fake_python),
        AF_PREFIX_NOISE_ROOT=str(tmp_path / "output"),
        AF_PREFIX_NOISE_CONFIG=str(config),
        AF_CASADI_ROOT=str(tmp_path / "deps"),
        AF_ACCOUNT="test",
        AF_CONCURRENCY="2",
    )
    return environment


def test_delta_shell_preflight_and_overrides(tmp_path, shell_environment):
    result = subprocess.run(
        ["bash", f"scripts/hpc/submit_{STEM}_delta.sh"],
        env=shell_environment, check=True, capture_output=True, text=True,
    )
    invocation = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    argv = invocation["argv"]
    for option, value in (
        ("--root", tmp_path / "output"),
        ("--config", tmp_path / "config.json"),
        ("--account", "test"),
        ("--concurrency", "2"),
    ):
        assert argv[argv.index(option) + 1] == str(value)
    assert "--source" not in argv and "--inputs" not in argv
    assert "checking code integrity" in result.stdout
    assert "checking numerical imports" in result.stdout
    assert invocation["env"]["PYTHONPATH"].endswith(str(tmp_path / "deps"))
    assert all(
        invocation["env"][name] == "1"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )


@pytest.mark.parametrize("missing", ["config.json", "python"])
def test_delta_shell_missing_prerequisites_fail_early(
    tmp_path, shell_environment, missing
):
    (tmp_path / missing).unlink()
    failed = subprocess.run(
        ["bash", f"scripts/hpc/submit_{STEM}_delta.sh"],
        env=shell_environment, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "checking numerical imports" not in failed.stdout


@pytest.mark.parametrize("stage", ["prepare", "run", "report"])
def test_worker_routes_stage_and_array_index(tmp_path, shell_environment, stage):
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh").resolve()
    (tmp_path / "SOURCE_COMMIT").write_text("test-commit\n")
    shell_environment.update(
        AF_REPO_ROOT=str(tmp_path),
        AF_OUTPUT_ROOT=str(tmp_path / "output"),
        AF_COMMIT="test-commit",
        SLURM_ARRAY_TASK_ID="3",
    )
    result = subprocess.run(
        ["bash", str(worker), stage], env=shell_environment,
        check=True, capture_output=True, text=True,
    )
    invocations = [json.loads(line)["argv"] for line in result.stdout.splitlines()]
    assert ("verify" if stage == "prepare" else stage) in invocations[0]
    assert invocations[0][-1] == ("3" if stage == "run" else str(tmp_path / "output"))
    if stage == "prepare":
        assert len(invocations) == 2 and "pytest" in invocations[1]
    else:
        assert len(invocations) == 1


def test_worker_rejects_changed_code_identity(tmp_path, shell_environment):
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh").resolve()
    (tmp_path / "SOURCE_COMMIT").write_text("changed-commit\n")
    shell_environment.update(
        AF_REPO_ROOT=str(tmp_path),
        AF_OUTPUT_ROOT=str(tmp_path / "output"),
        AF_COMMIT="test-commit",
    )
    failed = subprocess.run(
        ["bash", str(worker), "report"], env=shell_environment,
        capture_output=True, text=True,
    )
    assert failed.returncode != 0 and failed.stdout == ""


def test_worker_rejects_changed_source_checksum(tmp_path, shell_environment):
    worker = Path(f"scripts/hpc/run_{STEM}_delta.sh").resolve()
    (tmp_path / "SOURCE_COMMIT").write_text("test-commit\n")
    (tmp_path / "source.py").write_text("changed source")
    digest = hashlib.sha256(b"submitted source").hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  source.py\n")
    shell_environment.update(
        AF_REPO_ROOT=str(tmp_path),
        AF_OUTPUT_ROOT=str(tmp_path / "output"),
        AF_COMMIT="test-commit",
    )
    failed = subprocess.run(
        ["bash", str(worker), "report"], env=shell_environment,
        capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert not any(line.startswith("{") for line in failed.stdout.splitlines())


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
    assert {
        "tests/test_prefix_initialization.py",
        "tests/test_prefix_execution.py",
        "tests/test_prefix_multiroot_controls.py",
        "tests/test_prefix_multiroot_campaign.py",
        "tests/test_prefix_portable_comparison.py",
        "tests/test_prefix_predictive_support.py",
        "tests/test_prefix_noise_reliability.py",
        "tests/test_submit_prefix_noise_reliability.py",
    } <= tests
    assert all(Path(path).is_file() for path in tests)
    assert "pip install" not in worker
