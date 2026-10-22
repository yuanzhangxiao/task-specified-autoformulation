"""Frozen parent/child CPU launchers preserve identities and ambiguous receipts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting.public_fitting import content_sha256
from scripts import compare_construction_fitting as cli
from scripts import submit_review_continuation as scheduler

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, code = tmp_path / "results", tmp_path / "bundle"
    plan = {
        "tasks": [{"index": i, "task_id": f"model-{i}"} for i in range(24)],
        "source_sha256": "frozen-source",
    }
    campaign = SimpleNamespace(verify=lambda *a, **k: (plan, {}))
    monkeypatch.setattr(cli, "_campaign", lambda: campaign)
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    for name in cli.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(cli, "REPO", code)
    monkeypatch.delenv("AF_PLATFORM", raising=False)
    monkeypatch.delenv("AF_ACCOUNT", raising=False)
    return root, code, plan, campaign


def accepting(monkeypatch):
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    return commands


def test_twenty_four_cpu_models_eight_parallel_afterany_and_idempotent(
    prepared, monkeypatch
):
    root, _, plan, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/historical/unrelated")
    result = cli.submit(root)
    assert result["jobs"] == {"run": "101", "report": "102"}
    assert result["tasks"] == 24 and result["maximum_parallel_tasks"] == 8
    assert result["gpus"] == result["llm_calls"] == 0
    assert result["test_data_opened"] is False
    assert result["automatic_followup"] is False
    assert "--array=0-23%8" in commands[0]
    assert "--time=03:00:00" in commands[0]
    assert "--time=00:15:00" in commands[1]
    assert "--dependency=afterany:101" in commands[1]
    for command in commands:
        assert "--account=156264627414" in command
        assert "--partition=cpu" in command
        assert "--cpus-per-task=1" in command
        assert "--mem=8G" in command
        assert "--exclude=ac042" in command
        assert not any("gpu" in value for value in command)
    assert os.environ["AF_PAIRED_FIT_ROOT"] == str(root)
    assert os.environ["AF_OUTPUT_ROOT"] == "/historical/unrelated"
    assert cli.submit(root) == result and len(commands) == 2
    identity = cli.verify_execution(root)
    assert identity["source_sha256"] == "frozen-source"
    assert identity["plan_sha256"] == content_sha256(plan)


def test_delta_uses_its_cpu_account_and_no_aces_exclusion(prepared, monkeypatch):
    root, _, _, _ = prepared
    monkeypatch.setenv("AF_PLATFORM", "delta")
    commands = accepting(monkeypatch)
    cli.submit(root)
    assert "--account=bibo-delta-cpu" in commands[0]
    assert not any("--exclude" in part for part in commands[0])
    monkeypatch.setenv("AF_PLATFORM", "aces")
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)


@pytest.mark.parametrize("count", [0, 23, 25])
def test_wrong_task_count_never_contacts_scheduler(prepared, monkeypatch, count):
    root, _, plan, _ = prepared
    plan["tasks"] = [{"index": i} for i in range(count)]
    commands = accepting(monkeypatch)
    with pytest.raises(ValueError, match="requires 24"):
        cli.submit(root)
    assert not commands


@pytest.mark.parametrize("failure", ["blank", "nonzero", "timeout"])
def test_uncertain_reply_never_renews_submission(prepared, monkeypatch, failure):
    root, _, _, _ = prepared
    commands = []

    def fail(argv, **kwargs):
        commands.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(
            returncode=int(failure == "nonzero"), stdout="", stderr="uncertain"
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fail)
    with pytest.raises(ValueError, match=r"uncertain|unconfirmed"):
        cli.submit(root)
    with pytest.raises(ValueError, match="uncertain"):
        cli.submit(root)
    assert len(commands) == 1
    assert (root / cli.SUBMISSIONS / "run.reply.json").exists()
    assert not (root / cli.SUBMISSIONS / "report.intent.json").exists()


def test_confirmed_array_is_not_resubmitted_after_uncertain_report(
    prepared, monkeypatch
):
    root, _, _, _ = prepared
    commands = []

    def reply(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout="103\n" if len(commands) == 1 else "", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", reply)
    with pytest.raises(ValueError, match="unconfirmed"):
        cli.submit(root)
    with pytest.raises(ValueError, match="uncertain report"):
        cli.submit(root)
    assert len(commands) == 2


@pytest.mark.parametrize("change", ["source", "entrypoint", "runtime", "plan"])
def test_changed_execution_cannot_submit_or_run(prepared, monkeypatch, change):
    root, code, plan, _ = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    if change == "source":
        plan["source_sha256"] = "different-source"
    elif change == "entrypoint":
        (code / cli.ENTRYPOINTS[1]).write_text("changed worker\n")
    elif change == "plan":
        plan["tasks"][0]["task_id"] = "a-different-model"
    else:
        monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)
    with pytest.raises(ValueError):
        cli.submit(root)
    assert len(commands) == 2


def test_saved_manifest_does_not_bypass_confirmed_receipts(prepared, monkeypatch):
    root, _, _, _ = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    receipt = root / cli.SUBMISSIONS / "run.reply.json"
    receipt.write_text(json.dumps({"returncode": 0, "stdout": "999\n"}))
    with pytest.raises(ValueError, match="unconfirmed run"):
        cli.submit(root)
    assert len(commands) == 2


def test_run_requires_execution_verification_before_numerical_work(
    prepared, monkeypatch
):
    root, _, _, campaign = prepared
    calls = []

    def rejected(root):
        calls.append("verify")
        raise ValueError("launcher identity differs")

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(cli, "verify_execution", rejected)
    campaign.run_task = lambda *a: pytest.fail("numerical work before verification")
    with pytest.raises(ValueError, match="identity differs"):
        cli.main(["run", "--root", str(root), "--index", "1"])
    assert calls == ["verify"]


def test_run_refuses_login_shell_before_numerical_work(prepared, monkeypatch):
    root, _, _, campaign = prepared
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    campaign.run_task = lambda *a: pytest.fail("no login-node numerical work")
    with pytest.raises(ValueError, match="compute allocation"):
        cli.main(["run", "--root", str(root), "--index", "1"])


def test_prepare_reads_sealed_portable_inputs_without_submitting(
    prepared, tmp_path, monkeypatch, capsys
):
    root, _, plan, campaign = prepared
    inputs = tmp_path / "inputs.json"
    config = tmp_path / "config.json"
    seal(inputs, {"models": "frozen-portable-inputs"})
    config.write_text('{"seed": 0}')
    calls = []
    campaign.Policy = SimpleNamespace(model_validate_json=json.loads)
    campaign.prepare = lambda *a: calls.append(a) or plan
    monkeypatch.setattr(cli, "submit", lambda _: pytest.fail("no scheduler in prepare"))
    cli.main([
        "prepare", "--root", str(root), "--inputs", str(inputs),
        "--config", str(config),
    ])
    assert calls == [(root, {"models": "frozen-portable-inputs"}, {"seed": 0})]
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == 24 and "models" not in result


def test_runtime_binds_sympy_version_and_selected_overlay(monkeypatch):
    monkeypatch.setenv("AF_FIT_RUNTIME", "native")
    monkeypatch.setenv("AF_CASADI_ROOT", "/explicit/deps")
    monkeypatch.setattr(cli, "_runtime_identity", lambda: {"packages": {}})
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "1.14.0")
    result = cli.runtime_identity()
    assert result["packages"]["sympy"] == "1.14.0"
    assert result["casadi_root"] == "/explicit/deps"
    monkeypatch.setenv("AF_FIT_RUNTIME", "container")
    with pytest.raises(ValueError, match="native"):
        cli.runtime_identity()


def shell_environment(tmp_path):
    """Exercise dispatch without scientific work or contacting the real scheduler."""
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    python.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    squeue.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    env.update(
        AF_REPO_ROOT=str(REPO),
        AF_PAIRED_FIT_ROOT=str(tmp_path / "paired-results"),
        AF_OUTPUT_ROOT=str(tmp_path / "stale-unrelated"),
        AF_PYTHON=str(python),
        AF_ARGUMENT_LOG=str(tmp_path / "args.txt"),
        PATH=f"{tmp_path}:{env['PATH']}",
    )
    return env


def test_shell_resolves_bundle_from_any_directory_and_preserves_historical_root(
    tmp_path,
):
    env = shell_environment(tmp_path)
    launcher = REPO / "scripts/hpc/start_construction_paired_fit.sh"
    worker = REPO / "scripts/hpc/run_construction_paired_fit.sh"
    for path in (launcher, worker):
        subprocess.run(["bash", "-n", str(path)], check=True)
    subprocess.run(["bash", str(launcher), "run"], cwd=tmp_path, env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert "prepare" in args and "submit" in args
    assert str(REPO / "inputs/paired-inputs.json") in args
    assert str(REPO / "configs/construction_paired_fit_v1.json") in args
    assert env["AF_PAIRED_FIT_ROOT"] in args
    assert env["AF_OUTPUT_ROOT"] not in args
    assert args.count("-c") == 2 and "sympy" in args[1]
    assert not {"run_task", "--index"} & set(args)


@pytest.mark.parametrize("command", ["status", "report", "inspect"])
def test_readonly_shell_commands_do_not_prepare_submit_or_fit(tmp_path, command):
    env = shell_environment(tmp_path)
    launcher = REPO / "scripts/hpc/start_construction_paired_fit.sh"
    subprocess.run(["bash", str(launcher), command], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert command in args
    assert not {"run", "prepare", "submit", "--index"} & set(args)


def test_worker_array_index_and_report_placeholder(tmp_path):
    env = shell_environment(tmp_path) | {
        "SLURM_ARRAY_TASK_ID": "23", "SLURM_JOB_ID": "100", "AF_PLATFORM": "aces",
    }
    worker = REPO / "scripts/hpc/run_construction_paired_fit.sh"
    subprocess.run(["bash", str(worker), "run", "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--index") + 1] == "23"
    (tmp_path / "args.txt").unlink()
    subprocess.run(["bash", str(worker), "report", "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert "report" in args and "0" not in args and "--index" not in args


def test_worker_rejects_run_without_compute_allocation(tmp_path):
    env = shell_environment(tmp_path)
    env.pop("SLURM_JOB_ID", None)
    worker = REPO / "scripts/hpc/run_construction_paired_fit.sh"
    result = subprocess.run(
        ["bash", str(worker), "run", "0"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "compute allocation" in result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert "--index" not in args


def test_aces_preflight_loads_compiler_before_python_when_required(tmp_path):
    env = shell_environment(tmp_path)
    python = Path(env["AF_PYTHON"])
    python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n'
        'if [[ "$1" == -c && ! -f "$AF_PROBE_MARKER" ]]; then\n'
        '  touch "$AF_PROBE_MARKER"; exit 1\nfi\n'
    )
    env.update(
        AF_PROBE_MARKER=str(tmp_path / "probe-failed"),
        AF_MODULE_LOG=str(tmp_path / "modules.txt"),
        AF_PLATFORM="aces",
    )
    worker = REPO / "scripts/hpc/run_construction_paired_fit.sh"
    command = (
        'module() { printf "%s\\n" "$*" >> "$AF_MODULE_LOG"; }; '
        'export -f module; bash "$1" status'
    )
    subprocess.run(["bash", "-c", command, "bash", str(worker)], env=env, check=True)
    assert (tmp_path / "modules.txt").read_text().strip() == (
        "load GCCcore/13.2.0 Python/3.11.5"
    )
    assert (tmp_path / "args.txt").read_text().splitlines().count("-c") == 2


@pytest.mark.parametrize("index", [-1, 24])
def test_run_rejects_unplanned_indices_before_verification(
    prepared, monkeypatch, index
):
    root, _, _, campaign = prepared
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setattr(
        cli, "verify_execution", lambda *a: pytest.fail("invalid task")
    )
    campaign.run_task = lambda *a: pytest.fail("invalid task")
    with pytest.raises(ValueError, match="between 0 and 23"):
        cli.main(["run", "--root", str(root), "--index", str(index)])


def test_status_exposes_counts_without_model_packets(prepared):
    root, _, _, campaign = prepared
    campaign.report = lambda _: {
        "status": "incomplete", "expected": 24, "recorded": 4,
        "numerically_verified": 3, "validation_available": 2,
        "rows": [{"secretly_large_model_packet": "not for stdout"}],
    }
    result = cli.status(root)
    assert result["expected"] == 24 and result["recorded"] == 4
    assert result["numerically_verified"] == 3
    assert result["validation_available"] == 2 and "rows" not in result
    assert result["jobs"] == {}


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("aces", "/scratch/group/p.nairr260351.000/u.yx126462/phase-c-paired-fit-v1"),
        ("delta", "/work/hdd/bibo/yxiao2/phase_c/construction-paired-fit-v1"),
    ],
)
def test_platform_default_output_root_is_distinct(tmp_path, platform, expected):
    env = shell_environment(tmp_path)
    env.pop("AF_PAIRED_FIT_ROOT")
    env["AF_PLATFORM"] = platform
    launcher = REPO / "scripts/hpc/start_construction_paired_fit.sh"
    subprocess.run(["bash", str(launcher), "report"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--root") + 1] == expected
