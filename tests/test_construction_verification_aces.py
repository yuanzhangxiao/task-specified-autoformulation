"""Independent CPU verification submission and its durable scheduler receipts."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import submit_review_continuation as scheduler
from scripts import verify_construction_parameters as cli
from tests.test_construction_fit_aces import worker_environment

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    plan = {"candidates": [{"task": {"task_id": str(i)}} for i in range(8)]}
    monkeypatch.setattr(cli.campaign, "verify", lambda _: (plan, {}))
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    monkeypatch.setenv("AF_ACCOUNT", "156264627414")
    return tmp_path / "results"


def test_submission_is_cpu_only_and_idempotent(prepared, monkeypatch):
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    value = cli.submit(prepared)
    assert value["jobs"] == {"verify": "101", "report": "102"}
    assert "--array=0-7%4" in commands[0]
    assert "--dependency=afterany:101" in commands[1]
    for command in commands:
        assert "--time=00:15:00" in command
        assert "--cpus-per-task=1" in command
        assert "--partition=cpu" in command
        assert not any("gpu" in part for part in command)
    assert cli.submit(prepared) == value
    assert len(commands) == 2
    assert cli.verify_execution(prepared)["gpus"] == 0
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(prepared)


def test_ambiguous_scheduler_reply_cannot_submit_again(prepared, monkeypatch):
    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="\n", stderr="Socket timed out")

    monkeypatch.setattr(scheduler.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="unconfirmed"):
        cli.submit(prepared)
    with pytest.raises(ValueError, match="uncertain"):
        cli.submit(prepared)
    assert len(calls) == 1
    directory = prepared / "submissions/output-verification-1"
    assert (directory / "verify.reply.json").exists()
    assert not (directory / "report.intent.json").exists()


def test_worker_uses_only_saved_endpoint_cli(tmp_path):
    environment = worker_environment(tmp_path)
    environment["SLURM_ARRAY_TASK_ID"] = "7"
    worker = REPO / "scripts/hpc/run_construction_verification_aces.sh"
    subprocess.run(["bash", "-n", str(worker)], check=True)
    subprocess.run(["bash", str(worker), "verify", "0"], env=environment, check=True)
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert "construction_fit_verification" in args[1]
    assert "sympy" not in args[1]
    assert args[args.index("--index") + 1] == "7"
    assert "verify-execution" in args
    assert str(REPO / "scripts/verify_construction_parameters.py") in args
    assert str(REPO / "scripts/qualify_construction_fitting.py") not in args


def test_wrapper_uses_separate_root_and_frozen_source(tmp_path):
    environment = worker_environment(tmp_path)
    environment.pop("AF_OUTPUT_ROOT")
    wrapper = REPO / "scripts/hpc/start_construction_verification_aces.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", str(wrapper), "run"], env=environment, check=True)
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert str(tmp_path / "phase-c-construction-verification-v1") in args
    assert str(REPO / "inputs/source") in args
    assert "prepare" in args and "submit" in args


def test_actual_verifier_imports_without_sympy(tmp_path):
    import os
    import sys

    probe = """
import importlib.abc
import sys
class NoSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sympy' or fullname.startswith('sympy.'):
            raise ImportError('no sympy')
sys.meta_path.insert(0, NoSympy())
from autoformalism.research import construction_verification_campaign
assert 'sympy' not in sys.modules
"""
    environment = os.environ | {"PYTHONPATH": str(REPO / "src")}
    subprocess.run([sys.executable, "-c", probe], env=environment, check=True)
