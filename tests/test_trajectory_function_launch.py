"""Exercise Slurm dependencies, resumability and train/validation packaging."""

import json
import sys
from types import ModuleType

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import package_trajectory_functions as delivery
from scripts import submit_trajectory_functions as launch


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    cli = ModuleType("scripts.trajectory_functions")
    status = {"status": "incomplete", "generation_complete": False}
    plan = {
        "protocol": launch.gpu.TRAJECTORY_FUNCTIONS,
        "maximum_fit_slots": 36,
        "maximum_generation_requests": 24,
        "artifact_sha256": "frozen",
    }
    cli.status = lambda root: status
    cli.load = lambda root: (plan, {})
    monkeypatch.setitem(sys.modules, "scripts.trajectory_functions", cli)
    import scripts

    monkeypatch.setattr(scripts, "trajectory_functions", cli, raising=False)
    root = tmp_path / "results"
    root.mkdir()
    jobs = []

    def queue(directory, argv):
        jobs.append(argv)
        return str(200 + len(jobs))

    monkeypatch.setattr(launch.gpu, "submit", lambda *args, **kw: {"job": "100"})
    monkeypatch.setattr(launch, "_submit_job", queue)
    return root, status, jobs


def test_chain_fits_then_validates_with_distinct_resources(campaign, monkeypatch):
    root, status, jobs = campaign
    result = launch.submit(root)
    assert result["jobs"] == {"generate": "100", "fit": "201", "validate": "202"}
    fit, validate = jobs
    assert "--array=0-35%8" in fit and "--dependency=afterok:100" in fit
    assert "--dependency=afterany:201" in validate
    assert not any(s.startswith("--array") for s in validate)
    for command in jobs:
        assert "--partition=cpu" in command
        assert "--kill-on-invalid-dep=yes" in command
        assert all("--gres=" not in s for s in command)
    assert launch.submit(root) == result
    monkeypatch.setattr(launch.gpu, "_finished", lambda _: False)
    assert launch.submit(root, resume=True) == result
    assert len(jobs) == 2
    monkeypatch.setattr(launch.gpu, "_finished", lambda _: True)
    monkeypatch.setattr(
        launch.gpu, "submit", lambda *args, **kw: {"status": "already_complete"}
    )
    rerun = launch.submit(root, resume=True)
    assert rerun["jobs"] == {"generate": None, "fit": "203", "validate": "204"}
    assert not any(s.startswith("--dependency=") for s in jobs[2])
    status.update(status="complete", generation_complete=True)
    assert launch.submit(root)["status"] == "complete"
    assert len(jobs) == 4


def test_lost_ack_adopts_same_array_job(campaign, monkeypatch):
    root, _, jobs = campaign

    def lost(directory, argv):
        jobs.append(argv)
        (directory / "propose.intent.json").write_text(json.dumps({"argv": argv}))
        raise ValueError("Socket timed out")

    def accounting(argv):
        active = jobs[-1]
        tag = launch.scheduler._value(active, "comment")
        name = launch.scheduler._value(active, "job-name")
        return (
            ""
            if argv[0] == "squeue"
            else "\n".join(f"401_{i}|{tag}|{name}" for i in range(36))
            if active[-1] == "fit"
            else f"402|{tag}|{name}"
        )

    monkeypatch.setattr(launch, "_submit_job", lost)
    monkeypatch.setattr(launch.scheduler, "_query", accounting)
    assert launch.submit(root)["jobs"] == {
        "generate": "100",
        "fit": "401",
        "validate": "402",
    }
    assert len(jobs) == 2
    assert launch.submit(root)["jobs"]["fit"] == "401"
    assert len(jobs) == 2


def test_missing_ack_preserves_partial_cpu_chain(campaign, monkeypatch):
    root, _, jobs = campaign

    def uncertain(directory, argv):
        jobs.append(argv)
        (directory / "propose.intent.json").write_text(json.dumps({"argv": argv}))
        raise ValueError("unknown")

    monkeypatch.setattr(launch, "_submit_job", uncertain)
    monkeypatch.setattr(launch.scheduler, "_query", lambda command: "")
    with pytest.raises(ValueError, match="Awaiting accounting"):
        launch.submit(root)
    with pytest.raises(ValueError, match="Awaiting accounting"):
        launch.submit(root, resume=True)
    assert len(jobs) == 1


def test_resumed_generation_gets_new_cpu_dependencies(campaign, monkeypatch):
    root, _, jobs = campaign
    assert launch.submit(root)["jobs"]["generate"] == "100"
    monkeypatch.setattr(launch.gpu, "submit", lambda *args, **kw: {"job": "110"})
    assert launch.submit(root, resume=True)["jobs"] == {
        "generate": "110",
        "fit": "203",
        "validate": "204",
    }
    assert "--dependency=afterok:110" in jobs[2]


def test_gpu_declaration_uses_one_h100_and_bounded_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(launch.gpu.aces.harmony_vocab, "verify", lambda _: {})
    model = launch.gpu.aces.MODELS["20b"]
    settings = {"model": model["model"], "max_output_tokens": 4096}
    plan = {
        "artifact_sha256": "plan",
        "settings": settings,
        "entrypoints": {},
        "maximum_generation_requests": 24,
        "protocol": launch.gpu.TRAJECTORY_FUNCTIONS,
    }
    record = launch.gpu.execution(plan, "image", {})
    assert record["model_revision"] == model["revision"]
    assert record["tensor_parallel_size"] == 1
    assert record["maximum_generation_requests"] == 24
    assert record["scheduler_walltime_seconds"] == 3600


def test_package_requires_matching_separate_validation(tmp_path, monkeypatch):
    train = sealed_write(tmp_path / "train.json", {"validation_files": {"one": "sha"}})
    valid = {
        "split": "val",
        "training_intake_sha256": train["artifact_sha256"],
        "source_files": train["validation_files"],
    }
    out = tmp_path / "out"
    sealed_write(tmp_path / "validation.json", valid)
    monkeypatch.setattr(delivery, "package_committed", lambda *a, **k: (a, k))
    args, kwargs = delivery.package(out, tmp_path, tmp_path)
    assert {
        "src",
        "scripts/trajectory_functions.py",
        "scripts/replay_trajectory_function_pointwise.py",
        "scripts/replay_trajectory_function_revisions.py",
    }.issubset(args[2])
    assert args[3] == "inputs/train.json"
    assert kwargs["extra_inputs"] == (
        ("inputs/validation.json", tmp_path / "validation.json"),
    )
    (tmp_path / "validation.json").unlink()
    sealed_write(
        tmp_path / "validation.json", valid | {"training_intake_sha256": "other"}
    )
    with pytest.raises(ValueError, match="Validation packet does not match"):
        delivery.package(out, tmp_path, tmp_path)
