"""Verify the matched 20B GPU allocation, reconciliation and training-only bundle."""

import json
import shlex
import sys
from datetime import UTC, datetime
from types import ModuleType

import pytest

import scripts
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import package_trajectory_functions as packaging
from scripts import submit_t2_breadth as launcher


def fake_trial(monkeypatch):
    trial = ModuleType("scripts.pointwise_repair_trial")
    monkeypatch.setitem(sys.modules, "scripts.pointwise_repair_trial", trial)
    monkeypatch.setattr(scripts, "pointwise_repair_trial", trial, raising=False)
    return trial


def test_lost_reply_and_retry_keep_single_matched_job(tmp_path, monkeypatch):
    root = tmp_path / "results"
    monkeypatch.setenv("AF_REPO_ROOT", str(launcher.campaign.REPO))
    monkeypatch.setenv("AF_OUTPUT_ROOT", str(root))
    trial = fake_trial(monkeypatch)
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": launcher.POINTWISE_REPAIR,
            "entrypoints": {},
            "maximum_generation_requests": 18,
            "settings": {"model": "openai/gpt-oss-20b", "max_output_tokens": 4096},
        },
    )
    trial.load = lambda root: (plan, {})
    status = {"status": "incomplete"}
    trial.status = lambda root: status
    image = tmp_path / "image"
    image.write_text("local GPU image fixture")
    monkeypatch.setattr(launcher.aces, "_assets", lambda size: (image, {"size": size}))
    monkeypatch.setattr(launcher.aces.harmony_vocab, "verify", lambda _: {})
    jobs = []

    def lost(wave, argv):
        jobs.append(argv)
        (wave / "propose.intent.json").write_text(json.dumps({"argv": argv}))
        (wave / "propose.reply.json").write_text(
            json.dumps(
                {"returncode": 0, "stdout": "", "stderr": "sbatch: Socket timed out"}
            )
        )
        raise ValueError("Acknowledgment unconfirmed")

    monkeypatch.setattr(launcher, "_submit_job", lost)

    def accounting(argv):
        if argv[0] == "squeue":
            return ""
        assert argv[0] == "sacct"
        user = launcher.scheduler.getpass.getuser()
        now = datetime.now(UTC).isoformat()
        return f"984|{user}|phasec-witness-20b|RUNNING|{now}|{shlex.join(jobs[0])}"

    monkeypatch.setattr(launcher.scheduler, "_query", accounting)
    result = launcher.submit(root)
    assert result["job"] == "984"
    assert "--gres=gpu:h100:1" in jobs[0] and "--time=00:45:00" in jobs[0]
    assert "--dependency=singleton" in jobs[0]
    assert jobs[0][-1] == "pointwise-repair"
    exe = sealed_read(root / "execution.json")
    assert (exe["model_size"], exe["snapshot"], exe["maximum_generation_requests"]) == (
        "20b",
        {"size": "20b"},
        18,
    )
    assert exe["compiler_cache_policy"] == "node-local-job-temporary-1"
    assert exe["optimizer_calls"] == 0
    assert launcher.submit(root)["job"] == "984"
    monkeypatch.setattr(launcher, "_finished", lambda _: False)
    assert launcher.submit(root, resume=True)["job"] == "984"
    status["status"] = "complete"
    assert launcher.submit(root, resume=True)["status"] == "already_complete"
    assert len(jobs) == 1
    with pytest.raises(ValueError, match="18-call"):
        launcher.execution({**plan, "maximum_generation_requests": 19}, "image", {})


def test_delivery_registers_only_seven_authenticated_source_files(
    tmp_path, monkeypatch
):
    trial = fake_trial(monkeypatch)
    verified = []
    trial.read_source = verified.append
    monkeypatch.setattr(packaging, "package_committed", lambda *args, **kw: (args, kw))
    args, kw = packaging.package(
        tmp_path / "out", tmp_path, tmp_path, pointwise_repair=True
    )
    assert verified == [tmp_path]
    assert args[1] == "phase-c-pointwise-repair-20b-v1"
    tracked = set(args[2])
    assert {
        "scripts/pointwise_repair_trial.py",
        "scripts/pointwise_repair_fitting.py",
        "scripts/trajectory_boundary_trial.py",
        "scripts/hpc/start_pointwise_repair.sh",
        "scripts/hpc/run_meal_insulin_revision_aces.sh",
        "scripts/t2_breadth_scheduler.py",
        "scripts/submit_t2_breadth.py",
        "scripts/trajectory_functions.py",
    } <= tracked
    files = [(args[3], args[4]), *kw["extra_inputs"]]
    assert {name for name, _ in files} == {
        f"inputs/source/{s}"
        for s in (
            "admission.json",
            "plan.json",
            "generation.json",
            "intake.json",
            "candidates/002.json",
            "candidates/014.json",
            "candidates/015.json",
        )
    }
    for name, p in files:
        assert p == tmp_path / name.removeprefix("inputs/source/")


def test_optional_fit_inputs_come_only_from_original_training_campaign(
    tmp_path, monkeypatch
):
    trial = fake_trial(monkeypatch)
    trial.read_source = lambda _: {}
    fitter = ModuleType("scripts.pointwise_repair_fitting")
    checked = []
    fitter.parents_from_archive = checked.append
    fitter.SLOTS = (2, 14, 15)
    monkeypatch.setitem(sys.modules, "scripts.pointwise_repair_fitting", fitter)
    monkeypatch.setattr(
        packaging, "package_committed", lambda *args, **kw: kw["extra_inputs"]
    )
    source = tmp_path / "original"
    files = packaging.package(
        tmp_path / "out",
        tmp_path / "replay",
        tmp_path,
        pointwise_repair=True,
        fit_campaign=source,
    )
    assert checked == [source]
    selected = {n: p for n, p in files if n.startswith("inputs/parents/")}
    assert len(selected) == 10
    for n, p in selected.items():
        assert source / n.removeprefix("inputs/parents/") == p
    assert not any("validation" in n or "/val/" in n or "/test/" in n for n, _ in files)
    with pytest.raises(ValueError, match="Parent fit receipts"):
        packaging.package(tmp_path / "out", tmp_path, tmp_path, fit_campaign=source)
