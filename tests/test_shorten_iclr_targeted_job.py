"""Safe scheduler-window deployment without replacing scientific checkpoints."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import shorten_iclr_targeted_job as short


def bundle(folder, *, replacement=False):
    values = {
        "SOURCE_COMMIT": ("b" if replacement else "a") * 40 + "\n",
        "scripts/iclr_controlled_search.py": "new CLI" if replacement else "old CLI",
        "src/model.py": "frozen scientific code",
        "configs/study.json": "frozen configuration",
        "public/train.json": "frozen data",
    }
    if replacement:
        values[short.PATCHES[1]] = "new deadline helper"
    for name, value in values.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (folder / "SHA256SUMS").write_text(
        "".join(
            f"{short.digest(value.encode())}  {name}\n"
            for name, value in values.items()
        )
    )


class Scheduler:
    def __init__(self, root, code):
        self.root, self.code = root, code
        self.calls = []
        self.held = False
        self.limit = "12:00:00"
        self.running = False
        self.race = False
        self.bad_owner = False
        self.fail_update = False
        self.ignore_update = False

    def __call__(self, args):
        self.calls.append(args)
        if args[0] == "squeue":
            return "\n".join(
                f"2376062_{i}|{'RUNNING' if self.running and i == 0 else 'PENDING'}"
                for i in range(6)
            )
        if args[1:4] == ["-o", "show", "job"]:
            uid = os.getuid() + int(self.bad_owner)
            return (
                "JobId=2376062 ArrayJobId=2376062 ArrayTaskId=0-5%6 "
                f"UserId=test({uid}) SubmitTime=2026-11-18T01:00:00 "
                "JobName=iclr-targeted-20b Account=156264627414 Partition=gpu "
                f"Command={self.code}/scripts/hpc/run_iclr_controlled_search_aces.sh "
                f"WorkDir={self.code} ExcNodeList=ac[042,096] "
                f"StdOut={self.root}/logs/wave2-%A_%a.out "
                f"StdErr={self.root}/logs/wave2-%A_%a.err JobState=PENDING "
                f"Reason={'JobHeldUser' if self.held else 'Priority'} "
                f"TimeLimit={self.limit}"
            )
        if args[1] == "hold":
            self.held = True
            if self.race:
                self.running = True
        elif args[1] == "update":
            if self.fail_update:
                raise subprocess.TimeoutExpired(args, 45)
            if not self.ignore_update:
                self.limit = short.LIMIT
        elif args[1] == "release":
            self.held = False
        else:
            raise AssertionError(args)
        return ""


@pytest.fixture
def setup(tmp_path):
    root, code, replacement = (tmp_path / name for name in ("results", "old", "new"))
    bundle(code)
    bundle(replacement, replacement=True)
    wave = root / "submissions/wave2"
    wave.mkdir(parents=True)
    (wave / "array.id").write_text("2376062\n")
    argv = [
        "sbatch",
        "--account=156264627414",
        "--partition=gpu",
        "--job-name=iclr-targeted-20b",
        "--exclude=ac042,ac096",
        "--array=0,1,2,3,4,5%6",
        f"--output={root}/logs/wave2-%A_%a.out",
        f"--error={root}/logs/wave2-%A_%a.err",
        str(code / "scripts/hpc/run_iclr_controlled_search_aces.sh"),
    ]
    (wave / "array.intent.json").write_text(json.dumps({"argv": argv}))
    scheduler = Scheduler(root, code)
    return root, code, replacement, scheduler


def invoke(setup, *, apply=True):
    root, code, replacement, scheduler = setup
    return short.update(root, "2376062", code, replacement, apply=apply, run=scheduler)


def test_dry_run_and_apply_preserve_submission_and_scientific_files(setup):
    root, code, replacement, scheduler = setup
    manifest = (code / "SHA256SUMS").read_bytes()
    intent = (root / "submissions/wave2/array.intent.json").read_bytes()
    assert invoke(setup, apply=False)["status"] == "ready"
    assert not scheduler.held and not list(root.rglob("deployment.json"))
    assert (code / short.PATCHES[0]).read_text() == "old CLI"
    result = invoke(setup)
    assert result["status"] == "updated" and result["same_submission"]
    assert not scheduler.held and scheduler.limit == short.LIMIT
    assert (code / "SHA256SUMS").read_bytes() == manifest
    assert (root / "submissions/wave2/array.intent.json").read_bytes() == intent
    assert (code / "src/model.py").read_text() == "frozen scientific code"
    assert all(
        (code / name).read_bytes() == (replacement / name).read_bytes()
        for name in short.PATCHES
    )
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["phase"] == "complete"
    calls = len(scheduler.calls)
    assert invoke(setup)["status"] == "already_applied"
    assert len(scheduler.calls) == calls


@pytest.mark.parametrize("defect", ["running", "owner", "data", "code", "job"])
def test_rejects_before_any_hold_or_patch(setup, defect):
    root, code, replacement, scheduler = setup
    if defect == "running":
        scheduler.running = True
    elif defect == "owner":
        scheduler.bad_owner = True
    elif defect == "data":
        path = replacement / "public/train.json"
        old = path.read_bytes()
        path.write_text("changed data")
        manifest = replacement / "SHA256SUMS"
        manifest.write_text(
            manifest.read_text().replace(
                short.digest(old), short.digest(path.read_bytes())
            )
        )
    elif defect == "code":
        (code / short.PATCHES[0]).write_text("unrecognized edit")
    else:
        (root / "submissions/wave2/array.id").write_text("999")
    with pytest.raises(ValueError):
        invoke(setup)
    assert not scheduler.held
    assert not any(args[1] in ("hold", "update") for args in scheduler.calls)


def test_race_to_running_releases_only_our_hold_and_does_not_patch(setup):
    _, code, _, scheduler = setup
    scheduler.race = True
    with pytest.raises(ValueError, match="started during hold"):
        invoke(setup)
    assert not scheduler.held and scheduler.limit == "12:00:00"
    assert (code / short.PATCHES[0]).read_text() == "old CLI"
    assert not (code / short.PATCHES[1]).exists()


@pytest.mark.parametrize("ignored", [False, True])
def test_unconfirmed_limit_remains_held_and_resume_reconciles(setup, ignored):
    _, code, _, scheduler = setup
    scheduler.fail_update = not ignored
    scheduler.ignore_update = ignored
    with pytest.raises(RuntimeError, match="may remain held"):
        invoke(setup)
    assert scheduler.held
    assert (code / short.PATCHES[0]).read_text() == "new CLI"
    scheduler.fail_update = scheduler.ignore_update = False
    assert invoke(setup)["status"] == "updated"
    assert not scheduler.held and scheduler.limit == short.LIMIT


def test_corrupted_backup_prevents_resume(setup):
    root, _, _, scheduler = setup
    scheduler.fail_update = True
    with pytest.raises(RuntimeError):
        invoke(setup)
    (root / "submissions/wave2/short-allocation-2376062/original-cli.py").write_text(
        "changed"
    )
    with pytest.raises(ValueError, match="saved original"):
        invoke(setup)
    assert scheduler.held


def test_unrelated_user_hold_is_not_released(setup):
    _, _, _, scheduler = setup
    scheduler.held = True
    with pytest.raises(ValueError, match="unrelated user hold"):
        invoke(setup)
    assert scheduler.held
    assert not any(args[1] in ("hold", "release", "update") for args in scheduler.calls)


def test_lost_release_acknowledgment_reconciles_after_job_starts(setup):
    root, code, replacement, scheduler = setup

    def uncertain(args):
        result = scheduler(args)
        if args[1] == "release":
            scheduler.running = True
            raise subprocess.TimeoutExpired(args, 45)
        return result

    with pytest.raises(RuntimeError, match="may remain held"):
        short.update(root, "2376062", code, replacement, apply=True, run=uncertain)
    assert not scheduler.held and scheduler.running
    before = len(scheduler.calls)
    result = invoke(setup)
    assert result["release_reconciled"]
    assert not any(args[1] in ("hold", "update") for args in scheduler.calls[before:])


def test_lost_hold_acknowledgment_is_actionable_and_resumable(setup):
    root, code, replacement, scheduler = setup

    def uncertain(args):
        result = scheduler(args)
        if args[1] == "hold":
            raise subprocess.TimeoutExpired(args, 45)
        return result

    with pytest.raises(RuntimeError, match=r"Hold acknowledgment.*unconfirmed"):
        short.update(root, "2376062", code, replacement, apply=True, run=uncertain)
    assert scheduler.held and scheduler.limit == "12:00:00"
    assert (code / short.PATCHES[0]).read_text() == "old CLI"
    assert invoke(setup)["status"] == "updated"
    assert not scheduler.held


def test_submit_time_change_prevents_release(setup):
    root, code, replacement, scheduler = setup

    def changed(args):
        result = scheduler(args)
        if scheduler.limit == short.LIMIT and args[1:4] == ["-o", "show", "job"]:
            result = result.replace("2026-11-18T01:00:00", "2026-11-18T02:00:00")
        return result

    with pytest.raises(RuntimeError, match="may remain held"):
        short.update(root, "2376062", code, replacement, apply=True, run=changed)
    assert scheduler.held
    assert not any(args[1] == "release" for args in scheduler.calls)


def test_partial_install_and_lost_hold_preserve_phase_and_resume(setup, monkeypatch):
    root, code, replacement, scheduler = setup
    original_atomic = short.atomic

    def partial(path, value):
        if path == code / short.PATCHES[0]:
            raise OSError("simulated CLI installation failure")
        original_atomic(path, value)

    monkeypatch.setattr(short, "atomic", partial)
    with pytest.raises(RuntimeError, match="may remain held"):
        invoke(setup)
    receipt = root / "submissions/wave2/short-allocation-2376062/deployment.json"
    assert json.loads(receipt.read_text())["phase"] == "installing"
    assert (code / short.PATCHES[1]).read_text() == "new deadline helper"
    assert (code / short.PATCHES[0]).read_text() == "old CLI"
    monkeypatch.setattr(short, "atomic", original_atomic)

    def uncertain(args):
        result = scheduler(args)
        if args[1] == "hold":
            raise subprocess.TimeoutExpired(args, 45)
        return result

    with pytest.raises(RuntimeError, match="may remain held"):
        short.update(root, "2376062", code, replacement, apply=True, run=uncertain)
    assert json.loads(receipt.read_text())["phase"] == "installing"
    assert scheduler.held
    assert invoke(setup)["status"] == "updated"
    assert not scheduler.held
