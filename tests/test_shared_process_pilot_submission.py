"""Scheduler receipt safety and previous-round barriers without cluster access."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_shared_process_pilot as smoke
from scripts import submit_shared_process_pilot as submitter


def setup(tmp_path, monkeypatch):
    root = tmp_path / "run"
    plan = smoke.fixture(root)
    python = tmp_path / "python"
    image = tmp_path / "image.sif"
    python.touch()
    image.touch()
    for key, value in {
        "AF_PYTHON": python,
        "AF_VLLM_IMAGE": image,
        "AF_HF_HOME": tmp_path / "hf",
        "AF_COMPUTE_CACHE_ROOT": tmp_path / "cache",
        "AF_IPC_TMP_ROOT": "/tmp/ipc",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("AF_COMMIT", raising=False)
    monkeypatch.setattr(submitter, "source_commit", lambda repo: "a" * 40)
    return root, plan


def test_one_round_no_duplicate_jobs_or_automatic_followup(tmp_path, monkeypatch):
    root, _ = setup(tmp_path, monkeypatch)
    calls = []

    def fake(directory, key, options, worker, stage, index):
        calls.append((stage, options))
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", fake)
    manifest = submitter.submit(root, 0)
    assert len(calls) == 4
    assert not manifest["automatic_next_round"]
    assert "--gres=gpu:h100:1" in calls[1][1]
    assert "--cpus-per-task=1" in calls[2][1]
    assert submitter.submit(root, 0) == manifest
    assert len(calls) == 4
    with pytest.raises(ValueError, match="incomplete"):
        submitter.submit(root, 1)
    assert len(calls) == 4


def test_partial_or_uncertain_submission_is_not_repeated(tmp_path, monkeypatch):
    root, _ = setup(tmp_path, monkeypatch)

    def uncertain(*args):
        raise ValueError("uncertain response")

    monkeypatch.setattr(submitter, "submit_job", uncertain)
    with pytest.raises(ValueError, match="uncertain response"):
        submitter.submit(root, 0)
    with pytest.raises(ValueError, match="submission exists"):
        submitter.submit(root, 0)


def test_round_one_requires_all_sealed_predecessors(tmp_path, monkeypatch):
    root, plan = setup(tmp_path, monkeypatch)
    for task in plan["tasks"]:
        sealed_write(
            root / "results" / task["task_id"] / "round_00/result.json",
            {
                "task": task,
                "round": 0,
                "status": "construction_failed",
                "selected": None,
            },
        )
    monkeypatch.setattr(submitter, "submit_job", lambda *args: "123")
    result = submitter.submit(root, 1)
    assert result["round"] == 1
    assert json.loads(
        (root / "submission-intent/round-1/prerequisites.json").read_text()
    )["results"]


def test_archive_marker_requires_full_hash(tmp_path):
    (tmp_path / "SOURCE_COMMIT").write_text("abc")
    with pytest.raises(ValueError, match="full commit"):
        submitter.source_commit(tmp_path)
    (tmp_path / "SOURCE_COMMIT").write_text("a" * 40 + "\n")
    assert submitter.source_commit(tmp_path) == "a" * 40
