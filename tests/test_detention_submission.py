"""Scheduler receipts: no duplicate arrays after success or uncertain replies."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "detention_submit",
    Path(__file__).resolve().parents[1] / "scripts/submit_detention_benchmark.py",
)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


@pytest.fixture
def scheduler(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("AF_PYTHON", __import__("sys").executable)
    monkeypatch.setenv("AF_COMMIT", "b" * 40)
    for key in ("AF_REPO_ROOT", "AF_OUTPUT_ROOT", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setattr(SUBMIT._support, "source_commit", lambda _: "b" * 40)

    def queue(directory, key, options, worker, stage, index):
        calls.append((key, options))
        return str(100 + len(calls))

    monkeypatch.setattr(SUBMIT._support, "submit_job", queue)
    return tmp_path, calls


def test_cpu_matrix_dependencies_and_unchanged_resume(scheduler):
    root, calls = scheduler
    first = SUBMIT.submit(root)
    assert [c[0] for c in calls] == ["prepare", "fit", "report"]
    assert "--array=0-7%4" in calls[1][1]
    assert "--dependency=afterok:101" in calls[1][1]
    assert "--dependency=afterany:102" in calls[2][1]
    assert all("--cpus-per-task=1" in c[1] for c in calls)
    assert SUBMIT.submit(root) == first and len(calls) == 3


def test_uncertain_submission_is_not_automatically_repeated(scheduler, monkeypatch):
    root, _ = scheduler

    def uncertain(*args):
        raise ValueError("uncertain")

    monkeypatch.setattr(SUBMIT._support, "submit_job", uncertain)
    with pytest.raises(ValueError, match="uncertain"):
        SUBMIT.submit(root)
    with pytest.raises(ValueError, match="partial/uncertain"):
        SUBMIT.submit(root)
