"""The revision launcher requests one bounded H100 allocation with no fit jobs."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import submit_construction_revision_aces as submission


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": submission.PROTOCOL,
            "cases": [
                {"case_id": f"parent_{i}__revision_seed{s}"}
                for i in range(8)
                for s in range(3)
            ],
            "settings": deepcopy(submission.SETTINGS),
            "input_bundle_sha256": "a" * 64,
            "runtime_source_sha256": "b" * 64,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )
    monkeypatch.setattr(submission.campaign, "verify", lambda root: deepcopy(plan))
    monkeypatch.setattr(
        submission.harmony_vocab, "verify", lambda path: {"sha256": "test"}
    )
    image, cache = tmp_path / "image.sif", tmp_path / "cache"
    image.write_bytes(b"test image")
    snapshot = (
        cache
        / "hub/models--openai--gpt-oss-20b/snapshots"
        / submission.REVISION
        / "config.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    for key, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
    }.items():
        monkeypatch.setenv(key, value)
    return root, plan


def test_one_bounded_allocation_idempotent(setup, monkeypatch):
    root, plan = setup
    seen = []

    def queue(directory, key, options, worker, stage, index):
        seen.append(options)
        assert worker.name == "run_construction_revision_aces.sh"
        assert key == stage == "propose" and index == 0
        return "12345"

    monkeypatch.setattr(submission, "submit_job", queue)
    first = submission.submit(root)
    assert submission.submit(root) == first
    assert len(seen) == 1
    assert first["maximum_generation_requests"] == 72
    assert first["maximum_charged_tokens"] == 24 * 98304
    assert {"--gres=gpu:h100:1", "--time=02:00:00", "--mem=64G"} <= set(seen[0])
    current, execution = submission.verify(root)
    assert current == plan
    assert execution["worker_seconds"] == 5400
    assert execution["optimizer_calls"] == 0
    assert execution["automatic_followup"] is False
    with pytest.raises(ValueError, match="one-job bound"):
        submission.submit(root, "another-wave")


@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_submission_not_automatically_repeated(setup, monkeypatch, timeout):
    root, _ = setup
    from scripts import submit_review_continuation

    seen = []

    def scheduler(argv, **kwargs):
        seen.append(argv)
        if timeout:
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(argv, 0, "\n", "Socket timed out")

    monkeypatch.setattr(submit_review_continuation.subprocess, "run", scheduler)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        submission.submit(root)
    with pytest.raises(ValueError):
        submission.submit(root, "retry")
    assert len(seen) == 1
    assert (root / "submissions/revision-1/propose.intent.json").exists()


def test_expanded_requests_are_rejected(setup, monkeypatch):
    root, plan = setup
    plan["settings"]["maximum_requests"] = 4
    monkeypatch.setattr(submission, "submit_job", lambda *a: pytest.fail("no submit"))
    with pytest.raises(ValueError):
        submission.submit(root)


def test_shell_entrypoints_validate_and_use_shared_server():
    for name in (
        "start_construction_revision_aces.sh",
        "run_construction_revision_aces.sh",
    ):
        path = submission.REPO / "scripts/hpc" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
    worker = (
        submission.REPO / "scripts/hpc/run_construction_revision_aces.sh"
    ).read_text()
    assert "run_construction_functions_aces.sh" in worker
    assert "AF_FUNCTION_WORKER_SECONDS=5400" in worker
    start = (
        submission.REPO / "scripts/hpc/start_construction_revision_aces.sh"
    ).read_text()
    assert "phase-c-construction-revision-v1" in start
    assert "inputs/revision-inputs.json" in start
    assert '"${AF_GCCCORE_MODULE:-GCCcore/13.2.0}"' in start


def test_execution_identity_cannot_be_changed(setup, monkeypatch):
    root, _ = setup
    monkeypatch.setattr(submission, "submit_job", lambda *a: "12345")
    submission.submit(root)
    path = root / "execution.json"
    value = json.loads(path.read_text())
    value["maximum_sequences"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        submission.verify(root)
