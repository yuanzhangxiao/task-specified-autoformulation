"""Delta placement preserves scientific budgets, paired fits, and sealed resume."""

import hashlib
import subprocess
import sys

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_sign_repair as campaign
from scripts import submit_dalla_sign_repair as submitter
from scripts.smoke_dalla_sign_repair import direction_transport, fixture
from tests.test_process_pruning_campaign import backend


def setup(tmp_path, monkeypatch):
    source, config, root = fixture(tmp_path, protocol="dalla-sign-repair-2")
    image = tmp_path / "vllm.sif"
    image.write_bytes(b"fake image for scheduler tests")
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_VLLM_IMAGE", str(image))
    monkeypatch.setenv("AF_HF_HOME", str(tmp_path / "new-cache"))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setenv("AF_GPU_ACCOUNT", "test-gpu")
    monkeypatch.setenv("AF_CPU_ACCOUNT", "test-cpu")
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    return source, config, root, image


def test_delta_submit_accounts_resources_and_exact_resume(tmp_path, monkeypatch):
    source, config, root, image = setup(tmp_path, monkeypatch)
    original = public._read(config)
    calls = {}

    def queue(directory, key, options, worker, stage, index):
        calls[key] = options
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, root, config=config, delta=True)
    assert result["platform"] == "delta-a40x1"
    assert result["maximum_llm_calls"] == 6
    assert result["array_tasks"] == 1 and result["fit_gpus"] == 0
    assert len(calls) == 4
    for stage, options in calls.items():
        assert "--constraint=projects&work" in options
        assert "--account=test-" + ("gpu" if stage == "review" else "cpu") in options
        assert not any("h100" in arg for arg in options)
    assert "--partition=gpuA40x4" in calls["review"]
    assert "--gpus-per-node=1" in calls["review"]
    assert "--partition=cpu" in calls["fit"]
    assert "--time=04:30:00" in calls["fit"]
    assert "--dependency=afterok:102" in calls["fit"]
    assert result == submitter.submit(source, root, config=config, delta=True)
    assert len(calls) == 4
    frozen = campaign.verify(root)["config"]
    assert (
        frozen["serving_image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    )
    assert public._read(config) == original
    assert {
        k: v for k, v in frozen.items() if k not in ("platform", "serving_image_sha256")
    } == {
        k: v
        for k, v in original.items()
        if k not in ("platform", "serving_image_sha256")
    }


def test_delta_image_change_is_not_silent_resume(tmp_path, monkeypatch):
    _, config, root, image = setup(tmp_path, monkeypatch)
    path = submitter.delta_config(config, root, image)
    before = path.read_bytes()
    image.write_bytes(b"replacement image")
    with pytest.raises(ValueError, match="configuration changed"):
        submitter.delta_config(config, root, image)
    assert path.read_bytes() == before


def test_delta_paired_fit_and_resume(tmp_path, monkeypatch):
    source, config, root, image = setup(tmp_path, monkeypatch)
    frozen = submitter.delta_config(config, root, image)
    campaign.freeze(source, frozen, root)
    response = campaign.review_one(
        root, 0, base_url="http://mock", transport=direction_transport
    )
    assert response["status"] == "repaired" and response["physical_requests"] == 2
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.fit_one(root, 0)
    assert result["status"] == "complete" and result["sign_integrity"]["passed"]
    assert len(calls) == 6
    assert campaign.fit_one(root, 0) == result
    assert len(calls) == 6
    assert campaign.report(root)["status"] == "complete"
    check = subprocess.run(
        [
            "bash",
            "scripts/hpc/run_staged_topology_server.sh",
            "--check-config",
            str(frozen),
        ],
        cwd=campaign.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert check.stdout.strip() == "dalla_sign_repair.py"


def test_missing_delta_image_does_not_queue(tmp_path, monkeypatch):
    source, config, root, image = setup(tmp_path, monkeypatch)
    image.unlink()
    monkeypatch.setattr(
        submitter, "submit_job", lambda *a: pytest.fail("unexpected submission")
    )
    with pytest.raises(FileNotFoundError):
        submitter.submit(source, root, config=config, delta=True)
    assert not (root / "submission_manifest.json").exists()
