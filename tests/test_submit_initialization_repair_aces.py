"""The bounded graph retest cannot duplicate an uncertain scheduler submission."""

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import submit_initialization_repair_aces as submission


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    plan = sealed_write(root / "plan.json", {"protocol": "test-plan"})
    monkeypatch.setattr(submission.campaign, "verify", lambda _: plan)
    image = tmp_path / "image.sif"
    image.write_bytes(b"test image")
    for name, value in {
        "AF_PYTHON": "/test/python",
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": "/test/cache",
        "AF_REPO_ROOT": str(submission.REPO),
        "AF_OUTPUT_ROOT": str(root),
        "AF_INITIALIZATION_WORKER": "choice",
    }.items():
        monkeypatch.setenv(name, value)
    return root, image


def test_confirmed_submission_is_idempotent_and_selects_graph_worker(
    tmp_path, monkeypatch
):
    root, _ = fixture(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(options)
        assert Path(worker).name == "run_initializer_choice_aces.sh"
        assert stage == "propose"
        assert "--gres=gpu:h100:1" in options
        return "1234"

    monkeypatch.setattr(submission, "submit_job", submit)
    first = submission.submit(root)
    assert first == submission.submit(root)
    assert len(calls) == 1 and first["job"] == "1234"
    assert not first["automatic_followup"]
    assert submission.os.environ["AF_INITIALIZATION_WORKER"] == "graph-repair"


def test_uncertain_intent_never_resubmits(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    folder = root / "submissions" / "repair-1"
    folder.mkdir(parents=True)
    (folder / "propose.intent.json").write_text(json.dumps({"pending": True}))
    monkeypatch.setattr(
        submission, "submit_job", lambda *args: pytest.fail("must not resubmit")
    )
    with pytest.raises(ValueError, match="uncertain"):
        submission.submit(root)


def test_changed_image_cannot_reuse_execution(tmp_path, monkeypatch):
    root, image = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(submission, "submit_job", lambda *args: "1234")
    submission.submit(root)
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError):
        submission.submit(root)
