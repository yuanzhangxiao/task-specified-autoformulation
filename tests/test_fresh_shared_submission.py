"""One-GPU dispatch, final CPU pruning and safe uncertain-submission recovery."""

import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import smoke_fresh_shared as smoke
from scripts import submit_fresh_shared as submitter


def setup(tmp_path, monkeypatch):
    root = tmp_path / "run"
    plan = smoke.fixture(root)
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        path = tmp_path / key
        path.touch()
        monkeypatch.setenv(key, str(path))
    monkeypatch.delenv("AF_COMMIT", raising=False)
    monkeypatch.setattr(submitter, "source_commit", lambda p: "a" * 40)
    return root, plan


def fake_scheduler(calls, *, uncertain=False):
    def queue(directory, key, options, worker, stage, index):
        calls.append((stage, options))
        public._write(
            directory / f"{key}.intent.json",
            {
                "argv": [
                    "sbatch",
                    "--parsable",
                    *options,
                    str(worker),
                    stage,
                    str(index),
                ]
            },
        )
        if uncertain and len(calls) == 1:
            public._write(
                directory / f"{key}.reply.json",
                {"returncode": 0, "stdout": "", "stderr": "Socket timed out"},
            )
            raise ValueError("unconfirmed scheduler reply")
        job = str(100 + len(calls))
        (directory / f"{key}.id").write_text(job + "\n")
        return job

    return queue


def test_dispatch_and_final_cpu_pruning_do_not_duplicate(tmp_path, monkeypatch):
    root, plan = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(submitter, "submit_job", fake_scheduler(calls))
    first = submitter.submit(root, 0)
    assert [c[0] for c in calls] == ["prepare", "propose", "fit", "finish", "dispatch"]
    assert "--gres=gpu:h100:1" in calls[1][1]
    assert "--dependency=afterok:104" in calls[4][1]
    assert submitter.submit(root, 0) == first and len(calls) == 5
    with pytest.raises(ValueError, match="previous round missing"):
        submitter.submit(root, 1)
    for task in plan["tasks"]:
        sealed_write(
            root / "results" / task["task_id"] / "round_00/result.json",
            {
                "task": task,
                "round": 0,
                "selected": None,
            },
        )
    last = submitter.submit(root, 1)
    assert last["all_rounds_submitted"]
    assert [c[0] for c in calls[5:]] == ["propose", "fit", "finish", "prune", "report"]
    assert "--dependency=afterok:108" in calls[8][1]
    assert not any("gpu" in option for option in calls[8][1])
    assert "--dependency=afterany:109" in calls[9][1]


def test_uncertain_reply_is_adopted_only_after_scheduler_verification(
    tmp_path, monkeypatch
):
    root, _ = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(submitter, "submit_job", fake_scheduler(calls, uncertain=True))
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(root, 0)
    with pytest.raises(ValueError, match="Unconfirmed prepare-0"):
        submitter.submit(root, 0)
    assert len(calls) == 1
    verified = []

    def verify(job, argv, *, since):
        verified.append(job)
        assert job == "201" and argv[-2:] == ["prepare", "0"]
        return {"job": job, "source": "scontrol"}

    monkeypatch.setattr(submitter, "scheduler_record", verify)
    value = submitter.submit(root, 0, {"prepare-0": "201"})
    assert verified == ["201"] and len(calls) == 5
    assert value["jobs"]["prepare-0"] == "201"
    receipt = root / "submission-intent/round-0/prepare-0.reply.json"
    assert json.loads(receipt.read_text())["stderr"] == "Socket timed out"
    assert "--dependency=afterok:201" in calls[1][1]


def test_changed_intent_is_rejected_before_adopting(tmp_path, monkeypatch):
    root, _ = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(submitter, "submit_job", fake_scheduler(calls, uncertain=True))
    with pytest.raises(ValueError):
        submitter.submit(root, 0)
    path = root / "submission-intent/round-0/prepare-0.intent.json"
    path.write_text(json.dumps({"argv": ["sbatch", "wrong-path"]}))
    monkeypatch.setattr(
        submitter,
        "scheduler_record",
        lambda *a, **k: pytest.fail("adopted wrong command"),
    )
    with pytest.raises(ValueError, match="command differs"):
        submitter.submit(root, 0, {"prepare-0": "201"})
