"""Scheduler recovery preserves the old scientific worker and completed visits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_review_deadline_v2.py"
spec = importlib.util.spec_from_file_location("review_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    repo = tmp_path / "old-checkout"
    repo.mkdir()
    plan = {
        "protocol": "review-deadline-2",
        "artifact_sha256": "frozen-plan",
        "config": {"rounds": 3},
        "tasks": [{"index": 0}, {"index": 1}],
    }
    manifest = {
        "protocol": "review-deadline-2",
        "commit": "old-scientific-commit",
        "jobs": {"prepare-0": "101", "finish-0": "102", "submit-next-1": "103"},
        "planned_rounds": 3,
        "submitted_through_round": 0,
    }
    recovery.write(root / "plan.json", plan)
    recovery.write(root / "submission-round-0.json", manifest)
    recovery.write(root / "submission_manifest.json", manifest)
    old_intent = root / "submission-intent/round-1"
    old_intent.mkdir(parents=True)
    (root / "completed-round-zero-evidence").write_text("preserve exactly")
    monkeypatch.setenv("AF_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("AF_REPO_ROOT", str(repo))
    monkeypatch.setenv("AF_PYTHON", "/old/python")
    monkeypatch.delenv("AF_RECOVERY_SCRIPT_SHA256", raising=False)
    state = {"submits": [], "commands": [], "states": {}, "history": "", "active": ""}

    def run(argv, *, env=None):
        state["commands"].append((argv, env))
        if argv[1] == "-c":
            if state.get("missing_prior"):
                raise FileNotFoundError("prior result missing")
            return '{"prior-task": "sealed-result"}'
        if argv[0] == "git":
            return "" if "status" in argv else "old-scientific-commit"
        if "review_deadline.py" in argv[1]:
            assert argv[1] == str(repo / "scripts/review_deadline.py")
            assert argv[2] == "verify"
            assert env["PYTHONPATH"] == str(repo / "src")
            return "verified"
        if argv[0] == "sacct":
            if "-j" not in argv:
                return state["history"]
            job = argv[argv.index("-j") + 1]
            status = state["states"].get(job, "FAILED" if job == "103" else "COMPLETED")
            code = "1:0" if status == "FAILED" else "0:0"
            return f"{job}|{status}|{code}|2026-09-16T04:45:00"
        if argv[0] == "squeue":
            return state["active"]
        assert argv[0] == "sbatch"
        # Simulate an active controller that has forgotten ALL old dependencies.
        assert not any("afterok:101" in x or "afterany:102" in x for x in argv)
        state["submits"].append((argv, env))
        if state.get("reject_at") == len(state["submits"]):
            raise RuntimeError("scheduler rejected")
        return str(200 + len(state["submits"]))

    monkeypatch.setattr(recovery, "run", run)
    return root, repo, state


def test_recovery_queues_only_unstarted_visits_from_original_code(campaign):
    root, repo, state = campaign
    original = (root / "plan.json").read_bytes()
    first = recovery.recover(1)
    assert len(state["submits"]) == 4
    propose, fit, finish, dispatch = [a for a, _ in state["submits"]]
    worker = str(repo / "scripts/hpc/run_review_deadline_aces.sh")
    assert propose[-3:] == [worker, "propose", "1"]
    assert not any(x.startswith("--dependency") for x in propose)
    assert "--array=0,1%16" in fit
    assert "--dependency=afterany:201" in fit
    assert "--dependency=afterany:202" in finish
    assert "--dependency=afterany:203" in dispatch
    assert str(SCRIPT) in dispatch[-1]
    assert "submit_review_deadline_v2_aces.sh" not in dispatch[-1]
    assert first["commit"] == "old-scientific-commit"
    assert first["jobs"]["submit-next-1"] == "103"  # Failed ID remains in history.
    assert not first["all_rounds_submitted"]
    assert recovery.recover(1) == first
    assert len(state["submits"]) == 4
    second = recovery.recover(2)
    assert len(state["submits"]) == 7
    assert second["submission_complete"]
    assert second["scheduler_recovery"]["prior_finish"]["job"] == "203"
    assert (root / "plan.json").read_bytes() == original
    assert (root / "completed-round-zero-evidence").read_text() == "preserve exactly"
    assert (root / "submission-intent/round-1").is_dir()
    assert list((root / "submission-intent/round-1").iterdir()) == []


@pytest.mark.parametrize(
    "job,state", [("101", "FAILED"), ("102", "RUNNING"), ("103", "RUNNING")]
)
def test_unready_prerequisites_do_not_consume_recovery_attempt(campaign, job, state):
    root, _, fake = campaign
    fake["states"][job] = state
    with pytest.raises(ValueError):
        recovery.recover(1)
    assert not (root / "scheduler-recovery/attempt-1").exists()
    assert not fake["submits"]


@pytest.mark.parametrize("known", ["history", "active", "id", "result"])
def test_recovery_refuses_possible_duplicate_work(campaign, known):
    root, _, state = campaign
    if known in {"history", "active"}:
        state[known] = "an existing job"
    elif known == "id":
        (root / "submission-intent/propose-1.id").write_text("999")
    else:
        path = root / "results/cell/round_01/result.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    with pytest.raises(ValueError):
        recovery.recover(1)
    assert not state["submits"]


def test_partial_rejection_keeps_known_ids_and_blocks_blind_retry(campaign):
    root, _, state = campaign
    state["reject_at"] = 2
    with pytest.raises(RuntimeError, match="scheduler rejected"):
        recovery.recover(1)
    assert (root / "scheduler-recovery/attempt-1/propose-1.id").read_text() == "201\n"
    with pytest.raises(ValueError, match="intent exists"):
        recovery.recover(1)
    assert len(state["submits"]) == 2
    assert not (root / "submission-round-1.json").exists()


def test_recovery_binds_dispatcher_source(campaign, monkeypatch):
    _, _, state = campaign
    monkeypatch.setenv("AF_RECOVERY_SCRIPT_SHA256", "different-source")
    with pytest.raises(ValueError, match="source changed"):
        recovery.recover(1)
    assert not state["submits"]


def test_original_source_mismatch_cannot_be_re_frozen(campaign):
    root, _, state = campaign
    path = root / "submission-round-0.json"
    manifest = json.loads(path.read_text())
    manifest["commit"] = "another-commit"
    recovery.write(path, manifest)
    with pytest.raises(ValueError, match="ORIGINAL pinned"):
        recovery.recover(1)
    assert not state["submits"]


def test_missing_prior_results_block_recovery(campaign):
    root, _, state = campaign
    state["missing_prior"] = True
    with pytest.raises(FileNotFoundError, match="prior result missing"):
        recovery.recover(1)
    assert not state["submits"]
    assert not (root / "scheduler-recovery/attempt-1").exists()


def test_venv_python_symlink_is_preserved(campaign, tmp_path, monkeypatch):
    _, _, state = campaign
    target = tmp_path / "system-python"
    target.touch()
    python = tmp_path / "venv-python"
    python.symlink_to(target)
    monkeypatch.setenv("AF_PYTHON", str(python))
    recovery.recover(1)
    verifies = [a for a, _ in state["commands"] if "verify" in a]
    assert verifies[0][0] == str(python)
