"""Scheduler recovery preserves the old scientific worker and completed visits."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_review_deadline_v2.py"
spec = importlib.util.spec_from_file_location("review_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
REAL_RUN = recovery.run


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

    def run(argv, *, env=None, receipt=None):
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
            if any("JobName" in a for a in argv):
                assert "--array" in argv
                command = state["submits"][int(job) - 201][0]
                name = next(
                    a.split("=", 1)[1] for a in command if a.startswith("--job-name=")
                )
                if job == "202":
                    children = [
                        f"{job}_{i}|{name}|"
                        + (
                            "FAILED|1:0"
                            if i == state.get("failed_child")
                            else "COMPLETED|0:0"
                        )
                        for i in state.get("materialized_array", [0, 1])
                    ]
                    if state.get("array_parent"):
                        children.insert(0, f"{job}|{name}|COMPLETED|0:0")
                    return "\n".join(children)
                return f"{job}|{state.get('wrong_name', name)}|{status}|{code}"
            return f"{job}|{status}|{code}|2026-09-16T04:45:00"
        if argv[0] == "squeue":
            return state["active"]
        assert argv[0] == "sbatch"
        # Simulate an active controller that has forgotten ALL old dependencies.
        assert not any("afterok:101" in x or "afterany:102" in x for x in argv)
        state["submits"].append((argv, env))
        if state.get("reject_at") == len(state["submits"]):
            raise RuntimeError("scheduler rejected")
        assert "--wrap" not in argv
        if "--job-name=review-v2-submit-next-2" in argv:
            assert Path(argv[-1]).is_file()
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
    dispatcher = Path(dispatch[-1]).read_text()
    assert str(SCRIPT) in dispatcher
    assert "submit_review_deadline_v2_aces.sh" not in dispatcher
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


@pytest.mark.parametrize("finish_state", ["PENDING", "COMPLETED"])
def test_dispatcher_only_repair_adopts_jobs_without_resubmitting(
    campaign, finish_state
):
    root, _, state = campaign
    state["reject_at"] = 4
    with pytest.raises(RuntimeError):
        recovery.recover(1)
    state["states"]["203"] = finish_state
    before = (root / "scheduler-recovery/attempt-1/evidence.json").read_bytes()
    repaired = recovery.recover(1, complete_dispatcher_only=True)
    assert (
        len(state["submits"]) == 5
    )  # three accepted + failed dispatcher + replacement
    command, _ = state["submits"][-1]
    assert "--job-name=review-v2-submit-next-2" in command
    assert ("--dependency=afterany:203" in command) == (finish_state == "PENDING")
    assert repaired["jobs"]["propose-1"] == "201"
    assert repaired["jobs"]["fit-1"] == "202"
    assert repaired["jobs"]["finish-1"] == "203"
    assert repaired["jobs"]["submit-next-2"] == "205"
    assert (root / "scheduler-recovery/attempt-1/evidence.json").read_bytes() == before
    assert recovery.recover(1, complete_dispatcher_only=True) == repaired
    assert len(state["submits"]) == 5


@pytest.mark.parametrize(
    "fault", ["orphan", "name", "failed", "identity", "missing_id", "repeat"]
)
def test_dispatcher_repair_blocks_uncertain_or_mismatched_state(campaign, fault):
    root, _, state = campaign
    state["reject_at"] = 4
    with pytest.raises(RuntimeError):
        recovery.recover(1)
    attempt = root / "scheduler-recovery/attempt-1"
    if fault == "orphan":
        state["active"] = "an already accepted dispatcher"
    elif fault == "name":
        state["wrong_name"] = "unrelated job"
    elif fault == "failed":
        state["states"]["203"] = "FAILED"
    elif fault == "identity":
        evidence = recovery.read(attempt / "evidence.json")
        evidence["plan_sha256"] = "wrong"
        recovery.write(attempt / "evidence.json", evidence)
    elif fault == "missing_id":
        (attempt / "fit-1.id").unlink()
    else:
        (attempt / "dispatcher-repair").mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        recovery.recover(1, complete_dispatcher_only=True)
    assert len(state["submits"]) == 4


def test_site_wrapper_argument_splitting_accepts_script_submission(tmp_path):
    """Reproduce the ACES wrapper's unquoted forwarding without bypassing it."""
    underlying = tmp_path / "underlying"
    underlying.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "assert '--wrap' not in sys.argv\n"
        "assert Path(sys.argv[-1]).read_text().startswith('#!/bin/bash')\n"
        "print(12345)\n"
    )
    underlying.chmod(0o755)
    wrapper = tmp_path / "sbatch"
    wrapper.write_text(
        '#!/bin/sh\njob_file="$@"\n'
        f"submission_output=$({shlex.quote(str(underlying))} $job_file)\n"
        "echo $submission_output\n"
    )
    wrapper.chmod(0o755)
    script = tmp_path / "dispatch.sh"
    script.write_text("#!/bin/bash\nset -euo pipefail\ntrue\n")
    receipt = tmp_path / "reply.json"
    reply = REAL_RUN([str(wrapper), "--parsable", str(script)], receipt=receipt)
    assert reply == "12345"
    assert recovery.read(receipt)["returncode"] == 0
    # The same wrapper breaks a command string with spaces, even with exit 0.
    reply = REAL_RUN(
        [str(wrapper), "--parsable", "--wrap", "bash -lc true"], receipt=receipt
    )
    assert reply == ""
    assert recovery.read(receipt)["returncode"] == 0
    assert "AssertionError" in recovery.read(receipt)["stderr"]


def test_failed_scheduler_reply_is_saved_before_raising(tmp_path):
    receipt = tmp_path / "reply.json"
    with pytest.raises(subprocess.CalledProcessError):
        REAL_RUN(
            [
                sys.executable,
                "-c",
                "import sys; print('rejected', file=sys.stderr); sys.exit(1)",
            ],
            env=os.environ.copy(),
            receipt=receipt,
        )
    assert recovery.read(receipt)["stderr"].strip() == "rejected"


@pytest.mark.parametrize("indices", [[0, 1], [0], [0, 1, 1], [0, 2]])
def test_dispatcher_repair_checks_materialized_array_tasks(campaign, indices):
    _, _, state = campaign
    state["reject_at"] = 4
    with pytest.raises(RuntimeError):
        recovery.recover(1)
    state["materialized_array"] = indices
    if indices == [0, 1]:
        result = recovery.recover(1, complete_dispatcher_only=True)
        assert (
            result["scheduler_recovery"]["adopted_jobs"]["fit-1"]["state"]
            == "ARRAY_CONFIRMED"
        )
    else:
        with pytest.raises(ValueError, match="complete saved array"):
            recovery.recover(1, complete_dispatcher_only=True)
        assert len(state["submits"]) == 4


@pytest.mark.parametrize("fault", ["failed_child", "missing_child", "parent_only"])
def test_array_parent_cannot_hide_bad_or_missing_children(campaign, fault):
    _, _, state = campaign
    state["reject_at"] = 4
    with pytest.raises(RuntimeError):
        recovery.recover(1)
    state["array_parent"] = True
    if fault == "failed_child":
        state["failed_child"] = 0
    else:
        state["materialized_array"] = [0] if fault == "missing_child" else []
    with pytest.raises(ValueError):
        recovery.recover(1, complete_dispatcher_only=True)
    assert len(state["submits"]) == 4


@pytest.mark.parametrize(
    "receipt",
    [
        {"returncode": 0, "stdout": "205\n", "stderr": ""},
        {
            "returncode": 1,
            "stdout": "Submitted batch job 205",
            "stderr": "post-submit error",
        },
        {"returncode": 0, "stdout": "", "stderr": ""},
        {"returncode": -9, "stdout": "", "stderr": ""},
        {"returncode": 137, "stdout": "", "stderr": ""},
        {"status": "timeout_reply_uncertain"},
    ],
)
def test_local_uncertain_reply_blocks_retry_even_with_empty_scheduler(
    campaign, receipt
):
    root, _, state = campaign
    state["reject_at"] = 4
    with pytest.raises(RuntimeError):
        recovery.recover(1)
    recovery.write(
        root / "scheduler-recovery/attempt-1/submit-next-2.reply.json", receipt
    )
    with pytest.raises(ValueError, match="acceptance or uncertainty"):
        recovery.recover(1, complete_dispatcher_only=True)
    assert len(state["submits"]) == 4
