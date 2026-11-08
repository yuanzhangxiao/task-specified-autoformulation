"""ACES acknowledgment recovery and preservation of the original shallow call budget."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import submit_t2_breadth as submitter
from scripts import t2_breadth as pilot
from scripts import t2_breadth_history as history
from scripts import t2_breadth_scheduler as scheduler
from tests import test_t2_breadth
from tests.test_construction_initialization_graph_cli import reply, tokenizer

experiment = test_t2_breadth.experiment


def intent(root, *, depth=False):
    name = "phasec-t2-breadth"
    wave = root / "submissions/wave-0001"
    wave.mkdir(parents=True, exist_ok=True)
    argv = [
        "sbatch",
        "--parsable",
        "--job-name=" + name,
        f"--output={root}/logs/propose-%j.out",
        f"--error={root}/logs/propose-%j.err",
        *(["--comment=depth3-a"] if depth else []),
        "/pinned/scripts/hpc/run_meal_insulin_revision_aces.sh",
        "t2-breadth",
    ]
    (wave / "propose.intent.json").write_text(json.dumps({"argv": argv}))
    (wave / "propose.reply.json").write_text(
        json.dumps(
            {
                "returncode": 0,
                "stdout": "\n",
                "stderr": "sbatch: error: Socket timed out",
            }
        )
    )
    return wave, argv


def database(monkeypatch, argv, *, queue="", lines=(), controller="", offline=False):
    user = scheduler.getpass.getuser()
    now = datetime.now(UTC).isoformat()

    def query(command):
        if command[0] == "sacct":
            if offline:
                raise ValueError("Accounting service unavailable")
            return "\n".join(
                f"{job}|{user}|phasec-t2-breadth|{state}|{now}|{cmd}"
                for job, state, cmd in lines
            )
        if command[0] == "scontrol":
            return controller
        return f"{queue}|{user}|phasec-t2-breadth|PENDING|{now}" if queue else ""

    monkeypatch.setattr(scheduler, "_query", query)


def test_timeout_adopts_exact_existing_job_even_when_both_jobs_share_a_name(
    tmp_path, monkeypatch
):
    wave, argv = intent(tmp_path, depth=True)
    other = [a.replace("depth3-a", "different") for a in argv]
    database(
        monkeypatch,
        argv,
        queue="912",
        lines=[
            ("904", "COMPLETED", " ".join(other)),
            ("912", "PENDING", " ".join(argv)),
        ],
    )
    result = scheduler.reconcile(tmp_path, wave)
    assert (result["job"], result["state"]) == ("912", "PENDING")


def test_live_owner_and_job_paths_when_accounting_is_down(tmp_path, monkeypatch):
    wave, argv = intent(tmp_path)
    controller = (
        f"JobId=921 JobName=phasec-t2-breadth UserId={scheduler.getpass.getuser()}(11) "
        f"Command={argv[-2]} StdOut={tmp_path}/logs/propose-921.out "
        f"StdErr={tmp_path}/logs/propose-921.err"
    )
    database(monkeypatch, argv, queue="921", controller=controller, offline=True)
    assert scheduler.reconcile(tmp_path, wave)["job"] == "921"
    database(
        monkeypatch,
        argv,
        queue="921",
        controller=controller.replace("921.out", "922.out"),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        scheduler.reconcile(tmp_path, wave)


def test_missing_or_ambiguous_records_never_become_a_job(tmp_path, monkeypatch):
    wave, argv = intent(tmp_path)
    database(
        monkeypatch,
        argv,
        lines=[("912", "RUNNING", " ".join(argv)), ("913", "PENDING", " ".join(argv))],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        scheduler.reconcile(tmp_path, wave)
    database(monkeypatch, argv)
    with pytest.raises(ValueError, match="No matching job yet"):
        scheduler.reconcile(tmp_path, wave)
    then = time.time() - 300
    for path in wave.iterdir():
        os.utime(path, (then, then))
    assert (
        scheduler.reconcile(tmp_path, wave)["status"]
        == "absent_in_queue_and_accounting"
    )
    database(monkeypatch, argv, offline=True)
    with pytest.raises(ValueError, match="Cannot establish absence"):
        scheduler.reconcile(tmp_path, wave)


def test_definitive_rejection_reported_verbatim(tmp_path, monkeypatch):
    wave, argv = intent(tmp_path)
    error = "sbatch: error: Job violates accounting/QOS policy"
    (wave / "propose.reply.json").write_text(
        json.dumps({"returncode": 0, "stdout": "", "stderr": error})
    )
    database(monkeypatch, argv)
    assert scheduler.reconcile(tmp_path, wave)["scheduler_stderr"] == error
    assert scheduler.reconcile(tmp_path, wave)["status"] == "scheduler_rejected"


def test_repeat_run_recovers_before_another_sbatch(experiment, monkeypatch, tmp_path):
    image = tmp_path / "image"
    image.write_bytes(b"mock")
    monkeypatch.setattr(submitter.aces, "_assets", lambda: (image, {}))
    monkeypatch.setattr(
        submitter.aces.harmony_vocab, "verify", lambda d: {"sha256": "test"}
    )
    seen = []

    def timeout(wave, argv):
        assert "--dependency=singleton" in argv
        seen.append(argv)
        (wave / "propose.intent.json").write_text(json.dumps({"argv": argv}))
        (wave / "propose.reply.json").write_text(
            json.dumps({"returncode": 0, "stderr": "socket timeout", "stdout": ""})
        )
        raise ValueError("reply was unconfirmed")

    monkeypatch.setattr(submitter, "_submit_job", timeout)
    database(monkeypatch, [])
    assert submitter.submit(experiment)["status"] == "acknowledgment_pending"
    database(monkeypatch, seen[0], lines=[("912", "COMPLETED", " ".join(seen[0]))])
    assert submitter.submit(experiment)["job"] == "912"
    assert (
        sealed_read(experiment / "submissions/wave-0001/adoption.json")["job"] == "912"
    )
    assert len(seen) == 1


def legacy(experiment, monkeypatch, tmp_path):
    """Seed a v1-shaped partial campaign with authentic provider cache artifacts."""
    original = tmp_path / "legacy"
    intake = tmp_path / "intake.json"
    inputs = sealed_read(intake)
    sealed_write(
        original / "inputs.json",
        {k: v for k, v in inputs.items() if k != "artifact_sha256"},
    )
    plan = sealed_write(
        original / "plan.json", {"protocol": "synthetic pinned predecessor"}
    )
    monkeypatch.setattr(history, "V1_PLAN", plan["artifact_sha256"])
    system, packet = pilot._packet(inputs, "c15")
    raw = {"alternatives": [{"broken": True}] * 5}
    pilot._ask(
        original,
        plan,
        "c15-five",
        0,
        system + "\n" + pilot.FIVE,
        packet,
        pilot.FiveInsulinRepairs,
        "http://127.0.0.1",
        lambda *a: reply(raw),
        tokenizer,
        {"slots": {}, "physical_requests": 0},
        27,
    )
    new = tmp_path / "upgraded"
    pilot.prepare(new, intake, original)
    return original, new


@pytest.mark.parametrize("inflight", [False, True])
def test_completed_and_interrupted_v1_calls_share_a_single_total(
    experiment, monkeypatch, tmp_path, inflight
):
    original, new = legacy(experiment, monkeypatch, tmp_path)
    if inflight:
        slot = original / "steps/c15-five"
        (slot / "reply.json").unlink()
        physical = next((slot / "calls").glob("*.json"))
        record = json.loads(physical.read_text())
        record.update(status="inflight", observed_total_tokens=None)
        record.pop("raw_response")
        physical.write_text(json.dumps(record))
    names = []

    def unavailable(url, body, timeout):
        names.append(body["response_format"]["json_schema"]["name"])
        assert names[-1] != "FiveInsulinRepairs"
        raise TimeoutError("fixture")

    result = pilot.run(
        new, "http://127.0.0.1", transport=unavailable, token_transport=tokenizer
    )
    assert result["inherited_physical_requests"] == 1
    assert result["physical_requests"] == len(names) + 1
    assert result["first_27_physical_calls"]["through_call"] == len(names) + 1
    assert (
        sealed_read(new / "call-order.json")["calls"][0]["source_plan"]
        == history.V1_PLAN
    )
    assert result["groups"]["c15"] == (
        {"unavailable": 5} if inflight else {"unavailable": 1, "rejected": 4}
    )
    assert (
        pilot.run(
            new,
            "http://127.0.0.1",
            transport=lambda *a: pytest.fail("duplicate"),
            token_transport=tokenizer,
        )
        == result
    )


def test_source_prompt_drift_fails_before_duplicate_generation(
    experiment, monkeypatch, tmp_path
):
    _, new = legacy(experiment, monkeypatch, tmp_path)
    monkeypatch.setattr(pilot, "FIVE", pilot.FIVE + " Changed.")
    with pytest.raises(ValueError, match="Original prompt differs"):
        pilot.run(
            new,
            "http://127.0.0.1",
            transport=lambda *a: pytest.fail("generation"),
            token_transport=tokenizer,
        )
