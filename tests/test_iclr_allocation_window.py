"""Allocation changes preserve science and defer work before consuming budgets."""

import json
import subprocess
import time
from datetime import UTC, datetime

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from scripts import iclr_allocation_window as window
from scripts import iclr_controlled_search as cli

PLAN = {
    "targeted_protocol": window.PROTOCOL,
    "artifact_sha256": "plan-unchanged",
    "source_sha256": "source-unchanged",
}
ENV = {
    "SLURM_JOB_ID": "2376062",
    "SLURM_ARRAY_JOB_ID": "2376062",
    "SLURM_ARRAY_TASK_ID": "0",
    "AF_JETSTREAM_API_KEY": "must-not-be-recorded",
}
END = "2026-11-18T18:00:00+00:00"
END_EPOCH = datetime(2026, 11, 18, 18, tzinfo=UTC).timestamp()


def scheduler(
    monkeypatch,
    *,
    end=END,
    state="RUNNING",
    limit="04:00:00",
    job="2376062",
    array_job="2376062",
    array_task="0",
):
    """Supply the actual shortened allocation independently of worker arguments."""
    calls = []

    def query(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                f"JobId={job} JobName=iclr-targeted-20b JobState={state} "
                + (
                    f"ArrayJobId={array_job} ArrayTaskId={array_task} "
                    if array_job is not None
                    else ""
                )
                + f"TimeLimit={limit} EndTime={end}\n"
            ),
        )

    monkeypatch.setattr(window.subprocess, "run", query)
    return calls


def test_shortened_allocation_subtracts_startup_and_preserves_scientific_identity(
    tmp_path, monkeypatch
):
    calls = scheduler(monkeypatch)
    # A four-hour allocation spent 20 minutes on startup; keep two hours to drain.
    effective = window.allowance(
        tmp_path, PLAN, 0, 36000, environment=ENV, now=END_EPOCH - 13200
    )
    assert effective == 6000
    assert calls[0][0] == ["scontrol", "-o", "show", "job", "2376062_0"]
    assert calls[0][1]["timeout"] == 30
    path = tmp_path / "runtime/allocation-2376062.json"
    receipt = json.loads(path.read_text())
    assert receipt["slurm_time_limit_seconds"] == 14400
    assert receipt["requested_seconds"] == 36000
    assert receipt["effective_seconds"] == 6000
    assert receipt["completion_margin_seconds"] == 7200
    assert receipt["completion_margin_is_hard_bound"] is False
    assert receipt["scientific_budgets_modified"] is False
    assert receipt["plan_sha256"] == PLAN["artifact_sha256"]
    assert receipt["source_sha256"] == PLAN["source_sha256"]
    assert receipt["array_task"] == "0"
    assert "must-not-be-recorded" not in path.read_text()


def test_array_child_query_is_specific_and_numeric_job_identity_is_checked(
    tmp_path, monkeypatch
):
    calls = scheduler(monkeypatch, job="2376123", array_task="4")
    env = {**ENV, "SLURM_JOB_ID": "2376123", "SLURM_ARRAY_TASK_ID": "4"}
    assert (
        window.allowance(
            tmp_path, PLAN, 4, 36000, environment=env, now=END_EPOCH - 14400
        )
        == 7200
    )
    assert calls[0][0][-1] == "2376062_4"
    receipt = json.loads((tmp_path / "runtime/allocation-2376123.json").read_text())
    assert receipt["job"] == "2376123"
    assert receipt["slurm_query_id"] == "2376062_4"


@pytest.mark.parametrize(
    "changes",
    [{"job": "999"}, {"array_job": "999"}, {"array_task": "5"}],
)
def test_array_allocation_mismatch_fails_closed(tmp_path, monkeypatch, changes):
    scheduler(monkeypatch, **changes)
    with pytest.raises(ValueError, match="ambiguous or mismatched"):
        window.allowance(tmp_path, PLAN, 0, 36000, environment=ENV)


@pytest.mark.parametrize(
    "env",
    [
        {"SLURM_JOB_ID": "2376062", "SLURM_ARRAY_JOB_ID": "2376062"},
        {"SLURM_JOB_ID": "2376062", "SLURM_ARRAY_TASK_ID": "0"},
        {**ENV, "SLURM_ARRAY_JOB_ID": ""},
        {**ENV, "SLURM_ARRAY_TASK_ID": "0-5"},
    ],
)
def test_incomplete_or_invalid_array_environment_fails_before_query(
    tmp_path, monkeypatch, env
):
    calls = scheduler(monkeypatch)
    with pytest.raises(ValueError, match="array identifiers"):
        window.allowance(tmp_path, PLAN, 0, 36000, environment=env)
    assert calls == []


def test_nonarray_allocation_queries_numeric_job(tmp_path, monkeypatch):
    calls = scheduler(monkeypatch, array_job=None)
    assert (
        window.allowance(
            tmp_path,
            PLAN,
            0,
            36000,
            environment={"SLURM_JOB_ID": "2376062"},
            now=END_EPOCH - 14400,
        )
        == 7200
    )
    assert calls[0][0][-1] == "2376062"


def test_smaller_requested_window_is_never_increased(tmp_path, monkeypatch):
    scheduler(monkeypatch)
    assert (
        window.allowance(
            tmp_path, PLAN, 0, 1200, environment=ENV, now=END_EPOCH - 14400
        )
        == 1200
    )


@pytest.mark.parametrize("remaining", [7200, 7199, -1])
def test_insufficient_margin_defers_without_running_search(
    tmp_path, monkeypatch, remaining
):
    scheduler(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", ENV["SLURM_JOB_ID"])
    monkeypatch.setattr(window.time, "time", lambda: END_EPOCH - remaining)
    monkeypatch.setattr(cli, "sealed_read", lambda path: PLAN)

    def forbidden(*args):
        pytest.fail("a drained allocation must not enter proposal, critic or fitting")

    monkeypatch.setattr(cli.study, "run_block", forbidden)
    assert cli.run(tmp_path, 0, "http://localhost:1", 36000) == {
        "status": "pending",
        "block": 0,
        "reason": "allocation_draining",
    }
    receipt = json.loads((tmp_path / "runtime/allocation-2376062.json").read_text())
    assert receipt["effective_seconds"] == 0


def test_deferred_provider_call_returns_pending_but_other_failures_propagate(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "2376062")
    monkeypatch.setattr(cli, "sealed_read", lambda path: PLAN)
    monkeypatch.setattr(window, "allowance", lambda *args: 6000)
    calls = []

    def defer(root, block, url, seconds):
        calls.append(seconds)
        raise DeferredCall("allocation is draining")

    monkeypatch.setattr(cli.study, "run_block", defer)
    assert cli.run(tmp_path, 2, "http://localhost:1", 36000)["status"] == "pending"
    assert calls == [6000]

    def fail(*args):
        raise ValueError("real fit or source failure")

    monkeypatch.setattr(cli.study, "run_block", fail)
    with pytest.raises(ValueError, match="real fit or source failure"):
        cli.run(tmp_path, 2, "http://localhost:1", 36000)


@pytest.mark.parametrize("plan,env", [({}, ENV), (PLAN, {})])
def test_other_protocols_and_non_slurm_calls_are_unchanged(
    tmp_path, monkeypatch, plan, env
):
    def forbidden(*args, **kwargs):
        pytest.fail("unrelated execution must not query Slurm")

    monkeypatch.setattr(window.subprocess, "run", forbidden)
    assert window.allowance(tmp_path, plan, 0, 36000, environment=env) == 36000
    assert not (tmp_path / "runtime").exists()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if env:
        monkeypatch.setenv("SLURM_JOB_ID", env["SLURM_JOB_ID"])
    monkeypatch.setattr(cli, "sealed_read", lambda path: plan)
    monkeypatch.setattr(cli.study, "run_block", lambda *args: {"seconds": args[-1]})
    assert cli.run(tmp_path, 0, "http://localhost:1", 36000) == {"seconds": 36000}


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"end": "Unknown"}, "EndTime"),
        ({"end": "None"}, "EndTime"),
        ({"end": "2026-99-18T18:00:00"}, "EndTime"),
        ({"limit": "UNLIMITED"}, "TimeLimit"),
        ({"limit": "04:99:00"}, "TimeLimit"),
        ({"state": "PENDING"}, "RUNNING"),
        ({"state": "COMPLETING"}, "RUNNING"),
    ],
)
def test_invalid_scheduler_values_fail_closed(tmp_path, monkeypatch, changes, match):
    scheduler(monkeypatch, **changes)
    with pytest.raises(ValueError, match=match):
        window.allowance(tmp_path, PLAN, 0, 36000, environment=ENV)
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("scontrol"), subprocess.TimeoutExpired("scontrol", 30)],
)
def test_unavailable_scheduler_fails_closed(tmp_path, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(window.subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="Cannot verify"):
        window.allowance(tmp_path, PLAN, 0, 36000, environment=ENV)
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize(
    "output",
    [
        "JobId=2376062 JobState=RUNNING TimeLimit=04:00:00",
        f"JobId=999 JobState=RUNNING TimeLimit=04:00:00 EndTime={END}",
        (
            f"JobId=2376062 JobState=RUNNING TimeLimit=04:00:00 EndTime={END} "
            "JobId=2376063"
        ),
    ],
)
def test_missing_or_ambiguous_allocation_fails_closed(tmp_path, monkeypatch, output):
    monkeypatch.setattr(
        window.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout=output),
    )
    with pytest.raises(ValueError, match="ambiguous or mismatched"):
        window.allowance(tmp_path, PLAN, 0, 36000, environment=ENV)


def test_slurm_naive_time_uses_local_timezone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("test needs POSIX local timezone support")
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "UTC0")
            time.tzset()
            utc = window.end_timestamp("2026-11-18T18:00:00")
            patch.setenv("TZ", "HST10")
            time.tzset()
            local = window.end_timestamp("2026-11-18T18:00:00")
            assert local == utc + 36000
            assert window.end_timestamp(END) == utc
    finally:
        time.tzset()


def test_finite_slurm_day_limit():
    assert window.limit_seconds("1-04:00:00") == 100800
