"""Opt-in clocks preserve failures, publication and supervised budgets."""

import pytest

from autoformalism.fitting import mesh_refinement_process as process
from autoformalism.fitting import operation_timing as timing
from autoformalism.fitting import public_fitting as public


def test_disabled_and_exception_transparency(monkeypatch):
    @timing.timed("work")
    def work(fail=False):
        if fail:
            raise TimeoutError("original deadline")
        return 17

    def no_clock():
        pytest.fail("disabled instrumentation read clock")

    monkeypatch.setattr(timing, "monotonic", no_clock)
    assert work() == 17
    with pytest.raises(TimeoutError, match="original deadline"):
        work(True)


def test_cpu_wall_separate_bounded_nested_spans(monkeypatch):
    wall = iter(range(1000))
    cpu = iter(x / 10 for x in range(1000))
    monkeypatch.setattr(timing, "monotonic", lambda: next(wall))
    monkeypatch.setattr(timing, "process_time", lambda: next(cpu))
    recorder = timing.Recorder()

    @timing.timed("inner")
    def inner():
        raise ValueError("preserve")

    @timing.timed("outer")
    def outer():
        inner()

    with recorder.active():
        for _ in range(30):
            with pytest.raises(ValueError, match="preserve"):
                outer()
    assert recorder.groups["inner"]["failures"] == 30
    assert recorder.groups["outer"]["wall_seconds"] == 90
    assert recorder.groups["outer"]["cpu_seconds"] == pytest.approx(9)
    assert len(recorder.longest) == 20
    assert recorder.snapshot()["spans_are_inclusive"]


def test_checkpoint_bytes_unchanged(tmp_path):
    from autoformalism.rebuttal.fitter_diagnostic import write_json

    data = {"parameters": {"a": 0.1}, "status": "pending"}
    recorder = timing.Recorder()
    for writer in (public._write, write_json):
        writer(tmp_path / "plain.json", data)
        with recorder.active():
            writer(tmp_path / "timed.json", data)
        assert (tmp_path / "plain.json").read_bytes() == (
            tmp_path / "timed.json"
        ).read_bytes()
    assert recorder.groups["checkpoint_write"]["count"] == 2


def test_real_worker_failure_timing_and_resume(tmp_path):
    # Deliberately invalid payload: instrumented entrypoint records then re-raises.
    result = process.invoke(
        "recovery_rollout", {}, tmp_path, 30, diagnostic_timing=True
    )
    assert result["status"] == "failed" and result["termination_confirmed"]
    data = public._read(tmp_path / "timing.json")
    assert data["status"] == "raised"
    assert data["groups"]["imports"]["wall_seconds"] > 0
    assert data["worker_wall_seconds"] >= data["groups"]["imports"]["wall_seconds"]
    assert result["artifacts_sha256"]["timing.json"] == public.content_sha256(data)
    assert (
        process.invoke("recovery_rollout", {}, tmp_path, 30, diagnostic_timing=True)
        == result
    )
    with pytest.raises(ValueError, match="identity"):
        process.invoke("recovery_rollout", {}, tmp_path, 30)
    public._write(tmp_path / "timing.json", data | {"status": "changed"})
    with pytest.raises(ValueError, match="output differs"):
        process.invoke("recovery_rollout", {}, tmp_path, 30, diagnostic_timing=True)


def test_killed_worker_timing_missing_not_zero(tmp_path):
    result = process.invoke(
        "recovery_rollout", {}, tmp_path, 0.001, diagnostic_timing=True
    )
    assert result["status"] == "timeout" and result["termination_confirmed"]
    assert "timing.json" not in result["artifacts_sha256"]
