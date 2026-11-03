"""Persistence policy changes writes, never numerical selection or allowances."""

import json
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import compact_search_checkpoint as c
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.research.construction_fit_qualification import _operation


class Clock:
    def __init__(self):
        self.time = 1000.0

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(c, "monotonic", clock)
    monkeypatch.setattr(verification, "monotonic", clock)
    return clock


def test_slow_sink_throttles_counters_but_preserves_every_improved_endpoint(
    monkeypatch,
):
    clock = fake_clock(monkeypatch)
    budget = verification._Budget(
        verification.ExecutionPolicy(numerical_seconds=60, overall_seconds=300)
    )
    snapshots = []

    def slow_sink(snapshot):
        snapshots.append(deepcopy(snapshot))
        clock.advance(0.7)

    writer = c.SearchProgress(c.SearchCheckpointPolicy(), slow_sink)
    result = {
        "best": None,
        "root_profile_certificate": {"large_static": "x" * 10000},
        "accepted_shared_iteration_history": [],
    }
    writer.attach(result)
    best_losses = []
    for i in range(200):
        result["search_calls_attempted"] = i + 1
        writer.save(result, budget)
        clock.advance(0.01)
        improved = i in (0, 17, 80, 199)
        if improved:
            result["best"] = {
                "parameters": {"p": float(i)},
                "training_nmse": 1 / (i + 1),
                "root_solve": [{"conditional_optimality": True}],
            }
            best_losses.append(result["best"]["training_nmse"])
        writer.save(result, budget, force=improved, event="incumbent")
    budget.callback(lambda: writer.final(deepcopy(result)))
    assert writer.stats["writes_completed"] == 6
    assert writer.stats["skipped_routine_requests"] == 395
    assert writer.stats["writes_failed"] == 0
    assert [
        x["best"]["training_nmse"]
        for x in snapshots
        if x["compact_progress"]["event"] == "incumbent" and x["best"] is not None
    ] == best_losses
    assert snapshots[-1]["best"] == result["best"]
    assert all("root_profile_certificate" not in x for x in snapshots)
    assert snapshots[-1]["compact_progress"]["omitted_static_sha256"] == {
        "root_profile_certificate": content_sha256(result["root_profile_certificate"])
    }
    assert writer.stats["serialized_bytes_completed"] == sum(
        len(json.dumps(x, sort_keys=True, separators=(",", ":")).encode())
        for x in snapshots
    )
    assert budget.snapshot()["callback_wall_seconds"] == pytest.approx(4.2)
    assert budget.snapshot()["numerical_active_wall_seconds"] == pytest.approx(2)


def test_periodic_progress_and_bounded_history_keep_latest_state(monkeypatch):
    clock = fake_clock(monkeypatch)
    budget = verification._Budget(
        verification.ExecutionPolicy(numerical_seconds=120, overall_seconds=300)
    )
    snapshots = []
    writer = c.SearchProgress(
        c.SearchCheckpointPolicy(interval_seconds=30, history_limit=2), snapshots.append
    )
    result = {"best": {"parameters": {"x": 1}}, "accepted_history": list(range(20))}
    writer.attach(result)
    writer.save(result, budget)
    clock.advance(29)
    writer.save(result, budget)
    clock.advance(1)
    writer.save(result, budget)
    assert len(snapshots) == 2
    assert snapshots[-1]["accepted_history"] == [18, 19]
    assert snapshots[-1]["compact_progress"]["accepted_history_total"] == 20
    assert result["accepted_history"] == list(range(20))
    assert "compact_progress" not in result


def test_failed_forced_write_is_accounted_and_propagated(monkeypatch):
    fake_clock(monkeypatch)
    budget = verification._Budget(verification.ExecutionPolicy())

    def unavailable(_):
        raise OSError("persistent storage unavailable")

    writer = c.SearchProgress(c.SearchCheckpointPolicy(), unavailable)
    result = {"best": {"parameters": {"x": 1}}}
    writer.attach(result)
    with pytest.raises(verification._CheckpointFailure):
        writer.save(result, budget, force=True, event="incumbent")
    assert writer.stats["writes_started"] == writer.stats["writes_failed"] == 1
    assert writer.stats["writes_completed"] == 0
    assert writer.last_written is None


@pytest.mark.parametrize("field", ["best", "certificate"])
def test_payload_preparation_failure_is_accounted_before_sink(monkeypatch, field):
    fake_clock(monkeypatch)
    budget = verification._Budget(verification.ExecutionPolicy())
    snapshots = []
    writer = c.SearchProgress(c.SearchCheckpointPolicy(), snapshots.append)
    result = {field: {"unsupported_value": object()}}
    writer.attach(result)
    with pytest.raises(verification._CheckpointFailure):
        writer.save(result, budget, force=True, event="incumbent")
    assert snapshots == []
    assert writer.stats["writes_started"] == writer.stats["writes_failed"] == 1
    assert writer.stats["writes_completed"] == 0
    assert budget.callback_failures == 1
    assert writer.last_written is None


def test_typed_checkpoint_policy_required():
    with pytest.raises(TypeError, match="typed SearchCheckpointPolicy"):
        c.SearchProgress({}, lambda _: None)


def test_hard_interruption_retains_compact_best_without_renewing_reservation(
    monkeypatch, tmp_path
):
    fake_clock(monkeypatch)
    identity = {"campaign": "compact-progress-test"}
    directory = tmp_path / "operations" / "search"
    bound = {"campaign": identity, "stage": "search", "allowance_seconds": 123.0}
    seal(directory / "started.json", bound)
    budget = verification._Budget(verification.ExecutionPolicy())
    writer = c.SearchProgress(
        c.SearchCheckpointPolicy(),
        lambda snapshot: public._write(directory / "progress.json", snapshot),
    )
    result = {
        "best": {"parameters": {"p": 2.0}, "training_nmse": 0.25},
        "search_calls_attempted": 10,
        "certificate": {"proof": "already-qualified"},
    }
    writer.attach(result)
    writer.save(result, budget, force=True, event="incumbent")
    # A killed process cannot finalize. A repeated operation sees only the
    # preexisting reservation and compact progress, and must not rerun work.
    calls = []

    def forbidden(_):
        calls.append(True)
        raise AssertionError("interrupted budget must not restart")

    recovered = _operation(tmp_path, identity, "search", 123.0, forbidden)
    assert calls == []
    assert recovered["status"] == "interrupted"
    assert not recovered["budget_restarted"] and not recovered["accounting_complete"]
    assert recovered["budget_charge_seconds"] == 123.0
    assert recovered["observed_progress"]["best"] == result["best"]
    assert "certificate" not in recovered["observed_progress"]
    assert _operation(tmp_path, identity, "search", 123.0, forbidden) == recovered
