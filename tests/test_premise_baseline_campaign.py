"""Bounded continuous-control persistence, integrity, and execution accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from autoformalism.research import premise_baseline_campaign as campaign


def scientific(report: dict) -> dict:
    """Keep nondeterministic measured execution time outside scientific equality."""
    return {
        key: value for key, value in report.items() if key != "execution_accounting"
    }


def snapshot(root: Path, *, include_runtime: bool = True) -> dict[str, bytes]:
    """Read durable bytes, optionally restricting to deterministic artifacts."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and (include_runtime or path.parent.name != "runtime")
    }


@pytest.fixture
def small_roster(monkeypatch):
    """A sealed, explicit small schedule for inexpensive persistence tests."""
    problem = campaign.core.problems()[0]
    monkeypatch.setattr(campaign.core, "problems", lambda: (problem,))
    monkeypatch.setattr(campaign.core, "BUDGETS", (8, 16))
    return problem


def forbidden(*_args, **_kwargs):
    raise AssertionError("inspection or completed resume started a new search")


def test_zero_checkpoint_report_preserves_frozen_full_denominator(
    tmp_path, monkeypatch
):
    report = campaign.run_campaign(tmp_path, max_new_checkpoints=0)
    assert report["expected_jobs"] == 8
    assert report["expected_checkpoints"] == 32
    assert report["recorded_checkpoints"] == 0
    assert len(report["missing_checkpoints"]) == 32
    assert report["completed_jobs"] == [] and not report["complete"]
    assert not (tmp_path / "summary.json").exists()
    before = snapshot(tmp_path)
    monkeypatch.setattr(campaign.core, "initialize", forbidden)
    monkeypatch.setattr(campaign.core, "advance", forbidden)
    assert campaign.report_campaign(tmp_path) == report
    assert snapshot(tmp_path) == before


def test_partial_report_does_not_fill_missing_rows(tmp_path, monkeypatch):
    partial = campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    assert partial["recorded_checkpoints"] == 1
    assert len(partial["missing_checkpoints"]) == 31
    assert partial["rows"][0]["budget"] == 64
    before = snapshot(tmp_path)
    monkeypatch.setattr(campaign.core, "initialize", forbidden)
    monkeypatch.setattr(campaign.core, "advance", forbidden)
    assert campaign.report_campaign(tmp_path) == partial
    assert snapshot(tmp_path) == before


def test_resume_scientific_bytes_equal_uninterrupted_run(tmp_path, small_roster):
    resumed_root, full_root = tmp_path / "resumed", tmp_path / "full"
    partial = campaign.run_campaign(resumed_root, max_new_checkpoints=2)
    assert partial["expected_jobs"] == 2
    assert partial["expected_checkpoints"] == 4
    assert partial["recorded_checkpoints"] == 2
    assert len(partial["completed_jobs"]) == 1
    first_receipts = snapshot(resumed_root)
    resumed = campaign.run_campaign(resumed_root)
    full = campaign.run_campaign(full_root)
    assert scientific(resumed) == scientific(full)
    assert snapshot(resumed_root, include_runtime=False) == snapshot(
        full_root, include_runtime=False
    )
    assert all(
        (resumed_root / path).read_bytes() == value
        for path, value in first_receipts.items()
    )
    assert resumed["complete"] and resumed["missing_checkpoints"] == []
    accounting = resumed["execution_accounting"]
    assert accounting["actual_total_execution_cost"] == resumed["logical_cost"]
    assert accounting["unfinished_attempts"] == []
    assert accounting["actual_total_elapsed_seconds"] >= 0


def test_completed_resume_and_report_do_not_recompute(
    tmp_path, small_roster, monkeypatch
):
    initial = campaign.run_campaign(tmp_path)
    before = snapshot(tmp_path)
    monkeypatch.setattr(campaign.core, "initialize", forbidden)
    monkeypatch.setattr(campaign.core, "advance", forbidden)
    assert campaign.run_campaign(tmp_path) == initial
    assert campaign.report_campaign(tmp_path) == initial
    assert snapshot(tmp_path) == before


def test_failed_attempt_preserves_observed_work_and_deterministic_resume(
    tmp_path, small_roster, monkeypatch
):
    resumed_root, full_root = tmp_path / "resumed", tmp_path / "full"
    original = campaign.core.advance

    def interrupted(problem, state, budget):
        original(problem, state, budget)
        raise RuntimeError("after work, before checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(campaign.core, "advance", interrupted)
        with pytest.raises(RuntimeError, match="after work"):
            campaign.run_campaign(resumed_root, max_new_checkpoints=1)
    partial = campaign.report_campaign(resumed_root)
    assert partial["recorded_checkpoints"] == 0
    assert partial["execution_accounting"]["interrupted_attempts"] == ["000:000"]
    assert partial["execution_accounting"]["observed_execution_cost"]["boxes"] > 0
    assert partial["logical_cost"]["boxes"] == 0
    resumed = campaign.run_campaign(resumed_root)
    full = campaign.run_campaign(full_root)
    assert scientific(resumed) == scientific(full)
    assert snapshot(resumed_root, include_runtime=False) == snapshot(
        full_root, include_runtime=False
    )
    assert (
        resumed["execution_accounting"]["observed_execution_cost"]["boxes"]
        > resumed["logical_cost"]["boxes"]
    )


def test_unfinished_attempt_cannot_be_reported_as_zero_cost(tmp_path, small_roster):
    campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    (tmp_path / "runtime/attempt-000-000-finish.json").unlink()
    before = snapshot(tmp_path)
    report = campaign.report_campaign(tmp_path)
    accounting = report["execution_accounting"]
    assert accounting["unfinished_attempts"] == ["000:000"]
    assert accounting["actual_total_elapsed_seconds"] is None
    assert accounting["actual_total_execution_cost"] is None
    assert accounting["missing_completed_checkpoint_timings"] == [0]
    assert report["recorded_checkpoints"] == 1
    assert snapshot(tmp_path) == before
    resumed = campaign.run_campaign(tmp_path)
    assert resumed["complete"]
    assert resumed["execution_accounting"]["actual_total_execution_cost"] is None


def test_missing_all_timing_receipts_remains_unknown(tmp_path, small_roster):
    campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    for path in (tmp_path / "runtime").iterdir():
        path.unlink()
    accounting = campaign.report_campaign(tmp_path)["execution_accounting"]
    assert accounting["missing_completed_checkpoint_timings"] == [0]
    assert accounting["actual_total_elapsed_seconds"] is None
    assert accounting["actual_total_execution_cost"] is None


@pytest.mark.parametrize("kind", ["digest", "hole", "unexpected"])
def test_corrupted_or_noncontiguous_checkpoints_fail_closed(
    tmp_path, small_roster, kind
):
    campaign.run_campaign(tmp_path, max_new_checkpoints=2)
    path = tmp_path / "checkpoint-000.json"
    if kind == "digest":
        value = json.loads(path.read_text())
        value["value"]["budget"] = 9
        path.write_text(json.dumps(value))
    elif kind == "hole":
        path.unlink()
    else:
        (tmp_path / "checkpoint-unexpected.json").write_bytes(path.read_bytes())
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        campaign.report_campaign(tmp_path)
    with pytest.raises(ValueError):
        campaign.run_campaign(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["method", "problem", "chain", "report"])
def test_resealed_semantic_checkpoint_changes_are_rejected(
    tmp_path, small_roster, kind
):
    campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    path = tmp_path / "checkpoint-000.json"
    value = campaign.persistence._read(path)
    if kind == "method":
        value["state"]["method"] = "direct"
    elif kind == "problem":
        value["state"]["problem"]["observations"][0] = "999"
    elif kind == "chain":
        value["previous_sha256"] = "wrong"
    else:
        value["report"]["frontier_cells"] += 1
    path.unlink()
    campaign.persistence._seal(path, value)
    with pytest.raises(ValueError):
        campaign.report_campaign(tmp_path)


def test_stale_summary_is_rejected_without_search(tmp_path, small_roster, monkeypatch):
    campaign.run_campaign(tmp_path)
    (tmp_path / "checkpoint-003.json").unlink()
    monkeypatch.setattr(campaign.core, "initialize", forbidden)
    monkeypatch.setattr(campaign.core, "advance", forbidden)
    with pytest.raises(ValueError):
        campaign.report_campaign(tmp_path)
    with pytest.raises(ValueError):
        campaign.run_campaign(tmp_path)


@pytest.mark.parametrize("field", ["source_sha256", "runtime"])
def test_changed_source_or_runtime_rejects_resume(
    tmp_path, small_roster, monkeypatch, field
):
    campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    original = campaign._plan

    def changed():
        plan = original()
        plan[field] = {"changed": True}
        return plan

    before = snapshot(tmp_path)
    monkeypatch.setattr(campaign, "_plan", changed)
    with pytest.raises(ValueError):
        campaign.run_campaign(tmp_path)
    with pytest.raises(ValueError):
        campaign.report_campaign(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_checkpoint_allowance_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        campaign.run_campaign(tmp_path, max_new_checkpoints=value)


def test_real_roster_profiled_and_direct_first_budget_replay(tmp_path):
    campaign.run_campaign(tmp_path, max_new_checkpoints=1)
    report = campaign.run_campaign(tmp_path, max_new_checkpoints=4)
    assert report["recorded_checkpoints"] == 5
    assert report["expected_checkpoints"] == 32
    problem = campaign.core.problems()[0]
    for index, method in ((0, "profiled"), (4, "direct")):
        expected = campaign.core.initialize(problem, method)
        campaign.core.advance(problem, expected, 64)
        recorded = campaign.persistence._read(tmp_path / f"checkpoint-{index:03d}.json")
        assert recorded["state"] == expected
        assert recorded["report"] == campaign.core.summarize(
            problem, copy.deepcopy(expected)
        )
