"""Routing, construction-wide budget gates, cached resume and actual compilation."""

import json

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.search import process_revision_runner as runner
from scripts import smoke_process_revisions as smoke


def args():
    brief, context, source = smoke.fixture()
    return brief, context, source, "term_0_0", {}


def test_real_compile_and_cached_topology_repair(tmp_path):
    assert smoke.run(tmp_path)["status"] == "passed"


def test_routing_uses_remaining_budget_instead_of_granting_new_attempts(tmp_path):
    calls = []
    c = smoke.client(
        tmp_path,
        calls,
        [{"expression": "u", "parameters": [], "revise_dependencies": True}],
    )
    result = runner.run(
        *args(),
        c,
        tmp_path / "repair",
        initial_error="dependency revision required",
        attempt_offset=1,
    )
    assert result["status"] == "repair_exhausted"
    assert result["attempts_consumed"] == 2
    assert [e["scope"] for e in result["events"]] == ["local", "topology"]
    assert all(e["error_category"] == "public_pathway" for e in result["events"])
    assert len(calls) == 2
    assert not calls[0]["ordinary_companion_slots"]
    assert "term_1_0" in calls[1]["ordinary_companion_slots"]
    assert "term_1_1" not in calls[1]["ordinary_companion_slots"]
    assert (
        runner.run(
            *args(),
            c,
            tmp_path / "repair",
            initial_error="dependency revision required",
            attempt_offset=1,
        )
        == result
    )
    assert len(calls) == 2


def test_deferred_resume_reuses_first_rejection(tmp_path):
    calls = []
    replies = [
        {"expression": "u", "parameters": [], "revise_dependencies": True},
        {"expression": "x+u+crest", "parameters": []},
    ]
    c = smoke.client(tmp_path, calls, replies, can_start=lambda: len(calls) < 1)
    with pytest.raises(DeferredCall):
        runner.run(
            *args(),
            c,
            tmp_path / "repair",
            initial_error="dependency revision required",
        )
    progress = json.loads((tmp_path / "repair/progress.json").read_text())
    assert len(progress["events"]) == 1
    c = smoke.client(tmp_path, calls, replies)
    result = runner.run(
        *args(), c, tmp_path / "repair", initial_error="dependency revision required"
    )
    assert result["status"] == "repaired"
    assert len(calls) == 2


def test_no_remaining_attempts_and_manifest_drift(tmp_path):
    c = smoke.client(tmp_path, [], [{}])
    result = runner.run(
        *args(), c, tmp_path / "repair", initial_error="x", attempt_offset=3
    )
    assert result["attempts_consumed"] == 0
    assert result["status"] == "repair_exhausted"
    with pytest.raises(ValueError, match="frozen artifact differs"):
        runner.run(*args(), c, tmp_path / "repair", initial_error="x", attempt_offset=0)


def test_unsafe_expression_is_recorded_and_repaired(tmp_path):
    calls = []
    c = smoke.client(
        tmp_path,
        calls,
        [
            {"expression": "__import__('os')", "parameters": []},
            {"expression": "x+u+crest", "parameters": []},
        ],
    )
    result = runner.run(*args(), c, tmp_path / "repair", initial_error="syntax")
    assert result["status"] == "repaired"
    assert not result["events"][0]["accepted"]
    assert len(calls) == 2


def test_existing_construction_call_cap_is_not_reset(tmp_path):
    c = smoke.client(tmp_path, [], [{}])
    c.settings = c.settings.model_copy(update={"maximum_requests": 1})
    c.records.append({"budget_charge": 0})
    with pytest.raises(ValueError, match="total provider request budget"):
        runner.run(
            *args(),
            c,
            tmp_path / "repair",
            initial_error="dependency revision required",
        )
