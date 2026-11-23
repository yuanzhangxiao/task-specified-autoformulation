"""Replay verifies the old calls and never attributes a route to the earlier model."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import replay_trajectory_function_revisions as replay
from scripts import trajectory_functions as trial
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_trajectory_function_routing_campaign import (  # noqa: F401
    candidate,
    inputs,
)


def test_replay_checks_original_receipts_and_spends_no_requests(inputs, monkeypatch):  # noqa: F811
    original, train, _ = inputs
    old_plan = trial.prepare(original, train)
    calls = 0

    def old_provider(url, body, timeout):
        nonlocal calls
        calls += 1
        if "shortlist" in json.loads(body["messages"][1]["content"]):
            return reply({"candidate": {"shortlist_id": "A"}})
        wire = candidate()["candidate"]
        del wire["scientific_reason"], wire["route"]
        return reply({"candidate": wire})

    trial.run(
        original, "http://localhost", transport=old_provider, token_transport=tokenizer
    )
    assert calls == 8
    root = original.parent / "replay"
    with pytest.raises(ValueError, match="frozen ACES"):
        replay.replay(original, root)
    monkeypatch.setattr(replay, "ARCHIVED_PLAN", old_plan["artifact_sha256"])
    gen = sealed_read(original / "generation.json")
    monkeypatch.setattr(replay, "ARCHIVED_GENERATION", gen["artifact_sha256"])
    result = replay.replay(original, root)
    assert result["generation_complete"] and result["physical_requests"] == 0
    assert replay.replay(original, root) == result
    routed = sealed_read(root / "candidates/000.json")
    assert (
        routed["status"] == "eligible"
        and routed["archived_status"] == "invalid_formula"
    )
    assert routed["reason_source"] == "retrospective_audit"
    assert "revised_graph" in routed
    assert sealed_read(root / "candidates/004.json")["route"] == "fixed_graph"
    with pytest.raises(ValueError, match="no generation allowance"):
        trial.run(root, "http://localhost", transport=old_provider)
    assert calls == 8
    (physical,) = (original / "steps/slot-000/calls").glob("*.json")
    physical.write_text(physical.read_text() + "\n")
    with pytest.raises(ValueError, match="generation receipt"):
        replay.replay(original, root)
