"""The public construction-to-feedback edge; historical timeout logs are synthetic."""

from __future__ import annotations

import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import prefit_numerical_sibling as sibling
from scripts.smoke_prefit_numerical_sibling import client_for, fixture, smoke


def test_real_feedback_to_child_fit_and_deterministic_resume(tmp_path):
    result = smoke(tmp_path)
    assert result["status"] == "passed"
    assert result["real_frozen_child_fitting"] and result["resume_unchanged"]


@pytest.mark.parametrize("action", ["no_change", "topology_revision_needed"])
def test_unresolved_evidence_may_end_without_child(tmp_path, monkeypatch, action):
    root = fixture(tmp_path)
    assert sibling.replay(root)["status"] == "ready"
    plan, residual = sibling.verify(root)
    calls = []
    client = client_for(root, plan, calls, action=action)
    state = sibling.run_episode(root, plan, residual, client)
    assert state["stop_reason"] == action
    monkeypatch.setattr(
        public, "_run_backend", lambda *args: pytest.fail("no child permitted")
    )
    result = sibling.fit_child(root)
    assert result["status"] == "complete" and result["child"] is None
    assert not (root / "child_fit").exists()
    assert sibling.run_episode(root, plan, residual, client) == state
    assert len(calls) == 1
    payload = json.loads(calls[0]["messages"][1]["content"].split("\n", 1)[1])
    assert (
        payload["retained_fitted_parameters"] == residual["seed"]["state"]["parameters"]
    )
    assert (
        "validation" not in payload
        and "seed" not in payload
        and "parent" not in payload
    )
    packet = payload["training_evidence"]
    assert packet["split"] == "train"
    assert all(r["trajectory_id"].startswith("train") for r in packet["rows"])
    assert not packet["numerical_status"]["structural_failure_established"]


def test_consumed_replay_blocks_llm_and_fit(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    frozen = public._read(root / "evidence/freeze.json")
    public._write(root / "evidence/started.json", {"identity": frozen["identity"]})
    assert sibling.replay(root)["status"] == "interrupted"
    plan, residual = sibling.verify(root)
    calls = []
    client = client_for(root, plan, calls)
    state = sibling.run_episode(root, plan, residual, client)
    assert state["stop_reason"] == "replay_unavailable" and calls == []
    monkeypatch.setattr(
        public, "_run_backend", lambda *args: pytest.fail("no new budget")
    )
    assert sibling.fit_child(root)["child"] is None


def test_report_rejects_valid_child_from_different_episode(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    sibling.replay(root)
    plan, residual = sibling.verify(root)
    sibling.run_episode(root, plan, residual, client_for(root, plan, []))
    # Prepare a valid child without spending its numerical allocation.
    monkeypatch.setattr(sibling, "execute_child_fit", lambda *args: None)
    assert sibling.fit_child(root)["status"] == "partial"
    old, request, train, val = sibling_fit._load(root / "child_fit")
    foreign = sibling_fit._freeze(
        public.PublicFitRequest.model_validate(old["parent_request"]),
        request,
        old["parent_parameters"],
        train,
        val,
        {"different_episode": "sealed_elsewhere"},
    )
    public._write(root / "child_fit/freeze.json", foreign)
    assert sibling_fit.inspect_child_fit(root / "child_fit")["result"] is None
    with pytest.raises(ValueError, match=r"child|decision|lineage"):
        sibling.report(root)
