"""The public construction-to-feedback edge; historical timeout logs are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import prefit_numerical_sibling as sibling
from scripts.smoke_prefit_numerical_sibling import client_for, fixture, smoke


@pytest.mark.parametrize("policy", ["optional-review-1", "routed-hypothesis-2"])
def test_real_feedback_to_child_fit_and_deterministic_resume(tmp_path, policy):
    result = smoke(tmp_path, feedback_policy=policy)
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


def test_citation_audit_is_read_only_and_does_not_resume_old_campaign(tmp_path):
    root = fixture(tmp_path)
    sibling.replay(root)
    plan, residual = sibling.verify(root)
    calls = []
    sibling.run_episode(
        root, plan, residual, client_for(root, plan, calls, action="no_change")
    )
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    audit = sibling.audit_saved_citations(root)
    assert audit["attempts"][0]["routed-hypothesis-2"]["citation_contract_valid"]
    assert audit["live_llm_calls"] == 0
    assert {p: p.read_bytes() for p in original} == original
    assert sibling.audit_saved_citations(root) == audit
    path = root / "evidence/result.json"
    envelope = json.loads(path.read_text())
    envelope["result"]["packet"]["normalized_mse"] += 1
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest"):
        sibling.audit_saved_citations(root)


def test_policy_allocations_are_separate_and_cannot_be_renewed_by_path(tmp_path):
    from autoformalism.schemas.residual_feedback import FeedbackSelection

    selection = FeedbackSelection()
    source = tmp_path / "parent/fit"
    config = sibling.SiblingConfig.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "configs/prefit_numerical_sibling_v2.json"
        ).read_text()
    )
    old = config.model_copy(update={"feedback_policy": "optional-review-1"})
    sibling._reserve(tmp_path / "v1", source, old)
    sibling._reserve(tmp_path / "v2", source, config)
    sibling._reserve(tmp_path / "v2", source, config)
    assert sibling._reservation_directory(
        source, selection, old.feedback_policy
    ) != sibling._reservation_directory(source, selection, config.feedback_policy)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        sibling._reserve(tmp_path / "v2-extra", source, config)
