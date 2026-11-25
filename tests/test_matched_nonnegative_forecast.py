"""Verify the matched refit uses the original slots, budget and freeze order."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import matched_nonnegative_forecast as refit


def test_prepare_preserves_formula_and_refuses_changed_baseline(tmp_path, monkeypatch):
    parent, root = tmp_path / "old", tmp_path / "new"
    identity = {"sources": {"model": "fixed"}, "versions": {}, "source_package": "old"}
    baseline = sealed_write(
        parent / "plan.json",
        {
            "implementation": identity,
            "fit_budget": {"starts": 2},
            "slots": [{"fit_key": "first", "formula": {"eq": "a"}}],
            "cohort": {"rows": []},
            "protocol": "previous",
        },
    )
    monkeypatch.setattr(refit, "PARENT", baseline["artifact_sha256"])
    monkeypatch.setattr(refit.original, "implementation", lambda: deepcopy(identity))
    plan = refit.prepare(root, parent)
    assert plan["slots"] == baseline["slots"]
    assert plan["fit_budget"] == baseline["fit_budget"]
    assert plan["parent_plan"] == baseline["artifact_sha256"]
    assert (
        sealed_read(root / "eligibility-rule.json")["plan"] == plan["artifact_sha256"]
    )
    assert refit.prepare(root, parent) == plan
    with pytest.raises(ValueError, match="separate results"):
        refit.prepare(parent, parent)
    monkeypatch.setattr(refit, "PARENT", "different")
    with pytest.raises(ValueError, match="registered"):
        refit.prepare(root, parent)


def test_slot_charges_only_once_and_reconciles_interruption(tmp_path, monkeypatch):
    calls = []

    def fit(row, slot, budget):
        calls.append(slot["fit_key"])
        return {"status": "fitted", "training_nmse": 0.1}

    monkeypatch.setattr(refit, "_fit", fit)
    slot = {"fit_key": "a", "formula": {"eq": "x"}}
    run = {"root": tmp_path, "digest": "p", "budget": {"starts": 2}}
    refit._slot(({}, slot), **run)
    refit._slot(({}, slot), **run)
    assert calls == ["a"]
    slot = {**slot, "fit_key": "b"}
    sealed_write(
        tmp_path / "reservations/b/start.json",
        {"plan": "p", "slot_sha256": refit.content_hash(slot)},
    )
    refit._slot(({}, slot), **run)
    assert sealed_read(tmp_path / "fits/b.json")["result"]["status"] == "interrupted"
    assert calls == ["a"]


def test_development_precondition_blocks_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(refit.original, "development", lambda *a: pytest.fail("Scored"))
    sealed_write(tmp_path / "frozen.json", {"fit_receipts": {"a": "old"}})
    sealed_write(
        tmp_path / "eligibility-freeze.json",
        {"fit_receipts": {"a": "other"}, "plan": "p"},
    )
    with pytest.raises(ValueError, match="share all"):
        refit.forecast(tmp_path, {"artifact_sha256": "p"})
