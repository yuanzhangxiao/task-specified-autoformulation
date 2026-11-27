"""Training-only policy inputs, family evidence and frozen held-out replay."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import compare_evidence_decisions as run


def test_check_cannot_read_fit_quality_and_tension_only_refutes_precision():
    assert run.observation({"certified_reject": True}, "check", "forecast") == "tension"
    assert run.observation({"certified_reject": True}, "check", "precision") == "reject"
    candidate = {"raw_selection_loss": 0.004, "selection_loss": None}
    assert run.observation(candidate, "fit", "forecast") == "2"
    assert run.observation(candidate, "fit", "precision") == "failed"
    assert run.loss_bin(0.1) == "3"
    with pytest.raises(ValueError):
        run.loss_bin(float("inf"))


def test_transfer_preserves_joint_patterns_and_all_novel_legal_outcomes():
    example = {
        "candidates": {
            "simple": {"certified_reject": True, "selection_loss": None},
            "extended": {"certified_reject": False, "selection_loss": 0.00001},
        }
    }
    worlds = run.build_scenarios([example], "precision", 0.75)
    assert sum(w.weight for w in worlds) == pytest.approx(1)
    assert all(w.weight > 0 for w in worlds)
    observed = {
        "check_simple": "reject",
        "fit_simple": "failed",
        "check_extended": "open",
        "fit_extended": "0",
    }
    assert sum(w.weight for w in worlds if w.outcomes == observed) > 0.75
    possible = {
        tuple(w.outcomes[k] for k in ("check_simple", "fit_simple")) for w in worlds
    }
    assert ("open", "failed") in possible and ("open", "6") in possible
    assert ("reject", "0") not in possible


def test_validate_after_choices_and_account_for_abstentions(tmp_path, monkeypatch):
    checked = []

    def validation(panel, model):
        checked.append((panel, model))
        return {"fit_receipt": "f1", "status": "qualified", "nmse": 0.125}

    monkeypatch.setattr(run.data, "validation", validation)
    plan = sealed_write(tmp_path / "plan.json", {"design": run.design()})
    with pytest.raises(FileNotFoundError):
        run.evaluate(tmp_path)
    assert not checked
    outcome = {
        "model": "simple",
        "endpoint": "f1",
        "checks": 0,
        "fits": 1,
        "cost_units": 4,
        "optimizer_evaluations_replayed": 25,
        "selection_nmse": 0.1,
    }
    rec = sealed_write(
        tmp_path / "decisions/example.json",
        {
            "plan": plan["artifact_sha256"],
            "panel": "basin/a",
            "benchmark": "basin",
            "mode": "forecast",
            "budget": 6,
            "transfer": 0.0,
            "decisions": {
                "first": outcome,
                "second": outcome | {"model": None, "endpoint": None},
            },
        },
    )
    sealed_write(
        tmp_path / "choices.json",
        {
            "plan": plan["artifact_sha256"],
            "receipts": [["example", rec["artifact_sha256"]]],
        },
    )
    result = run.evaluate(tmp_path)
    assert result == run.evaluate(tmp_path)
    a, b = result["rows"]
    assert a["validation_nmse"] == 0.125 and a["loss_with_cost"] == 0.245
    assert b["validation_status"] == "abstained" and b["terminal_loss"] == 1
    assert checked == [("basin/a", "simple")]
