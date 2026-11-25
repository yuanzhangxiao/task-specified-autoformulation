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
            "cohort": {
                "rows": [{"case": "case"}],
                "cases": {
                    "case": {
                        "public_brief": {
                            "public_variables": [{"name": "x", "data_role": "target"}],
                            "scientific_context": (
                                "Target channels:\n- x(t): nonnegative target"
                            ),
                        }
                    }
                },
            },
            "protocol": "previous",
        },
    )
    monkeypatch.setattr(refit, "PARENT", baseline["artifact_sha256"])
    monkeypatch.setattr(refit.original, "implementation", lambda: deepcopy(identity))
    plan = refit.prepare(root, parent)
    assert plan["slots"] == baseline["slots"]
    assert plan["fit_budget"] == baseline["fit_budget"]
    assert plan["parent_plan"] == baseline["artifact_sha256"]
    assert plan["constraint"]["targets"] == ["x"]
    assert (
        plan["constraint"]["observed_target_bounds"]["x"]["basis"] == "explicit_public"
    )
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

    def fit(row, slot, budget, targets):
        assert targets == ["from_plan"]
        calls.append(slot["fit_key"])
        return {"status": "fitted", "training_nmse": 0.1}

    monkeypatch.setattr(refit, "_fit", fit)
    slot = {"fit_key": "a", "formula": {"eq": "x"}}
    run = {
        "root": tmp_path,
        "digest": "p",
        "budget": {"starts": 2},
        "targets": ["from_plan"],
    }
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


def test_completed_v1_archive_is_verifiable_without_rerunning(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    slot = {"fit_key": "a"}
    plan = sealed_write(root / "plan.json", {"slots": [slot]})["artifact_sha256"]
    monkeypatch.setattr(refit, "LEGACY_PLAN", plan)
    receipt = sealed_write(
        root / "fits/a.json",
        {
            "plan": plan,
            "slot_sha256": refit.content_hash(slot),
            "result": {"status": "fitted"},
        },
    )
    frozen = sealed_write(
        root / "frozen.json",
        {"plan": plan, "fit_receipts": {"a": receipt["artifact_sha256"]}},
    )
    sealed_write(root / "development.json", {"freeze": frozen["artifact_sha256"]})
    assert refit.status(root)["verified_fits"] == 1
    with pytest.raises(ValueError, match="read-only"):
        refit.prepare(root, tmp_path / "no-current-runtime-or-parent-needed")
    (root / "fits/a.json").unlink()
    with pytest.raises(ValueError, match="frozen fit receipts"):
        refit.status(root)


def test_only_inherited_targets_restrict_the_numerical_fit(monkeypatch):
    graph = {
        "initializers": {
            k: {"mode": "observed_initial", "channel": k} for k in ["mass", "change"]
        },
        "variable_types": {"mass": "differential", "change": "differential"},
        "sources": {"mass": [], "change": []},
    }
    train = {
        "trajectory_id": "train_00",
        "time": [0, 1, 2],
        "targets": {"mass": [1, 2, 3], "change": [0, -1, -2]},
        "covariates": {},
        "inputs": {},
        "auxiliaries": {},
    }
    monkeypatch.setattr(
        refit.adapter, "_training", lambda *_: {"evidence": {"trajectories": [train]}}
    )
    result = refit._fit(
        {"graph": graph, "case": "case", "dataset": "data"},
        {
            "formula": {
                "equations": {"mass": "1", "change": "-1"},
                "initializers": {},
                "parameters": {},
            },
            "seed": 0,
        },
        {"starts": 1, "max_evaluations": 4, "wall_seconds": 30},
        ["mass"],
    )
    assert result["status"] == "fitted"
    assert result["constrained_targets"] == ["mass"]
    assert result["negative_samples"] == {"mass": 0, "change": 2}
