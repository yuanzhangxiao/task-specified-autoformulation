"""Check the order experiment's fit constraints, derivatives and resume boundaries."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import fit_t2_initial_disposal_order as trial
from tests.test_validate_positive_insulin_transfer import toy


def candidate(monkeypatch):
    row, model = toy("train_000")
    row.auxiliaries["Uii"] = [1.0] * 4
    row.auxiliaries["Gt"] = [0.0] * 4
    row.targets["U"] = [6.0, 15.0, 15.0, 15.0]
    two = deepcopy(vars(row))
    two["trajectory_id"] = "train_001"
    two["auxiliaries"]["Gt"] = [1.0] * 4
    two["targets"]["U"] = [8.0, 5.0, 5.0, 5.0]
    f = model["formula"]
    f["equations"]["U"] = "offset+gain*Gt+afcoef13*I"
    model["graph"]["sources"]["U"] = ["Gt", "I"]
    f["parameters"].update(offset=10.0, gain=-2.0, afcoef13=0.001)
    model_id = trial.content_hash([model["graph"], f["equations"], f["initializers"]])
    monkeypatch.setattr(trial.base, "GRAPHS", {model_id})
    return model, [vars(row), two]


@pytest.mark.parametrize(
    "arm,bound", [("control", None), ("small_gap", 0.02), ("half_gap", 1.0)]
)
def test_matches_trajectory_objective_and_margin(monkeypatch, arm, bound):
    model, rows = candidate(monkeypatch)
    result = trial.fit(model, rows, arm, 40, "warm")
    assert result["solver_success"] and result["forward_evaluations"] <= 40
    gain = result["parameters"]["gain"]
    if bound is None:
        assert gain < 0
    else:
        assert gain == pytest.approx(bound, abs=1e-8)
    assert result["parameters"]["afcoef13"] == 0.001


def test_infeasible_start_does_not_enter_constrained_result(monkeypatch):
    model, rows = candidate(monkeypatch)
    result = trial.fit(model, rows, "half_gap", 1, "warm")
    assert result["parameters"] is None
    assert not result["solver_success"] and result["forward_evaluations"] == 1
    rows[-1]["trajectory_id"] = "val_001"
    with pytest.raises(ValueError):
        trial.fit(model, rows, "control", 40, "warm")


def test_order_pairs_omit_ties():
    high, low = trial.order_pairs(np.array([1.0, 3.0, 3.0, 0.0]))
    assert list(zip(high, low, strict=True)) == [(0, 3), (1, 0), (1, 3), (2, 0), (2, 3)]
    with pytest.raises(ValueError):
        trial.order_pairs(np.array([np.nan]))


def test_immutable_receipts_resume_without_reoptimizing(tmp_path, monkeypatch):
    dummy_training = tmp_path / "train.json"
    dummy_training.write_text("[]")
    monkeypatch.setattr(trial.coupled, "INPUTS", tmp_path)
    monkeypatch.setattr(trial.coupled.data, "load_training", lambda _: [])
    freeze = {
        "artifact_sha256": trial.FREEZE,
        "validation_opened_at_freeze": False,
        "models": {"delayed_refit": {"id": "source"}},
    }
    monkeypatch.setattr(
        trial, "sealed_read", lambda p: freeze if p == trial.SOURCE else sealed_read(p)
    )
    calls = []

    def fitted(_, rows, arm, budget, start):
        calls.append((arm, start))
        return {"parameters": {"slope": 1}, "training_nmse": 1.25}

    monkeypatch.setattr(trial, "fit", fitted)
    monkeypatch.setattr(
        trial,
        "qualify",
        lambda m, r, a, record, root, k: record | {"eligible": True, "pooled": 1.25},
    )
    path = tmp_path / "experiment"
    receipt = trial.run(path, budget=2)
    assert receipt["arms"]["half_gap"]["start"] == "warm"
    assert receipt == trial.run(path, budget=2)
    assert len(calls) == 6
    assert not receipt["validation_opened"] and not receipt["test_opened"]
    with pytest.raises(ValueError, match="frozen artifact"):
        trial.run(path, budget=3)
