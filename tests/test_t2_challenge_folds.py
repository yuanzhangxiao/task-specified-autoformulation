from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import t2_challenge_folds as trial


def event(i):
    return {
        "trajectory_id": f"train_{i:03}",
        "time": [0.0, 1.0, 2.0],
        "targets": {"I": [2.0, 2.0 + i, 2.0 + 2 * i]},
        "auxiliaries": {"Gt": [4.0 + i] * 3},
        "inputs": {trial.family.DRUG: [0.0] * 3, trial.family.MEAL: [0.0] * 3},
    }


def test_contiguous_whole_event_blocks_cover_all_challenges_once():
    data = [event(i) for i in reversed(range(16))]
    splits = trial.folds(data)
    assert splits[0]["held"] == [f"train_{i:03}" for i in range(1, 6)]
    assert (
        sum(len(s["held"]) for s in splits)
        == len(set().union(*(s["held"] for s in splits)))
        == 15
    )
    for s in splits:
        assert trial.ANCHOR in s["fit"] and len(s["fit"]) == 11
        assert set(s["fit"]).isdisjoint(s["held"])
    with pytest.raises(ValueError, match="training"):
        trial.folds([*data, event(2)])
    assert trial.fasting_reference(data)["Gt"] == 4
    assert trial.fasting_reference([event(1)]) is None
    active = event(0)
    active["inputs"][trial.family.DRUG][1] = 0.1
    assert trial.fasting_reference([active]) is None


def test_old_fitted_parameters_and_held_out_event_cannot_set_starts():
    data = [event(i) for i in range(7)]
    graph = {"variable_types": {"I": "differential"}}
    form = {
        "parameters": {"afcoef1": 1000, "afcoef2": -500},
        "equations": {"I": "afcoef1+afcoef2*Gt"},
    }
    split = trial.folds(data)[0]

    def subset():
        return [r for r in data if r["trajectory_id"] in split["fit"]]

    initial = trial.cold_parameters(graph, form, subset())
    assert initial["afcoef2"] == pytest.approx(1.0, abs=0.001)
    assert initial["afcoef1"] == pytest.approx(-4.0, abs=0.001)
    data[1]["targets"]["I"] = [1e6] * 3
    form["parameters"] = {"afcoef1": -77.0, "afcoef2": 12.0}
    assert trial.cold_parameters(graph, form, subset()) == initial


def test_pooling_weights_samples_and_reports_incomplete_cohort():
    def result(n, obs, pred):
        return trial.event_metrics(
            {
                "trajectory_id": str(n),
                "time": list(range(n)),
                "observed": {"I": [obs] * n},
                "predicted": {"I": [pred] * n},
            },
            {"I": 2},
        )

    first, second = result(2, 10, 0), result(4, 10, 8)
    x = trial.aggregate([first, second], 2)
    assert x["pooled_nmse"] == pytest.approx((2 * 25 + 4) / 6)
    assert x["available_rmse"]["I"] == pytest.approx(np.sqrt((2 * 100 + 4 * 4) / 6))
    assert x["near_zero_I_with_observed_above_one"] == 2
    partial = trial.aggregate(
        [first, {"event": "unfit", "status": "no_verified_fit"}], 2
    )
    assert partial["pooled_nmse"] is None
    assert partial["available_pooled_nmse"] == 25
    with pytest.raises(ValueError, match="Duplicate"):
        trial.aggregate([first, first], 2)


def test_no_fasting_reference_abstains_for_all_arms_and_resumes(monkeypatch, tmp_path):
    data = [event(i) for i in range(1, 7)]
    monkeypatch.setattr(
        trial, "_sources", lambda _: (data, {k: ({}, {}, "id") for k in trial.SLOTS})
    )
    monkeypatch.setattr(trial.original, "_code", lambda: {})
    monkeypatch.setattr(
        trial.original, "_fit", lambda *_: pytest.fail("Fitting without fasting anchor")
    )
    root = tmp_path / "study"
    plan, rows = trial.prepare(Path("source"), root)
    assert len(plan["models"]) == 12
    assert trial.report(root)["status"] == "incomplete"
    for i in range(12):
        one = trial.run_one(root, plan, rows, i)
        assert one["status"] == "no_qualified_fasting_reference"
        assert trial.run_one(root, plan, rows, i) == one
    report = trial.report(root)
    assert report["status"] == "complete" and report["evaluations"] == 0
    assert report["groups"]["2"]["restoring"]["failed_events"] == 6
    assert trial.prepare(Path("source"), root)[0] == plan


def test_fit_receives_only_its_training_ids(monkeypatch, tmp_path):
    data = [event(i) for i in range(7)]
    s = trial.folds(data)[0]
    plan = sealed_write(
        tmp_path / "plan.json",
        {
            "models": [{"status": "prepared", "slot": 2, "arm": "bounded", "fold": 0}],
            "folds": [s],
        },
    )

    def fit(root, plan, index, rows):
        assert [r["trajectory_id"] for r in rows] == s["fit"]
        return {
            "artifact_sha256": "a",
            "status": "assessed",
            "evaluations": 1,
            "selected_option": None,
            "options": [],
        }

    monkeypatch.setattr(trial.original, "_fit", fit)
    data = {r["trajectory_id"]: r for r in data}
    r = trial.run_one(tmp_path, plan, data, 0)
    assert {e["event"] for e in r["events"]} == set(s["held"])
    monkeypatch.setattr(
        trial.original, "_fit", lambda *_: pytest.fail("Repeated fitting")
    )
    assert trial.run_one(tmp_path, plan, data, 0) == r
