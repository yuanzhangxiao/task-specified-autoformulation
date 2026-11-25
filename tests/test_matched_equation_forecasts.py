"""Check the matched estimator, disjoint event scores, and resume contracts."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import matched_equation_forecasts as study

GRAPH = {
    "variable_types": {"q": "differential"},
    "sources": {"q": ["v"]},
    "initializers": {"q": {"mode": "observed_initial", "channel": "q"}},
}
FORM = {"equations": {"q": "v"}, "initializers": {}, "parameters": {}}


def event(j):
    return {
        "trajectory_id": f"train_{j}",
        "time": [0, 1, 2],
        "targets": {"q": [1, 2, 3]},
        "inputs": {"v": [1, 1, 1]},
        "auxiliaries": {},
        "covariates": {},
    }


def test_fit_replay_and_disjoint_scoring(tmp_path, monkeypatch):
    kwargs = {"panel_id": "g0_odd", "arm": "a", "graph": GRAPH}
    fit_args = {
        **kwargs,
        "train_rows": [event(0)],
        "formulas": [FORM, None],
        "nonnegative_targets": ["q"],
    }
    result = study.fit_arm(tmp_path, **fit_args)
    assert (
        result["fit_statuses"] == ["fitted", "no_formula"]
        and result["selected_slot"] == 0
    )
    monkeypatch.setattr(study, "_fit_one", lambda *a: pytest.fail("Repeated fit"))
    assert study.fit_arm(tmp_path, **fit_args) == result
    with pytest.raises(ValueError, match="overlap"):
        study.score_arm(tmp_path, **kwargs, held_rows=[event(0)])
    score = study.score_arm(tmp_path, **kwargs, held_rows=[event(1)])
    assert score["status"] == "scored"
    assert score["pooled_nmse"] == pytest.approx(0)
    assert score["targets"]["q"]["negative_samples"] == 0
    assert score["targets"]["q"]["comparable_peaks"] == 0
    assert (
        score["events"][0]["targets"]["q"]["observed_peak"]["status"]
        == "right_boundary"
    )
    assert study.score_arm(tmp_path, **kwargs, held_rows=[event(1)]) == score
    changed = event(1)
    changed["targets"]["q"][-1] = 3.01
    with pytest.raises(ValueError, match="Different held events"):
        study.score_arm(tmp_path, **kwargs, held_rows=[changed])
    kwargs["arm"] = "b"
    with pytest.raises(ValueError, match="frozen artifact differs"):
        study.fit_arm(tmp_path, **{**fit_args, "arm": "b", "train_rows": [event(2)]})
    with pytest.raises(ValueError, match="frozen artifact differs"):
        study.fit_arm(tmp_path, **{**fit_args, "nonnegative_targets": []})


def test_invalid_candidates_and_missing_fits_do_not_enter_scores(tmp_path):
    formula = deepcopy(FORM)
    formula["equations"]["q"] = "missing"
    kwargs = {"panel_id": "graph", "arm": "arm", "graph": GRAPH}
    selected = study.fit_arm(
        tmp_path,
        **kwargs,
        train_rows=[event(0)],
        formulas=[formula, None],
        nonnegative_targets=["q"],
    )
    assert selected["fit_statuses"] == ["invalid_formula", "no_formula"]
    assert selected["selected_slot"] is None
    assert (
        study.score_arm(tmp_path, **kwargs, held_rows=[event(2)])["status"]
        == "no_training_winner"
    )


def test_metric_uses_initial_change_for_late_recovery_and_earliest_peak_tie():
    curves = [
        {
            "trajectory_id": "train_9",
            "time": [0, 1, 2, 3, 4],
            "observed": {"q": [1, 3, 3, 1, 1]},
            "predicted": {"q": [2, 3, 4, 2, 2]},
        }
    ]
    result = study._diagnostics(curves, {"q": 2})
    target = result["targets"]["q"]
    assert target["nmse"] == np.mean([0, 1, 1, 1]) / 4
    assert target["mean_peak_absolute_time_error"] == 1
    assert target["mean_event_late_recovery_mae_scaled"] == 0
    assert target["observed_interior_peaks"] == target["comparable_peaks"] == 1
    with pytest.raises(ValueError, match="identifier"):
        study._safe_ids("good", "../bad")
