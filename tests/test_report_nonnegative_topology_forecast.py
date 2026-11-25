"""Check qualification, paired graph coverage, and insulin peak reporting."""

from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import report_nonnegative_topology_forecast as report
from scripts.report_nonnegative_topology_forecast import (
    _match_selection,
    _peaks,
    render,
    verify_matched_plans,
)
from tests.test_report_topology_forecast_match import _fixture


def _campaign(path: Path) -> dict:
    _, dev = _fixture(path)
    sealed_write(path / "development.json", dev)
    return sealed_read(path / "plan.json")


def test_compare_same_models_with_failed_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        report, "_replay_minima", lambda *a: {"2": {"I": 0}, "4": {"I": 0}}
    )
    old, new = tmp_path / "old", tmp_path / "new"
    _campaign(old)
    _campaign(new)
    html = render(old, new).read_text()
    assert "paired" in html.lower()
    assert "../old/fits/f0-1.json" in html
    assert (new / "insulin-t2_fixture.png").is_file()
    result = sealed_read(new / "comparison-summary.json")
    assert len(result["optimizer_controls"]) == 1
    assert result["optimizer_controls"][0]["reached_original"] is True
    groups = {g["arm"]: g for g in result["groups"]}
    a = groups["algorithm"]["constrained"]
    assert a["eligible_graphs"] == a["eligible_slots"] == 1
    assert a["median_common"] == 0.3
    assert a["qualified_finalists_scored"] == 0
    assert a["best_eligible_train"]["development_nmse"] == 0.4
    assert a["best_eligible_train"]["insulin_peaks"][0]["predicted"]["value"] == 2
    assert groups["20b_repair"]["constrained"]["median_common"] is None
    assert groups["20b_repair"]["constrained"]["best_eligible_train"] is None


def test_unmatched_formulas_budget_and_development_choice(tmp_path):
    plan = _campaign(tmp_path)
    other = deepcopy(plan)
    other["slots"][0]["rank"] = 2
    with pytest.raises(ValueError, match="Formula slots differ"):
        verify_matched_plans(plan, other)
    other = deepcopy(plan)
    other["fit_budget"]["starts"] = 3
    with pytest.raises(ValueError, match="Unequal fitting budgets"):
        verify_matched_plans(plan, other)
    selected = {"inventory": "i", "arm": "algorithm", "graph_id": "g", "rank": 0}
    with pytest.raises(ValueError, match="differs from its training selection"):
        _match_selection([selected], [{**selected, "rank": 1}])
    assert _peaks(
        {
            "time": [0, 1, 2],
            "observed": {"I": [0, 1, 1]},
            "predicted": {"I": [2, 0, 1]},
            "trajectory_id": "val_a",
        }
    )["observed"] == {"time": 1, "value": 1}


def test_optimizer_control_only_counts_verified_comparable_vectors(
    tmp_path, monkeypatch
):
    plan = _campaign(tmp_path)
    base = report._load(tmp_path, plan)
    trial = deepcopy(base)
    trial["receipts"]["f0-1"]["result"]["training_nmse"] += 0.2
    monkeypatch.setattr(
        report, "_replay_minima", lambda *a: {"2": {"I": 0}, "4": {"I": 1}}
    )
    controls = report.optimizer_control(plan, base, trial)
    assert [c["reached_original"] for c in controls] == [False]
    monkeypatch.setattr(
        report, "_replay_minima", lambda *a: {"2": {"I": -2}, "4": {"I": 1}}
    )
    controls = report.optimizer_control(plan, base, trial)
    assert [c["reached_original"] for c in controls] == [None]
