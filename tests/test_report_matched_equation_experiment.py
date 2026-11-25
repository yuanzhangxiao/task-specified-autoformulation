"""Count complete paired graphs and missing outcomes on their original denominators."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import report_matched_equation_experiment as study


def panel(graph, fold, arm, score):
    return {
        "panel_id": f"graph{graph}-fold{fold}",
        "inventory": f"id{graph}",
        "origin": "held",
        "graph_id": str(graph),
        "fold": fold,
        "arm": arm,
        "held_nmse": score,
        "training_nmse": score,
        "scored": score is not None,
        "held_nonnegative": score is not None and score < 1,
        "source": "a" if score is not None else None,
        "fit_statuses": ["fitted", "fitted"]
        if score is not None
        else ["no_formula"] * 2,
        "targets": {
            "x": {
                "nmse": score,
                "negative_samples": 0,
                "observed_interior_peaks": 2,
                "comparable_peaks": 1,
                "mean_peak_absolute_time_error": 5,
                "mean_event_late_recovery_mae_scaled": 0.4,
            }
        }
        if score is not None
        else {},
    }


def test_incomplete_graph_retains_coverage_and_paired_fold_scores():
    rows = [
        panel(graph, fold, arm, scores[fold])
        for graph, reference, other in [
            (0, (1.0, 0.4), (0.2, 0.4)),
            (1, (0.3, 0.7), (0.75, None)),
        ]
        for arm, scores in (("reference", reference), ("new", other))
        for fold in (0, 1)
    ]
    summary = study.aggregate(rows, ["reference", "new"])
    new = summary["arms"]["new"]
    assert new["slots_expected"] == 8
    assert new["fit_statuses"] == {"fitted": 6, "no_formula": 2}
    assert new["graphs_both_folds_scored"] == new["graphs_both_folds_nonnegative"] == 1
    assert new["panels_expected"] == 4 and new["panels_scored"] == 3
    assert new["median_graph_held_nmse"] == pytest.approx(0.3)
    assert new["targets"]["x"]["missed_interior_peaks"] == 3
    paired = summary["paired"][0]
    assert paired["graphs"]["paired"] == paired["graphs"]["alternative_wins"] == 1
    assert paired["panels"]["paired"] == 3
    assert (
        paired["panels"]["alternative_wins"]
        == paired["panels"]["ties"]
        == paired["panels"]["reference_wins"]
        == 1
    )
    with pytest.raises(ValueError, match="complementary folds"):
        study.aggregate(rows[:-1], ["reference", "new"])


def test_final_report_rejects_partial_experiment(tmp_path):
    sealed_write(
        tmp_path / "plan.json",
        {
            "arms": ["control"],
            "nonnegative_targets": ["a"],
            "panels": [
                {
                    "panel_id": f"g-f{fold}",
                    "inventory": "a",
                    "origin": "b",
                    "graph_id": "g",
                    "fold": fold,
                }
                for fold in (0, 1)
            ],
        },
    )
    with pytest.raises(ValueError, match="every arm/panel"):
        study.report(tmp_path)
    result = study.report(tmp_path, partial=True)
    assert result["arms"]["control"]["fit_statuses"] == {"pending": 4}
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.partial.json").is_file()
