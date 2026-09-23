"""Medians over the planned roster, not only over what happened to succeed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "roster",
    Path(__file__).resolve().parent.parent / "scripts" / "roster_median_report.py",
)
roster = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roster)


def test_failing_the_hard_cells_no_longer_improves_the_median() -> None:
    """The whole point: a gap must cost a method, not flatter it."""
    scored = [0.1, 0.2, 0.3, 0.4]
    assert roster.roster_median(scored, 4) == 0.2
    # the same four scores, but six conditions were planned: the two gaps rank
    # worst, so the middle of the roster sits further out
    assert roster.roster_median(scored, 6) == 0.3
    # once more than half the roster is unscored there is no value at the
    # median rank at all, which is a stronger statement than a large number
    assert roster.roster_median(scored, 12) is None
    assert roster.roster_median([], 12) is None


def test_both_figures_are_reported_with_their_denominators() -> None:
    """A reader must be able to see which convention produced which number."""
    rows = [
        {"method": "pysr", "status": "complete", "target_nmse": 1.0},
        {"method": "pysr", "status": "complete", "target_nmse": 3.0},
        {"method": "pysr", "status": "missing_timed_out"},
        {"method": "pysr", "status": "missing_timed_out"},
    ]
    summary = roster.collect(rows, {"pysr": 4})
    assert summary["pysr"]["scored"] == 2
    assert summary["pysr"]["unscored"] == 2
    assert summary["pysr"]["median_conditional_on_success"] == 2.0
    # ranking the two gaps worst moves the median outward
    assert summary["pysr"]["median_over_roster"] == 3.0
    assert summary["pysr"]["status_counts"]["missing_timed_out"] == 2


def test_a_score_is_found_under_any_of_the_names_used() -> None:
    """Different stages of this pipeline spell the endpoint differently."""
    for field in ("target_nmse", "normalized_mse", "nmse"):
        assert roster.score_of({field: 2.5}) == 2.5
    assert roster.score_of({"target_prediction": {"normalized_mse": 4.0}}) == 4.0
    assert roster.score_of({"status": "missing_timed_out"}) is None
    # a null score is absent, not zero
    assert roster.score_of({"target_nmse": None}) is None


def test_the_planned_count_comes_from_the_sealed_report(tmp_path: Path) -> None:
    """The denominator is what was planned, never what happened to arrive."""
    report = tmp_path / "external_baseline_report.json"
    report.write_text(
        json.dumps(
            {
                "by_method": {"sindy": {"planned": 120}, "pysr": {"planned": 120}},
                "rows": [{"method": "sindy", "status": "complete", "target_nmse": 9.0}],
            }
        )
    )
    payload = json.loads(report.read_text())
    summary = roster.collect(
        payload["rows"],
        {name: counts["planned"] for name, counts in payload["by_method"].items()},
    )
    assert summary["sindy"]["planned"] == 120
    assert summary["sindy"]["unscored"] == 119
    # one score out of 120 planned cannot reach the middle of the roster
    assert summary["sindy"]["median_over_roster"] is None
    # a method with no rows at all is still reported against its denominator
    assert summary["pysr"]["planned"] == 120 and summary["pysr"]["scored"] == 0
