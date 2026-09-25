"""Reporting compliance and complexity, not only predictive error.

Both are computed per subject by the frozen evaluator and were never
surfaced: the report carried target NMSE alone. Mechanism compliance is the
point of a task-specified benchmark, and complexity is what a parsimony claim
rests on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "summarize",
    Path(__file__).resolve().parent.parent
    / "scripts" / "summarize_external_baseline_evaluation.py",
)
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


def _row(**overrides) -> dict:
    base = {
        "evaluation": "x", "request_id": "r", "method_id": "m",
        "benchmark_id": "c", "tier": "easy", "repetition": 0,
        "state": "evaluated", "terminal_status": "", "reason": "",
        "runtime_valid": True, "target_nmse": 1.0, "target_status": "available",
        "target_message": "", "evaluation_protocol": None,
        "mechanism_compliance": 1.0, "mechanism_coverage": 1.0,
        "mechanism_status": "available", "complexity_states": 2,
        "complexity_latent_states": 0, "complexity_parameters": 3,
        "complexity_terms": 5,
    }
    base.update(overrides)
    return base


def test_compliance_ranks_a_missing_model_at_the_bottom() -> None:
    """Higher is better, so the roster convention is mirrored.

    Using the NMSE helper unchanged would rank an absent model *best*, which
    is the opposite of what the convention exists to prevent.
    """
    assert summarize._worst_first_median([0.9, 0.5, 0.1], 3) == 0.5
    # two of five planned produced nothing: the median moves down, not up
    assert summarize._worst_first_median([0.9, 0.5, 0.1], 5) == 0.1
    # past halfway there is no value at the median rank
    assert summarize._worst_first_median([0.9], 5) is None
    assert summarize._worst_first_median([], 5) is None


def test_complexity_is_reported_over_what_exists() -> None:
    """A missing model is not infinitely complex, so it is not ranked worst."""
    subset = [
        _row(complexity_terms=4, complexity_parameters=2),
        _row(complexity_terms=10, complexity_parameters=6),
        _row(complexity_terms=None, complexity_parameters=None,
             complexity_states=None, complexity_latent_states=None),
    ]
    got = summarize._complexity_summary(subset)
    assert got["complexity_measured"] == 2
    assert got["complexity_terms_median"] == 7.0
    assert got["complexity_parameters_median"] == 4.0


def test_a_method_with_no_models_reports_nothing_rather_than_zero() -> None:
    """Zero complexity would rank an absent method as the most parsimonious."""
    got = summarize._complexity_summary(
        [_row(complexity_terms=None, complexity_parameters=None,
              complexity_states=None, complexity_latent_states=None)]
    )
    assert got["complexity_measured"] == 0
    assert got["complexity_terms_median"] is None


def test_the_report_carries_compliance_and_complexity_per_method() -> None:
    rows = [
        _row(method_id="a", target_nmse=1.0, mechanism_compliance=0.8,
             complexity_terms=4),
        _row(method_id="a", target_nmse=3.0, mechanism_compliance=0.4,
             complexity_terms=8),
        _row(method_id="b", target_nmse=2.0, mechanism_compliance=1.0,
             complexity_terms=2),
    ]
    report = summarize.summarize(rows)
    first = report["by_method"]["a"]
    assert first["mechanism_compliance_scored"] == 2
    assert first["mechanism_compliance_median_full_roster"] == 0.8
    assert first["complexity_terms_median"] == 6.0
    assert first["complexity_measured"] == 2
    assert report["by_method"]["b"]["mechanism_compliance_median_full_roster"] == 1.0
