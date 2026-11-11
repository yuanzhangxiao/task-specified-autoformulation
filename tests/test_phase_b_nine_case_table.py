"""The nine-case comparison table: failures count, pending tasks do not."""

from __future__ import annotations

import math

import pytest

from autoformalism.rebuttal import phase_b_nine_case_d3 as d3_family
from autoformalism.rebuttal import phase_b_nine_case_spec_sindy as spec_sindy
from autoformalism.rebuttal import phase_b_nine_case_table as tabulate

T1 = "phase_b_dalla_man_t1_canonical_named_easy"
CSTR = "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy"


def _d3(case: str, repetition: int, value: float | None, status: str = "complete"):
    rollout = (
        {"status": "complete", "normalized_mse": value}
        if value is not None
        else {"status": "rollout_failed"}
    )
    return {
        "protocol": d3_family.PROTOCOL,
        "benchmark_id": case,
        "repetition": repetition,
        "status": status,
        "evaluation": {"phase_b_rollout": rollout},
    }


def _sindy(case: str, repetition: int, value: float | None, status: str):
    row = {
        "protocol": spec_sindy.PROTOCOL,
        "benchmark_id": case,
        "repetition": repetition,
        "status": status,
    }
    if value is not None:
        row["validation_normalized_mse"] = value
    return row


def test_a_failed_rollout_counts_as_infinite_and_the_median_keeps_it():
    summary = {
        "rows": [
            _d3(T1, 2, 3.0),
            _d3(T1, 0, 1.0),
            _d3(T1, 1, None),
            _d3(CSTR, 0, None),
            _d3(CSTR, 1, None),
            _d3(CSTR, 2, 2.0),
        ]
    }
    result = tabulate.summarize_method(summary)
    assert result["per_case"][T1]["scores"] == [1.0, math.inf, 3.0]
    assert result["per_case"][T1]["median"] == 3.0
    assert result["per_case"][CSTR]["median"] == math.inf
    assert math.isnan(result["per_case"][CSTR]["deviation"])
    assert result["median"] == math.inf
    assert result["finite_share"] == pytest.approx(3 / 6)


def test_the_aggregate_is_the_median_over_cases_with_its_deviation():
    rows = [_d3(T1, k, v) for k, v in enumerate((1.0, 2.0, 3.0))]
    rows += [_d3(CSTR, k, v) for k, v in enumerate((5.0, 6.0, 7.0))]
    rows += [_d3("third", k, v) for k, v in enumerate((0.5, 0.5, 0.5))]
    result = tabulate.summarize_method({"rows": rows})
    assert result["median"] == 2.0
    assert result["deviation"] == 1.5
    assert result["finite_share"] == 1.0


def test_a_pending_task_leaves_the_method_incomplete_not_failed():
    rows = [_sindy(T1, 0, 1.7, "complete"), _sindy(T1, 1, None, "pending")]
    rows.append(_sindy(CSTR, 0, None, "rollout_failed"))
    result = tabulate.summarize_method({"rows": rows})
    assert result["per_case"][T1]["scores"] == [1.7, None]
    assert result["per_case"][T1]["complete"] is False
    assert result["per_case"][CSTR]["median"] == math.inf
    assert result["complete"] is False
    assert result["median"] is None


def test_the_table_places_literature_models_by_case_and_marks_the_rest():
    literature = {
        "rows": [
            {
                "benchmark_id": T1,
                "model": "dm07_t1_easy",
                "label": "x",
                "validation_nmse": 0.003,
                "unfitted_validation_nmse": 0.2,
                "complete": True,
            },
            {
                "benchmark_id": CSTR,
                "model": "cstr",
                "label": "y",
                "validation_nmse": 18.0,
                "unfitted_validation_nmse": None,
                "complete": True,
            },
        ]
    }
    run = {"rows": [_d3(T1, 0, 1.0), _d3(CSTR, 0, None)]}
    result = tabulate.table({"D3": run}, literature)
    assert result["methods"]["literature"]["median"] == pytest.approx(9.0015)
    text = tabulate.markdown(result)
    assert "| T1-easy canonical named | 1 [0] | 0.003 |" in text
    assert "| CSTR easy named | failed | 18 |" in text
    assert "| **Median over cases [MAD]** | **failed** (2 cases) |" in text
    assert "| Alien device easy functional | n/a | n/a |" in text
