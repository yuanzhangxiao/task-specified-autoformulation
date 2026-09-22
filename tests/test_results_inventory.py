"""Inventorying campaign roots written by differently shaped campaigns."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "inventory", Path(__file__).resolve().parent.parent
    / "scripts" / "inventory_phase_b_results.py"
)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def test_the_roster_median_ranks_missing_work_worst() -> None:
    """A median over successes alone rewards failing on the hard cells."""
    scored = [0.1, 0.2, 0.3]
    assert inventory.roster_median(scored, 3) == 0.2
    # two of five planned never produced a score: the median moves outward
    assert inventory.roster_median(scored, 5) == 0.3
    # nothing scored at all is not a score of zero
    assert inventory.roster_median([], 5) is None


def test_a_campaign_root_reports_coverage_and_frozen_models(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "results" / "0").mkdir(parents=True)
    (root / "results" / "1").mkdir(parents=True)
    (root / "results" / "0" / "result.json").write_text(
        json.dumps({"status": "complete", "benchmark_id": "a", "normalized_mse": 0.5})
    )
    (root / "results" / "0" / "native-selection.json").write_text("{}")
    (root / "results" / "1" / "result.json").write_text(
        json.dumps({"status": "inexpressible", "benchmark_id": "b"})
    )
    record = inventory.inventory_root(root)
    assert record["results"]["with_result_json"] == 2
    assert record["results"]["with_frozen_model"] == 1
    assert record["results"]["scored"] == 1
    assert record["results"]["status_counts"] == {"complete": 1, "inexpressible": 1}


def test_a_summary_is_grouped_by_whatever_the_campaign_calls_the_method(
    tmp_path: Path,
) -> None:
    """Campaigns label methods source_kind, method or cohort; all must group."""
    root = tmp_path / "eval"
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "saved-baseline-validation-1",
                "status": "complete",
                "rows": [
                    {"source_kind": "pysr", "status": "complete",
                     "benchmark_id": "a", "normalized_mse": 1.0},
                    {"source_kind": "pysr", "status": "source_unavailable",
                     "benchmark_id": "b"},
                    {"method": "sindy", "status": "complete",
                     "benchmark_id": "a", "normalized_mse": 2.0},
                ],
            }
        )
    )
    groups = inventory.inventory_root(root)["summary.json"]["groups"]
    assert groups["pysr"]["rows"] == 2 and groups["pysr"]["scored"] == 1
    # one of two planned is unscored, so the roster median is the worse value
    assert groups["pysr"]["median_over_roster"] == 1.0
    assert groups["sindy"]["scored"] == 1


def test_an_unreadable_or_missing_root_does_not_stop_the_sweep(
    tmp_path: Path,
) -> None:
    """One bad directory must not cost the inventory of every other."""
    assert inventory.inventory_root(tmp_path / "nope")["exists"] is False
    root = tmp_path / "broken"
    root.mkdir()
    (root / "summary.json").write_text("{not json")
    record = inventory.inventory_root(root)
    assert "__unreadable__" not in record  # scalar fields simply absent
    assert record["summary.json"]["sha256"]


@pytest.mark.parametrize("name", ["plan.json", "summary.json"])
def test_identity_fields_are_carried_through(tmp_path: Path, name: str) -> None:
    """A transferred inventory must still name the plan it came from."""
    root = tmp_path / name.split(".")[0]
    root.mkdir()
    (root / name).write_text(
        json.dumps({"plan_sha256": "d" * 64, "test_data_opened": False, "rows": []})
    )
    entry = inventory.inventory_root(root)[name]
    assert entry["plan_sha256"] == "d" * 64
    assert entry["test_data_opened"] is False
