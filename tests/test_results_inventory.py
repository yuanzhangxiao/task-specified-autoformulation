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


def test_a_frozen_test_evaluation_root_is_discovered(tmp_path: Path) -> None:
    """It writes none of plan.json, summary.json or results/.

    A scan that looked only for those reported the absence of any test
    evaluation as a fact, when the instrument simply could not see one.
    """
    root = tmp_path / "external-baseline-test-v1"
    root.mkdir()
    (root / "external_baseline_freeze.json").write_text(
        json.dumps({"execution_authorized": True, "planned": 480})
    )
    (root / "external_baseline_roster.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"method": "pysr", "status": "complete",
                 "benchmark_id": "a", "normalized_mse": 1.0},
                {"method": "pysr", "status": "source_unavailable",
                 "benchmark_id": "b"},
                {"method": "d3", "status": "complete",
                 "benchmark_id": "a", "normalized_mse": 2.5},
            ]
        )
    )
    assert any(
        (root / marker).exists() for marker in inventory.ROOT_MARKERS
    ), "a scan would skip this directory entirely"

    record = inventory.inventory_root(root)
    assert record["external_baseline_freeze.json"]["execution_authorized"] is True
    groups = record["external_baseline_roster.jsonl"]["groups"]
    assert groups["pysr"]["rows"] == 2 and groups["pysr"]["scored"] == 1
    assert groups["d3"]["scored"] == 1


def test_a_roster_line_that_will_not_parse_does_not_lose_the_rest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    root.mkdir()
    (root / "external_baseline_roster.jsonl").write_text(
        '{"method": "pysr", "status": "complete", "normalized_mse": 1.0}\n'
        "{ not json\n"
        '{"method": "pysr", "status": "complete", "normalized_mse": 3.0}\n'
    )
    entry = inventory.inventory_root(root)["external_baseline_roster.jsonl"]
    assert entry["lines"] == 3
    assert entry["groups"]["pysr"]["scored"] == 2


def test_artifacts_in_subdirectories_are_found(tmp_path: Path) -> None:
    """The frozen evaluation writes into frozen/ and final-evaluation/.

    Looking only at the top level found nothing and reported that as an
    absence, twice: first because the filenames were unknown, then because
    the depth was.
    """
    root = tmp_path / "external-baseline-evaluation-v1"
    (root / "frozen").mkdir(parents=True)
    (root / "final-evaluation").mkdir()
    (root / "frozen" / "execution_record.json").write_text(
        json.dumps({"execution_authorized": True})
    )
    (root / "frozen" / "external_baseline_roster.jsonl").write_text(
        json.dumps({"method": "sindy", "status": "complete",
                    "benchmark_id": "a", "normalized_mse": 1.5})
    )
    (root / "final-evaluation" / "external_baseline_report.json").write_text(
        json.dumps({"status": "complete"})
    )

    record = inventory.inventory_root(root)
    assert record["execution_record.json"]["path"] == "frozen/execution_record.json"
    assert record["execution_record.json"]["execution_authorized"] is True
    assert record["external_baseline_report.json"]["status"] == "complete"
    roster = record["external_baseline_roster.jsonl"]
    assert roster["path"] == "frozen/external_baseline_roster.jsonl"
    assert roster["groups"]["sindy"]["scored"] == 1


def test_a_scan_discovers_a_root_whose_artifacts_are_nested(tmp_path: Path) -> None:
    """--scan must reach the same roots inventory_root can describe."""
    root = tmp_path / "external-baseline-evaluation-v1"
    (root / "frozen").mkdir(parents=True)
    (root / "frozen" / "external_baseline_freeze.json").write_text("{}")
    discovered = [
        child
        for child in sorted(tmp_path.iterdir())
        if child.is_dir()
        and any(
            (child / name).exists()
            or any(
                (grandchild / name).exists()
                for grandchild in child.iterdir()
                if grandchild.is_dir()
            )
            for name in inventory.ROOT_MARKERS
        )
    ]
    assert discovered == [root]
