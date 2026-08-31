"""Tests for arm-preserving sealed search-ablation endpoint reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from autoformalism.rebuttal.search_integration_ablation import (
    FrozenSearchAblationSource,
)
from scripts.summarize_phase_b_search_ablation_evaluation import (
    build_report,
    write_report,
)


def _source(
    arm_id: Literal["paired_question_consensus", "no_judge"],
) -> FrozenSearchAblationSource:
    return FrozenSearchAblationSource(
        request_id=f"{arm_id}__benchmark__easy__rep0",
        method_label=f"autoformalism:{arm_id}",
        arm_id=arm_id,
        benchmark_id="benchmark",
        tier="easy",
        repetition=0,
        source_path="/frozen/summary.json",
        artifact_status="missing",
    )


def test_report_retains_each_arm_when_all_sources_fail_adaptation(
    tmp_path: Path,
) -> None:
    sources = (
        _source("paired_question_consensus"),
        _source("no_judge"),
    )
    outcomes = tuple(
        SourceAdapterOutcome(
            request_id=item.request_id,
            source_kind="autoformalism",
            source_path=item.source_path,
            status="failed",
            error_type="ValueError",
            error="fixture failure",
        )
        for item in sources
    )

    report = build_report(sources, outcomes, ())
    write_report(report, tmp_path)

    assert report["weighted_overall_score_defined"] is False
    assert {item["method_id"] for item in report["groups"]} == {
        "paired_question_consensus",
        "no_judge",
    }
    assert report["paired_trials"] == [
        {
            "benchmark_id": "benchmark",
            "tier": "easy",
            "repetition": 0,
            "both_sources_adapted": False,
            "judge_source_status": "failed",
            "no_judge_source_status": "failed",
            "judge_runtime_valid": None,
            "no_judge_runtime_valid": None,
            "judge_target_test_nmse": None,
            "no_judge_target_test_nmse": None,
            "judge_mechanism_compliance": None,
            "no_judge_mechanism_compliance": None,
        }
    ]
    assert (tmp_path / "search_ablation_endpoint_report.json").is_file()
    assert (tmp_path / "search_ablation_endpoint_report.md").is_file()
    assert (tmp_path / "search_ablation_endpoint_subjects.csv").is_file()
