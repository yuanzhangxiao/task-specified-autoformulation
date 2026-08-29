#!/usr/bin/env python3
"""Audit matched search execution and judge-call coverage without test data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.search_integration_ablation import (
    build_search_integration_tasks,
    load_search_integration_plan,
)


def collect_search_audit(plan_path: Path, search_root: Path) -> dict[str, object]:
    """Collect arm outcomes and validate the matched initial-request contract."""
    plan = load_search_integration_plan(plan_path)
    root = search_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    by_trial: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for task in build_search_integration_tasks(plan):
        run = (
            root
            / "searches"
            / task.arm_id
            / "runs"
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        summary_path = run / "summary.json"
        proposer_events = _read_events(run / "proposer_events.jsonl")
        hybrid_events = _read_events(run / "hybrid_pair_events.jsonl")
        if not task.use_judge and hybrid_events:
            raise ValueError(f"no-judge arm emitted hybrid judge events: {run}")
        proposer_successes = [
            item
            for item in proposer_events
            if item.get("event") == "llm_response" and item.get("role") == "proposer"
        ]
        judge_successes = [
            item for item in hybrid_events if item.get("event") == "llm_response"
        ]
        rounds = sorted((run / "checkpoints").glob("round_*.json"))
        challenge_count = 0
        valid_round_count = 0
        for path in rounds:
            payload = _read_object(path)
            if payload.get("stage") == "complete" and payload.get("valid") is True:
                valid_round_count += 1
            if payload.get("incumbent_challenge") is not None:
                challenge_count += 1
        summary = _read_object(summary_path) if summary_path.is_file() else None
        row = {
            "arm_id": task.arm_id,
            "benchmark_id": task.benchmark_id,
            "tier": task.tier,
            "repetition": task.repetition,
            "source_complete": summary is not None,
            "selection_policy": task.selection_policy,
            "selected_candidate_id": (
                None
                if summary is None
                else _selected_candidate_id(summary.get("selected_candidate"))
            ),
            "selection_validation_normalized_mse": (
                None
                if summary is None
                else summary.get("selection_validation_normalized_mse")
            ),
            "attempted_round_count": len(rounds),
            "valid_round_count": valid_round_count,
            "challenge_count": challenge_count,
            "proposer_response_count": len(proposer_successes),
            "proposer_cache_hit_count": sum(
                item.get("cache_hit") is True for item in proposer_successes
            ),
            "initial_proposer_request_hash": (
                None
                if not proposer_successes
                else proposer_successes[0].get("request_hash")
            ),
            "initial_proposer_cache_hit": (
                None
                if not proposer_successes
                else proposer_successes[0].get("cache_hit")
            ),
            "judge_stage_response_count": len(judge_successes),
            "judge_stage_cache_hit_count": sum(
                item.get("cache_hit") is True for item in judge_successes
            ),
            "judge_failure_event_count": sum(
                item.get("event") == "llm_failure" for item in hybrid_events
            ),
        }
        rows.append(row)
        by_trial[(task.benchmark_id, task.tier, task.repetition)].append(row)

    matched_rows = []
    for key, trial_rows in sorted(by_trial.items()):
        by_arm = {str(item["arm_id"]): item for item in trial_rows}
        judge = by_arm["paired_question_consensus"]
        no_judge = by_arm["no_judge"]
        hashes_available = all(
            item["initial_proposer_request_hash"] is not None
            for item in (judge, no_judge)
        )
        request_match = (
            hashes_available
            and judge["initial_proposer_request_hash"]
            == no_judge["initial_proposer_request_hash"]
        )
        if hashes_available and not request_match:
            raise ValueError(f"initial proposer request differs across arms: {key}")
        if request_match and no_judge["initial_proposer_cache_hit"] is not True:
            raise ValueError(f"no-judge arm did not reuse the initial request: {key}")
        matched_rows.append(
            {
                "benchmark_id": key[0],
                "tier": key[1],
                "repetition": key[2],
                "both_sources_complete": (
                    judge["source_complete"] and no_judge["source_complete"]
                ),
                "initial_request_comparable": request_match,
            }
        )
    return {
        "schema_version": "phase-b-search-integration-ablation-audit-1",
        "development_only": True,
        "test_data_opened": False,
        "task_count": len(rows),
        "matched_trial_count": len(matched_rows),
        "source_completion_count": sum(item["source_complete"] for item in rows),
        "matched_source_completion_count": sum(
            item["both_sources_complete"] for item in matched_rows
        ),
        "initial_request_comparable_count": sum(
            item["initial_request_comparable"] for item in matched_rows
        ),
        "judge_stage_response_count": sum(
            int(item["judge_stage_response_count"]) for item in rows
        ),
        "judge_failure_event_count": sum(
            int(item["judge_failure_event_count"]) for item in rows
        ),
        "tasks": rows,
        "matched_trials": matched_rows,
    }


def write_search_audit(report: dict[str, object], output_root: Path) -> None:
    """Write machine-readable, tabular, and concise Markdown audit outputs."""
    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "search_integration_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["tasks"]
    assert isinstance(rows, list)
    with (output / "search_integration_tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(rows[0]) if rows else ["arm_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Search integration ablation audit",
        "",
        "This development audit uses no test or private benchmark data.",
        "",
        f"- planned tasks: {report['task_count']}",
        f"- completed sources: {report['source_completion_count']}",
        f"- matched trials complete: {report['matched_source_completion_count']}",
        f"- matched initial requests: {report['initial_request_comparable_count']}",
        f"- successful judge stages: {report['judge_stage_response_count']}",
        f"- judge failure events: {report['judge_failure_event_count']}",
    ]
    (output / "search_integration_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _selected_candidate_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    identifier = value.get("candidate_id")
    return str(identifier) if identifier is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = collect_search_audit(args.plan, args.search_root)
    write_search_audit(report, args.output_root)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"tasks", "matched_trials"}
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
